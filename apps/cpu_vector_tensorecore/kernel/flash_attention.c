/**
 * Flash-Attention：TensorCore GEMM 封装 + softmax。
 *
 * Softmax 与 apps/softmax 一致：对每一行将 logits 填入连续 float 缓冲区，调用与
 * `kernel/softmax.c` 中 `softmax_vec` 相同的 RVV float32 路径（`__exp_2xf32`、vfdiv），
 * 再量化为 int8 写入 P（L2 上 A 块布局，供 GEMM2 读取）。避免双精度 `__exp_1xf64`/`vfredosum`
 * 在部分仿真环境中卡死的问题。
 */

#include "flash_attention.h"

#include "../../softmax/lib/exp.h"

#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>

#if defined(SPIKE) || defined(ARA_LINUX)
#include <stdio.h>
#else
#include "printf.h"
#endif

#define BLOCK_BYTES_A  4096
#define BLOCK_BYTES_D  16384

/** softmax 单行最大长度；缓冲区用 static 放入 .bss，避免大数组占用栈导致裸机栈溢出 */
#define FA_SOFTMAX_ROW_MAX 1024

#ifndef NR_LANES
#define NR_LANES 4
#endif

/**
 * 与 apps/softmax/kernel/softmax.c 中 `softmax_vec` 等价（供本文件内联，避免链接 softmax 子目录）。
 * 对形状 [channels × innerSize] 沿 channels 维做 softmax；Flash-Attention 每行 softmax 使用
 * channels=N、innerSize=1，即一行 N 个 logits。
 */
static void fa_softmax_vec(const float *i, const float *o, uint64_t channels,
                           uint64_t innerSize) {
  volatile int temp;
  asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(temp));

  asm volatile("vmv.v.i  v0, 0");
  asm volatile("vmv.v.i  v8, 0");
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v24, 0");

  size_t avl = innerSize;
  size_t vl;

  float *_i = (float *)i;
  float *_o = (float *)o;
  float *__i = (float *)i;
  float *__o = (float *)o;

  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  for (vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = __riscv_vsetvl_e32m1(avl);

    max_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
    __i += innerSize;
    for (uint64_t ch = 1; ch < channels; ++ch) {
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      __i += innerSize;
      max_chunk_v = __riscv_vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    __i = _i;

    den_chunk_v = __riscv_vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < channels; ++ch) {
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      buf_chunk_v = __riscv_vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, buf_chunk_v, vl);
      den_chunk_v = __riscv_vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      __i += innerSize;
      __o += innerSize;
    }
    __i = _i;
    __o = _o;

    for (uint64_t ch = 0; ch < channels; ++ch) {
      num_chunk_v = __riscv_vle32_v_f32m1(__o, vl);
      res_chunk_v = __riscv_vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, res_chunk_v, vl);
      __o += innerSize;
    }
    _i += vl;
    _o += vl;
    __i = _i;
    __o = _o;
  }
}

/** 行缓冲：与 apps/softmax 一致对齐到 4*NR_LANES，便于向量加载 */
static float s_fa_row_in[FA_SOFTMAX_ROW_MAX]
    __attribute__((aligned(4 * NR_LANES)));
static float s_fa_row_out[FA_SOFTMAX_ROW_MAX]
    __attribute__((aligned(4 * NR_LANES)));

uint64_t tc_d_block_idx(uint64_t m, uint64_t n, uint64_t Ncols) {
  uint64_t mb = m / 64;
  uint64_t nb = n / 64;
  uint64_t Nb = (Ncols + 63) / 64;
  uint64_t nl = n % 64;
  uint64_t ml = m % 64;
  return (mb * Nb + nb) * (64 * 64) + nl * 64 + ml;
}

uint64_t tc_a_block_idx(uint64_t m, uint64_t k, uint64_t Kdim) {
  uint64_t mb = m / 64;
  uint64_t kb = k / 64;
  uint64_t Kb = (Kdim + 63) / 64;
  uint64_t kl = k % 64;
  uint64_t ml = m % 64;
  return (mb * Kb + kb) * (64 * 64) + kl * 64 + ml;
}

void fa_softmax_scores_to_p(const int32_t *S_score, uint64_t M, uint64_t N,
                            uint64_t K_head, int8_t *P_packed, uint32_t causal) {
  const float inv_sqrt_k = 1.0f / sqrtf((float)K_head);
  const float mask_neg = -1.0e9f;

  if (N > FA_SOFTMAX_ROW_MAX || M > FA_SOFTMAX_ROW_MAX) {
    printf("[FA][softmax] 错误：M 或 N 超过 FA_SOFTMAX_ROW_MAX，跳过\n");
    return;
  }

  printf("[FA][softmax] 开始：与 apps/softmax 相同 softmax_vec(in,out,N,1)，"
         "M=%llu N=%llu K_head=%llu，P 写入 A 块布局（GEMM2 读 L2）\n",
         (unsigned long long)M, (unsigned long long)N,
         (unsigned long long)K_head);

  const uint32_t use_causal = (causal != 0U && M == N) ? 1U : 0U;

  for (uint64_t i = 0; i < M; i++) {
    float *row_in = s_fa_row_in;
    float *row_out = s_fa_row_out;

    for (uint64_t j = 0; j < N; j++) {
      uint64_t ix = tc_d_block_idx(i, j, N);
      float v = (float)S_score[ix] * inv_sqrt_k;
      if (use_causal && j > i)
        v += mask_neg;
      row_in[j] = v;
    }

    /* 与 softmax/main.c 中 softmax_vec 一致：沿通道维 N、innerSize=1 → 单行 softmax */
    fa_softmax_vec(row_in, row_out, N, 1ULL);

    for (uint64_t j = 0; j < N; j++) {
      float p = row_out[j] * 127.0f;
      int32_t q = (int32_t)(p + 0.5f);
      if (q > 127)
        q = 127;
      if (q < -128)
        q = -128;
      uint64_t pi = tc_a_block_idx(i, j, N);
      P_packed[pi] = (int8_t)q;
    }
  }

  printf("[FA][softmax] 完成：共 %llu 行，权值矩阵 P 已就绪\n",
         (unsigned long long)M);
}

void fa_send_gemm(const char *phase_label, uint64_t M, uint64_t N, uint64_t K,
                  const int8_t *A, const int8_t *B, const int32_t *Din,
                  int32_t *Dout) {
  uint32_t M_block = (uint32_t)((M + 63) / 64);
  uint32_t N_block = (uint32_t)((N + 63) / 64);
  uint32_t K_block = (uint32_t)((K + 63) / 64);

  /* 与 apps/tensorcore_to_cpu/main.c 相同：Dout = A*B + Din，A[M×K]，B[K×N] */
  mma_instruction_t inst = {0};

  inst.fields.Din.info.base_addr = (uint32_t)(uintptr_t)Din >> 8;
  inst.fields.Din.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Din.info.stride_minor = BLOCK_BYTES_D * N_block;

  inst.fields.Dout.info.base_addr = (uint32_t)(uintptr_t)Dout >> 8;
  inst.fields.Dout.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Dout.info.stride_minor = BLOCK_BYTES_D * N_block;

  inst.fields.A.info.base_addr = (uint32_t)(uintptr_t)A >> 8;
  inst.fields.A.info.stride_major = BLOCK_BYTES_A;
  inst.fields.A.info.stride_minor = BLOCK_BYTES_A * K_block;

  /* B：块 (Kb,Nb,64,64)，n 为内层；stride_minor=一块；stride_major=N_block 块沿 n 再走 k */
  inst.fields.B.info.base_addr = (uint32_t)(uintptr_t)B >> 8;
  inst.fields.B.info.stride_major = BLOCK_BYTES_A * N_block;
  inst.fields.B.info.stride_minor = BLOCK_BYTES_A;

  inst.fields.mma_meta.info.M = M_block;
  inst.fields.mma_meta.info.N = N_block;
  inst.fields.mma_meta.info.K = K_block;
  inst.fields.mma_meta.info.if_B_transpose = 0;
  inst.fields.mma_meta.info.if_A_transpose = 0;
  inst.fields.mma_meta.info.instruction_type = 0;

  printf("[FA][%s] send_tensorcore_instruction: M_block=%u N_block=%u K_block=%u\n",
         phase_label, (unsigned)M_block, (unsigned)N_block, (unsigned)K_block);
  send_tensorcore_instruction(&inst);
  printf("[FA][%s] send_tensorcore_instruction 完成，轮询 TensorCore 状态...\n",
         phase_label);

  uint64_t tc_state;
  while ((tc_state = load_tensorcore_state()) == 0) {
  }
  printf("[FA][%s] TensorCore 完成，state=%llu\n", phase_label,
         (unsigned long long)tc_state);
  __sync_synchronize();
}

int fa_verify_i32_block(const int32_t *got, const int32_t *gold, uint64_t rows,
                        uint64_t cols) {
  int first = -1;
  for (uint64_t n = 0; n < cols; n++) {
    for (uint64_t m = 0; m < rows; m++) {
      uint64_t idx = tc_d_block_idx(m, n, cols);
      if (got[idx] != gold[idx]) {
        if (first < 0)
          first = (int)idx;
      }
    }
  }
  return first;
}

int fa_verify_i8_a_block(const int8_t *got, const int8_t *gold, uint64_t rows,
                         uint64_t Kdim) {
  int first = -1;
  for (uint64_t k = 0; k < Kdim; k++) {
    for (uint64_t m = 0; m < rows; m++) {
      uint64_t idx = tc_a_block_idx(m, k, Kdim);
      if (got[idx] != gold[idx]) {
        if (first < 0)
          first = (int)idx;
      }
    }
  }
  return first;
}
