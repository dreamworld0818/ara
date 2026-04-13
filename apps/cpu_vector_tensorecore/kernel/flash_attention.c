/**
 * Flash-Attention：TensorCore GEMM 封装 + softmax。
 *
 * Softmax 与 apps/softmax 一致：对每一行将 logits 填入连续 float 缓冲区，调用同目录
 * `kernel/softmax.c` 中的 `fa_softmax_vec`（与 apps/softmax 的 softmax_vec 同算法：RVV float32、
 * `__exp_2xf32`、vfdiv），再量化为 int8 写入 P（L2 上 A 块布局，供 GEMM2 读取）。
 */

#include "flash_attention.h"
#include "softmax.h"

#include <math.h>
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

/** 行缓冲：与 apps/softmax 一致对齐到 4*NR_LANES，便于向量加载 */
static float s_fa_row_in[FA_SOFTMAX_ROW_MAX]
    __attribute__((aligned(4 * NR_LANES)));
static float s_fa_row_out[FA_SOFTMAX_ROW_MAX]
    __attribute__((aligned(4 * NR_LANES)));

/** 调试：一行 float 的 min / max / sum 与首部若干元素。
 *  使用 %f：apps/common 的 tiny printf 仅支持浮点的 %f，不支持 %g（会原样打出 “g” 且不取参数）。
 */
static void fa_dbg_print_frow(const char *stage, uint64_t row_i, uint64_t n,
                              const float *row) {
  if (n == 0U)
    return;
  float mn = row[0], mx = row[0];
  double sum = 0.0;
  for (uint64_t j = 0; j < n; j++) {
    float v = row[j];
    if (v < mn)
      mn = v;
    if (v > mx)
      mx = v;
    sum += (double)v;
  }
  const uint64_t head = n < 4U ? n : 4U;
  printf("[FA][softmax][i=%llu][%s] n=%llu min=%.6f max=%.6f sum=%.8f | head:",
         (unsigned long long)row_i, stage, (unsigned long long)n,
         (double)mn, (double)mx, sum);
  for (uint64_t j = 0; j < head; j++)
    printf("%s%.6f", j ? "," : " ", (double)row[j]);
  printf("\n");
}

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
/**
 * S_scores: GEMM1 的输出，D 块布局
 * M: 矩阵 A 的行数
 * N: 矩阵 A 的列数
 * K_head: 每个 head 的特征维
 * causal: 是否使用因果掩码
 * P_packed: 输出矩阵 P 的指针
 * causal: 是否使用因果掩码
 */
void fa_softmax_scores_to_p(const int32_t *S_score, uint64_t M, uint64_t N,
                            uint64_t K_head, int8_t *P_packed, uint32_t causal) {
  const float inv_sqrt_k = 1.0f / sqrtf((float)K_head);
  const float mask_neg = -1.0e9f;

  if (N > FA_SOFTMAX_ROW_MAX || M > FA_SOFTMAX_ROW_MAX) {
    printf("[FA][softmax] 错误：M 或 N 超过 FA_SOFTMAX_ROW_MAX，跳过\n");
    return;
  }

  printf("[FA][softmax] 开始：与 apps/softmax 相同 fa_softmax_vec(in,out,N,1)，"
         "M=%llu N=%llu K_head=%llu，P 写入 A 块布局（GEMM2 读 L2）\n",
         (unsigned long long)M, (unsigned long long)N,
         (unsigned long long)K_head);

  const uint32_t use_causal = (causal != 0U && M == N) ? 1U : 0U;

  for (uint64_t i = 0; i < M; i++) {
    float *row_in = s_fa_row_in;
    float *row_out = s_fa_row_out;

    printf("[FA][softmax][i=%llu] ---------- 行开始 ----------\n",
           (unsigned long long)i);

    /* 阶段1：S_score（D 块布局）→ 连续 logits：缩放 + 可选 causal */
    for (uint64_t j = 0; j < N; j++) {
      uint64_t ix = tc_d_block_idx(i, j, N);
      float v = (float)S_score[ix] * inv_sqrt_k;
      if (use_causal && j > i)
        v += mask_neg;
      row_in[j] = v;
    }
    fa_dbg_print_frow("1 logits(row_in)", i, N, row_in);

    /* 阶段2：与 softmax/main.c 一致 fa_softmax_vec(in,out,N,1) → 行概率 */
    printf("[FA][softmax][i=%llu][2 fa_softmax_vec] 调用 N=%llu inner=1\n",
           (unsigned long long)i, (unsigned long long)N);
    fa_softmax_vec(row_in, row_out, N, 1ULL);
    fa_dbg_print_frow("3 prob(row_out)", i, N, row_out);

    /* 阶段3：prob → int8，scatter 到 P（A 块布局） */
    int32_t qmin = 127;
    int32_t qmax = -128;
    int32_t head_q[4];
    const uint64_t head_n = N < 4U ? N : 4U;
    for (uint64_t j = 0; j < N; j++) {
      float p = row_out[j] * 127.0f;
      int32_t q = (int32_t)(p + 0.5f);
      if (q > 127)
        q = 127;
      if (q < -128)
        q = -128;
      if (q < qmin)
        qmin = q;
      if (q > qmax)
        qmax = q;
      if (j < head_n)
        head_q[j] = q;
      uint64_t pi = tc_a_block_idx(i, j, N);
      P_packed[pi] = (int8_t)q;
    }
    printf("[FA][softmax][i=%llu][4 quant->P] qmin=%ld qmax=%ld | head int8:",
           (unsigned long long)i, (long)qmin, (long)qmax);
    for (uint64_t j = 0; j < head_n; j++)
      printf("%s%ld", j ? "," : " ", (long)head_q[j]);
    printf("\n");
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
