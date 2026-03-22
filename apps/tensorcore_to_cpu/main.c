#include <string.h>

#include "kernel/gemm.h"
#include "runtime.h"
#include "util.h"

#if defined(SPIKE) || defined(ARA_LINUX)
#include <stdio.h>
#else
#include "printf.h"
#endif

/* Input matrices (A, B, Din) go only to this file, not to the terminal. */
#if defined(SPIKE) || defined(ARA_LINUX)
#define TC_INPUT_LOG_PATH "/home/zhoujinwei/pulp/ara/apps/tensorcore_to_cpu/main.log"
#endif

// Dout = A*B + Din, A[M×K], B[K×N], Din/Dout/G[M×N]
// 数据段由 gen_data.py / 链接脚本决定；基址由 TC_*_IN_L2 宏决定（未定义时默认 1，即 Case 1 全 L2）
extern uint64_t M;
extern uint64_t N;
extern uint64_t K;
extern int8_t A[];
extern int8_t B[];
extern int32_t Din[];
extern int32_t Dout[];
extern int32_t G[];

#ifndef TC_A_IN_L2
#define TC_A_IN_L2 1
#endif
#ifndef TC_B_IN_L2
#define TC_B_IN_L2 1
#endif
#ifndef TC_DIN_IN_L2
#define TC_DIN_IN_L2 1
#endif

// 块大小（字节）：A/B 单块 64*64*1=4096，Din/Dout 单块 64*64*4=16384
#define BLOCK_BYTES_A  4096
#define BLOCK_BYTES_D  16384

/* Din/Dout/G in L2 follow TensorCore block layout (gen_data.py): n-major, m-minor.
 * Element (m,n) => flat index idx = n * M + m — NOT C row-major (m*N+n). */
static inline uint64_t tc_d_idx(uint64_t m, uint64_t n, uint64_t Mrows) {
  return n * Mrows + m;
}

#if defined(SPIKE) || defined(ARA_LINUX)
static void print_matrix_i32(FILE *out, const char *tag, const int32_t *mat,
                             uint64_t rows, uint64_t cols) {
  fprintf(out, "BEGIN_%s rows=%llu cols=%llu layout=n-major,m-minor idx=n*M+m "
               "(print order: outer n, inner m)\n",
          tag, (unsigned long long)rows, (unsigned long long)cols);
  for (uint64_t n = 0; n < cols; n++) {
    for (uint64_t m = 0; m < rows; m++) {
      uint64_t idx = tc_d_idx(m, n, rows);
      fprintf(out, "m=%llu n=%llu idx=%llu val=%ld\n", (unsigned long long)m,
              (unsigned long long)n, (unsigned long long)idx, (long)mat[idx]);
    }
  }
  fprintf(out, "END_%s\n", tag);
}
#endif

/* Compare Dout vs G; only print entries that differ. Returns -1 if all match,
 * else TC flat idx (n*M+m) of first mismatch. */
static int verify_dout_print_mismatches(int32_t *dout, int32_t *gold, uint64_t rows,
                                      uint64_t cols) {
  int first_mismatch_idx = -1;
  for (uint64_t n = 0; n < cols; n++) {
    for (uint64_t m = 0; m < rows; m++) {
      uint64_t idx = tc_d_idx(m, n, rows);
      if (dout[idx] != gold[idx]) {
        printf("[SW][MISMATCH] m=%llu n=%llu idx=%llu Dout=%ld G=%ld\n",
               (unsigned long long)m, (unsigned long long)n,
               (unsigned long long)idx, (long)dout[idx], (long)gold[idx]);
        if (first_mismatch_idx < 0) first_mismatch_idx = (int)idx;
      }
    }
  }
  return first_mismatch_idx;
}

/* UART dump: outer n (0..cols-1), inner m (0..rows-1); idx = n*M+m. */
static void print_matrix_i32_terminal(const char *tag, const int32_t *mat,
                                      uint64_t rows, uint64_t cols) {
  printf("BEGIN_%s rows=%llu cols=%llu (TC layout idx=n*M+m; print outer n, inner "
         "m)\n",
         tag, (unsigned long long)rows, (unsigned long long)cols);
  for (uint64_t n = 0; n < cols; n++) {
    for (uint64_t m = 0; m < rows; m++) {
      uint64_t idx = tc_d_idx(m, n, rows);
      printf("m=%llu n=%llu idx=%llu val=%ld\n", (unsigned long long)m,
             (unsigned long long)n, (unsigned long long)idx, (long)mat[idx]);
    }
  }
  printf("END_%s\n", tag);
}

#if defined(SPIKE) || defined(ARA_LINUX)
static void print_matrix_i8_k_major_m_minor(FILE *out, const char *tag,
                                            const int8_t *mat, uint64_t rows_m,
                                            uint64_t cols_k) {
  // Flat layout is [k-major][m-minor], i.e., idx = k*rows_m + m.
  fprintf(out, "BEGIN_%s rows=%llu cols=%llu layout=k-major,m-minor\n", tag,
          (unsigned long long)rows_m, (unsigned long long)cols_k);
  for (uint64_t m = 0; m < rows_m; m++) {
    for (uint64_t k = 0; k < cols_k; k++) {
      uint64_t idx = k * rows_m + m;
      fprintf(out, "m=%llu k=%llu idx=%llu val=%d\n", (unsigned long long)m,
              (unsigned long long)k, (unsigned long long)idx, (int)mat[idx]);
    }
  }
  fprintf(out, "END_%s\n", tag);
}
#endif

int main(void) {
  uint32_t M_block = (uint32_t)((M + 63) / 64);
  uint32_t N_block = (uint32_t)((N + 63) / 64);
  uint32_t K_block = (uint32_t)((K + 63) / 64);

  /* A, B, Din: only written to main.log (host path), not printed on terminal. */
// #if defined(SPIKE) || defined(ARA_LINUX)
//   {
//     FILE *inlog = fopen(TC_INPUT_LOG_PATH, "w");
//     if (inlog) {
//       print_matrix_i8_k_major_m_minor(inlog, "A", A, M, K);
//       print_matrix_i8_k_major_m_minor(inlog, "B", B, N, K);
//       print_matrix_i32(inlog, "DIN", Din, M, N);
//       fclose(inlog);
//     }
//   }
// #endif

  // 按 FLOW_README：base_addr 为相对所在区域的字节偏移（L2 或 RRAM）
  mma_instruction_t inst = {0};

  inst.fields.Din.info.base_addr    = (uint32_t)(uintptr_t)Din >> 8;
  inst.fields.Din.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Din.info.stride_minor = BLOCK_BYTES_D * N_block;

  // // 软件侧打印 Din 地址（原始指针 + 指令中写入的 base_addr 字段）
  // {
  //   uint64_t din_ptr = (uint64_t)(uintptr_t)Din;
  //   uint64_t din_base_addr_field = (uint64_t)inst.fields.Din.info.base_addr;
  //   printf("[DEBUG] Din ptr = 0x%llx (bin): ", (unsigned long long)din_ptr);
  //   print_bin_u64_grouped(din_ptr, 64, 8);
  //   TC_PUTCHAR('\n');
  //   printf("[DEBUG] Din base_addr_field = 0x%llx (bin): ",
  //          (unsigned long long)din_base_addr_field);
  //   print_bin_u64_grouped(din_base_addr_field, 32, 8);
  //   TC_PUTCHAR('\n');
  // }

  inst.fields.Dout.info.base_addr    = (uint32_t)(uintptr_t)Dout >> 8; /* Dout 始终在 L2 */
  inst.fields.Dout.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Dout.info.stride_minor = BLOCK_BYTES_D * N_block;

  inst.fields.A.info.base_addr    = (uint32_t)(uintptr_t)A >> 8;
  inst.fields.A.info.stride_major = BLOCK_BYTES_A;
  inst.fields.A.info.stride_minor = BLOCK_BYTES_A * K_block;

  inst.fields.B.info.base_addr    = (uint32_t)(uintptr_t)B >> 8;
  inst.fields.B.info.stride_major = BLOCK_BYTES_A;
  inst.fields.B.info.stride_minor = BLOCK_BYTES_A * K_block;

  inst.fields.mma_meta.info.M = M_block;
  inst.fields.mma_meta.info.N = N_block;
  inst.fields.mma_meta.info.K = K_block;
  inst.fields.mma_meta.info.if_B_transpose = 0;
  inst.fields.mma_meta.info.if_A_transpose = 0;
  inst.fields.mma_meta.info.instruction_type = 0;

  send_tensorcore_instruction(&inst);
  printf("[DEBUG] send_tensorcore_instruction completed.\n");

  // 轮询：TensorCore 完成时 slv_register_output_data 置 1
  uint64_t tc_state;
  printf("[DEBUG] load_tensorcore_state: TensorCore executing, polling...\n");
  while ((tc_state = load_tensorcore_state()) == 0) {
    /* spin */
  }
  printf("[DEBUG] load_tensorcore_state completed.\n");
  printf("TensorCore state (after poll): %llu\n", (unsigned long long)tc_state);
  __sync_synchronize(); /* ensure DMA write to Dout is visible before CPU read */

  int first_bad = verify_dout_print_mismatches(Dout, G, M, N);
  /* Gold matrix for cross-check with RTL [TC][SA][DOUT_OUT] / software Dout. */
  // print_matrix_i32_terminal("G", G, M, N);
  // print_matrix_i32_terminal("Dout", Dout, M, N);
  if (first_bad < 0) {
    printf("PASS (matrix 64x64)\n");
    return 0;
  }
  printf("FAIL (matrix 64x64, first error index: %d)\n", first_bad);
  return 1;
}
