#include <string.h>

#include "kernel/gemm.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Character output abstraction:
// - Linux/Spike: use libc putchar
// - Baremetal: use Ara's _putchar from printf.h
#if defined(SPIKE) || defined(ARA_LINUX)
#define TC_PUTCHAR(c) putchar((c))
#else
#define TC_PUTCHAR(c) _putchar((c))
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

static void print_matrix_i32(const char *tag, const int32_t *mat, uint64_t rows,
                             uint64_t cols) {
  // Fixed, easy-to-parse format:
  // BEGIN_<TAG> rows=<r> cols=<c>
  // i=<row> j=<col> idx=<flat_idx> val=<int32>
  // ...
  // END_<TAG>
  printf("BEGIN_%s rows=%llu cols=%llu\n", tag, (unsigned long long)rows,
         (unsigned long long)cols);
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      uint64_t idx = i * cols + j;  
      printf("i=%llu j=%llu idx=%llu val=%ld\n", (unsigned long long)i,
             (unsigned long long)j, (unsigned long long)idx,
             (long)mat[idx]);
    }
  }
  printf("END_%s\n", tag);
}

static void print_bin_u64_grouped(uint64_t v, int bits, int group_bits) {
  if (bits <= 0 || bits > 64) bits = 64;
  if (group_bits <= 0) group_bits = 8;
  for (int i = bits - 1; i >= 0; --i) {
    TC_PUTCHAR(((v >> i) & 1ULL) ? '1' : '0');
    if (i != 0 && (i % group_bits) == 0) TC_PUTCHAR('_');
  }
}

static int verify_matrix(int32_t *result, int32_t *gold, uint64_t rows, uint64_t cols) {
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      uint64_t idx = i * cols + j;
      if (result[idx] != gold[idx]) {
        return (int)(i == 0 && j == 0 ? -1 : idx);
      }
    }
  }
  return 0;
}

int main(void) {
  uint32_t M_block = (uint32_t)((M + 63) / 64);
  uint32_t N_block = (uint32_t)((N + 63) / 64);
  uint32_t K_block = (uint32_t)((K + 63) / 64);

  printf("[INPUT] A(0,0)=%d B(0,0)=%d Din(0,0)=%ld\n", (int)A[0], (int)B[0],
         (long)Din[0]);

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

  // For this debug experiment we only care about the top-left scalar.
  // Expected: Dout(0,0) == G(0,0).
  printf("[CHECK] Dout(0,0)=%ld, Expected G(0,0)=%ld\n", (long)Dout[0], (long)G[0]);
  if (Dout[0] == G[0]) {
    printf("PASS (scalar)\n");
    return 0;
  }
  printf("FAIL (scalar)\n");
  // print_matrix_i32("DOUT", Dout, M, N);
  // print_matrix_i32("G", G, M, N);
  return 1;

  int err = verify_matrix(Dout, G, M, N);
  if (err == 0) {
    printf("PASS\n");
    return 0;  
  }
  printf("FAIL (first error index: %d)\n", err);
  // Dump matrices for Verilator-side log extraction.
  // Keep dumps after FAIL so scripts can grep BEGIN_/END_ markers reliably.
  // print_matrix_i32("DOUT", Dout, M, N);
  // print_matrix_i32("G", G, M, N);
  return 1;
}
