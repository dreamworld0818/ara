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

  // 按 FLOW_README：base_addr 为相对所在区域的字节偏移（L2 或 RRAM）
  mma_instruction_t inst = {0};
  uint32_t l2_base = L2_BASE, rram_base = RRAM_BASE;

  inst.fields.Din.info.base_addr    = (uint32_t)(uintptr_t)Din - (TC_DIN_IN_L2 ? l2_base : rram_base);
  inst.fields.Din.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Din.info.stride_minor = BLOCK_BYTES_D * N_block;

  inst.fields.Dout.info.base_addr    = (uint32_t)(uintptr_t)Dout - l2_base; /* Dout 始终在 L2 */
  inst.fields.Dout.info.stride_major = BLOCK_BYTES_D;
  inst.fields.Dout.info.stride_minor = BLOCK_BYTES_D * N_block;

  inst.fields.A.info.base_addr    = (uint32_t)(uintptr_t)A - (TC_A_IN_L2 ? l2_base : rram_base);
  inst.fields.A.info.stride_major = BLOCK_BYTES_A;
  inst.fields.A.info.stride_minor = BLOCK_BYTES_A * K_block;

  inst.fields.B.info.base_addr    = (uint32_t)(uintptr_t)B - (TC_B_IN_L2 ? l2_base : rram_base);
  inst.fields.B.info.stride_major = BLOCK_BYTES_A * N_block;
  inst.fields.B.info.stride_minor = BLOCK_BYTES_A;

  inst.fields.mma_meta.info.M = M_block;
  inst.fields.mma_meta.info.N = N_block;
  inst.fields.mma_meta.info.K = K_block;
  inst.fields.mma_meta.info.if_B_transpose = 0;
  inst.fields.mma_meta.info.if_A_transpose = 0;
  inst.fields.mma_meta.info.instruction_type = 1;

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


  int err = verify_matrix(Dout, G, M, N);
  if (err == 0) {
    printf("PASS\n");
    return 0;
  }
  printf("FAIL (first error index: %d)\n", err);
  return 1;
}
