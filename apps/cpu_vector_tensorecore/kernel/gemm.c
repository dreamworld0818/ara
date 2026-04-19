/**
 * TensorCore MMIO 发送与状态轮询。
 * 与 apps/tensorcore_to_cpu/kernel/gemm.c 一致：逐槽写入后加内存屏障，确保硬件完整锁存指令。
 */

#include "gemm_instruction.h"
#include "gemm.h"
#if defined(SPIKE) || defined(ARA_LINUX)
#include <stdio.h>
#else
#include "printf.h"
#endif

void send_tensorcore_instruction(mma_instruction_t *inst) {
  // printf("[TC-INST] send_tensorcore_instruction start\n");
  volatile uint64_t *tensorcore_base_addr = (volatile uint64_t *)TENSORCORE_BASE_ADDR;
  // printf("[TC-INST] send_tensorcore_instruction get tensorcore_base_addr\n");
  uint64_t *raw64 = inst->raw64;
  // printf("[TC-INST] send_tensorcore_instruction get raw64\n");
  /* 仿真卡死时 RTL 打印的 pc_commit 可与 apps/bin/*.dump 中本符号附近汇编对照 */
  __asm__ volatile(".globl fa_tc_mmio_inst_region\n"
                   "fa_tc_mmio_inst_region:\n" ::
                       : "memory");
  for (int i = 0; i < (int)(sizeof(mma_instruction_t) / sizeof(uint64_t)); i++) {
    tensorcore_base_addr[i] = raw64[i];
    // printf("[TC-INST] store word 1\n");
    asm volatile("" ::: "memory");
    /* After this line: MMIO word i completed (AXI B for this store). */
    // printf("[TC-INST] store word 2\n");
    /* After this line: MMIO word i completed (AXI B for this store). */
    // printf("[TC-INST] store word 3\n");
    // __sync_synchronize();
  }

  // tensorcore_base_addr[(int)(sizeof(mma_instruction_t) / sizeof(uint64_t))-1] = raw64[(int)(sizeof(mma_instruction_t) / sizeof(uint64_t))-1];
  // printf("[TC-INST] fence ok, send_tensorcore_instruction1\n");
  __sync_synchronize();
  // printf("[TC-INST] fence ok, send_tensorcore_instruction return\n");
}




uint64_t load_tensorcore_state(void) {
  // printf("[TC-INST] load_tensorcore_state\n");
  volatile uint64_t *tensorcore_state_addr =
      (volatile uint64_t *)TENSORCORE_STATE_ADDR;
  uint64_t s = *tensorcore_state_addr;
  // printf("[TC-INST] load_tensorcore_state -> %llu\n", (unsigned long long)s);
  return s;
}
