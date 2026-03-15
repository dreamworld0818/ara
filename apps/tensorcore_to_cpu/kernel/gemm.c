#include "gemm_instruction.h"
#include "gemm.h"

void send_tensorcore_instruction(mma_instruction_t *inst) {
  volatile uint64_t *tensorcore_base_addr = (volatile uint64_t *)TENSORCORE_BASE_ADDR;
  uint64_t *raw64 = inst->raw64;
  for (int i = 0; i < sizeof(mma_instruction_t) / sizeof(uint64_t); i++) {
    tensorcore_base_addr[i] = raw64[i];
    asm volatile("" ::: "memory");
  }
  __sync_synchronize();
}

uint64_t load_tensorcore_state(void) {
  volatile uint64_t *tensorcore_state_addr = (volatile uint64_t *)TENSORCORE_STATE_ADDR;
  return *tensorcore_state_addr;
}
