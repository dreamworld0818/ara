#include "gemm_instruction.h"
#include "gemm.h"

void send_tensorcore_instruction(mma_instruction_t *inst) {
    volatile uint64_t *tensorcore_base_addr = (uint64_t *)TENSORCORE_BASE_ADDR;
    uint64_t *raw64 = inst->raw64;
    for (int i = 0; i < sizeof(mma_instruction_t) / sizeof(uint64_t); i++) {
        tensorcore_base_addr[i] = raw64[i];
        asm volatile("" ::: "memory"); // compiler memory barrier
    }
    __sync_synchronize(); // hardware memory barrier
}

uint64_t load_tensorcore_state(void) {
    volatile uint64_t *tensorcore_state_addr = (uint64_t *)TENSORCORE_STATE_ADDR;
    return *tensorcore_state_addr;
}