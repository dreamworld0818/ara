#ifndef GEMM_H
#define GEMM_H

#include "gemm_instruction.h"

#define TENSORCORE_BASE_ADDR 0xD0001000
#define TENSORCORE_STATE_ADDR 0xD0002000

void send_tensorcore_instruction(mma_instruction_t *inst);

uint64_t load_tensorcore_state(void);

#endif
