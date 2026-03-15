#ifndef GEMM_H
#define GEMM_H

#include "gemm_instruction.h"

#define TENSORCORE_BASE_ADDR  0xD0001000
/* RTL returns slv_register_output_data only when ar.addr == SLV_REGISTER_ADDR (0xD000_1000) */
#define TENSORCORE_STATE_ADDR 0xD0001000

#define L2_BASE   0x80000000
#define RRAM_BASE 0x10000000

void send_tensorcore_instruction(mma_instruction_t *inst);

uint64_t load_tensorcore_state(void);

#endif
