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

// Define Matrix dimensions:
// Dout = AB + Din with A=[MxK], B=[KxN], Din=[MxN], Dout=[MxN]
extern uint64_t M;
extern uint64_t N;
extern uint64_t K;
extern int8_t A[] __attribute__((aligned(4096), section(".l2")));
extern int32_t Din[] __attribute__((aligned(16384), section(".l2")));
extern int32_t Dout[] __attribute__((aligned(16384), section(".l2")));
extern int8_t B[] __attribute__((aligned(4096), section(".rram")));
// Gold results
extern int32_t G[] __attribute__((aligned(16384), section(".l2")));

// Verify the matrix
int verify_matrix(int32_t *result, int32_t *gold, size_t R, size_t C) {
  for (uint64_t i = 0; i < R; ++i) {
    for (uint64_t j = 0; j < C; ++j) {
      uint64_t idx = i * C + j;
      if (result[idx] != gold[idx]) {
        return (i + j) == 0 ? -1 : idx;
      }
    }
  }
  return 0;
}


int main() {
    uint32_t M_block = (M + 63) / 64;
    uint32_t N_block = (N + 63) / 64;
    uint32_t K_block = (K + 63) / 64;
    mma_instruction_t inst = {0};
    inst.fields.Din.info.stride_minor = N_block;
    inst.fields.Din.info.stride_major = 1;
    inst.fields.Din.info.base_addr = (uint32_t)Din / 16384;
    inst.fields.B.info.stride_minor = N_block;
    inst.fields.B.info.stride_major = 1;
    inst.fields.B.info.base_addr = (uint32_t)B / 4096;
    inst.fields.A.info.stride_minor = K_block;
    inst.fields.A.info.stride_major = 1;
    inst.fields.A.info.base_addr = (uint32_t)A / 4096;
    inst.fields.Dout.info.stride_minor = N_block;
    inst.fields.Dout.info.stride_major = 1;
    inst.fields.Dout.info.base_addr = (uint32_t)Dout / 16384;
    inst.fields.mma_meta.info.K = K_block;
    inst.fields.mma_meta.info.N = N_block;
    inst.fields.mma_meta.info.M = M_block;
    inst.fields.mma_meta.info.if_B_transpose = 0;
    inst.fields.mma_meta.info.if_A_transpose = 0;
    inst.fields.mma_meta.info.instruction_type = 1;
    send_tensorcore_instruction(&inst);  
    uint64_t state = load_tensorcore_state();
    printf("tensorcore state: %016lx\n", state);

    printf("M: %d, N: %d, K: %d\n", M, N, K);
    printf("M_block: %d, N_block: %d, K_block: %d\n", M_block, N_block, K_block);
    printf("Din: stride_minor: %d, stride_major: %d, base_addr: 0x%06x, real_addr: %p\n", N_block, 1, (uint32_t)Din / 16384, Din);
    printf("B: stride_minor: %d, stride_major: %d, base_addr: 0x%06x, real_addr: %p\n", N_block, 1, (uint32_t)B / 4096, B);
    printf("A: stride_minor: %d, stride_major: %d, base_addr: 0x%06x, real_addr: %p\n", K_block, 1, (uint32_t)A / 4096, A);
    printf("Dout: stride_minor: %d, stride_major: %d, base_addr: 0x%06x, real_addr: %p\n", N_block, 1, (uint32_t)Dout / 16384, Dout);
    printf("K: %d, N: %d, M: %d, if_B_transpose: %d, if_A_transpose: %d, instruction_type: %d\n", K_block, N_block, M_block, 0, 0, 1);
    printf("sizeof(mma_instruction_512_t): %ld\n", sizeof(mma_instruction_t));
    printf("inst: \n");

    for (int i = sizeof(mma_instruction_t) - 1; i >= 0; i--) {
        printf("%02x ", inst.raw[i]);
        if (i % 8 == 0) {
            printf("\n");
        }
        // for (int j = 7; j >= 0; j--) {
        //     printf("%d", (inst.raw[i] >> j) & 1);
        // }
        // printf(" ");
    }
    printf("\n");
    return 0;
}