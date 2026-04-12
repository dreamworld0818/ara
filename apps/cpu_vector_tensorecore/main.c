/**
 * cpu_vector_tensorecore：Flash-Attention 端到端验证。
 *
 * 维度 M、N、K 与 apps/tensorcore_to_cpu 相同，由 gen_data.py 与 default_args.mk 配置：
 *   - GEMM1：S_score = Q * K^T + Din1，逻辑为 A[M×K] * B[K×N]（与通用 TensorCore 记号一致）
 *   - Softmax：P = softmax(S_score / sqrt(K) + mask)
 *   - GEMM2：O = P * V + Din2，逻辑为 A[M×N] * B[N×K] → M×K
 */

#include "kernel/flash_attention.h"
#include "kernel/gemm.h"
#include "runtime.h"

#if defined(SPIKE) || defined(ARA_LINUX)
#include <stdio.h>
#else
#include "printf.h"
#endif

extern uint64_t M;
extern uint64_t N;
extern uint64_t K;
extern uint64_t FA_CAUSAL;

extern int8_t Q[];
extern int8_t B1[];
extern int8_t V[];
extern int8_t P_work[];
extern int32_t Din1[];
extern int32_t S_out[];
extern int32_t Gold_S[];
extern int32_t Din2[];
extern int32_t O_out[];
extern int32_t Gold_O[];
extern int8_t Gold_P[];

static void print_first_i32_mismatches(const int32_t *got, const int32_t *gold,
                                       uint64_t rows, uint64_t cols, int max_print) {
  int printed = 0;
  for (uint64_t n = 0; n < cols && printed < max_print; n++) {
    for (uint64_t m = 0; m < rows && printed < max_print; m++) {
      uint64_t idx = tc_d_block_idx(m, n, cols);
      if (got[idx] != gold[idx]) {
        printf("[FA] mismatch m=%llu n=%llu idx=%llu got=%ld gold=%ld\n",
               (unsigned long long)m, (unsigned long long)n,
               (unsigned long long)idx, (long)got[idx], (long)gold[idx]);
        printed++;
      }
    }
  }
}

static void print_first_i8_mismatches(const int8_t *got, const int8_t *gold,
                                      uint64_t rows, uint64_t Kdim, int max_print) {
  int printed = 0;
  for (uint64_t k = 0; k < Kdim && printed < max_print; k++) {
    for (uint64_t m = 0; m < rows && printed < max_print; m++) {
      uint64_t idx = tc_a_block_idx(m, k, Kdim);
      if (got[idx] != gold[idx]) {
        printf("[FA] P mismatch m=%llu k=%llu idx=%llu got=%d gold=%d\n",
               (unsigned long long)m, (unsigned long long)k,
               (unsigned long long)idx, (int)got[idx], (int)gold[idx]);
        printed++;
      }
    }
  }
}

int main(void) {
  uint64_t m = M;
  uint64_t n = N;
  uint64_t k = K;
  uint32_t causal = (FA_CAUSAL != 0U) ? 1U : 0U;

  printf("=== cpu_vector_tensorecore: Flash-Attention (M,N,K 与 tensorcore_to_cpu 一致) ===\n");
  printf("[FA] M=%llu N=%llu K=%llu causal=%u\n", (unsigned long long)m,
         (unsigned long long)n, (unsigned long long)k, (unsigned)causal);

  /* GEMM1：A[M,K]=Q，B[K,N]=K^T，Dout[M,N]=S_out */
  printf("[FA] GEMM1 Q*K^T: MMA (M,N,K)=(%llu,%llu,%llu)\n",
         (unsigned long long)m, (unsigned long long)n, (unsigned long long)k);
  fa_send_gemm("GEMM1_QxKt", m, n, k, Q, B1, Din1, S_out);

  int bad_s = fa_verify_i32_block(S_out, Gold_S, m, n);
  if (bad_s >= 0) {
    printf("FAIL Gold_S (GEMM1), first block idx %d\n", bad_s);
    print_first_i32_mismatches(S_out, Gold_S, m, n, 4);
    return 1;
  }
  printf("PASS checkpoint Gold_S (GEMM1)\n");

  printf("=== cpu_vector_tensorecore: Flash-Attention softmax 阶段开始(scores -> P) ===\n");
  fa_softmax_scores_to_p(S_out, m, n, k, P_work, causal);
  printf("===  softmax 阶段结束(scores -> P) ===\n");

  int bad_p = fa_verify_i8_a_block(P_work, Gold_P, m, n);
  if (bad_p >= 0) {
    printf("FAIL Gold_P (softmax), first idx %d\n", bad_p);
    print_first_i8_mismatches(P_work, Gold_P, m, n, 4);
    return 2;
  }
  printf("PASS checkpoint Gold_P (softmax)\n");

  /* GEMM2：P[M,N]*V[N,K]->O[M,K] ⇒ fa_send_gemm(..., M, K, N, P, V, ...) */
  printf("[FA] GEMM2 P*V: MMA (M,N,K)=(%llu,%llu,%llu)\n",
         (unsigned long long)m, (unsigned long long)k, (unsigned long long)n);
  fa_send_gemm("GEMM2_PxV", m, k, n, P_work, V, Din2, O_out);

  int bad_o = fa_verify_i32_block(O_out, Gold_O, m, k);
  if (bad_o >= 0) {
    printf("FAIL Gold_O (GEMM2), first block idx %d\n", bad_o);
    print_first_i32_mismatches(O_out, Gold_O, m, k, 4);
    return 3;
  }
  printf("PASS checkpoint Gold_O (GEMM2)\n");

  printf("PASS Flash-Attention end-to-end\n");
  return 0;
}
