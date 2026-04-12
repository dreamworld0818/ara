#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <stdint.h>

#include "gemm.h"

/**
 * Flash-Attention 验证：TensorCore 双段 GEMM + 中间 softmax（RISC-V Vector RVV 实现）。
 *
 * 内存布局与 script/gen_data.py 一致：
 * - int32 累加矩阵（Dout/Din）：tc_d_block_idx（块内 n 主序）
 * - int8 A 矩阵：tc_a_block_idx（块内 k 主序、m 次序）
 */

/**
 * D 矩阵（int32，M×N）块排布下的线性下标。
 * @param m     行 [0,M)
 * @param n     列 [0,N)
 * @param Ncols 列数 N
 * @return      块布局下的元素下标
 */
uint64_t tc_d_block_idx(uint64_t m, uint64_t n, uint64_t Ncols);

/**
 * A 矩阵（int8，M×K）块排布下的线性下标（与 gen_data _pack_a 一致）。
 */
uint64_t tc_a_block_idx(uint64_t m, uint64_t k, uint64_t Kdim);

/**
 * 对得分矩阵 S_score（GEMM1 输出，形状 M×N）按行做 softmax（与 apps/softmax 的 softmax_vec 相同：
 * float32、__exp_2xf32、vfdiv）。
 * P[i,j] = quantize(127 * softmax(...))，int8 写入 P_packed（L2 A 块布局）；黄金由 gen_data 中 float32
 * 舍入 + math.exp 生成。
 *
 * 缩放用 K_head（与 A 的内积维一致）；因果掩码仅当 causal≠0 且 M==N 时施加（j>i）。
 *
 * @param S_score  M×N，int32 块布局（列数 Ncols=N）
 * @param M        query 行数
 * @param N        key 列数（每行 softmax 长度）
 * @param K_head   head 维（sqrt 缩放分母）
 * @param P_packed M×N，int8 A 块布局（tc_a_block_idx，第二维为 N）
 * @param causal   非 0 且 M==N 时启用因果 mask
 */
void fa_softmax_scores_to_p(const int32_t *S_score, uint64_t M, uint64_t N,
                            uint64_t K_head, int8_t *P_packed, uint32_t causal);

/**
 * 下发一次 TensorCore MMA：Dout = A*B + Din（填指令方式与 tensorcore_to_cpu/main.c 一致）。
 * @param phase_label 阶段名，用于串口打印（如 "GEMM1" / "GEMM2"）
 * @param M,N,K       与通用 GEMM 记号一致：A[M×K]·B[K×N]，Dout[M×N]
 * @param A,B         int8 操作数基址
 * @param Din         int32 累加输入
 * @param Dout        int32 输出
 */
void fa_send_gemm(const char *phase_label, uint64_t M, uint64_t N, uint64_t K,
                  const int8_t *A, const int8_t *B, const int32_t *Din,
                  int32_t *Dout);

/**
 * 比对 int32 块矩阵与黄金参考（全等）。
 * @return 一致返回 -1，否则返回首个块线性下标
 */
int fa_verify_i32_block(const int32_t *got, const int32_t *gold, uint64_t rows,
                        uint64_t cols);

/**
 * 比对 int8 A 块矩阵与黄金参考（全等）。
 */
int fa_verify_i8_a_block(const int8_t *got, const int8_t *gold, uint64_t rows,
                         uint64_t Kdim);

#endif
