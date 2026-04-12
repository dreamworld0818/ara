#ifndef GEMM_H
#define GEMM_H

/**
 * TensorCore 软件接口：通过 MMIO 下发 MMA 指令并读取完成状态。
 * 基址与 SoC 地址映射一致；仿真/板上若偏移变化，只需改下列宏。
 */

#include "gemm_instruction.h"

/** TensorCore 指令寄存器映射起始地址（按 uint64 槽写入 5 个字） */
#define TENSORCORE_BASE_ADDR  0xD0001000
/** 与 RTL 约定：轮询该地址直到非 0 表示本次计算完成 */
#define TENSORCORE_STATE_ADDR 0xD0001000

/** 片上 L2 与 RRAM 典型基址（仅文档/调试参考，本应用以链接脚本实际放置为准） */
#define L2_BASE   0x80000000
#define RRAM_BASE 0x10000000

/**
 * 将 40 字节 MMA 指令按 5 个 uint64 写入 TensorCore 寄存器区。
 * @param inst 已填好的指令（含 Din/A/B/Dout 地址与 stride、M/N/K 块数）
 */
void send_tensorcore_instruction(mma_instruction_t *inst);

/**
 * 读取 TensorCore 状态字；为 0 表示仍在执行，非 0 表示可读取 Dout。
 * @return 硬件返回的 64 位状态（具体语义见 RTL）
 */
uint64_t load_tensorcore_state(void);

#endif
