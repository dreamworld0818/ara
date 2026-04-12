#ifndef GEMM_INSTRUCTION_H
#define GEMM_INSTRUCTION_H

/**
 * TensorCore MMA 指令在软件侧的 40 字节布局（与硬件 MMIO 写入顺序一致）。
 * 用于描述 Din/B/A/Dout 的基址与步长，以及 M/N/K 块数与转置等元数据。
 */

#include <assert.h>
#include <stdint.h>

#pragma pack(push, 1)

/**
 * 单个操作数（Din、A、B 或 Dout）在指令中的 64 位描述字段。
 * base_addr：物理地址右移若干位后的字段（与 RTL DMA 约定一致）；
 * stride_major / stride_minor：块内/块间步长（字节级语义见 tensorcore_to_cpu 文档）。
 */
typedef union {
  uint64_t raw;
  struct {
    uint32_t stride_minor : 20;
    uint32_t stride_major : 20;
    uint32_t base_addr : 24;
  } info;
} op_info_t;

/**
 * MMA 元数据：以 64×64 子块为单位的 M、N、K 块个数，以及是否对 A/B 做逻辑转置。
 */
typedef union {
  uint32_t raw;
  struct {
    uint8_t K : 8;
    uint8_t N : 8;
    uint8_t M : 8;
    uint8_t reserved : 1;
    uint8_t if_B_transpose : 1;
    uint8_t if_A_transpose : 1;
    uint8_t instruction_type : 5;
  } info;
} mma_meta_t;

/**
 * 完整 MMA 指令：5×uint64 = 40 字节，按小端写入 MMIO 端口。
 */
typedef union {
  uint8_t raw[40];
  uint64_t raw64[5];
  struct {
    op_info_t Din;
    op_info_t B;
    op_info_t A;
    op_info_t Dout;
    mma_meta_t mma_meta;
    uint32_t padding;
  } fields;
} mma_instruction_t;

#ifndef __cplusplus
_Static_assert(sizeof(mma_instruction_t) == 40,
               "MMA instruction must be 40 bytes (320 bits)");
#endif

#pragma pack(pop)

#endif
