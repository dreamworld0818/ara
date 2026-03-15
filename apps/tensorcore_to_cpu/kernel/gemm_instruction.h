#ifndef GEMM_INSTRUCTION_H
#define GEMM_INSTRUCTION_H

#include <assert.h>
#include <stdint.h>

#pragma pack(push, 1) // 1字节对齐，避免编译器填充

typedef union {
  uint64_t raw;
  struct {
    uint32_t stride_minor : 20; //沿“行”方向的步长，等于该矩阵的列数
    uint32_t stride_major : 20; //沿“列”方向的步长，等于该矩阵的行数
    uint32_t base_addr : 24; //基地址
  } info;
} op_info_t;

typedef union {
  uint32_t raw;
  struct {
    uint8_t K : 8;
    uint8_t N : 8;
    uint8_t M : 8;
    uint8_t reserved : 1;   //保留
    uint8_t if_B_transpose : 1;  //B是否转置
    uint8_t if_A_transpose : 1;  //A是否转置
    uint8_t instruction_type : 5;  //指令类型
  } info;
} mma_meta_t;

// 完整的512位MMA指令结构体
// 注意：小端序，低地址存储低字节
typedef union {
//   uint8_t raw[64];
//   uint64_t raw64[8];
  uint8_t raw[40];
  uint64_t raw64[5];
  struct {
    // 位 [63:0] - Din矩阵信息 (最低64位，存储在最低地址)
    op_info_t Din;

    // 位 [127:64] - B矩阵信息
    op_info_t B;

    // 位 [191:128] - A矩阵信息
    op_info_t A;

    // 位 [255:192] - Dout矩阵信息
    op_info_t Dout;

    // 位 [287:256] - MMA元数据 (最高32位，存储在最高地址)
    mma_meta_t mma_meta;

    // // 填充到512位 (64字节)
    // // 已使用: 256位(Din+B+A+Dout) + 32位(meta) = 288位
    // // 填充: 512 - 288 = 224位 = 28字节
    // uint8_t padding[28];
    // 填充到320位 (40字节)
    // 已使用: 256位(Din+B+A+Dout) + 32位(meta) = 288位
    // 填充: 320 - 288 = 32位 = 4字节
    uint32_t padding;
  } fields;
// } mma_instruction_512_t;
} mma_instruction_t;

#ifndef __cplusplus
// _Static_assert(sizeof(mma_instruction_512_t) == 64,
//                "MMA instruction must be 64 bytes (512 bits)");
_Static_assert(sizeof(mma_instruction_t) == 40,
               "MMA instruction must be 40 bytes (320 bits)");
#endif

#pragma pack(pop) // 恢复默认对齐

#endif
