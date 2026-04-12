// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include <math.h>
#include <string.h>

#include "riscv_vector.h"

#include "../softmax/lib/exp.h"

/* 硬件除法前若向量寄存器含 NaN/Inf 可能导致异常；调试时可打开 RESET_VREGS 用汇编清零向量寄存器。
 * 正式性能测试不要依赖此宏。 */
#define RESET_VREGS

/* 标量 Softmax（沿通道维），算法与 OpenCV dnn softmax 类似：
 * https://github.com/opencv/opencv/blob/master/modules/dnn/src/layers/softmax_layer.cpp
 * 对每个 inner 下标 j：在 c=0..C-1 上算 softmax(i[c*innerSize+j])。
 */
void softmax(const float *i, const float *o, const float *buf,
             uint64_t channels, uint64_t innerSize) {

  /* srcPtr 输入；bufPtr 存每列最大值与各列 exp 之和；dstPtr 输出（与 OpenCV 变量名对应） */
  float *srcPtr = (float *)i;
  float *bufPtr = (float *)buf;
  float *dstPtr = (float *)o;

  /* 本实现固定 batch/outer 维为 1，只处理一块 [channels × innerSize] */
  size_t outerSize = 1;

  /* outerStep：一整块展平后的步长；cnStep：相邻通道同一 inner 下标之间的间隔（即 innerSize） */
  size_t outerStep = channels * innerSize;
  size_t cnStep = innerSize;

  /* 1) 沿通道求每列最大值，写入 buf，用于数值稳定的减 max */
  for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {

    size_t srcOffset = outerDim * outerStep;
    size_t bufOffset = outerDim * cnStep;

    memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

    for (size_t cnDim = 1; cnDim < channels; cnDim++) {
      for (size_t i = 0; i < innerSize; i++) {
        bufPtr[bufOffset + i] =
            fmax(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
      }
    }

    /* 减 max（数值稳定） */
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;
      size_t bufOffset = outerDim * cnStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = srcPtr[offset + i] - bufPtr[bufOffset + i];
      }
    }

    /* exp */
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = exp(dstPtr[offset + i]);
      }
    }

    /* 各 inner 位置对通道上 exp 求和，再归一化 */
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;
      size_t bufOffset = outerDim * cnStep;

      /* 累加 exp 到 buf */
      for (size_t i = 0; i < innerSize; i++)
        bufPtr[bufOffset + i] = 0.f;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          bufPtr[bufOffset + i] += dstPtr[offset + i];
      }

      /* 除以总和 */
      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] /= bufPtr[bufOffset + i];
      }
    }
  }
}

/* RVV 向量实现：对 innerSize 做 strip-mine（每次处理 vl 个元素），通道维在标量循环里展开。 */
void softmax_vec(const float *i, const float *o, uint64_t channels,
                 uint64_t innerSize) {

  /* 调试：避免向量寄存器残留非法值影响 vfdiv；发布测试可关 RESET_VREGS */
#ifdef RESET_VREGS
  volatile int temp;
  asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(temp));

  asm volatile("vmv.v.i  v0, 0");
  asm volatile("vmv.v.i  v8, 0");
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v24, 0");
#endif

  /* avl：当前块剩余元素数；vl：本条向量指令实际长度（由 vsetvl 决定） */
  size_t avl = innerSize;
  size_t vl;

  /* _i/_o：strip 块起始；__i/__o：在当前 vl 块内沿各通道移动 */
  float *_i = (float *)i;
  float *_o = (float *)o;
  float *__i = (float *)i;
  float *__o = (float *)o;

  /* max_chunk_v：当前块每列最大值；buf_chunk_v：暂存加载/减 max/exp；num/den：分子与分母（各通道 exp 之和） */
  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  /* 外层：沿 inner 维分块；内层：通道上求 max → 减 max+exp+累加和 → 除以和 */
  for (vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = __riscv_vsetvl_e32m1(avl);

    /* 沿通道求当前 vl 个位置上的最大值 */
    max_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
    __i += innerSize;
    for (uint64_t ch = 1; ch < channels; ++ch) {
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      __i += innerSize;
      max_chunk_v = __riscv_vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    __i = _i;

    /* 各通道：减 max、exp，写入 __o 作为分子，并累加到 den（分母） */
    den_chunk_v = __riscv_vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < channels; ++ch) {
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      buf_chunk_v = __riscv_vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, buf_chunk_v, vl);
      den_chunk_v = __riscv_vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      __i += innerSize;
      __o += innerSize;
    }
    __i = _i;
    __o = _o;

    /* 各通道分子除以同一分母 den */
    for (uint64_t ch = 0; ch < channels; ++ch) {
      num_chunk_v = __riscv_vle32_v_f32m1(__o, vl);
      res_chunk_v = __riscv_vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, res_chunk_v, vl);
      __o += innerSize;
    }
    _i += vl;
    _o += vl;
    __i = _i;
    __o = _o;
  }
}
