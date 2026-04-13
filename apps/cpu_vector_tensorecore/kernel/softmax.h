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

#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

/* 标量 Softmax：沿「通道」维做 softmax，形状视作 [channels × innerSize]，对每条 inner 位置在通道上归一化。
 * i      输入张量（展平为一维，布局与数据生成脚本一致）
 * o      输出张量（与 i 同长度；头文件里写 const 仅为接口形式，实现中会写入）
 * buf    临时缓冲区，长度至少 channels*innerSize，用于存每维最大值与各位置 exp 之和
 * channels  通道数 C（softmax 沿该维聚合）
 * innerSize  每条「横向」上的元素个数（向量化时按该长度分块处理）
 */
void fa_softmax(const float *i, const float *o, const float *buf,
             uint64_t channels, uint64_t innerSize);

/* 向量（RVV）Softmax：与 softmax 相同的数学与数据布局，使用 RISC-V Vector 加速。
 * i, o, channels, innerSize 含义同上；不需要单独 buf，中间结果用向量寄存器与输出区暂存。
 */
void fa_softmax_vec(const float *i, const float *o, uint64_t channels,
                 uint64_t innerSize);

#endif
