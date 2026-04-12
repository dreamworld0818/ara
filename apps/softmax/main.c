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

#include <stdint.h>
#include <string.h>

#include "kernel/softmax.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

/* 打开 CHECK：标量/向量结果用 THRESHOLD 做近似相等比较 */
#define CHECK

/* SANITY_CHECK：逐元素必须 bitwise 相等（看两种实现是否完全一致，通常过不了） */
// #define SANITY_CHECK

/* PRINT_RESULTS：打印每下标上向量与标量输出的十六进制，便于细查 */
// #define PRINT_RESULTS

/* 浮点比较允许的最大绝对误差 */
#define THRESHOLD 0.0001

/* 由 gen_data.py 写入 .data：通道数、内层长度、输入与三块缓冲（对齐到 NR_LANES） */
extern uint64_t channels;
extern uint64_t innerSize;
extern float i[] __attribute__((aligned(4 * NR_LANES)));
extern float buf[] __attribute__((aligned(4 * NR_LANES)));
extern float o_s[] __attribute__((aligned(4 * NR_LANES)));
extern float o_v[] __attribute__((aligned(4 * NR_LANES)));

/* 依次跑标量与向量 Softmax，计时并可选校验 o_s 与 o_v。返回值：0 成功，非 0 有误差。 */
int main() {
  printf("\n");
  printf("=============\n");
  printf("=  SOFTMAX  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");

  printf("Channels: %lu\nInner Size: %lu\n", channels, innerSize);

  int64_t runtime;
  int error = 0;

  printf("Scalar Softmax...\n");
  start_timer();
  softmax(i, o_s, buf, channels, innerSize);
  stop_timer();

  runtime = get_timer();
  printf("The scalar SOFTMAX execution took %d cycles.\n", runtime);

  printf("Vector Softmax...\n");
  start_timer();
  softmax_vec(i, o_v, channels, innerSize);
  stop_timer();

  runtime = get_timer();
  printf("The vector Softmax execution took %d cycles.\n", runtime);

#ifdef PRINT_RESULTS
  for (uint64_t k = 0; k < channels * innerSize; ++k) {
    printf("%lu) Vector, Scalar: %x, %x\n", k, *((uint32_t *)&(o_v[k])),
           *((uint32_t *)&(o_s[k])));
  }
#endif

#ifdef CHECK
  for (uint64_t k = 0; k < channels * innerSize; ++k) {
#ifdef SANITY_CHECK
    if (o_s[k] != o_v[k]) {
#else
    if (!similarity_check(o_s[k], o_v[k], THRESHOLD)) {
#endif
      error = 1;
      printf("Error at index %d. %f != %f\n", k, o_v[k], o_s[k]);
    }
  }
  if (!error)
    printf("Check okay. No errors.\n");
#endif

  return error;
}
