#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 用法: gen_data.py <channels> <innerSize>
# 生成链接进程序的 .data 段：channels、innerSize、输入 i 及 buf/o_s/o_v 占位（与硬件对齐宏一致）。

import random as rand
import numpy as np
import sys

def emit(name, array, alignment='8'):
  """输出汇编符号：全局标号 name,按 alignment 对齐，把 array 逐 word 写成 .word。"""
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)

def rand_matrix(N, dtype):
  """生成长度为 N 的随机浮点向量（用于输入 i）。"""
  return np.random.rand(N).astype(dtype)

############
## SCRIPT ##
############

if len(sys.argv) == 3:
  channels = int(sys.argv[1])
  innerSize = int(sys.argv[2])
else:
  print("Error. Give me two arguments: the number of channels and the inner size.")
  sys.exit()

# 展平形状 [channels, innerSize] 的随机输入
i = rand_matrix(channels * innerSize, np.float32).astype(np.float32)

# 与 main 中用途对应；汇编里 buf/o_s/o_v 用与 i 同形状的占位数据保证长度与对齐
buf = np.zeros(channels * innerSize, dtype=np.float32)
o_s = np.zeros(channels * innerSize, dtype=np.float32)
o_g = np.zeros(channels * innerSize, dtype=np.float32)

print(".section .data,\"aw\",@progbits")
emit("channels", np.array(channels, dtype=np.uint64))
emit("innerSize", np.array(innerSize, dtype=np.uint64))
emit("i", i, 'NR_LANES*4')
emit("buf", i, 'NR_LANES*4')
emit("o_s", i, 'NR_LANES*4')
emit("o_v", i, 'NR_LANES*4')
