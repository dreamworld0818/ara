#!/usr/bin/env python3

from ctypes import alignment
import random as rand
import numpy as np
import sys

def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)

def block_decomposition(matrix, block_shape=(64, 64), pad_mode='constant', pad_value=0):
  """
  将矩阵分解为指定形状的块
  
  参数:
  - matrix: 输入矩阵
  - block_shape: 块形状，默认为(64, 64)
  - pad_mode: 填充模式
  - pad_value: 填充值
  
  返回:
  - 块状矩阵和填充信息
  """
  block_h, block_w = block_shape
  rows, cols = matrix.shape
  
  # 计算需要填充的数量
  pad_rows = (block_h - rows % block_h) % block_h
  pad_cols = (block_w - cols % block_w) % block_w
  
  # 填充
  padded = np.pad(matrix, 
                  ((0, pad_rows), (0, pad_cols)), 
                  mode=pad_mode, 
                  constant_values=pad_value)
  
  # 计算块数量
  n_blocks_rows = padded.shape[0] // block_h
  n_blocks_cols = padded.shape[1] // block_w
  
  # 重塑
  blocks = padded.reshape(n_blocks_rows, block_h, n_blocks_cols, block_w)
  blocks = blocks.transpose(0, 2, 1, 3)  # (n_blocks_rows, n_blocks_cols, block_h, block_w)
  
  return blocks, (pad_rows, pad_cols)

############
## SCRIPT ##
############

if len(sys.argv) == 4:
  M = int(sys.argv[1])
  N = int(sys.argv[2])
  K = int(sys.argv[3])
else:
  print("Error. Give me three argument: M, N, K.")
  print("Dout = AB + Din with A=[MxK], B=[KxN], Din=[MxN], Dout=[MxN]")
  sys.exit()

input_dtype = np.int8
acc_dtype = np.int32

UPPER_LIMIT = 128
LOWER_LIMIT = -127

A = np.random.randint(LOWER_LIMIT, UPPER_LIMIT, size=(M, K)).astype(input_dtype)
B = np.random.randint(LOWER_LIMIT, UPPER_LIMIT, size=(K, N)).astype(input_dtype)
Din = np.random.randint(LOWER_LIMIT, UPPER_LIMIT, size=(M, N)).astype(acc_dtype)
Dout = np.zeros([M, N], dtype=acc_dtype)
# Golden result matrix
G = np.matmul(A, B) + Din

A_blocks, A_pad = block_decomposition(A)
B_blocks, B_pad = block_decomposition(B)
Din_blocks, Din_pad = block_decomposition(Din)
Dout_blocks, Dout_pad = block_decomposition(Dout)
G_blocks, G_pad = block_decomposition(G)
# alignment
input_balign = str(64 * 64) # 1 bytes per element
acc_balign = str(64 * 64 * 4) # 4 bytes per element

# Create the file
print(".section .l2,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("K", np.array(K, dtype=np.uint64))

emit("A", A_blocks, input_balign)
emit("Din", Din_blocks, acc_balign)
emit("Dout", Dout_blocks, acc_balign)
emit("G", G_blocks, acc_balign)

# Case 1 (all L2)：B 也放在 .l2，便于仅硬件仿真；其他 case 需改此处为 .rram 并设 TC_B_IN_L2=0
print(".section .l2,\"aw\",@progbits")
emit("B", B_blocks, input_balign)
