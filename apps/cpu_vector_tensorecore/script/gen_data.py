#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 cpu_vector_tensorecore 生成汇编数据文件 data.S（供链接进裸机 ELF）。

【命令行参数：与 apps/tensorcore_to_cpu 一致】
  三个正整数：M  N  K（均须为 64 的倍数），由 Makefile 传入，默认值见
  apps/common/default_args.mk 中的 def_args_cpu_vector_tensorecore。

  与通用 TensorCore GEMM 记号一致：
    Dout = A * B + Din，逻辑形状 A 为 [M×K]，B 为 [K×N]，Dout 为 [M×N]。

【映射到 Flash-Attention】
  - Q：查询 [M×K]（M 个 query token，特征维 K）
  - Kmat：键 [N×K]（N 个 key token；与 Q 共享同一内积维 K，即 head 维）
  - 得分：S_score = Q * Kmat^T → 形状 [M×N]
  - Softmax：对每一行（长度 N）做 softmax，缩放系数 1/sqrt(K)
  - V：值 [N×K]，输出 O = P * V → [M×K]
  - 因果掩码：仅当 M==N 时，可对 j>i 位置加掩码（见 FA_CAUSAL）；M≠N 时为全连接注意力（不做因果三角）

【与硬件验证的对应关系】
  ① GEMM1：S_score = Q * K^T + Din1（TensorCore：M,N,K 即上式）
  ② Softmax：P = softmax(S_score / sqrt(K) + mask)
  ③ GEMM2：O = P * V + Din2（逻辑上 A=P[M×N], B=V[N×K]，对应第二次调用 fa_send_gemm(M, K, N, …)）
"""

import math
import struct
import sys


def _as_f32(x: float) -> float:
  """与 IEEE float32 往返一致，便于黄金与 C 侧 float 运算对齐。"""
  return struct.unpack("f", struct.pack("f", float(x)))[0]


def emit(name, array, alignment="8"):
  """
  把一段二进制数据写成汇编：定义全局符号 name，并按指定字节对齐放入 .word。

  作用：链接后 C 里可以用 extern 拿到数组首地址；小端序，每 4 字节一个 .word。

  参数:
    name:      汇编中的符号名（如 "Q"、"Gold_S"）
    array:     bytes，长度应为 4 的倍数
    alignment: 汇编 .balign 的对齐值（字符串），常用 8 或 64*64 等
  """
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)


def _require_multiple_of_64(name, x):
  """
  检查维度是否为 64 的倍数。

  作用：TensorCore 以 64×64 为子块；不满足则直接报错，避免生成与 RTL 不一致的数据。
  参数 name: 变量名（用于报错信息）；x: 要检查的整数。
  """
  if x % 64 != 0:
    raise ValueError(f"{name} must be a multiple of 64, got {x}")


def _pack_u64(x: int) -> bytes:
  """把无符号 64 位整数打成 8 字节小端二进制（用于 M、N、K 等标量符号）。"""
  return struct.pack("<Q", int(x))


def _pack_i32(x: int) -> bytes:
  """把有符号 32 位整数打成 4 字节小端二进制（用于 int32 矩阵元素）。"""
  return struct.pack("<i", int(x))


def _pack_i8(x: int) -> bytes:
  """
  把数值收成 int8 再打成 1 字节。

  作用：Q/K/V/P 在硬件里是 int8；超出 [-128,127] 时钳位，避免 struct 打包报错。
  """
  v = int(x)
  if v > 127:
    v = 127
  if v < -128:
    v = -128
  return struct.pack("<b", v)


def _build_qkv(M: int, N: int, K: int):
  """
  构造 Q[M][K]、Kmat[N][K]、V[N][K]，元素为 int8。

  作用：用确定性公式填数（不依赖随机数），仿真可重复、日志便于对照。
  """
  Q = [[((i * 3 + d * 5) % 11) - 5 for d in range(K)] for i in range(M)]
  Kmat = [[((j * 7 + d * 2) % 9) - 4 for d in range(K)] for j in range(N)]
  V = [[((j * 13 + d * 17) % 15) - 7 for d in range(K)] for j in range(N)]
  return Q, Kmat, V


def _score_matmul(Q, Kmat, Din1, M: int, N: int, K: int):
  """
  软件模拟 GEMM1：S_score[i,j] = sum_d Q[i,d]*Kmat[j,d] + Din1[i,j]。

  与 TensorCore：A=Q[M×K]，B=K^T[K×N]，Dout[M×N] 一致。
  """
  Sscore = [[0 for _ in range(N)] for _ in range(M)]
  for i in range(M):
    for j in range(N):
      acc = 0
      for d in range(K):
        acc += int(Q[i][d]) * int(Kmat[j][d])
      Sscore[i][j] = int(acc + Din1[i][j])
  return Sscore


def _softmax_rows_to_p_int8(Sscore, M: int, N: int, K_head: int, causal: bool):
  """
  按行 softmax（每行长度 N），缩放 1/sqrt(K_head)；再量化为 int8。

  与 `kernel/flash_attention.c` 中 `fa_softmax_vec(row_in, row_out, N, 1)` 一致：logits 与中间值
  均经 _as_f32 舍入；exp 使用 math.exp（与 RVV __exp_2xf32 在多数样例上量化一致）。

  因果：仅当 causal 且 M==N 时，对 j>i 加大负偏置（自回归三角）。
  """
  inv_sqrt_k = _as_f32(1.0 / math.sqrt(float(K_head)))
  mask_neg = _as_f32(-1.0e9)
  P = [[0 for _ in range(N)] for _ in range(M)]
  use_causal = causal and (M == N)
  for i in range(M):
    logits = []
    for j in range(N):
      v = _as_f32(_as_f32(float(Sscore[i][j])) * inv_sqrt_k)
      if use_causal and j > i:
        v = _as_f32(v + mask_neg)
      logits.append(v)
    m = max(logits)
    exps = [_as_f32(math.exp(_as_f32(l - m))) for l in logits]
    ssum = float(sum(exps))
    if ssum <= 0.0:
      ssum = 1.0
    for j in range(N):
      p = 127.0 * float(exps[j]) / ssum
      pi = int(p + 0.5)
      if pi > 127:
        pi = 127
      if pi < -128:
        pi = -128
      P[i][j] = pi
  return P


def _matmul_pv(P, V, Din2, M: int, N: int, K: int):
  """O[i,d] = sum_j P[i,j]*V[j,d] + Din2[i,d]，形状 O[M][K]。"""
  O = [[0 for _ in range(K)] for _ in range(M)]
  for i in range(M):
    for d in range(K):
      acc = 0
      for j in range(N):
        acc += int(P[i][j]) * int(V[j][d])
      O[i][d] = int(acc + Din2[i][d])
  return O


def _pack_a_blocks_bytes(M: int, K: int, A_mk):
  """
  把左矩阵 A（M×K，int8）打成 TensorCore 「A 块」字节流（与 tc_a_block_idx 一致）。
  """
  _require_multiple_of_64("M", M)
  _require_multiple_of_64("K", K)
  Mb = M // 64
  Kb = K // 64
  out = bytearray()
  for mb in range(Mb):
    for kb in range(Kb):
      for kl in range(64):
        for ml in range(64):
          k = kb * 64 + kl
          m = mb * 64 + ml
          v = A_mk[m][k] if (m < M and k < K) else 0
          out += _pack_i8(v)
  return bytes(out)


def _pack_b_blocks_bytes(K: int, N: int, B_kn):
  """把右矩阵 B（K×N，int8）打成 「B 块」字节流。"""
  _require_multiple_of_64("K", K)
  _require_multiple_of_64("N", N)
  Kb = K // 64
  Nb = N // 64
  out = bytearray()
  for kb in range(Kb):
    for nb in range(Nb):
      for kl in range(64):
        for nl in range(64):
          k = kb * 64 + kl
          n = nb * 64 + nl
          v = B_kn[k][n] if (k < K and n < N) else 0
          out += _pack_i8(v)
  return bytes(out)


def _pack_mn_blocks_i32_bytes(M: int, N: int, D_mn):
  """Din/Dout（M×N，int32）块排布（与 tc_d_block_idx 一致）。"""
  _require_multiple_of_64("M", M)
  _require_multiple_of_64("N", N)
  Mb = M // 64
  Nb = N // 64
  out = bytearray()
  for mb in range(Mb):
    for nb in range(Nb):
      for nl in range(64):
        for ml in range(64):
          n = nb * 64 + nl
          m = mb * 64 + ml
          v = D_mn[m][n] if (m < M and n < N) else 0
          out += _pack_i32(v)
  return bytes(out)


# ---------------------------------------------------------------------------
# 主流程：M N K（与 tensorcore_to_cpu / default_args 一致）
# ---------------------------------------------------------------------------

if len(sys.argv) == 4:
  M = int(sys.argv[1])
  N = int(sys.argv[2])
  K = int(sys.argv[3])
else:
  print("用法: python3 gen_data.py <M> <N> <K>  （均为 64 的倍数）")
  print("与 TensorCore 记号一致: A[M×K] * B[K×N] + Din；本应用映射为 Flash-Attention。")
  print("示例见 apps/common/default_args.mk 中 def_args_cpu_vector_tensorecore")
  sys.exit(1)

_require_multiple_of_64("M", M)
_require_multiple_of_64("N", N)
_require_multiple_of_64("K", K)

Q, Kmat, V = _build_qkv(M, N, K)
Din1 = [[0 for _ in range(N)] for _ in range(M)]
Din2 = [[0 for _ in range(K)] for _ in range(M)]

Sscore = _score_matmul(Q, Kmat, Din1, M, N, K)
# 因果黄金仅在 M==N 时启用三角掩码；否则为全连接
P_gold = _softmax_rows_to_p_int8(Sscore, M, N, K, causal=True)
O_gold = _matmul_pv(P_gold, V, Din2, M, N, K)

# B1 = K^T，形状 [K][N]，B1[k][n] = Kmat[n][k]
B1 = [[Kmat[n][k] for n in range(N)] for k in range(K)]

# GEMM1：A=Q[M,K], B=B1[K,N] -> S_out[M,N]
A1_blocks = _pack_a_blocks_bytes(M, K, Q)
B1_blocks = _pack_b_blocks_bytes(K, N, B1)
Din1_blocks = _pack_mn_blocks_i32_bytes(M, N, Din1)
Sout_zero = _pack_mn_blocks_i32_bytes(M, N, [[0 for _ in range(N)] for _ in range(M)])
Gold_S_blocks = _pack_mn_blocks_i32_bytes(M, N, Sscore)

# Softmax 后 P[M,N]；GEMM2：A=P[M,N], B=V[N,K] -> O[M,K]
Gold_P_blocks = _pack_a_blocks_bytes(M, N, P_gold)
P_work_zero = [[0 for _ in range(N)] for _ in range(M)]
P_work_blocks = _pack_a_blocks_bytes(M, N, P_work_zero)
B2_blocks = _pack_b_blocks_bytes(N, K, V)
Din2_blocks = _pack_mn_blocks_i32_bytes(M, K, Din2)
Oout_zero = _pack_mn_blocks_i32_bytes(M, K, [[0 for _ in range(K)] for _ in range(M)])
Gold_O_blocks = _pack_mn_blocks_i32_bytes(M, K, O_gold)

input_balign = str(64 * 64)
acc_balign_mn = str(64 * 64 * 4)   # M×N int32
acc_balign_mk = str(64 * 64 * 4)   # M×K int32

causal_flag = struct.pack("<Q", 1)

print(".section .rram,\"a\",@progbits")
emit("Q", A1_blocks, input_balign)
emit("B1", B1_blocks, input_balign)
emit("V", B2_blocks, input_balign)

print(".section .l2,\"aw\",@progbits")
emit("M", _pack_u64(M))
emit("N", _pack_u64(N))
emit("K", _pack_u64(K))
emit("FA_CAUSAL", causal_flag)

emit("Din1", Din1_blocks, acc_balign_mn)
emit("S_out", Sout_zero, acc_balign_mn)
emit("Gold_S", Gold_S_blocks, acc_balign_mn)

emit("P_work", P_work_blocks, input_balign)
emit("Din2", Din2_blocks, acc_balign_mk)
emit("O_out", Oout_zero, acc_balign_mk)
emit("Gold_P", Gold_P_blocks, input_balign)
emit("Gold_O", Gold_O_blocks, acc_balign_mk)
