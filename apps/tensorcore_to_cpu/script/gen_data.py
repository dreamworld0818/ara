#!/usr/bin/env python3

import sys
import struct

def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  # "array" here is a raw bytes object.
  bs = array
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)

def _require_multiple_of_64(name, x):
  if x % 64 != 0:
    raise ValueError(f"{name} must be a multiple of 64, got {x}")

def _pack_u64(x: int) -> bytes:
  return struct.pack("<Q", int(x))

def _pack_i8(x: int) -> bytes:
  return struct.pack("<b", int(x))

def _pack_i32(x: int) -> bytes:
  return struct.pack("<i", int(x))

def _build_full_mk_matrix(M: int, K: int):
  # Small signed values to avoid overflow and keep logs readable.
  return [[((m * 3 + k * 5) % 11) - 5 for k in range(K)] for m in range(M)]

def _build_full_kn_matrix(K: int, N: int):
  return [[((k * 7 + n * 2) % 9) - 4 for n in range(N)] for k in range(K)]

def _build_full_mn_matrix(M: int, N: int):
  return [[(m * 13 + n * 17) % 97 - 48 for n in range(N)] for m in range(M)]

def _matmul_add(A_mk, B_kn, Din_mn, M: int, N: int, K: int):
  G = [[0 for _ in range(N)] for _ in range(M)]
  for m in range(M):
    for n in range(N):
      acc = 0
      for k in range(K):
        acc += int(A_mk[m][k]) * int(B_kn[k][n])
      G[m][n] = int(acc + Din_mn[m][n])
  return G

def _pack_a_blocks_bytes(M: int, K: int, A_mk) -> bytes:
  """Block grid (mb, kb) then k-major within each 64×64 — matches RTL DMA / old linear A."""
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

def _pack_b_blocks_bytes(K: int, N: int, B_kn) -> bytes:
  """Block grid (kb, nb) then k-major within each 64×64 (same as global k then n scan)."""
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

def _pack_mn_blocks_i32_bytes(M: int, N: int, D_mn) -> bytes:
  """Block grid (mb, nb) then n-major within 64×64: flat idx = n*64+m (see tc_buffer VERILATOR)."""
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

# Case (default via def_args_tensorcore_to_cpu): M=128, N=256, K=128.
# M, N, K must be multiples of 64 (TensorCore tile size).

# Fixed deterministic matrices (same formulas for any valid M,N,K):
# A[MxK], B[KxN], Din[MxN], G = A*B + Din
A_mk = _build_full_mk_matrix(M, K)
B_kn = _build_full_kn_matrix(K, N)
Din_mn = _build_full_mn_matrix(M, N)
G_mn = _matmul_add(A_mk, B_kn, Din_mn, M, N, K)

A_blocks = _pack_a_blocks_bytes(M, K, A_mk)
B_blocks = _pack_b_blocks_bytes(K, N, B_kn)
Din_blocks = _pack_mn_blocks_i32_bytes(M, N, Din_mn)
Dout_blocks = _pack_mn_blocks_i32_bytes(M, N, [[0 for _ in range(N)] for _ in range(M)])
G_blocks = _pack_mn_blocks_i32_bytes(M, N, G_mn)
# alignment
input_balign = str(64 * 64) # 1 bytes per element
acc_balign = str(64 * 64 * 4) # 4 bytes per element

# RRAM is read-only in this test: put all input matrices in .rram.
print(".section .rram,\"a\",@progbits")
emit("A", A_blocks, input_balign)
emit("B", B_blocks, input_balign)
emit("Din", Din_blocks, acc_balign)

# Create the file
print(".section .l2,\"aw\",@progbits")
emit("M", _pack_u64(M))
emit("N", _pack_u64(N))
emit("K", _pack_u64(K))

# emit("A", A_blocks, input_balign)
# emit("B", B_blocks, input_balign)
# emit("Din", Din_blocks, acc_balign)
emit("Dout", Dout_blocks, acc_balign)
emit("G", G_blocks, acc_balign)
# # Case 1 (all L2)：B 也放在 .l2，便于仅硬件仿真；其他 case 需改此处为 .rram 并设 TC_B_IN_L2=0
# print(".section .l2,\"aw\",@progbits")
# emit("B", B_blocks, input_balign)

