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

def pack_a_tc_layout_bytes(M: int, K: int, A_mk, bm=64, bk=64) -> bytes:
  # Layout: [M/bm, K/bk, bk, bm] => single block => order: k-major then m-minor
  _require_multiple_of_64("M", M)
  _require_multiple_of_64("K", K)
  if bm != 64 or bk != 64:
    raise ValueError(f"bm and bk must be 64, got bm={bm}, bk={bk}")
  out = bytearray()
  for k in range(K):
    for m in range(M):
      # Flat order for A is: k-major then m-minor.
      v = A_mk[m][k]
      out += _pack_i8(v)
  return bytes(out)

def pack_b_tc_layout_bytes(K: int, N: int, B_kn, bn=64, bk=64) -> bytes:
  # Layout: [N/bn, K/bk, bk, bn] => single block => order: k-major then n-minor
  _require_multiple_of_64("K", K)
  _require_multiple_of_64("N", N)
  if bn != 64 or bk != 64:
    raise ValueError(f"bn and bk must be 64, got bn={bn}, bk={bk}")
  out = bytearray()
  for k in range(K):
    for n in range(N):
      # Flat order for B is: k-major then n-minor.
      v = B_kn[k][n]
      out += _pack_i8(v)
  return bytes(out)

def pack_din_dout_tc_layout_bytes(M: int, N: int, D_mn, bm=64, bn=64) -> bytes:
  # Layout: [M/bm, N/bn, bn, bm] => single block => order: n-major then m-minor
  _require_multiple_of_64("M", M)
  _require_multiple_of_64("N", N)
  if bm != 64 or bn != 64:
    raise ValueError(f"bm and bn must be 64, got bm={bm}, bn={bn}")
  out = bytearray()
  for n in range(N):
    for m in range(M):
      # Flat order for Din/Dout is: n-major then m-minor.
      v = D_mn[m][n]
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

# Force a single 64x64x64 case.
M = 64
N = 64
K = 64

# Deterministic full 64x64 matrices:
# A[MxK], B[KxN], Din[MxN], G = A*B + Din
A_mk = _build_full_mk_matrix(M, K)
B_kn = _build_full_kn_matrix(K, N)
Din_mn = _build_full_mn_matrix(M, N)
G_mn = _matmul_add(A_mk, B_kn, Din_mn, M, N, K)

bm = 64
bn = 64
bk = 64

A_blocks = pack_a_tc_layout_bytes(M, K, A_mk, bm=bm, bk=bk)
B_blocks = pack_b_tc_layout_bytes(K, N, B_kn, bn=bn, bk=bk)
Din_blocks = pack_din_dout_tc_layout_bytes(M, N, Din_mn, bm=bm, bn=bn)
Dout_blocks = pack_din_dout_tc_layout_bytes(M, N, [[0 for _ in range(N)] for _ in range(M)], bm=bm, bn=bn)
G_blocks = pack_din_dout_tc_layout_bytes(M, N, G_mn, bm=bm, bn=bn)
# alignment
input_balign = str(64 * 64) # 1 bytes per element
acc_balign = str(64 * 64 * 4) # 4 bytes per element

# Create the file
print(".section .l2,\"aw\",@progbits")
emit("M", _pack_u64(M))
emit("N", _pack_u64(N))
emit("K", _pack_u64(K))

emit("A", A_blocks, input_balign)
emit("Din", Din_blocks, acc_balign)
emit("Dout", Dout_blocks, acc_balign)
emit("G", G_blocks, acc_balign)

# Case 1 (all L2)：B 也放在 .l2，便于仅硬件仿真；其他 case 需改此处为 .rram 并设 TC_B_IN_L2=0
print(".section .l2,\"aw\",@progbits")
emit("B", B_blocks, input_balign)
