# cpu_vector_tensorecore

**Flash-Attention** 端到端验证：维度 **M、N、K** 的配置方式与 `apps/tensorcore_to_cpu` 相同，通过 `apps/common/default_args.mk` 中的 `def_args_cpu_vector_tensorecore` 传入，并由 `script/gen_data.py` 生成 `data.S`。

## 记号（与通用 TensorCore GEMM 一致）

- **M、N、K**：与 `Dout = A*B + Din` 中 **A[M×K]、B[K×N]** 的记号一致。
- **Flash-Attention 映射**：
  - **GEMM1**：`Q[M×K] * K^T[K×N]` → 得分 `S_out[M×N]`（即 `A=Q`，`B=B1`）
  - **Softmax**：对 `S_out` 每一行（长度 **N**）做 softmax，缩放 `1/sqrt(K)`；**因果掩码**仅当 **M==N** 且 `FA_CAUSAL` 非 0 时启用（`j>i`）
  - **GEMM2**：`P[M×N] * V[N×K]` → `O_out[M×K]`（第二次 MMA 参数为 `(M, K, N)`）

## 构建

在仓库 `apps/` 目录：

```bash
make bin/cpu_vector_tensorecore
# 覆盖默认 M N K（须为 64 的倍数）
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="128 256 64"
```

默认见 `apps/common/default_args.mk`。

## Verilator 仿真

```bash
cd hardware && make verilate
cd ../apps && make bin/cpu_vector_tensorecore
cd ../hardware && make simv app=cpu_vector_tensorecore
```

## 行为说明

全部检查点通过时打印 `PASS Flash-Attention end-to-end`。

## 目录

| 路径 | 说明 |
|------|------|
| `kernel/gemm*.c/h` | TensorCore MMIO 与 MMA 指令布局 |
| `kernel/flash_attention.c/h` | 块下标、双段 GEMM、softmax、比对 |
| `script/gen_data.py` | `gen_data.py M N K` → `data.S` |
| `data_log/` | 可自行重定向仿真日志 |
