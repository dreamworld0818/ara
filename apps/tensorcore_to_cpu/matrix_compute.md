# TensorCore 分块矩阵乘：布局与遍历（硬件侧）

本文与 RTL `tc_controller.sv`、数据打包 `script/gen_data.py`、以及 `main.c` 中指令 stride 约定一致。运算为 **Dout = A×B + Din**，逻辑形状 **A[M×K]、B[K×N]、Din/Dout[M×N]**；硬件以 **64×64×1B（A/B）或 64×64×4B（Din/Dout）** 为一块。

---（主要看第3部分）

## 1. 指令中的 major / minor 与块地址

控制器用三个块下标 **`m_cnt`、`n_cnt`、`k_cnt`**（分别对应 M、N、K 方向的块索引，范围 `0 … M_block-1` 等）和 **stride_major / stride_minor** 计算 DMA 字节地址（见 `tc_controller.sv` 中 `addr_*`）。

**约定（无转置，`if_A_transpose=0`、`if_B_transpose=0`，与当前 `main.c` 一致）：**

| 矩阵 | 地址公式 | major 对应维度 | minor 对应维度 |
|------|----------|------------------|----------------|
| **A** | `base + k_cnt × stride_major_A + m_cnt × stride_minor_A` | **K**（沿 K 方向换块步进 4096 B） | **M**（沿 M 方向步进 `4096 × K_block`） |
| **B** | `base + k_cnt × stride_major_B + n_cnt × stride_minor_B` | **K**（沿 K 换块步进 `4096 × N_block`） | **N**（沿 N 换块步进 4096 B） |
| **Din / Dout** | `base + n_cnt × stride_major_D + m_cnt × stride_minor_D` | **N**（沿 N 换块步进 16384 B） | **M**（沿 M 换块步进 `16384 × N_block`） |

因此：

- **A**：块网格上 **先按 (m_block, k_block) 排布**；在地址公式里 **K 为 major 步长（单块 4KB）、M 为 minor 步长（跨一整条 K 向块带）**。
- **B**：块网格 **(k_block, n_block)**；**K 为 major、N 为 minor**。
- **Din/Dout**：块网格 **(m_block, n_block)**；与 `tc_d_block_idx` 注释一致：**N 为 major、M 为 minor**。

---

## 2. 块之间：硬件 FSM 的遍历顺序

主状态机在 `CTRL_IDLE` 收到指令后，从 `(m_cnt,n_cnt,k_cnt)=(0,0,0)` 开始。

对**每一个输出块** `(m_cnt, n_cnt)`：

1. **LOAD_DIN**：按 `addr_din_curr` 读当前块的 **Din**（16384 B 突发）。
2. **K 内循环**（`CTRL_LOOP_K_*` … `CTRL_LOOP_K_SA_START`）：
   - 读 **A** 块 `(m_cnt, k_cnt)`；
   - 读 **B** 块 `(k_cnt, n_cnt)`；
   - 启动 SA 做该 K 片的乘加；若尚未到最后一个 K，则 `k_cnt++`，回到读 A。
3. **最后一个 K**（`is_last_k`）：经固定延迟后 **STORE_DOUT**，把累加结果写回 **Dout** 的 `(m_cnt, n_cnt)` 块。
4. **CTRL_UPDATE_MN**：`k_cnt` 清零，`m_cnt/n_cnt` 更新为 `next_m_cnt/next_n_cnt`，再进入下一块的 LOAD_DIN（或结束）。

`next_n_cnt` / `next_m_cnt` 的组合逻辑决定：**`n_cnt` 先递增**；当 `n_cnt` 从 `N_block-1` 回到 0 时 **`m_cnt` 加 1**。因此块级访问顺序为：

```text
(m,n) = (0,0) → (0,1) → … → (0,N_block-1) → (1,0) → … → (M_block-1, N_block-1)
```

即 **外层：m_cnt（M 方向块）**；**内层：n_cnt（N 方向块）**。在每个 `(m,n)` 内，**最内层是 k_cnt（K 方向块）**，完成对该输出块在 K 上的归约。

用伪代码概括：

```text
for m_cnt = 0 .. M_block-1
  for n_cnt = 0 .. N_block-1
    load Din block (m_cnt, n_cnt)
    for k_cnt = 0 .. K_block-1
      load A block (m_cnt, k_cnt)
      load B block (k_cnt, n_cnt)
      MAC 累加到当前输出块
    store Dout block (m_cnt, n_cnt)
```

---

## 3. 块内部：64×64 子阵在内存中的布局
单块大小：A/B 为 **4096 B**（64×64×int8）；Din/Dout 为 **16384 B**（64×64×int32）。以下“块内/块间的内层-外层”均以 `gen_data.py` 的线性打包顺序为准（即谁在最内层循环里变化）。

先定义局部坐标（假设 M/N/K 都是 64 的倍数）：

- M 方向：`m = mb*64 + ml`，其中 `ml` 0..63
- K 方向：`k = kb*64 + kl`，其中 `kl` 0..63
- N 方向：`n = nb*64 + nl`，其中 `nl` 0..63

另外：`main.c` 里指令把 `base_addr` 传成 `ptr >> 8`，所以 RTL 里实际的字节基址是 `base_bytes = base_addr << 8`。

下面每个矩阵都给出 4 部分：
1. 块内维度：内层/外层（变化快/慢）
2. 块间维度：内层/外层（变化快/慢）
3. 完整遍历顺序伪代码（对应 `gen_data.py` 打包）
4. 硬件地址计算公式（对应 `tc_controller.sv` 的 `addr_*`）

### 3.1 A（逻辑 A[M×K]，int8）

**(1) 块内维度（64×64 内）**
- 内层（变化快）：`ml`（M 的局部行）
- 外层（变化慢）：`kl`（K 的局部列）
- 块内线性 offset（元素级）为：`offA = kl*64 + ml`
- 由于是 int8：块内字节 offset 为 `offA_byte = offA`

**(2) 块间维度（64×64 block 网格）**
- 内层（变化快）：`kb`（K-block）
- 外层（变化慢）：`mb`（M-block）

**(3) 完整遍历顺序伪代码（内存打包，等价于 gen_data.py）**
```text
for mb in 0..Mb-1:        # 外层（慢）
  for kb in 0..Kb-1:      # 内层（快）
    for kl in 0..63:      # 外层（块内慢）
      for ml in 0..63:    # 内层（块内快）
        write A[m=mb*64+ml][k=kb*64+kl] as int8
```

**(4) 硬件地址计算公式（DMA 读 A）**

`tc_controller.sv`：
- 块基址（byte）：
  `addr_a_block = base_A_bytes + k_cnt * stride_major_A + m_cnt * stride_minor_A`
  - 这里 `k_cnt=kb`，`m_cnt=mb`
- 块内元素字节地址：
  `addr_a_elem(m,k) = addr_a_block + (kl*64 + ml)`

在当前 `main.c`（无转置）下，`stride_major_A = 4096`，`stride_minor_A = 4096 * K_block`。

### 3.2 B（逻辑 B[K×N]，int8）

**(1) 块内维度（64×64 内）**
- 内层（变化快）：`nl`（N 的局部列）
- 外层（变化慢）：`kl`（K 的局部行/列索引中对应的局部 K）
- 块内线性 offset（元素级）为：`offB = kl*64 + nl`
- int8：`offB_byte = offB`

**(2) 块间维度（64×64 block 网格）**
- 内层（变化快）：`nb`（N-block）
- 外层（变化慢）：`kb`（K-block）

**(3) 完整遍历顺序伪代码（内存打包，等价于 gen_data.py）**
```text
for kb in 0..Kb-1:        # 外层（慢）
  for nb in 0..Nb-1:      # 内层（快）
    for kl in 0..63:      # 外层（块内慢）
      for nl in 0..63:    # 内层（块内快）
        write B[k=kb*64+kl][n=nb*64+nl] as int8
```

**(4) 硬件地址计算公式（DMA 读 B）**

`tc_controller.sv`：
- 块基址（byte）：
  `addr_b_block = base_B_bytes + k_cnt * stride_major_B + n_cnt * stride_minor_B`
  - 这里 `k_cnt=kb`，`n_cnt=nb`
- 块内元素字节地址：
  `addr_b_elem(k,n) = addr_b_block + (kl*64 + nl)`

在当前 `main.c`（无转置）下，`stride_major_B = 4096 * N_block`，`stride_minor_B = 4096`。

### 3.3 Din / Dout / G（逻辑 M×N，int32）

> Din/Dout/G 在打包里使用同一布局（元素类型 int32），所以块内/块间的内层外层完全一致。

**(1) 块内维度（64×64 内）**
- 内层（变化快）：`ml`（M 的局部行）
- 外层（变化慢）：`nl`（N 的局部列）
- 块内线性 offset（元素级）为：`offD = nl*64 + ml`
- int32：块内字节 offset 为 `offD_byte = offD * 4`

**(2) 块间维度（64×64 block 网格）**
- 内层（变化快）：`nb`（N-block）
- 外层（变化慢）：`mb`（M-block）

**(3) 完整遍历顺序伪代码（内存打包，等价于 gen_data.py）**
```text
for mb in 0..Mb-1:        # 外层（慢）
  for nb in 0..Nb-1:      # 内层（快）
    for nl in 0..63:      # 外层（块内慢）
      for ml in 0..63:    # 内层（块内快）
        write Din/Dout/G[m=mb*64+ml][n=nb*64+nl] as int32
```

**(4) 硬件地址计算公式（DMA 读 Din / 写 Dout）**

`tc_controller.sv`：
- Din 当前块读取块基址（byte）：
  `addr_din_block = base_D_bytes + n_cnt * stride_major_D + m_cnt * stride_minor_D`
  - `n_cnt=nb`，`m_cnt=mb`
- Dout 写回块基址（byte）：
  `addr_dout_block = base_Dout_bytes + n_cnt * stride_major_Dout + m_cnt * stride_minor_Dout`
  - `n_cnt=nb`，`m_cnt=mb`
- 块内元素字节地址（以 Din/Dout/G 通用表达）：
  `addr_d_elem(m,n) = addr_d_block + (nl*64 + ml) * 4`

在当前 `main.c`（无转置）下，`stride_major_D = 16384`，`stride_minor_D = 16384 * N_block`。

---

## 4. 小结表
| 矩阵 | 块内变化快维度 | 块内变化慢维度 | 块间变化快维度 | 块间变化慢维度 |
|------|------------------|------------------|------------------|------------------|
| **A** | `ml` | `kl` | `kb` | `mb` |
| **B** | `nl` | `kl` | `nb` | `kb` |
| **Din/Dout/G** | `ml` | `nl` | `nb` | `mb` |

---

## 5. 转置位（可选）

若 `if_A_transpose` 或 `if_B_transpose` 为 1，stride 与 DMA 转置标志会按 `tc_top_tb.sv` 中注释切换（例如 A 视为 K×M 块阵时 major/minor 对调）。当前 `tensorcore_to_cpu/main.c` 使用全 0 转置，上述即其实际路径。
