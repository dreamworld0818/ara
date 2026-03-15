# TensorCore 测试案例：A / B / Din 在 L2 与 RRAM 的 8 种组合

测试目标：覆盖 **A、B、Din** 分别放在 **L2** 或 **RRAM** 的所有组合，TensorCore 计算结果 **Dout 始终写回 L2**，由 CPU 从 L2 读 Dout 与黄金参考比对。

约定：
- **L2**：基址 `0x8000_0000`（DRAM）
- **RRAM**：基址 `0x1000_0000`
- **1** = 该矩阵在 L2，**0** = 该矩阵在 RRAM

---

## 测试案例一览

| 案例 | A 位置 | B 位置 | Din 位置 | Dout 位置 | TC_A_IN_L2 | TC_B_IN_L2 | TC_DIN_IN_L2 | 说明 |
|------|--------|--------|----------|-----------|-------------|-------------|---------------|------|
| 1    | L2     | L2     | L2       | L2        | 1           | 1           | 1             | 全 L2 |
| 2    | L2     | L2     | RRAM     | L2        | 1           | 1           | 0             | 仅 Din 在 RRAM |
| 3    | L2     | RRAM   | L2       | L2        | 1           | 0           | 1             | 仅 B 在 RRAM |
| 4    | L2     | RRAM   | RRAM     | L2        | 1           | 0           | 0             | B、Din 在 RRAM |
| 5    | RRAM   | L2     | L2       | L2        | 0           | 1           | 1             | 仅 A 在 RRAM |
| 6    | RRAM   | L2     | RRAM     | L2        | 0           | 1           | 0             | A、Din 在 RRAM |
| 7    | RRAM   | RRAM   | L2       | L2        | 0           | 0           | 1             | A、B 在 RRAM |
| 8    | RRAM   | RRAM   | RRAM     | L2        | 0           | 0           | 0             | 全 RRAM（仅 Dout 在 L2）|

---

## 参数含义（SoC 顶层）

`ara_soc` 提供三个参数（可在仿真/综合时覆盖），并传入 `ara_system` → `tc_top` → `tc_controller`：

- **TC_A_IN_L2**：1 = 矩阵 A 在 L2，0 = 矩阵 A 在 RRAM  
- **TC_B_IN_L2**：1 = 矩阵 B 在 L2，0 = 矩阵 B 在 RRAM  
- **TC_DIN_IN_L2**：1 = 矩阵 Din 在 L2，0 = 矩阵 Din 在 RRAM  

Dout 固定使用 L2，无需配置。运行某案例时，在仿真或综合中传入上表对应的一行参数即可（例如案例 8：`TC_A_IN_L2=0, TC_B_IN_L2=0, TC_DIN_IN_L2=0`）。

---

## 应用侧与数据布局

- 每个案例对应一种 **A/B/Din 的段与基址**：
  - 在 L2 的矩阵：链接到 `.l2`，指令中 `base_addr` 为相对 L2 的字节偏移（如 `(uint32_t)ptr - 0x80000000`）。
  - 在 RRAM 的矩阵：链接到 `.rram`，指令中 `base_addr` 为相对 RRAM 的字节偏移（如 `(uint32_t)ptr - 0x10000000`）。
- 黄金参考 G 与 Dout 均在 L2，CPU 在 L2 中比对 Dout 与 G。

运行某案例时：选用上表中对应的 **TC_A_IN_L2 / TC_B_IN_L2 / TC_DIN_IN_L2** 做一次 SoC 仿真（或 8 次仿真分别覆盖 8 行），应用侧数据布局与链接脚本与该案例一致。
