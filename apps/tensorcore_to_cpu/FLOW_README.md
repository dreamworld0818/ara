# TensorCore 与 CPU 完整流程验证说明（仅硬件仿真）

本文档描述 **仅做硬件仿真** 时，从 CPU 发指令到 TensorCore、TensorCore 执行 GEMM、结果写回内存、CPU 读取并与黄金参考比对的完整数据流，以及为完成该验证需要修改的文件与内容。

---

## 一、完整数据流概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CPU 侧 (CVA6, 运行在 L2/DRAM)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. 准备数据: A[M×K], B[K×N], Din[M×N] 放入内存 (L2)，生成黄金参考 G = A×B+Din   │
│  2. 构建 MMA 指令 (40 字节): base_addr/stride、M/N/K block、转置等               │
│  3. 通过 MMIO 写 0xD000_1000 将指令发给 TensorCore                               │
│  4. （可选）轮询状态：TensorCore 计算结束时 slv_register_output_data 置 1，读到非 0 即完成 │
│  5. 从内存读取 Dout[]，与 G[] 逐元素比对，报告 PASS/FAIL                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │ 指令 (MMIO 写)                    │ 结果 (读内存)
                    ▼                                  ▲
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TensorCore 侧 (AXI Slave + AXI Master)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • Slave: 接收 CPU 写入的指令 (0xD000_1000)，解析 M/N/K、Din/A/B/Dout 地址信息   │
│  • 控制 FSM: 按 K 循环读 A/B，读 Din，启动 SA，写 Dout                            │
│  • Master (DMA): 按解析出的地址从总线读 A、B、Din；算完后写 Dout 回总线           │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │ 读 A/B/Din、写 Dout (AXI 访问同一 L2/RRAM)
                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SoC 总线与内存                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • L2 (DRAM): 0x8000_0000, 32MB — 放 A, Din, Dout, G，CPU 与 TensorCore 共享    │
│  • RRAM:      0x1000_0000 — 可选放 B；若仅硬件仿真可把 B 也放 L2 以简化          │
│  • TC 从机: 0xD000_1000 写指令；读状态（返回 slv_register_output_data，置 1 表示计算结束） │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、分步数据流（与现有实现对应）

### 2.1 当前已实现部分

- **CPU 发送指令**：`send_tensorcore_instruction(&inst)` 向 `TENSORCORE_BASE_ADDR (0xD0001000)` 写入 40 字节指令。
- **TensorCore 接收并打印**：`tc_controller.sv` 在收到完整写事务后，将指令写入 `slv_register_input_data`，并在 `VERILATOR` 下打印收到的指令内容。
- **TensorCore 完成标志（已验证行为）**：`hardware/deps/tensor-core` 中 TensorCore 已通过验证。计算结束时 FSM 在 `CTRL_UPDATE_MN` 且 `is_last_mn` 时将 **`slv_register_output_data` 置 1**，表示 GEMM 已结束。CPU 通过**轮询**读该状态，直到读到非 0 再读 Dout，避免读到未写完的数据。

### 2.2 完整流程应具备的环节

| 步骤 | 位置 | 说明 |
|------|------|------|
| 1 | CPU | 在 L2 中准备 A、B、Din，并生成黄金参考 G = A×B + Din（软件 GEMM） |
| 2 | CPU | 按 `gemm_instruction.h` 构建 `mma_instruction_t`（Din/A/B/Dout 的 base_addr、stride_minor/major，以及 M/N/K block、instruction_type 等） |
| 3 | CPU | `send_tensorcore_instruction(&inst)` → MMIO 写 0xD000_1000 |
| 4 | TensorCore | 从 AXI Slave 收到写请求，写入内部寄存器，解析为 GEMM 参数 |
| 5 | TensorCore | 控制 FSM 启动：DMA 按**指令中的 base + strides** 得到字节地址从总线读 Din、A、B，写 Dout（TB 下 base 即 0-based 空间基址；SoC 下 base 为区域内的基址/偏移，RTL 需加 L2/RRAM 区域基址） |
| 6 | TensorCore | 执行 GEMM（Dout = A×B + Din），结果经 DMA 写回内存 |
| 7 | CPU | 可选：轮询读状态，TensorCore 将 slv_register_output_data 置 1 表示完成，读到非 0 后确认完成 |
| 8 | CPU | 从内存读 Dout，与 G 逐元素比较，输出验证结果 |

---

## 三、为何 TensorCore 已验证，在 SoC 里仍可能改 tc_controller？地址语义是什么？

TensorCore 在 **`hardware/deps/tensor-core/tb/tc_top_tb.sv`** 里已经按「指令中 base 即该段内存在地址空间中的基址、RTL 用 base + strides 得到字节地址」的约定验证过，RTL 本身是自洽的。

### 3.1 测试台 (TB) 里的约定

- **TB 内存为 0-based 地址空间**：仿真里有一块“假内存”响应用户的 AXI 访问，地址从 0 开始。
- **指令中的 base 即该段内存在该地址空间里的基址**：TB 里例如 `base_addr = 24'h10_0000`（A）、`24'h20_0000`（B）、`24'h30_0000`（Din）、`24'h40_0000`（Dout），即 **base 就是该矩阵在 0-based 地址空间中的基地址**；stride 为字节步长（如 A/B 用 4096，Din/Dout 用 16384）。
- **RTL 用 base + strides**：`tc_controller` 里 `addr = zero_extend(base) + n*stride_major + m*stride_minor`，得到的就是 DMA 要发的 AXI 字节地址。在 TB 里内存就在 0，所以**只做零扩展、不加其它基址是正确的**，与“base 即基地址”的语义一致。

所以在 TB 环境下：**RTL 用 base + strides 作为地址，是正确且已验证的。**

### 3.2 SoC 里为什么要在 RTL 加区域基址？

在 SoC 里：

- **L2 在 0x8000_0000**，RRAM 在 0x1000_0000，与 TB 的 0-based 空间不同。
- 指令里 **base_addr 仅 24 位**（`tc_pkg::ISA_ADDR_WIDTH = 24`），无法表示 0x80000000 或 0x10000000。因此 SoC 上软件无法在指令里直接填“SoC 物理基址”，只能填**该段在所在区域内的基址（即相对 L2 或 RRAM 的字节偏移）**。
- RTL 若仍只做 `addr = zero_extend(base) + strides`，发到总线的是 0x5000、0x6c0000 等，在 SoC 的 memory map 里**不落在 L2/RRAM**，访问会错或挂空。
- 因此 SoC 集成时，RTL 必须在发出 AXI 地址前加上**区域基址**：  
  `axi_addr = (xxx_IN_L2 ? L2_BASE_ADDR : RRAM_BASE_ADDR) + zero_extend(base) + n*stride_major + m*stride_minor`。  
  这样“指令中的 base”表示**区域内的基址**，RTL 加上区域基址后得到 SoC 物理地址，与 TB 的语义一致（TB 相当于区域基址为 0）。

结论：**TensorCore 核内“base + strides”的地址语义不变；为接到 SoC，必须在 RTL 中为 Din/A/B/Dout 分别加上 L2 或 RRAM 的区域基址。**

### 3.3 指令与地址约定（软件与 RTL 一致）

| 环境 | 指令中 base 的含义 | 软件填什么 | RTL 做什么 |
|------|-------------------|------------|------------|
| **TB** | 该段在 0-based 地址空间中的基址（字节） | TB 填 0x100000、0x200000 等 | `addr = zero_extend(base) + strides`，无需加基址 |
| **SoC** | 该段在所在区域（L2 或 RRAM）内的基址，即相对区域起点的字节偏移 | `base_addr = (uint32_t)ptr - 0x80000000`（L2）或减 RRAM 基址；stride 为字节步长（4096/16384） | `addr = L2_BASE/RRAM_BASE + zero_extend(base) + strides`，按 A_IN_L2/B_IN_L2/DIN_IN_L2 选基址；Dout 固定 L2 |

---

## 四、为实现完整验证需要修改的文件与内容

### 4.1 硬件 RTL：SoC 下加区域基址

**文件**：`hardware/deps/tensor-core/src/sv/tensor_core_components/tc_controller.sv`

- **地址语义（与 TB 一致）**：指令中的 base 表示该段在**所在地址空间中的基址**；RTL 计算 `addr = base + n*stride_major + m*stride_minor`（零扩展）。TB 下地址空间从 0 开始，故无需再加基址；SoC 下需加区域基址。
- **SoC 集成要求**：在 SoC 中，`tc_top` 已提供参数 `L2_BASE_ADDR`、`RRAM_BASE_ADDR`（默认 0，TB 兼容）、`A_IN_L2`、`B_IN_L2`、`DIN_IN_L2`。**tc_controller 中**对 Din/A/B/Dout 的地址计算需为：  
  `addr_* = (xxx_IN_L2 ? L2_BASE_ADDR : RRAM_BASE_ADDR) + zero_extend(base) + n*stride_major + m*stride_minor`。  
  Dout 固定写 L2，用 L2_BASE_ADDR。SoC 由 `ara_soc.sv` → `ara_system.sv` 传入上述参数。
- **软件约定**：SoC 上软件填 base 为**相对所在区域的字节偏移**（如 L2 内 `base_addr = (uint32_t)ptr - 0x80000000`），stride 为字节步长，与 3.3 一致。
- **注意**：`tc_controller.sv` 中 `VERILATOR` 下 `print_instruction()` 的 “Calculated Addresses” 为调试用显示，可能与实际 DMA 使用的 `addr_*` 形式不同；实际访问以 RTL 的 `addr_*` 为准。

### 4.2 TensorCore 完成标志与 CPU 轮询（已验证行为）

**文件**：`hardware/deps/tensor-core` 中 TensorCore 为已验证代码。

- **TensorCore 侧**：计算结束时，FSM 在 `CTRL_UPDATE_MN` 且 `is_last_mn` 时将 **`slv_register_output_data` 置 1**，表示 GEMM 已结束。该行为为 TensorCore 既有、已验证逻辑。
- **CPU 侧**：通过**轮询**读取该状态实现同步。从机返回的 `slv_register_output_data`；读到非 0（置 1 表示完成）后再从内存读 Dout，与黄金参考比对。轮询地址：与 `gemm.h` 中 `TENSORCORE_STATE_ADDR` 一致（如 0xD000_2000）；SoC 将整段 0xD000_1000..0xD000_2FFF 映射到 TensorCore，读该段内任意地址均返回同一状态寄存器。

### 4.3 SoC 地址映射与 TC 从机范围

**文件**：`hardware/src/ara_soc.sv`

- **确认**：TensorCore 从机基址为 `TCBase = 64'hD000_1000`，长度 `TCLength = 64'h2000`（即 0xD000_1000..0xD000_2FFF）。写指令写 0xD000_1000，读状态可用同一地址或 0xD000_2000（与 `tensorcore_gemm` 的 `TENSORCORE_STATE_ADDR` 一致）。

### 4.4 应用工程：`apps/tensorcore_to_cpu`

**目录**：`apps/tensorcore_to_cpu/`

- **main.c**  
  - 声明/定义 M、N、K 及 `A[]`、`B[]`、`Din[]`、`Dout[]`、`G[]`，与 `tensorcore_gemm` 类似；为便于**仅硬件仿真**，建议 A/B/Din/Dout/G 全部放在 **.l2**（L2），这样 TensorCore 与 CPU 共享同一块内存，避免 RRAM 参与。  
  - 调用 `gen_data.py` 生成的数据后，用软件 GEMM 计算黄金参考 `G = A*B + Din`（或直接使用 gen_data 生成的 G）。  
  - 填充 `mma_instruction_t`：指令中 **base 表示该段在所在地址空间中的基址**。SoC 上因 24 位无法表示 0x80000000，故填**相对 L2/RRAM 的字节偏移**（如 L2 内 `base_addr = (uint32_t)ptr - 0x80000000`），stride 为字节步长（A/B 用 4096，Din/Dout 用 16384）。  
  - 调用 `send_tensorcore_instruction(&inst)`。  
  - 轮询 `load_tensorcore_state()`，TensorCore 计算结束时 slv_register_output_data 置 1，读到非 0 即完成，再读 Dout。  
  - 从内存读取 `Dout[]`，与 `G[]` 逐元素比较，打印 PASS/FAIL 或第一个错误位置。

- **kernel/gemm.c、kernel/gemm.h、kernel/gemm_instruction.h**  
  - 保持与 `tensorcore_gemm` 一致即可（`TENSORCORE_BASE_ADDR`、`TENSORCORE_STATE_ADDR`、`send_tensorcore_instruction`、`load_tensorcore_state`）；若 SoC 将状态放在不同偏移，需在 `gemm.h` 中改 `TENSORCORE_STATE_ADDR`。

- **script/gen_data.py**  
  - 若希望**仅用 L2**（硬件仿真不跑 RRAM）：将 B 也生成到 `.l2`（与 `tensorcore_gemm` 的 A、Din、Dout、G 类似），并保证 A/B 按 4096 对齐、Din/Dout/G 按 16384 对齐，以便与 RTL 的块索引换算一致。  
  - 当前 `gen_data.py` 已把 A、Din、Dout、G 放在 `.l2`，B 放在 `.rram`；若 RTL 中 B 也访问 L2，则需在 gen_data 中把 B 放到 `.l2`，并在指令里 B 的 base_addr 使用相对 L2 的块索引。

- **链接脚本 / Makefile**  
  - 使用 `apps/common/arch.link.ld`，保证 `.l2`、`.rram` 与 SoC 的 L2/RRAM 基址一致。  
  - **硬件仿真**时使用仓库根目录下 `apps/Makefile` 的通用规则：在 `apps` 下执行 `make bin/tensorcore_to_cpu` 会生成 `bin/tensorcore_to_cpu`（ELF）。该规则会先根据 `def_args_tensorcore_to_cpu` 调用 `script/gen_data.py` 生成 `data.S`，再编译链接。`apps/common/default_args.mk` 中已包含 `def_args_tensorcore_to_cpu ?= "127 254 127"`（与 tensorcore_gemm 类似，M N K 参数）。

### 4.5 硬件仿真顶层 / Testbench

**文件**：如 `hardware/tb/ara_tb.sv` 或当前使用的仿真顶层

- **确认**：CPU 从 L2 取指、取数，TensorCore 通过 AXI 访问同一 L2（及可选 RRAM）；无需 Spike。  
- **加载镜像**：将 `tensorcore_to_cpu` 编译得到的 elf/hex 加载到 L2 的 0x8000_0000（或 SoC 定义的 DRAM 基址），保证 A、B、Din、Dout、G 的链接地址落在 L2 范围内，且满足上述对齐。

### 4.6 小结：修改清单

| 序号 | 文件 | 修改内容 |
|------|------|----------|
| 1 | `hardware/deps/tensor-core/.../tc_controller.sv` | **SoC 必须**：在发出的 AXI 地址上为 Din/A/B/Dout 分别加 L2_BASE 或 RRAM_BASE（与 3.1/3.2 的“base 即基址、SoC 上加区域基址”一致）。 |
| 2 | SoC/TC 从机 | TensorCore 已把完成标志放在 slv_register_output_data（置 1 表示完成） |
| 3| `apps/tensorcore_to_cpu/main.c` | 完整流程：准备数据、填指令、发指令、等完成、读 Dout、与 G 比对 |
| 4 | `apps/tensorcore_to_cpu/script/gen_data.py` | 可选：B 放 .l2，保证仅 L2 参与，便于硬件仿真 |
| 5 | `apps/Makefile`（通用规则） | 生成 `data.S`、编译链接，得到硬件仿真用 elf（`make bin/tensorcore_to_cpu`）。`apps/tensorcore_to_cpu/Makefile` 仅用于 Spike 设备插件，不参与 elf 构建。 |
| 6 | 仿真顶层 / Verilator | 在 `hardware/` 下 `make verilate` 后执行 `make simv app=tensorcore_to_cpu`，通过 `-E apps/bin/tensorcore_to_cpu` 将 elf 加载到 L2（0x8000_0000）与 RRAM（0x1000_0000），运行硬件仿真。 |
| 7 | `apps/common/default_args.mk` | 添加 `def_args_tensorcore_to_cpu ?= "M N K"` 供 data.S 生成使用 |

---

## 五、仅硬件仿真时的推荐配置

- **内存**：A、B、Din、Dout、G 全部放在 **L2 (0x8000_0000)**，RTL 中 Din/A/B/Dout 的基址均使用 L2_BASE，避免 RRAM 参与，简化地址与数据生成。
- **数据生成**：`gen_data.py` 对 A/B 使用 4096 对齐，对 Din/Dout/G 使用 16384 对齐；若 B 放 L2，则 B 也生成在 `.l2`。
- **验证**：CPU 发指令后通过轮询读状态（TensorCore 将 slv_register_output_data 置 1 表示完成），再读 Dout 与 G 比较；可输出比较结果到 UART 或 testbench 可观测的地址，便于自动化判断 PASS/FAIL。

按上述数据流与修改清单实现后，即可在**仅硬件仿真**下完成：CPU 发指令 → TensorCore 收指令、解析、执行 GEMM、写回内存 → CPU 读结果并与黄金参考比对的完整验证。
