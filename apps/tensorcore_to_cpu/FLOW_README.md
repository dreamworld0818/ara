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
│  4. （可选）轮询状态寄存器 0xD000_2000 等待完成                                   │
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
│  • TC 寄存器: 0xD000_1000 指令, 0xD000_2000 状态                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、分步数据流（与现有实现对应）

### 2.1 当前已实现部分

- **CPU 发送指令**：`send_tensorcore_instruction(&inst)` 向 `TENSORCORE_BASE_ADDR (0xD0001000)` 写入 40 字节指令。
- **TensorCore 接收并打印**：`tc_controller.sv` 在收到完整写事务后，将指令写入 `slv_register_input_data`，并在 `VERILATOR` 下打印收到的指令内容。

### 2.2 完整流程应具备的环节

| 步骤 | 位置 | 说明 |
|------|------|------|
| 1 | CPU | 在 L2 中准备 A、B、Din，并生成黄金参考 G = A×B + Din（软件 GEMM） |
| 2 | CPU | 按 `gemm_instruction.h` 构建 `mma_instruction_t`（Din/A/B/Dout 的 base_addr、stride_minor/major，以及 M/N/K block、instruction_type 等） |
| 3 | CPU | `send_tensorcore_instruction(&inst)` → MMIO 写 0xD000_1000 |
| 4 | TensorCore | 从 AXI Slave 收到写请求，写入内部寄存器，解析为 GEMM 参数 |
| 5 | TensorCore | 控制 FSM 启动：DMA 按**字节地址**从总线读 Din、A、B，写 Dout（地址需由“块索引×块大小+基址”得到） |
| 6 | TensorCore | 执行 GEMM（Dout = A×B + Din），结果经 DMA 写回内存 |
| 7 | CPU | 可选：读 0xD000_2000 状态寄存器确认完成 |
| 8 | CPU | 从内存读 Dout，与 G 逐元素比较，输出验证结果 |

---

## 三、指令与地址约定（软件与 RTL 需一致）

- **块大小**：A/B 为 64×64 int8 → 4096 字节/块；Din/Dout 为 64×64 int32 → 16384 字节/块。
- **软件侧**（如 `tensorcore_gemm/main.c`）：  
  `base_addr = (uint32_t)ptr / 对齐字节`（Din/Dout 除以 16384，A/B 除以 4096），即传的是**块索引**。
- **RTL 侧**：DMA 发出的是 **AXI 字节地址**，且需落在 L2 (0x8000_0000) 或 RRAM (0x1000_0000)。  
  因此 RTL 必须将“块索引 + stride”转换为字节地址，即：
  - **Din/Dout**：`byte_addr = L2_BASE + (base_addr + n*stride_major + m*stride_minor) * 16384`
  - **A/B**：若在 L2，`byte_addr = L2_BASE + block_index * 4096`；若在 RRAM，用 RRAM_BASE。

当前 `tc_controller.sv` 中 `addr_din_curr`、`addr_a`、`addr_b`、`addr_dout` 是直接用 24 位 base 与 20 位 stride 零扩展后相加，**没有**乘以块大小、也没有加 L2/RRAM 基址，因此与软件传的“块索引”不一致。要实现完整验证，必须在 RTL 中做上述**地址换算**（见第四节）。

---

## 四、为实现完整验证需要修改的文件与内容

### 4.1 硬件 RTL：地址换算与内存区域

**文件**：`hardware/deps/tensor-core/src/sv/tensor_core_components/tc_controller.sv`

- **问题**：当前 `addr_din_curr`、`addr_din_next`、`addr_a`、`addr_b`、`addr_dout` 仅做 `base + n*stride_major + m*stride_minor` 的零扩展，未按块大小与内存基址换算。
- **修改思路**：
  - 定义常量（或参数）：如 `L2_BASE = 64'h8000_0000`，`RRAM_BASE = 64'h1000_0000`；Din/Dout 块大小 16384，A/B 块大小 4096。
  - 对 Din/Dout：先计算块索引 `block_idx = base_addr + n*stride_major + m*stride_minor`，再 `addr = L2_BASE + block_idx * 16384`（或等价左移/乘法）。
  - 对 A/B：同样先得到块索引，再 `addr = L2_BASE + block_idx * 4096`（若 B 在 RRAM 则用 RRAM_BASE）。
- **注意**：与 SoC 中 L2/RRAM 的基址保持一致（参考 `ara_soc.sv` 中 `DRAMBase`、`RRAMBase`）。

### 4.2 硬件 RTL：状态寄存器与完成标志（可选但推荐）

**文件**：同上 `tc_controller.sv`（或 TensorCore 顶层/总线上挂的状态寄存器）

- **目的**：让 CPU 能轮询“计算是否完成”，再读 Dout，避免读到未写完的数据。
- **做法**：在 FSM 进入 `CTRL_IDLE` 且 `is_last_mn` 时，将“完成”状态写入可被 CPU 读的寄存器；该寄存器映射到 **0xD000_2000**（与 `TENSORCORE_STATE_ADDR` 一致）。当前 `slv_register_output_data <= '1` 等可视为内部状态，需确保 0xD000_2000 的读通道返回的是“完成”标志，且与 SoC 中 TC 从机的地址映射一致（若 0xD000_2000 属于同一从机，需在 TC 内实现对该地址的读响应）。

### 4.3 SoC 地址映射与 TC 从机范围

**文件**：`hardware/src/ara_soc.sv`

- **确认**：TensorCore 从机基址为 `TCBase = 64'hD000_1000`，长度 `TCLength = 64'h1000`，即 0xD000_1000–0xD000_1FFF。若状态寄存器 0xD000_2000 也由 TensorCore 提供，需在 SoC 中把 0xD000_2000 也划入 TC 从机范围，或单独增加一档映射；否则需在 TC 内用 0xD000_1000 的某偏移表示状态（例如 0xD000_1010），并在软件中把 `TENSORCORE_STATE_ADDR` 改为该偏移。

### 4.4 应用工程：`apps/tensorcore_to_cpu`

**目录**：`apps/tensorcore_to_cpu/`

- **main.c**（当前为空）  
  - 声明/定义 M、N、K 及 `A[]`、`B[]`、`Din[]`、`Dout[]`、`G[]`，与 `tensorcore_gemm` 类似；为便于**仅硬件仿真**，建议 A/B/Din/Dout/G 全部放在 **.l2**（L2），这样 TensorCore 与 CPU 共享同一块内存，避免 RRAM 参与。  
  - 调用 `gen_data.py` 生成的数据后，用软件 GEMM 计算黄金参考 `G = A*B + Din`（或直接使用 gen_data 生成的 G）。  
  - 按当前 `gemm_instruction.h` 和 `tensorcore_gemm/main.c` 的方式填充 `mma_instruction_t`（base_addr 为块索引：Din/Dout 用 `(uint32_t)ptr/16384`，A/B 用 `(uint32_t)ptr/4096`；stride 为 block 维度）。  
  - 调用 `send_tensorcore_instruction(&inst)`。  
  - 可选：轮询 `load_tensorcore_state()` 直到完成。  
  - 从内存读取 `Dout[]`，与 `G[]` 逐元素比较，打印 PASS/FAIL 或第一个错误位置。

- **kernel/gemm.c、kernel/gemm.h、kernel/gemm_instruction.h**  
  - 保持与 `tensorcore_gemm` 一致即可（`TENSORCORE_BASE_ADDR`、`TENSORCORE_STATE_ADDR`、`send_tensorcore_instruction`、`load_tensorcore_state`）；若 SoC 将状态放在不同偏移，需在 `gemm.h` 中改 `TENSORCORE_STATE_ADDR`。

- **script/gen_data.py**  
  - 若希望**仅用 L2**（硬件仿真不跑 RRAM）：将 B 也生成到 `.l2`（与 `tensorcore_gemm` 的 A、Din、Dout、G 类似），并保证 A/B 按 4096 对齐、Din/Dout/G 按 16384 对齐，以便与 RTL 的块索引换算一致。  
  - 当前 `gen_data.py` 已把 A、Din、Dout、G 放在 `.l2`，B 放在 `.rram`；若 RTL 中 B 也访问 L2，则需在 gen_data 中把 B 放到 `.l2`，并在指令里 B 的 base_addr 使用相对 L2 的块索引。

- **链接脚本 / Makefile**  
  - 使用 `apps/common/arch.link.ld`，保证 `.l2`、`.rram` 与 SoC 的 L2/RRAM 基址一致。  
  - **硬件仿真**时使用仓库根目录下 `apps/Makefile` 的通用规则：在 `apps` 下执行 `make bin/tensorcore_to_cpu` 会生成 `bin/tensorcore_to_cpu`（ELF）。该规则会先根据 `def_args_tensorcore_to_cpu` 调用 `script/gen_data.py` 生成 `data.S`，再编译链接。需在 `apps/common/default_args.mk` 中增加一行，例如：`def_args_tensorcore_to_cpu ?= "127 254 127"`（与 tensorcore_gemm 类似，M N K 参数）。

### 4.5 硬件仿真顶层 / Testbench

**文件**：如 `hardware/tb/ara_tb.sv` 或当前使用的仿真顶层

- **确认**：CPU 从 L2 取指、取数，TensorCore 通过 AXI 访问同一 L2（及可选 RRAM）；无需 Spike。  
- **加载镜像**：将 `tensorcore_to_cpu` 编译得到的 elf/hex 加载到 L2 的 0x8000_0000（或 SoC 定义的 DRAM 基址），保证 A、B、Din、Dout、G 的链接地址落在 L2 范围内，且满足上述对齐。

### 4.6 小结：修改清单

| 序号 | 文件 | 修改内容 |
|------|------|----------|
| 1 | `hardware/deps/tensor-core/.../tc_controller.sv` | 地址换算：块索引 × 块大小 + L2/RRAM 基址，得到 AXI 字节地址 |
| 2 | `hardware/deps/tensor-core/.../tc_controller.sv`（或 TC 顶层） | 实现/暴露 0xD000_2000 状态寄存器读，表示“GEMM 完成” |
| 3 | `hardware/src/ara_soc.sv` | 如需，扩展 TC 从机地址范围以包含 0xD000_2000 |
| 4 | `apps/tensorcore_to_cpu/main.c` | 完整流程：准备数据、填指令、发指令、等完成、读 Dout、与 G 比对 |
| 5 | `apps/tensorcore_to_cpu/script/gen_data.py` | 可选：B 放 .l2，保证仅 L2 参与，便于硬件仿真 |
| 6 | `apps/tensorcore_to_cpu/Makefile` | 生成 data.s、链接、目标为硬件仿真用 elf |
| 7 | 仿真脚本/顶层 | 加载 tensorcore_to_cpu 的 elf 到 L2，运行硬件仿真 |
| 8 | `apps/common/default_args.mk` | 添加 `def_args_tensorcore_to_cpu ?= "M N K"` 供 data.S 生成使用 |

---

## 五、仅硬件仿真时的推荐配置

- **内存**：A、B、Din、Dout、G 全部放在 **L2 (0x8000_0000)**，RTL 中 Din/A/B/Dout 的基址均使用 L2_BASE，避免 RRAM 参与，简化地址与数据生成。
- **数据生成**：`gen_data.py` 对 A/B 使用 4096 对齐，对 Din/Dout/G 使用 16384 对齐；若 B 放 L2，则 B 也生成在 `.l2`。
- **验证**：CPU 在发指令后轮询状态寄存器（若已实现），再读 Dout 与 G 比较；可输出比较结果到 UART 或 testbench 可观测的地址，便于自动化判断 PASS/FAIL。

按上述数据流与修改清单实现后，即可在**仅硬件仿真**下完成：CPU 发指令 → TensorCore 收指令、解析、执行 GEMM、写回内存 → CPU 读结果并与黄金参考比对的完整验证。
