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
- **CPU 侧**：通过**轮询**读取该状态实现同步。从机返回的 `slv_register_output_data`；读到非 0（置 1 表示完成）后再从内存读 Dout，与黄金参考比对。轮询地址：与 `kernel/gemm.h` 中 `TENSORCORE_STATE_ADDR` 一致（当前为 **`0xD0001000`**，与 `TENSORCORE_BASE_ADDR` 相同）；RTL 在从机地址匹配从机寄存器地址时返回完成状态。

### 4.3 SoC 地址映射与 TC 从机范围

**文件**：`hardware/src/ara_soc.sv`

- **确认**：TensorCore 从机基址为 `TCBase = 64'hD000_1000`，长度 `TCLength = 64'h2000`（即 0xD000_1000..0xD000_2FFF）。写指令与读完成状态均使用 `0xD000_1000`（见本应用 `gemm.h` 中的 `TENSORCORE_BASE_ADDR` / `TENSORCORE_STATE_ADDR`）。

### 4.4 应用工程：`apps/tensorcore_to_cpu`

**目录**：`apps/tensorcore_to_cpu/`

- **main.c**  
  - 声明/定义 M、N、K 及 `A[]`、`B[]`、`Din[]`、`Dout[]`、`G[]`；数据段由 `gen_data.py` 与链接脚本决定。当前默认脚本将 **`A`、`B`、`Din` 放在 `.rram`**，`Dout`、`G` 等在 `.l2`（见 4.4 下 `gen_data.py`）；若要做「全 L2」简化案例，需同步改 `gen_data.py` 与 SoC `TC_*_IN_L2` 参数（见 `README.md` / `TEST_CASE.md`）。  
  - 黄金参考 `G = A*B + Din` 由 `gen_data.py` 生成。  
  - 填充 `mma_instruction_t`：实现里将指针右移 8 位写入 `base_addr` 等字段（与 RTL 字段宽度一致），stride 为字节步长（A/B 用 4096，Din/Dout 用 16384）。  
  - 调用 `send_tensorcore_instruction(&inst)`。  
  - 轮询 `load_tensorcore_state()`，TensorCore 计算结束时 slv_register_output_data 置 1，读到非 0 即完成，再读 Dout。  
  - 从内存读取 `Dout[]`，与 `G[]` 逐元素比较，打印 PASS/FAIL 或第一个错误位置。

- **kernel/gemm.c、kernel/gemm.h、kernel/gemm_instruction.h**  
  - 保持与 `tensorcore_gemm` 一致即可（`TENSORCORE_BASE_ADDR`、`TENSORCORE_STATE_ADDR`、`send_tensorcore_instruction`、`load_tensorcore_state`）；若 SoC 将状态放在不同偏移，需在 `gemm.h` 中改 `TENSORCORE_STATE_ADDR`。

- **script/gen_data.py**  
  - **当前默认布局**：`A`、`B`、`Din` 放在 **`.rram`**（只读段）；`M`、`N`、`K`、`Dout`、`G` 放在 **`.l2`**。A/B 块按 `64*64` 字节对齐，Din/Dout/G 按 `64*64*4` 对齐。若要做「全 L2」案例，需按脚本末尾注释改为把输入矩阵放进 `.l2`，并保证 SoC 参数 `TC_*_IN_L2` 与链接布局一致（详见 `TEST_CASE.md`）。  
  - `M`、`N`、`K` 须为 64 的倍数（与 TensorCore 块大小一致）。

- **链接脚本 / Makefile**  
  - 使用 `apps/common/arch.link.ld`（经 `apps/Makefile` 的 `linker_script` 目标预处理），保证 `.l2`、`.rram` 与 SoC 的 L2/RRAM 基址一致。  
  - **硬件仿真 ELF**：在 **`$(ARA)/apps`** 下执行 `make bin/tensorcore_to_cpu` 或 `make tensorcore_to_cpu`（`apps/Makefile` 中应用名目标等价于生成 `bin/tensorcore_to_cpu`）。会先按 `def_args_tensorcore_to_cpu` 调用 `tensorcore_to_cpu/script/gen_data.py` 生成/覆盖 `tensorcore_to_cpu/data.S`，再编译链接得到 **`$(ARA)/apps/bin/tensorcore_to_cpu`**。  
  - `apps/common/default_args.mk` 中：`def_args_tensorcore_to_cpu ?= "128 256 128"`（M N K）。覆盖示例：`make bin/tensorcore_to_cpu def_args_tensorcore_to_cpu="64 64 64"`。  
  - **`apps/tensorcore_to_cpu/Makefile`** 仅构建 Spike 设备插件 `libtensorcore_to_cpu.so`，**不参与**上述裸机 ELF 的生成。

### 4.5 硬件仿真操作步骤（Verilator，与当前 Makefile 一致）

以下为与仓库中 `hardware/Makefile`、`apps/Makefile` 行为一致的流程；无需 Spike。

1. **依赖**：RISC-V 裸机工具链与 Verilator 已安装到 `$(ARA)/install`（`hardware/Makefile` 中 `INSTALL_DIR`、`veril_path` 指向 `install/verilator/bin`）；首次硬件构建需要可执行的 `hardware/bender`（Makefile 可在缺失时拉取）。

2. **编译应用 ELF**（在 `$(ARA)/apps`）：
   ```bash
   make bin/tensorcore_to_cpu
   ```
   产出：`$(ARA)/apps/bin/tensorcore_to_cpu`。仿真器通过 ELF 程序头把各段加载到 L2（如 `0x8000_0000`）与 RRAM（如 `0x1000_0000`），与链接脚本一致。

3. **Verilator 编译 RTL**（在 `$(ARA)/hardware`）：
   ```bash
   make verilate
   ```
   生成仿真可执行文件，例如 **`build/verilator/Vara_tb_verilator`**（`veril_top` 默认为 `ara_tb_verilator`）。若需要 FST 波形，应使用 **`make verilate trace=1`**（向 Verilator 传入 `--trace-fst`），再运行仿真时加 **`trace=1`**（见下）。

4. **运行仿真**（仍在 `hardware`）：
   ```bash
   make simv app=tensorcore_to_cpu
   ```
   等价于调用：
   ```text
   build/verilator/Vara_tb_verilator [-t] -E <app_path>/tensorcore_to_cpu
   ```
   其中 `app_path` 默认为 **`$(ARA)/apps/bin` 的绝对路径**（`hardware/Makefile` 变量 `app_path`）；`-t` 仅在 **`make simv ... trace=1`** 时加入，且需与 **`make verilate trace=1`** 配套。

5. **QuestaSim（可选）**：若已配置 `questa_cmd` 并成功 `make compile`，可 **`make simc app=tensorcore_to_cpu`**，通过 `+PRELOAD` 加载同一 ELF（非 Verilator 路径）。

6. **一键脚本（可选）**：`hardware/test.sh` 示例顺序为：`make verilate` → `cd ../apps && make tensorcore_to_cpu` → `cd ../hardware && make simv app=tensorcore_to_cpu`；可将输出重定向到 `apps/tensorcore_to_cpu/data.log/sim.log` 等便于留存。

7. **RTL 侧调试日志（可选）**：部分 TensorCore RTL 将调试信息写入固定路径，例如 `apps/tensorcore_to_cpu/log`、`datavalue.log`、`Dout.log`（源码中为绝对路径）。换机或需关闭时请自行修改对应 SystemVerilog 中的路径。

### 4.6 小结：修改清单

| 序号 | 文件 | 修改内容 |
|------|------|----------|
| 1 | `hardware/deps/tensor-core/.../tc_controller.sv` | **SoC 必须**：在发出的 AXI 地址上为 Din/A/B/Dout 分别加 L2_BASE 或 RRAM_BASE（与 3.1/3.2 的“base 即基址、SoC 上加区域基址”一致）。 |
| 2 | SoC/TC 从机 | TensorCore 已把完成标志放在 slv_register_output_data（置 1 表示完成） |
| 3| `apps/tensorcore_to_cpu/main.c` | 完整流程：准备数据、填指令、发指令、等完成、读 Dout、与 G 比对 |
| 4 | `apps/tensorcore_to_cpu/script/gen_data.py` | 按测试案例调整 `.l2` / `.rram` 段布局，与 `TC_*_IN_L2` 及 `TEST_CASE.md` 一致 |
| 5 | `apps/Makefile`（通用规则） | 生成 `data.S`、编译链接，得到 **`apps/bin/tensorcore_to_cpu`**。`apps/tensorcore_to_cpu/Makefile` 仅用于 Spike 插件 `libtensorcore_to_cpu.so`。 |
| 6 | `hardware/Makefile`：`verilate` / `simv` | `make verilate` 生成 `build/verilator/Vara_tb_verilator`；`make simv app=tensorcore_to_cpu` 以 `-E` 加载 ELF；可选 `trace=1` 与 `verilate trace=1` 配套。 |
| 7 | `apps/common/default_args.mk` | `def_args_tensorcore_to_cpu ?= "128 256 128"` 供 `gen_data.py` 生成 `data.S` |

---

## 五、仅硬件仿真时的推荐配置

- **内存（默认脚本）**：`gen_data.py` 将 **`A`、`B`、`Din` 置于 RRAM 段**，**`Dout`、`G`、`M/N/K` 置于 L2**；与 SoC 中 `TC_A_IN_L2` / `TC_B_IN_L2` / `TC_DIN_IN_L2` 的默认组合需一致（详见 `README.md`、`TEST_CASE.md`）。若希望输入与输出均在 L2、避免 RRAM 参与，可改为全 `.l2` 布局并调整上述 RTL 参数。
- **数据生成**：`gen_data.py` 对 A/B 块按 4096 字节对齐，对 Din/Dout/G 按 16384 字节对齐；`M`、`N`、`K` 为 64 的倍数。
- **验证**：CPU 发指令后轮询 `load_tensorcore_state()`（完成时 `slv_register_output_data` 非 0），再读 `Dout` 与 `G` 比较；程序在 UART 上打印 `PASS`/`FAIL`（或 mismatch 详情），退出码可用于脚本判断。

按上述数据流与修改清单实现后，即可在**仅硬件仿真**下完成：CPU 发指令 → TensorCore 收指令、解析、执行 GEMM、写回内存 → CPU 读结果并与黄金参考比对的完整验证。
