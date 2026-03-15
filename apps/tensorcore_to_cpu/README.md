# tensorcore_to_cpu — 硬件仿真说明

本应用用于在 **Ara SoC 硬件仿真** 中验证：CPU 通过 MMIO 向 TensorCore 下发 GEMM 指令 → TensorCore 执行 Dout = A×B + Din → 结果写回内存 → CPU 从 L2 读取 Dout 与黄金参考比对。

本文档说明 **如何编译应用、构建 Verilator 仿真、并运行硬件仿真**。

---

## 一、环境与依赖

- **仓库根目录**：记为 `$(ARA)`
- **RISC-V 工具链**：需已安装于 `$(ARA)/install`（如 `install/riscv-llvm`），供编译裸机 ELF
- **Verilator**：需已安装于 `$(ARA)/install/verilator`（或通过 `hardware/Makefile` 中 `veril_path` 指定）
- **Bender**：硬件依赖管理，`hardware/bender`，首次构建时会自动下载
- **Python 3**：用于 `script/gen_data.py` 生成测试数据（`data.S`）

若使用 QuestaSim 仿真，需另行安装 QuestaSim 并配置 `questa_cmd`。

---

## 二、步骤概览

1. **编译应用**：在 `apps/` 下生成 ELF `bin/tensorcore_to_cpu`
2. **（可选）打 TensorCore 补丁**：若使用 Bender 拉取的 tensor-core 依赖，可能需在 `hardware/` 下执行 `make tc-apply-patches`
3. **Verilator 编译**：在 `hardware/` 下执行 `make verilate`
4. **运行仿真**：在 `hardware/` 下执行 `make simv app=tensorcore_to_cpu`

---

## 三、详细步骤

### 3.1 编译应用（生成 ELF）

在 **`$(ARA)/apps`** 目录下执行：

```bash
make bin/tensorcore_to_cpu
```

- 会先根据 `def_args_tensorcore_to_cpu`（默认 `"127 254 127"`，即 M N K）调用 `tensorcore_to_cpu/script/gen_data.py` 生成 `tensorcore_to_cpu/data.S`
- 再编译、链接，得到 **`$(ARA)/apps/bin/tensorcore_to_cpu`**（ELF）
- 该 ELF 会被加载到仿真的 L2（0x8000_0000）与 RRAM（0x1000_0000）区域

修改矩阵规模时，可覆盖默认参数，例如：

```bash
make bin/tensorcore_to_cpu def_args_tensorcore_to_cpu="64 64 64"
```

### 3.2 打 TensorCore 补丁（若需要）

若 SoC 集成的是 `hardware/deps/tensor-core` 且仓库提供补丁，在 **`$(ARA)/hardware`** 下执行：

```bash
make tc-apply-patches
```

（若从未使用 Bender 更新/拉取依赖，可先执行 `make update`。）

### 3.3 Verilator 编译

在 **`$(ARA)/hardware`** 下执行：

```bash
make verilate
```

- 会使用 Bender 生成 Verilator 脚本并编译 RTL，生成可执行文件（如 `build/verilator/Vara_tb_verilator`）
- 首次或 RTL/配置变更后需要重新执行

### 3.4 运行硬件仿真

仍在 **`$(ARA)/hardware`** 下执行：

```bash
make simv app=tensorcore_to_cpu
```

- 会调用 `build/verilator/Vara_tb_verilator -E $(ARA)/apps/bin/tensorcore_to_cpu`，将 ELF 按段加载到 L2 与 RRAM
- 仿真运行直到程序退出；返回值由 SoC 的 `exit_o` 等决定，可用于脚本判断 PASS/FAIL

带波形（FST）时，可先编译时打开 trace，再运行，例如：

```bash
make verilate trace=1
make simv app=tensorcore_to_cpu trace=1
```

（具体是否支持 `trace=1` 以当前 `hardware/Makefile` 为准。）

---

## 四、SoC 参数与测试案例（A/B/Din 在 L2 或 RRAM）

TensorCore 的输入矩阵 A、B、Din 可分别放在 **L2** 或 **RRAM**，由 SoC 顶层参数决定：

- **TC_A_IN_L2**：1 = A 在 L2，0 = A 在 RRAM  
- **TC_B_IN_L2**：1 = B 在 L2，0 = B 在 RRAM  
- **TC_DIN_IN_L2**：1 = Din 在 L2，0 = Din 在 RRAM  

Dout 固定写回 L2。当前 Verilator 使用 `ara_soc.sv` / `ara_system.sv` 中的**默认参数**（如 TC_A_IN_L2=1, TC_B_IN_L2=0, TC_DIN_IN_L2=1）。若要跑不同测试案例（例如「全 L2」或「全 RRAM」），需要让 Verilator 使用不同参数：

- **方法一**：在 `hardware/Makefile` 的 `verilate` 目标中，为 Verilator 增加 `-GTC_A_IN_L2=1 -GTC_B_IN_L2=1 -GTC_DIN_IN_L2=1`（或 0）等，然后重新执行 `make verilate` 和 `make simv app=tensorcore_to_cpu`。
- **方法二**：在 `hardware/src/ara_soc.sv` 中修改上述参数的默认值，再重新 `make verilate`。

应用侧数据布局与链接脚本需与当前 SoC 参数一致（即：在 L2 的矩阵用 L2 基址 0x8000_0000，在 RRAM 的用 0x1000_0000）。8 种 A/B/Din 组合的对应关系见 **`TEST_CASE.md`**。

---

## 五、简要命令汇总

在仓库根目录下可按顺序执行：

```bash
# 1. 编译应用
cd apps && make bin/tensorcore_to_cpu && cd ..

# 2. （如需）打 TensorCore 补丁
cd hardware && make tc-apply-patches && cd ..

# 3. Verilator 编译
cd hardware && make verilate

# 4. 运行仿真
cd hardware && make simv app=tensorcore_to_cpu
```

仿真结束后，可通过控制台输出或 SoC 的 exit 码判断测试是否通过。更多数据流与软件/硬件约定见 **`FLOW_README.md`**，测试案例与参数含义见 **`TEST_CASE.md`**。
