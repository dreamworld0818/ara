# CPU+Vector+Tensorcore 验证执行计划（PLAN）

本文档依据 `dev/AGENTS.md` 编写，供 AI 与开发者在实现与回归时按阶段执行与验收。协作原则（中文注释、渐进开发等）以 `AGENTS.md` 为准。

## 1. 工程与路径

| 项 | 路径 |
|----|------|
| 应用与验证代码（主目录） | `/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore` |
| RTL | `/home/zhoujinwei/pulp/ara/hardware` |
| Verilator 仿真入口脚本 | `/home/zhoujinwei/pulp/ara/hardware/test.sh` |
| 参考实现示例 | `/home/zhoujinwei/pulp/ara/apps/tensorcore_to_cpu` |

目标目录结构（与 `AGENTS.md` 一致）：`kernel/`、`script/`、`data_log/`、`main.c`、`data.S`（由 `script/gen_data.py` 生成）、`Makefile`、`Readme.md`。

## 2. 验证目标

1. **功能正确性**：CPU 标量、Vector、Tensorcore 在独立与协同场景下结果正确。  
2. **数据一致性**：硬件输出与软件黄金模型（C/Python）逐元素比对，误差在预设阈值内。  
3. **系统协同（flash-attention）**：端到端验证 CPU 调度、Vector（softmax 等非线性）、Tensorcore（两段 GEMM）与回写。  
   - GEMM1：`S = Q * K^T`（Tensorcore）  
   - Softmax：`P = softmax((S * scale) + mask)`（Vector）  
   - GEMM2：`O = P * V`（Tensorcore）  
4. **回归稳定性**：固定种子与随机种子下可重复、可定位、可自动回归。

## 3. 验证范围与当前进度

**范围**：CPU 控制流与边界；Vector 访存/算术/lane/掩码；Tensorcore GEMM、分块、精度；三者与 L2/内存的交互；算子语义含 `scale`、`causal/non-causal mask`、softmax 数值稳定与行和校验。

**已验证（基线）**：

- CPU + Vector  
- CPU + Tensor：指令下发 → Tensorcore 解析 → 自 L2/RRAM 读数 → GEMM → 写回 L2 → CPU 读回并与黄金参考比对。

**待加强**：完整 flash-attention 三段链路与大规模/随机回归（见阶段 B/C）。

## 4. 分阶段执行（按顺序推进）

### Phase A：环境与基线

- [ ] 建立/对齐目录：`kernel/`、`script/`、`data_log/`、`Makefile`、`main.c`。  
- [ ] 打通链路：`gen_data.py` → `data.S` → Verilator 仿真 → `data_log/` 产出日志。  
- [ ] **交付**：1 个 `smoke case` + 可复现命令（见 §7）。

### Phase B：协同链路与一致性

- [ ] 端到端 case：CPU 调度两段 GEMM 与中间 softmax（见 §2 公式顺序）。  
- [ ] 检查点：对 `S`、`P`、`O` 与黄金模型比对；记录最大误差、首个错误索引。  
- [ ] 覆盖：两段 GEMM 的尺寸映射、步长、tile、稠密/稀疏数据分布。  
- [ ] **交付**：端到端 case 集 **≥15**，含自动化比对报告。

### Phase C：稳定性与回归

- [ ] 固定种子 + 多随机种子随机回归。  
- [ ] 压力：大矩阵、长循环、频繁换参。  
- [ ] 失败：日志归档、最小复现、修复后回归。  
- [ ] **交付**：每日/定期回归摘要，失败 TopN 归因。

## 5. 核心测试矩阵（摘要）

**尺寸（flash-attention 两段 GEMM）**

- **Prefill 主线**：`S=128,D=64` → GEMM1 `128×128×64`，GEMM2 `128×64×128`；`S=256,D=128` → GEMM1 `256×256×128`，GEMM2 `256×128×256`（`S`=序列长度，`D`=每 head 通道维）。  
- **Decode（M=1）**：如 `S=1024,D=128` → GEMM1 `1×1024×128`，GEMM2 `1×128×1024`。  
- **尾块/非对齐**：`S∈{127,255,511,1023}`，`D∈{80,96,128}` 等。  
- **压力**：`S=1024,D=128`（可扩 `S=2048`）。

**其他维度**：数据模式（全零/全一/递增/随机/边界）、访存（对齐/跨界/stride）、执行模式 CPU+Vector+Tensorcore、RTL 支持的精度格式（INT/FP 等）。

## 6. 通过标准（Exit Criteria）

1. 基线与端到端 `smoke` 全部通过。  
2. 功能 case 通过率 **≥98%**；核心 GEMM 主路径 **100%**。  
3. 黄金模型误差在阈值内，无未解释偏差。  
4. Softmax 行和（每行 `sum(P)≈1`）、mask 区域约束满足阈值。  
5. 最终输出 `O` 的最大/均值误差在阈值内。  
6. 最近 **3** 轮 nightly 无新增高优先级缺陷。

## 7. Verilator 仿真命令（与 `AGENTS.md` 一致）

```bash
rm -rf /home/zhoujinwei/pulp/ara/hardware/build
cd /home/zhoujinwei/pulp/ara/hardware && make verilate
cd /home/zhoujinwei/pulp/ara/apps && make cpu+vector+tensorcore
cd /home/zhoujinwei/pulp/ara/hardware && make simv app=cpu+vector+tensorcore
```

（亦可使用 `hardware/test.sh` 封装上述流程。）

## 8. 交付物清单

| 交付物 | 说明 |
|--------|------|
| `PLAN.md`（本文） | 阶段、矩阵、准入/退出标准 |
| `Readme.md` | 运行步骤、case 列表、日志说明 |
| `data_log/` | 终端与 `data.log` 等仿真/比对记录 |
| 黄金模型与快照 | C 或 Python；归档 `Q/K/V/mask/seed` |
| 回归汇总 | 通过率、失败分布、修复闭环 |

## 9. 风险与缓解

| 风险 | 应对 |
|------|------|
| 与黄金模型布局/舍入不一致 | 统一布局与舍入；先小矩阵对齐 |
| 随机 case 难复现 | 记录种子、输入快照、工具与 RTL 版本 |
| 仿真过慢 | 分 `smoke` / `regression` / `stress` 三档 |
| 接口协议歧义 | 文档化接口约束 + 断言 |

## 10. 文档关系

- **执行顺序与验收**：以本文 `§4–§6` 为准。  
- **AI 与代码规范**：以 `dev/AGENTS.md` 为准（中文、函数注释、错误处理等）。
