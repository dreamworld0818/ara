---
name:"CPU+Vector+Tensorcore 芯片AI验证"
description:"可以根据RTL代码，将用户对于RTL代码需要验证的要求，构建验证架构，对各种案例进行verilator仿真验证"
category：“AI 验证开发”
---

# CPU+Vector+Tensorcore 芯片AI验证

## 项目概述

CPU+Vector+Tensorcore 芯片AI验证是一个对已有的ARA(CPU+Vector)+tensorecore设计进行仿真验证的项目，它利用已有的RTL代码，设计对应的验证架构，设计对应的测试应用和案例，对主要的功能验证需求进行相应的软件测试代码编写、并调用工具进行verilator仿真。

## 工程路径：
/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore (主要代码)
/home/zhoujinwei/pulp/ara/hardware/test.sh （verilator仿真执行脚本）

## RTL代码：
/home/zhoujinwei/pulp/ara/hardware

## 代码实现示例：
/home/zhoujinwei/pulp/ara/apps/tensorcore_to_cpu

## 技术栈

## 
- **语言**:SystemC，Python
- **脚本**:Makefile

## 项目结构
'''
CPU+Vector+Tensorcore
|-- kernel
    |-- vector_xxx.c #有关vector的指令函数
    |-- vector_xxx.h #有关vector指令的头文件定义   
    |-- tensor_xxx.c #有关tensor的指令函数
    |-- tensor_xxx.h #有关tensor指令的头文件定义   
|-- script
    |-- gen_data.py #汇编指令的生成文件
|-- data_log
    |-- log         #终端打印信息 
    |-- data.log    #测试数据结果信息
|-- main.c          #主要软件测试代码
|-- data.S          #由gen_data.py文件生成的汇编指令文件
|-- PLAN.md         #AI 执行的plan计划
|-- AGENTS.md       #AI 助手配置
|-- Makefile        #部署脚本
|-- Readme.md       #测试流程描述与测试case
'''

## 开发指南

### 开发哲学

1. **代码质量优先**：编写清晰、可维护、可扩展的代码
2. **函数式编码**：优先使用函数式和声明式编程模式
3. **渐进增强**：从简单功能开始逐步完善

## CPU+Vector+Tensorcore 验证计划
### 1) 验证目标
1. **功能正确性**：CPU 标量路径、Vector 向量路径、Tensorcore 计算路径在协同场景下结果正确。  
2. **数据一致性**：硬件输出与软件黄金模型（C/Python）逐元素一致，误差满足预设阈值。  
3. **系统协同（flash-attention）**：通过 flash-attention 端到端链路验证 CPU 调度、Vector 主非线性计算（softmax）、Tensorcore 主线性计算（两段 GEMM）与结果回写稳定。  
   - GEMM1：`S = Q * K^T`（Tensorcore）  
   - Softmax：`P = softmax((S * scale) + mask)`（Vector）  
   - GEMM2：`O = P * V`（Tensorcore）  
4. **回归稳定性**：在固定种子和随机种子场景下可重复、可定位、可自动回归。  

### 2) 验证范围
1. **CPU 子系统**：控制流、地址计算、边界处理、异常输入防护。  
2. **Vector 子系统**：加载/存储、算术运算、lane 对齐、掩码和尾处理。  
3. **Tensorcore 子系统**：矩阵乘累加（GEMM）主路径、tile 分块、精度与饱和行为。  
4. **跨模块交互**：CPU+Vector+Tensorcore 中内存回写的接口时序和数据协议，三者共同作用实现 flash-attention 算子。  
5. **算子语义范围**：`scale(1/sqrt(d))`、`causal/non-causal mask`、softmax 数值稳定（max-subtract）与 row-sum 约束校验。

### 3) 已验证范围
- CPU+vector
- CPU+tensor ：CPU发送指令到Tensorcore，Tensorcore端需要看到接收到的指令，并解析指令，从内存（L2/RRAM）中读取数据，执行GEMM操作，然后计算的结果写会内存（L2），CPU从内存中读取结果，与CPU段软件侧跑出来的黄金参考进行比对。

### 4) 分阶段执行计划
#### Phase A: 环境与基线
- 建立目录与脚本基线：`kernel/`、`script/`、`data_log/`、`Makefile`、`main.c`。  
- 打通最小可运行链路：`gen_data.py` 生成 `data.S`，仿真完成并产生日志。  
- 产出：基线样例 `smoke case`（1 个）+ 可复现运行命令。 

#### Phase B: 协同链路与一致性验证
- 构建端到端 case：CPU 准备参数并调度两段 GEMM 与 softmax。  
  1) `S = Q * K^T`：CPU 下发 Tensorcore 指令，Tensorcore 读 Q/K 并完成 GEMM1；  
  2) `P = softmax((S * scale) + mask)`：CPU 下发 Vector 指令，Vector 完成缩放、掩码和 softmax；  
  3) `O = P * V`：CPU 再次下发 Tensorcore 指令，Tensorcore 完成 GEMM2，结果回写 L2；  
  4) CPU 读取 `S/P/O` 关键检查点并与黄金模型校验。  
- 引入黄金模型比对：逐元素 diff，记录最大误差、首个错误索引。  
- 覆盖关键维度：两段 GEMM 的矩阵尺寸映射关系、步长、tile 配置、数据分布（稀疏/稠密）。  
- 产出：端到端 case 集（>=15 个）+ 自动化比对报告。  

#### Phase C: 稳定性与回归
- 随机回归：固定随机种子 + 多随机种子模式。  
- 压力场景：大尺寸矩阵、长时间循环、频繁参数切换。  
- 缺陷闭环：失败日志归档、最小复现、修复后回归通过。  
- 产出：每日回归报告与失败 TopN 归因。  

### 5) 核心测试矩阵
1. **尺寸维度（按 flash-attention 两段 GEMM 组织）**：  
   - **功能主线（prefill）**：  
     - `S=128, D=64`：GEMM1=`128x128x64`，GEMM2=`128x64x128`     #（S = 序列长度（sequence length），也可理解为 token 数；D = 每个 head 的通道维度）
     - `S=256, D=128`：GEMM1=`256x256x128`，GEMM2=`256x128x256`  
   - **decode 主线（M=1）**：  
     - `S=1024, D=128`：GEMM1=`1x1024x128`，GEMM2=`1x128x1024`  
   - **尾块/边界（非 64 对齐）**：  
     - `S=127/255/511/1023`，`D=80/96/128`（覆盖 tail、mask、stride）  
   - **压力场景**：  
     - `S=1024, D=128`（必要时扩展到 `S=2048, D=128`）  
2. **数据模式**：全零、全一、递增、随机、有符号边界值。  
3. **访存模式**：对齐访问、跨界访问、stride 访问。  
4. **执行模式**：CPU+Vector+Tensorcore。  
5. **精度模式**：按当前 RTL 支持的数据格式逐一覆盖（如 INT/FP 变体）。  

### 6) 通过标准（Exit Criteria）
1. 基线与端到端 `smoke` 全通过。  
2. 功能 case 通过率 >= 98%，核心路径（GEMM 主流程）100% 通过。  
3. 黄金模型比对误差在阈值内，且无未解释偏差。  
4. softmax 行和约束满足阈值（每行 `sum(P)` 接近 1），mask 区域输出满足约束（被屏蔽位置接近 0）。  
5. flash-attention 最终输出 `O` 的最大误差与均值误差均在阈值内。  
6. 最近 3 轮 nightly 回归无新增高优先级缺陷。  

### 7) 交付物
1. `PLAN.md`：分阶段计划、测试矩阵、准入退出标准。  
2. `Readme.md`：运行步骤、case 列表、日志说明。  
3. `data_log/`：仿真日志、比对日志、失败样本。  
4. 黄金模型脚本（C/Python 任一实现）与输入快照（`Q/K/V/mask/seed`）归档。  
5. 回归汇总：每日通过率、失败分布、修复闭环状态。  

### 8) 风险与应对
1. **模型不一致风险**：统一数据布局和舍入规则，先对齐小矩阵样例。  
2. **随机 case 难复现**：强制记录随机种子、输入快照、版本号。  
3. **性能导致仿真过慢**：区分 `smoke/regression/stress` 三档回归集。  
4. **接口协议歧义**：补充接口约束文档和断言检查。  

## 硬件仿真

### 执行脚本
'''
/home/zhoujinwei/pulp/ara/hardware/test.sh （verilator仿真执行脚本）
'''

### 具体脚本代码
'''
rm -rf /home/zhoujinwei/pulp/ara/hardware/build
make verilate
cd ../apps/
make cpu+vector+tensorcore
cd ../hardware/
make simv app=cpu+vector+tensorcore
'''

## AI协作原则

### 基本原则
1. **中文优先**：所有交流和注释使用中文
2. **注释完整**：每个函数都必须包含清晰的中文注释
3. **渐进开发**：从核心功能开始，逐步添加特性
4. **用户体验**：优先考虑界面美观和交互流畅性

### 代码生成需求
- **函数注释**：每个函数都要有功能说明、参数描述和返回值说明
- **错误处理**：完善的错误处理和边界情况考虑

---

*此文件遵循 [ AGENTS.md 标准]（http://agentsmd.net）, 为AI编程助手提供统一的配置规范*
 