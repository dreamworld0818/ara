# CPU+Vector+Tensorcore 项目代码问答

## 功能流程验证

1. Q1：验证的案例是否按照CPU发出指令给tensorcore，tensorcore解析指令，从L2或者RRAM中读取q，k，然后计算出矩阵乘累加的结果p，然后CPU给vector发指令，vector解析指令，把先前计算出来的结果进行softmax操作。然后再启动tensorcore，对p和v操作，得到结果o，最后将最终结果o写回L2？

2. Q2：如何对以下各个待测任务进行验证？（详细的运行指令）   
- **功能主线（prefill）**：  
- `S=128, D=64`：GEMM1=`128x128x64`，GEMM2=`128x64x128`     #（S = 序列长度（sequence length），也可理解为 token 数；D = 每个 head 的通道维度）
- `S=256, D=128`：GEMM1=`256x256x128`，GEMM2=`256x128x256`  
- **decode 主线（M=1）**：  
  - `S=1024, D=128`：GEMM1=`1x1024x128`，GEMM2=`1x128x1024`  
- **尾块/边界（非 64 对齐）**：  
  - `S=127/255/511/1023`，`D=80/96/128`（覆盖 tail、mask、stride）  
- **压力场景**：  
  - `S=1024, D=128`（必要时扩展到 `S=2048, D=128`）  

---

## 参考答案（结合当前仓库实现，随代码更新）

### Q1：实际数据流是否与「CPU→TensorCore→CPU→Vector→TensorCore→L2」一致？

**一致或基本一致的方面**

1. **TensorCore 两段 GEMM（MMIO）**  
   `main.c` 通过 `fa_send_gemm()` → `send_tensorcore_instruction()`（基址见 `kernel/gemm.h`，与 `tensorcore_to_cpu` 同类）下发 MMA。语义：**GEMM1** `S_out = Q·K^T + Din1`；**GEMM2** `O_out = P·V + Din2`。硬件按指令从 **L2/RRAM** 取数、做乘加、写回。

2. **数据摆放**（`script/gen_data.py`）  
   **Q、B1（K^T）、V** 在 **`.rram`**；**M/N/K、Din1、S_out、P_work、Din2、O_out、Gold_*** 等在 **`.l2`**。第一段结果在 **S_out**（得分，int32 块布局）；第二段结果在 **O_out**（int32 块布局），与「最终 O 在 L2」一致。

3. **执行顺序**  
   GEMM1 → **softmax 得到 P → 写入 P_work** → GEMM2 → CPU 与 **Gold_S / Gold_P / Gold_O** 比对。

4. **Softmax 与「Vector」的关系（与旧版说明不同）**  
   Softmax 在 **`kernel/flash_attention.c`** 的 `fa_softmax_scores_to_p()` 中实现：对 **S_out** 按行处理时，**减最大值、∑exp、缩放** 等使用 **RISC-V Vector（RVV）** strip-mine（`vfsub`、`vfredosum`、`vfmul` 等），**exp** 使用与 **`apps/softmax/lib/exp.h`** 相同的 **`__exp_1xf64`**（向量 Cephes）。块布局下行内元素先 **标量 gather** 到连续缓冲区，再进入 RVV，再 **标量 scatter** 回 **P_work** 的 A 块布局。  
   因此：**不是**「第二个独立 Vector 加速器 MMIO 端口」；在 ARA 上 **Vector 指 RVV 指令在核内向量流水线上执行**，与「单独一块 Vector 外设寄存器」的叙述若严格对齐 RTL，需以 SoC 是否另有 Vector 从设备为准。

5. **与题干符号的对应**  
   题干里的「p」若指 **注意力权重**：实现中为 **P**（softmax 后 int8，在 **P_work**）。若指 **GEMM1 的得分矩阵**：实现中为 **S_out**（int32）。题干「对 p 和 v」对应 **GEMM2：P（P_work）与 V**。

**小结**  
端到端 **算子顺序、存储分段、TensorCore 双段、中间 softmax→P→第二段 GEMM** 与 Flash-Attention 及 QA 题干一致；**softmax 已由 RVV + `__exp_1xf64` 承担主要向量运算**；**尚无**与 TensorCore 平级的「独立 Vector MMIO 从设备」时，以 **RVV 路径** 作为「Vector 执行 softmax」的验证实现。

---

### Q2：各待测任务如何验证？（详细运行指令）

**参数约定**

- 应用使用 **`M N K`**（与 `apps/tensorcore_to_cpu` 一致），由  
  `make ... def_args_cpu_vector_tensorecore="M N K"`  
  传给 `script/gen_data.py`，且 **M、N、K 均为 64 的倍数**（脚本内校验）。
- 与 PLAN 中 **方阵 Prefill** 的 **S、D** 对应关系：**M=N=S**，**K=D**（head 维）。

**通用命令（在仓库根下按需替换 `M N K`）**

```bash
cd /home/zhoujinwei/pulp/ara/apps
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="M N K"

cd /home/zhoujinwei/pulp/ara/hardware
make verilate
cd /home/zhoujinwei/pulp/ara/apps
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="M N K"
cd /home/zhoujinwei/pulp/ara/hardware
make simv app=cpu_vector_tensorecore
```

默认 **`M N K`** 见 `apps/common/default_args.mk` 中的 `def_args_cpu_vector_tensorecore`。

**功能主线（prefill）**

| 叙述 (S, D) | GEMM1（逻辑） | GEMM2（逻辑） | `def_args_cpu_vector_tensorecore` |
|-------------|----------------|----------------|-----------------------------------|
| S=128, D=64 | 128×128×64 | 128×64×128 | `"128 128 64"` |
| S=256, D=128 | 256×256×128 | 256×128×256 | `"256 256 128"` |

```bash
cd /home/zhoujinwei/pulp/ara/apps
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="128 128 64"
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="256 256 128"
```

**Decode（M=1，如 1×1024×128）**

- 需要 **M=1**，但 **1 不是 64 的倍数**，当前 **`gen_data.py` 会直接报错**。要覆盖该形态需：**padding/改对齐策略/改 TensorCore 子块约定** 等，属后续增强。
- 在现有约束下可用 **M=N=64** 等做 **smoke**，不等价 decode 几何。

**尾块 / 非 64 对齐（如 S=127、D=80）**

- 同样受 **64 倍数** 限制，**不能直接**用当前脚本跑 PLAN 所列非对齐尺寸；需扩展 **gen_data + 掩码/黄金** 或硬件支持。

**压力（S=1024, D=128）**

```bash
cd /home/zhoujinwei/pulp/ara/apps
make bin/cpu_vector_tensorecore def_args_cpu_vector_tensorecore="1024 1024 128"
```

更大：**`"2048 2048 128"`**（仿真时间更长）。

**黄金模型**  
`gen_data.py` 中 softmax 使用 **`_exp_1xf64_scalar_gold`**，与固件侧 **`__exp_1xf64`** 同源 Cephes 公式，便于与 RVV 路径对齐。