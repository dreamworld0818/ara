# CPU+Vector+Tensorcore 项目补充纠正

# 代码实现纠正

目前项目的验证case没有遵循/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/dev/AGENTS.md中的flash-attention算子的需求。

通过 flash-attention 端到端链路验证 CPU 调度、Vector 主非线性计算（softmax）、Tensorcore 主线性计算（两段 GEMM）与结果回写稳定。  
   - GEMM1：`S = Q * K^T`（Tensorcore）  
   - Softmax：`P = softmax((S * scale) + mask)`（Vector）  
   - GEMM2：`O = P * V`（Tensorcore）  

## 流程

case：CPU 准备参数并调度两段 GEMM 与 softmax。  
  1) `S = Q * K^T`：CPU 下发 Tensorcore 指令，Tensorcore 读 Q/K 并完成 GEMM1；  
  2) `P = softmax((S * scale) + mask)`：CPU 下发 Vector 指令，Vector 完成缩放、掩码和 softmax；  
  3) `O = P * V`：CPU 再次下发 Tensorcore 指令，Tensorcore 完成 GEMM2，结果回写 L2；  
  4) CPU 读取 `S/P/O` 关键检查点并与黄金模型校验。  
- 引入黄金模型比对：逐元素 diff，记录最大误差、首个错误索引。

## vector执行softmax
### 目前现状：
- **Softmax 并非单独向 Vector 外设发指令**：当前实现里，softmax 在 **`kernel/flash_attention.c`** 的 `fa_softmax_scores_to_p()` 中完成，是 **CPU 上标量 + libm（double）** 路径，**没有** 类似 TensorCore 的单独「Vector MMIO 指令序列」。若验证目标必须是「Vector 硬件执行 softmax」，需要在后续把该段改为 **RVV 指令序列** 或 SoC 约定的 **Vector 加速器寄存器接口**，并与黄金模型对齐。
- **「先前计算出来的结果」**：得分在 **S_out**（int32 块布局）；softmax 读 **S_out**、写 **P_work**（int8）；第二段 TensorCore 读 **P_work** 与 **V**，写 **O_out**。逻辑上与叙述一致，只是 softmax 的执行单元在软件上目前是 **CPU 标量**，不是独立 Vector 指令流。

### 要求更改：
- **Softmax 需要由 vector 独立计算**：softmax算子操作需要由vector独立计算，而不能是cpu，请严格按照CPU发出指令给tensorcore，tensorcore解析指令，从L2或者RRAM中读取q，k，然后计算出矩阵乘累加的结果p，然后CPU给vector发指令，vector解析指令，把先前计算出来的结果进行softmax操作。然后再启动tensorcore，对p和v操作，得到结果o，写回L2的流程进行代码编写修正。

## 仿真运行时的问题
### 目前现状：
- **程序卡死**程序会卡在/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/main.c中的fa_send_gemm函数上，该函数的实现有问题。
- **tensorecore_to_cpu**的示例可以完成一次GEMM计算，没有问题。

### 要求更改：
- **tensorcore按照示例的方法发送指令和计算**：按照/home/zhoujinwei/pulp/ara/apps/tensorcore_to_cpu/main.c中的方式编写flash-attention的GEMM1和GEMM2阶段的函数定义和指令，通过send_tensorcore_instruction发送指令，并且每个阶段结束之后，需要打印显示。

## 仿真时softmax卡主问题
### 目前现状：
- **程序卡死**程序会卡在/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/main.c中的fa_softmax_scores_to_p函数上，该函数的实现有问题。
- **softmaxt**的示例可以完成一次softmax的计算，没有问题。

### 要求更改：
- **softmax按照示例的方法发送指令和计算**：按照/home/zhoujinwei/pulp/ara/apps/softmax中的方式编写flash-attention的softmax阶段的函数定义和指令，取GEMM1计算后的结果S进行softmax操作后变为权值矩阵P，并且每个阶段结束之后，需要打印显示。且softmax计算之后的结果是P存入L2中，让GEMM2能够取P进行第二次tensor计算
