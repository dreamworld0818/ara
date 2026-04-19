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

## 64*64*128卡死问题：
### 目前现状：
- **仿真卡死**程序卡在flash_attention的第二次GEMM2上，具体在/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/main.c的fa_send_gemm函数里的send_tensorcore_instruction函数上，运行到该函数时，终端一直不会打印出[FA][%s] MMIO done, polling TC state。
- **其他规模测试**相同的程序，改变规模为64*128*128和64*64*64均能正常执行完完整的flash_attention,终端也能正常打印出所有信息
- **终端打印信息**目前终端打印如下：
===  softmax 阶段结束(scores -> P) ===
PASS checkpoint Gold_P (softmax)
[FA] GEMM2 P*V: MMA (M,N,K)=(64,128,64)
[FA][GEMM2_PxV] send_tensorcore_instruction: M_block=1 N_block=2 K_block=1
[TC][DMA][AR] addr=0x000000008001c000 buf=0x4 transpose=0 len=127
[TC][BUF_DIN][DMA-WR] bank=0 addr=0 val=0
[TC][DMA][R0] req_addr=0x000000008001c000 buf=0x4 transpose=0 rdata[31:0]=0x00000000 rdata[7:0]=0x00
[TC][DMA][AR] addr=0x0000000080018000 buf=0x0 transpose=0 len=31
[TC][BUF_A][DMA-WR] bank=0 addr=0 data16=0x007f (k0=127 k1=0)
[TC][DMA][R0] req_addr=0x0000000080018000 buf=0x0 transpose=0 rdata[31:0]=0x7c00017f rdata[7:0]=0x7f
[TC][DMA][AR] addr=0x0000000010004000 buf=0x2 transpose=0 len=31
[TC][BUF_B][DMA-WR] bank=0 addr=0 data16=0x06f9 (k0=-7 k1=6)
[TC][DMA][R0] req_addr=0x0000000010004000 buf=0x2 transpose=0 rdata[31:0]=0xfffdfbf9 rdata[7:0]=0xf9
[TC][BUF_DIN][SA-RD] bank=0 addr=0 val=0
[TC][SA][DIN] addr=0 Din00=0
[FA][TC][BUF_B][SA-RD] bank=0 addr=0 part=0 val=-4
[TC][BUF_A][SA-RD] bank=0 addr=0 part=0 val=-2
[TC][SA][AB] addr=0 A00=0 B00=0
[TC][DMA][AR] addr=0x0000000080020000 buf=0x4 transpose=0 len=127
[[TC][BUF_DIN][DMA-WR] bank=0 addr=0 val=0
[TC][DMA][R0] req_addr=0x0000000080020000 buf=0x4 transpose=0 rdata[31:0]=0x00000000 rdata[7:0]=0x00
[TC][BUF_DOUT][SA-WR] bank=0 addr=0 val=-889
GEM[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=-889
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=126
M[TC][DMA][AR] addr=0x0000000080018000 buf=0x1 transpose=0 len=31
[TC][DMA][R0] req_addr=0x0000000080018000 buf=0x1 transpose=0 rdata[31:0]=0x7c00017f rdata[7:0]=0x7f
[TC][DMA][AR] addr=0x0000000010005000 buf=0x3 transpose=0 len=31
[TC][DMA][R0] req_addr=0x0000000010005000 buf=0x3 transpose=0 rdata[31:0]=0x07050301 rdata[7:0]=0x01
[TC][BUF_DIN][SA-RD] bank=0 addr=0 val=0
[TC][SA][DIN] addr=0 Din00=0
[TC][BUF_DOUT][SA-WR] bank=0 addr=0 val=127
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=127
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=174
然后一直卡在此处，没有新的打印信息
- **函数分析**结合终端打印信息，发现GEMM2运行到send_tensorcore_instruction函数时（函数定义位于/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/kernel/gemm.c），连该函数内部的所有print都没有打印，说明没有执行该函数？
- **最新测试**在/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/kernel/gemm.c的send_tensorcore_instruction函数中添加了各种printf之后能跑通

void send_tensorcore_instruction(mma_instruction_t *inst) {
  // printf("[TC-INST] send_tensorcore_instruction start\n");
  volatile uint64_t *tensorcore_base_addr = (volatile uint64_t *)TENSORCORE_BASE_ADDR;
  // printf("[TC-INST] send_tensorcore_instruction get tensorcore_base_addr\n");
  uint64_t *raw64 = inst->raw64;
  // printf("[TC-INST] send_tensorcore_instruction get raw64\n");
  for (int i = 0; i < (int)(sizeof(mma_instruction_t) / sizeof(uint64_t)); i++) {
    tensorcore_base_addr[i] = raw64[i];
    // printf("[TC-INST] store word 1\n");
    asm volatile("" ::: "memory");
    /* After this line: MMIO word i completed (AXI B for this store). */
    printf("[TC-INST] store word 2\n");
    /* After this line: MMIO word i completed (AXI B for this store). */
    printf("[TC-INST] store word 3\n");
  }
  __sync_synchronize();
  // printf("[TC-INST] fence ok, send_tensorcore_instruction return\n");
}

该函数只有在添加了   
   printf("[TC-INST] store word 2\n");
    /* After this line: MMIO word i completed (AXI B for this store). */
    printf("[TC-INST] store word 3\n");
这两行代码之后，能够正常跑通完整的仿真

- **初步分析** 经过debug能确定tensorcore接收指令正常、tensorcore读写矩阵数据正常，卡在send_tensorcore_instruction函数。现在需要看CPU侧具体卡在了哪条指令，编译之后，其中cva6执行的汇编指令为/home/zhoujinwei/pulp/ara/apps/bin/cpu_vector_tensorecore.dump。

### 要求更改：
- **CPU侧卡主指令定位** 打印CVA6 RTL的相关信号或者接口的信息，帮我确定cpu侧到底是卡在了哪条汇编指令处。













