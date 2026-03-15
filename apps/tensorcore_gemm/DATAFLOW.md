# TensorCore GEMM 数据流详解

## 数据流概览

```
CPU准备数据 → CPU发送指令 → TensorCore接收指令 → TensorCore读取数据 → 
TensorCore执行计算 → TensorCore写回结果 → CPU读取结果
```

## 详细数据流分析

### 1. 数据准备阶段（CPU端）

#### 1.1 数据生成和内存布局
- **数据生成**：`gen_data.py` 脚本生成测试数据
  - 矩阵 A: M×K (int8_t) - 存储在 `.l2` 段
  - 矩阵 B: K×N (int8_t) - 存储在 `.rram` 段（RRAM内存）
  - 矩阵 Din: M×N (int32_t) - 存储在 `.l2` 段
  - 矩阵 Dout: M×N (int32_t) - 存储在 `.l2` 段（输出缓冲区）
  - 矩阵 G: M×N (int32_t) - 存储在 `.l2` 段（黄金参考结果）

#### 1.2 内存地址映射
```c
// main.c 中的声明
extern int8_t A[] __attribute__((aligned(4096), section(".l2")));      // 4KB对齐
extern int8_t B[] __attribute__((aligned(4096), section(".rram")));    // 4KB对齐，RRAM
extern int32_t Din[] __attribute__((aligned(16384), section(".l2")));  // 16KB对齐
extern int32_t Dout[] __attribute__((aligned(16384), section(".l2"))); // 16KB对齐
```

**内存段说明**：
- `.l2`: L2缓存/共享内存，CPU和TensorCore都可以访问
- `.rram`: RRAM（Resistive RAM），可能是专用内存区域

### 2. 指令准备阶段（CPU端）

#### 2.1 构建MMA指令
在 `main.c` 中，CPU构建40字节的MMA指令：  

```c
mma_instruction_t inst = {0};
// 配置输入矩阵Din的信息
inst.fields.Din.info.base_addr = (uint32_t)Din / 16384;  // 地址转换为块号
inst.fields.Din.info.stride_minor = N_block;
inst.fields.Din.info.stride_major = 1;

// 配置矩阵B的信息
inst.fields.B.info.base_addr = (uint32_t)B / 4096;
inst.fields.B.info.stride_minor = N_block;
inst.fields.B.info.stride_major = 1;

// 配置矩阵A的信息
inst.fields.A.info.base_addr = (uint32_t)A / 4096;
inst.fields.A.info.stride_minor = K_block;
inst.fields.A.info.stride_major = 1;

// 配置输出矩阵Dout的信息
inst.fields.Dout.info.base_addr = (uint32_t)Dout / 16384;
inst.fields.Dout.info.stride_minor = N_block;
inst.fields.Dout.info.stride_major = 1;

// 配置计算元数据
inst.fields.mma_meta.info.M = M_block;
inst.fields.mma_meta.info.N = N_block;
inst.fields.mma_meta.info.K = K_block;
inst.fields.mma_meta.info.instruction_type = 1;
```

**关键点**：
- `base_addr` 存储的是**地址除以对齐大小后的值**（块号），不是实际地址
- TensorCore需要根据对齐大小（A/B: 4096字节，Din/Dout: 16384字节）还原实际地址

### 3. 指令传输阶段（CPU → TensorCore）

#### 3.1 CPU发送指令
```c
// gemm.c: send_tensorcore_instruction()
void send_tensorcore_instruction(mma_instruction_t *inst) {
    volatile uint64_t *tensorcore_base_addr = (uint64_t *)TENSORCORE_BASE_ADDR; // 0xD0001000
    uint64_t *raw64 = inst->raw64;
    // 将40字节指令分5次写入（每次8字节）
    for (int i = 0; i < 5; i++) {
        tensorcore_base_addr[i] = raw64[i];  // MMIO写入
        asm volatile("" ::: "memory");       // 编译器内存屏障
    }
    __sync_synchronize();                    // 硬件内存屏障
}
```

**传输方式**：
- 通过 **MMIO（Memory-Mapped I/O）** 写入到地址 `0xD0001000`
- 指令缓冲区大小为40字节（5个64位字）
- 使用内存屏障确保写入顺序

#### 3.2 TensorCore接收指令
在 `device.cxx` 中，设备插件模拟TensorCore硬件：

```cpp
// device.cxx: store() 函数
bool store(reg_t addr, size_t len, const uint8_t* bytes) {
    // 写入指令缓冲区 (0xD0001000 + offset, 0 <= offset < 40)
    if (offset < 0x28) {  // 40字节
        uint64_t idx = offset / 8;
        instruction_buffer[idx] = ...;  // 存储指令字
        
        // 当写入最后一个字时，触发指令执行
        if (idx == 4) {
            instruction_valid = true;
            execute_instruction();  // 执行指令
        }
    }
}
```

### 4. 数据读取阶段（TensorCore → 内存）

#### 4.1 TensorCore解析指令
TensorCore从指令中提取地址信息：

```cpp
// device.cxx: execute_instruction()
// 解析指令字段
op_info_t din = inst.fields.Din;
op_info_t b = inst.fields.B;
op_info_t a = inst.fields.A;
op_info_t dout = inst.fields.Dout;

// 计算实际地址（还原地址）
uint64_t din_addr = (uint64_t)din.info.base_addr * 16384;   // Din/Dout: 16KB对齐
uint64_t a_addr = (uint64_t)a.info.base_addr * 4096;        // A: 4KB对齐
uint64_t b_addr = (uint64_t)b.info.base_addr * 4096;        // B: 4KB对齐
uint64_t dout_addr = (uint64_t)dout.info.base_addr * 16384; // Dout: 16KB对齐
```

#### 4.2 TensorCore读取数据
**实际硬件中**，TensorCore会通过AXI总线从内存读取数据：
- 从 `a_addr` 读取矩阵A（M×K，int8_t）
- 从 `b_addr` 读取矩阵B（K×N，int8_t）
- 从 `din_addr` 读取矩阵Din（M×N，int32_t）

**数据来源**：
- 矩阵A和Din：从 `.l2` 段（L2缓存/共享内存）读取
- 矩阵B：从 `.rram` 段（RRAM内存）读取

### 5. 计算执行阶段（TensorCore内部）

#### 5.1 执行GEMM运算
TensorCore执行矩阵乘法运算：
```
Dout = A × B + Din
```

**计算过程**：
- 输入：A (M×K, int8), B (K×N, int8), Din (M×N, int32)
- 计算：矩阵乘法 A×B（int8×int8 → int32累加）
- 累加：加上Din矩阵
- 输出：Dout (M×N, int32)

**分块处理**：
- 矩阵被分解为64×64的块
- `M_block = (M + 63) / 64`
- `N_block = (N + 63) / 64`
- `K_block = (K + 63) / 64`

### 6. 结果写回阶段（TensorCore → 内存）

#### 6.1 TensorCore写回结果
TensorCore将计算结果写入内存：
- 写入地址：`dout_addr`（从指令中的Dout.base_addr计算得出）
- 写入数据：计算得到的Dout矩阵（M×N，int32_t）
- 存储位置：`.l2` 段的Dout数组

**重要**：结果直接写入到CPU可访问的内存区域（`.l2`段），**不需要CPU主动读取**。

### 7. 状态查询阶段（CPU ← TensorCore）

#### 7.1 CPU查询执行状态
```c
// main.c
uint64_t state = load_tensorcore_state();  // 读取状态寄存器
printf("tensorcore state: %016lx\n", state);
```

```c
// gemm.c: load_tensorcore_state()
uint64_t load_tensorcore_state(void) {
    volatile uint64_t *tensorcore_state_addr = (uint64_t *)TENSORCORE_STATE_ADDR; // 0xD0002000
    return *tensorcore_state_addr;  // MMIO读取
}
```

**状态寄存器**（地址：`0xD0002000`）：
- 用于指示TensorCore的执行状态
- CPU可以轮询此寄存器判断计算是否完成

### 8. 结果验证阶段（CPU端）

#### 8.1 CPU读取结果
CPU直接从内存读取Dout数组（因为结果已写入`.l2`段）：

```c
// 验证结果（虽然当前main.c中没有实现，但可以添加）
int verify_matrix(int32_t *result, int32_t *gold, size_t R, size_t C) {
    // 比较Dout和G（黄金结果）
    for (uint64_t i = 0; i < R; ++i) {
        for (uint64_t j = 0; j < C; ++j) {
            if (result[idx] != gold[idx]) {
                return idx;  // 返回不匹配的位置
            }
        }
    }
    return 0;  // 验证通过
}
```

## 数据流总结

### 指令流
```
CPU内存(指令结构体) 
  → MMIO写入(0xD0001000) 
  → TensorCore指令缓冲区 
  → TensorCore解析执行
```

### 数据流（读取）
```
内存(.l2/.rram段) 
  → AXI总线 
  → TensorCore内部缓存 
  → 计算单元
```

### 数据流（写回）
```
TensorCore计算单元 
  → TensorCore内部缓存 
  → AXI总线 
  → 内存(.l2段的Dout数组)
```

### 关键特点

1. **零拷贝**：数据不需要在CPU和TensorCore之间显式拷贝
   - TensorCore直接从共享内存（`.l2`）和RRAM（`.rram`）读取
   - 结果直接写回共享内存（`.l2`），CPU可直接访问

2. **异步执行**：CPU发送指令后可以继续执行其他任务
   - 通过状态寄存器查询执行状态
   - 结果写回后CPU可以直接读取

3. **地址转换**：指令中使用块号而非实际地址
   - 减少指令大小
   - TensorCore根据对齐大小还原实际地址

4. **内存对齐**：严格的内存对齐要求
   - A/B矩阵：4KB对齐（4096字节）
   - Din/Dout矩阵：16KB对齐（16384字节）
   - 确保TensorCore可以高效访问数据

## 内存映射图

```
内存地址空间：
┌─────────────────────────────────────┐
│  .l2段 (L2缓存/共享内存)            │
│  ├─ A[]      (4KB对齐)              │
│  ├─ Din[]    (16KB对齐)             │
│  ├─ Dout[]   (16KB对齐) ← 结果写回  │
│  └─ G[]      (16KB对齐)             │
├─────────────────────────────────────┤
│  .rram段 (RRAM内存)                 │
│  └─ B[]      (4KB对齐)              │
├─────────────────────────────────────┤
│  MMIO区域                            │
│  ├─ 0xD0001000: 指令缓冲区 (40字节)  │
│  └─ 0xD0002000: 状态寄存器 (8字节)   │
└─────────────────────────────────────┘
```


数据流核心要点
1. 指令传输（CPU → TensorCore）
CPU在内存中构建40字节的MMA指令结构体
通过MMIO写入到地址 0xD0001000（TensorCore的指令缓冲区）
指令包含：矩阵A、B、Din、Dout的地址信息和计算参数
2. 数据读取（内存 → TensorCore）
TensorCore从指令中解析出地址，通过AXI总线直接从内存读取数据
数据来源：
矩阵A和Din：从.l2段（L2缓存/共享内存）读取
矩阵B：从.rram段（RRAM内存）读取
不需要CPU显式传输数据，TensorCore直接访问共享内存
3. 计算执行（TensorCore内部）
TensorCore执行：Dout = A × B + Din
在TensorCore内部的计算单元完成
4. 结果写回（TensorCore → 内存）
TensorCore通过AXI总线将结果直接写入内存
写入位置：.l2段的Dout[]数组（16KB对齐）
结果直接写回CPU可访问的内存区域
5. CPU读取结果
CPU不需要主动从TensorCore读取数据
结果已写入.l2段的Dout[]，CPU可直接访问该内存区域
CPU可以读取状态寄存器（0xD0002000）查询执行状态
关键特点
零拷贝架构：数据不需要在CPU和TensorCore之间显式拷贝，TensorCore直接访问共享内存
异步执行：CPU发送指令后可继续执行其他任务，通过状态寄存器查询完成状态
地址编码：指令中使用块号（base_addr = 实际地址 / 对齐大小），TensorCore还原实际地址
完整的数据流文档已保存到 DATAFLOW.md，包含详细的内存映射图和代码分析。