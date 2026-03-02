# Tensor Core接入修改

## 1. DataWidth

- 原本的DataWidth为32*NR_LANES，现在改为TensorCore的DataWidth，即1024（width_byte=128）。

- 具体文件修改：

  - `hardware\tb\verilator\ara_tb.cpp` ：
  
    把 `memutil.RegisterMemoryArea` 函数参数从32*NR_LANES改为1024。

  - `hardware\tb\verilator\lowrisc_dv_verilator_memutil_dpi\cpp\dpi_memutil.cc` ：
  
    删掉一些限制位宽的assert，调整`minbuf`数组大小适配128的width_byte。

  - `hardware\deps\tech_cells_generic\src\rtl\tc_sram.cc` :
  
    把 `simutil_set_mem` 和 `simutil_get_mem` 函数的val数组大小从512改为1024。

  - `hardware\src\ara_soc.sv` :

    把 `ara_system` 模块的 `AxiWideDataWidth` 参数从32*NR_LANES改为1024。
    相当于把ara的数据宽度改成1024，所以也去掉了一个原本 `ara_system.sv` 里用来转换ara数据宽度的 `axi_dw_converter` 。现在ara_system里只有cva6的数据宽度不是1024。

## 2. Makefile
- 在原本的`make verilate`编译指令里加上 

  `-I$(ROOT_DIR)/deps/tensor-core/src/sv/mesh                                    \`

  `-I$(ROOT_DIR)/deps/tensor-core/src/sv/tensor_core_components                  \`

## 3. Tensor Core

- `hardware\deps\tensor_core\src\sv\tensor_core_components\tc_pkg.sv` ：

  原本的 `AXI_ID_WIDTH` 和 `AXI_USER_WIDTH` 会越界，分别改成5和1，和 `ara_soc.sv` 保持一致。

## Tensor Core接入测试

使用了 `apps` 文件夹下面的 `tensorcore_gemm` 作为测试。

测试正常得到的输出如下：
```
[TensorCore] Write instruction[0] = 0x0200030000100004
[TensorCore] Write instruction[1] = 0x0100000000100004
[TensorCore] Write instruction[2] = 0x0800050000100002
[TensorCore] Write instruction[3] = 0x02000b0000100004
[TensorCore] Write instruction[4] = 0x0000000008020402
[TensorCore] Complete instruction received, executing...

[TensorCore] ===== GEMM Instruction Received =====
[TensorCore] Instruction Type: 1
[TensorCore] Matrix Dimensions: M=2, N=4, K=2
[TensorCore] Transpose: A=0, B=0
[TensorCore] Din: base=0x020003, stride_minor=4, stride_major=1
[TensorCore] A:  base=0x080005, stride_minor=2, stride_major=1
[TensorCore] B:  base=0x010000, stride_minor=4, stride_major=1
[TensorCore] Dout: base=0x02000b, stride_minor=4, stride_major=1
[TensorCore] Calculated Addresses:
[TensorCore]   Din:  0x000000008000c000
[TensorCore]   A:    0x0000000080005000
[TensorCore]   B:    0x0000000010000000
[TensorCore]   Dout: 0x000000008002c000
[TensorCore] Instruction executed (count: 1)
[TensorCore] ======================================

tensorcore state: ca11ab1ebadcab1e
M: 127, N: 254, K: 127
M_block: 2, N_block: 4, K_block: 2
Din: stride_minor: 4, stride_major: 1, base_addr: 0x020003, real_addr: 000000008000C000
B: stride_minor: 4, stride_major: 1, base_addr: 0x010000, real_addr: 0000000010000000
A: stride_minor: 2, stride_major: 1, base_addr: 0x080005, real_addr: 0000000080005000
Dout: stride_minor: 4, stride_major: 1, base_addr: 0x02000b, real_addr: 000000008002C000
K: 2, N: 4, M: 2, if_B_transpose: 0, if_A_transpose: 0, instruction_type: 1
sizeof(mma_instruction_512_t): 40
inst: 
00 00 00 00 08 02 04 02 
02 00 0b 00 00 10 00 04 
08 00 05 00 00 10 00 02 
01 00 00 00 00 10 00 04 
02 00 03 00 00 10 00 04 

```