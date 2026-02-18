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

## 2. Makefile
- 在原本的`make verilate`编译指令里加上 

  `-I$(ROOT_DIR)/deps/tensor-core/src/sv/mesh                                    \`

  `-I$(ROOT_DIR)/deps/tensor-core/src/sv/tensor_core_components                  \`

## 3. Tensor Core

- `hardware\deps\tensor_core\src\sv\tensor_core_components\tc_pkg.sv` ：

  原本的 `AXI_ID_WIDTH` 和 `AXI_USER_WIDTH` 会越界，分别改成5和1，和 `ara_soc.sv` 保持一致。
