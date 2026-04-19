#!/usr/bin/env bash
# 一键：清理 Verilator 构建 → 重新 verilate → 编译 cpu_vector_tensorecore 应用 → 运行仿真。
# 用法：在 ara 仓库中执行  ./hardware/all_unit.sh
# 依赖：已安装并配置好 Makefile 所需的 riscv-llvm、verilator 等（见项目文档）。

rm -rf /home/zhoujinwei/pulp/ara/hardware/build
cd /home/zhoujinwei/pulp/ara/hardware 
make verilate cva6_stall_dbg=1
cd /home/zhoujinwei/pulp/ara/apps 
make cpu_vector_tensorecore
cd /home/zhoujinwei/pulp/ara/hardware 
make simv app=cpu_vector_tensorecore