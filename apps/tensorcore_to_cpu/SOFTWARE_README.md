## 简单指令流程
make verilate
cd ../apps/
make tensorcore_to_cpu
cd ../hardware/
make simv app=tensorcore_to_cpu(verilator)
make simc app=tensorcore_to_cpu(questasim)