# CPU+Vector+Tensorcore 项目优化补充

# 代码编写优化

## 代码注释：
/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore
所有.h/.c/.py 文件的各个重要参数和函数都需要通俗易懂的中文注释，解释其作用，注释的标点符号用英文半角

## 代码自动化
/home/zhoujinwei/pulp/ara/hardware/all_unit.sh

```bash
rm -rf /home/zhoujinwei/pulp/ara/hardware/build
cd /home/zhoujinwei/pulp/ara/hardware && make verilate
cd /home/zhoujinwei/pulp/ara/apps && make cpu+vector+tensorcore
cd /home/zhoujinwei/pulp/ara/hardware && make simv app=cpu+vector+tensorcore
```
将以上代码包装在all_unit.sh文件中