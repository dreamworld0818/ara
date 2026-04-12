rm -rf /home/zhoujinwei/pulp/ara/hardware/build
cd /home/zhoujinwei/pulp/ara/hardware 
make verilate
cd /home/zhoujinwei/pulp/ara/apps 
make softmax
cd /home/zhoujinwei/pulp/ara/hardware 
make simv app=softmax