#!/bin/bash
# set -euo pipefail
rm -rf /home/zhoujinwei/pulp/ara/hardware/build
make verilate
cd ../apps/
make tensorcore_to_cpu
cd ../hardware/
# LOGDIR="../apps/tensorcore_to_cpu/data.log"
# mkdir -p "${LOGDIR}"
# Keep both terminal output and a persistent log file for parsing BEGIN_/END_ blocks.
make simv app=tensorcore_to_cpu  # (verilator)
# make simv app=tensorcore_to_cpu 2>&1 | tee "${LOGDIR}/sim.log"  # (verilator)
# make simc app=tensorcore_to_cpu   (questasim)


# set -euo pipefail
# rm -rf /home/zhoujinwei/pulp/ara/hardware/build
# make verilate
# cd ../apps/
# make tensorcore_to_cpu
# cd ../hardware/
# LOGDIR="../apps/tensorcore_to_cpu/data.log"
# mkdir -p "${LOGDIR}"
# make simv app=tensorcore_to_cpu 2>&1 | tee "${LOGDIR}/sim.log"  # (verilator)