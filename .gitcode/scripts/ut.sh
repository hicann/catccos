#!/bin/bash

set -euxo pipefail

export log_path=$(pwd)/${pr_id}/
rm -rf "${log_path}"
mkdir -p "${log_path}"

export ASCEND_PROCESS_LOG_PATH=${log_path}/catccos
export ACLSHMEM_LOG_PATH=${log_path}/shmem

# 加载cann环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset SHMEM_HOME_PATH
npu-smi info

cat /usr/local/Ascend/driver/version.info
export SHMEM_LOG_TO_STDOUT=1

set +u
source examples/utils/setup.sh
cd ./examples
set -u

set +e
bash run_all_examples.sh
run_ret=$?
set -e

echo "[INFO] run_all_examples exit code = $run_ret"

if [ -f execution_summary.log ] && grep -q "failed" execution_summary.log; then
    echo "ut 用例失败"
    exit 1
fi

exit $run_ret
