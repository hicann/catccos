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

# CI 容器以 root 运行，但镜像内 CANN lib64 属主为 uid 1001；
# SHMEM 的 dlopen 安全检查要求库目录属主必须是当前用户或 root，
# 否则加载 hybm 失败(-4) → init_device_state 失败(-3)，所有用例连锁失败。
chown -R root:root "${ASCEND_HOME_PATH}/aarch64-linux/lib64" \
    || echo "[WARN] chown ${ASCEND_HOME_PATH}/aarch64-linux/lib64 failed, UT may fail on hybm load"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset SHMEM_HOME_PATH
npu-smi info

# cat /usr/local/Ascend/driver/version.info
export SHMEM_LOG_TO_STDOUT=1

set +u
source examples/utils/setup.sh
cd ./examples
set -u

mapfile -t device_ids < <(
    find /dev -maxdepth 1 -name 'davinci[0-9]*' -printf '%f\n' |
        sed -nE 's/^davinci([0-9]+)$/\1/p' |
        sort -n
)
if [ "${#device_ids[@]}" -lt 2 ]; then
    echo "[ERROR] UT requires two NPU devices; found ${#device_ids[@]}." >&2
    exit 1
fi
# aclrtSetDevice 接受的是逻辑卡号(0..N-1)，不能直接使用 /dev/davinciN 的物理卡号，
# 否则报 107001 (ACL_ERROR_RT_INVALID_DEVICEID)。这里将可见物理卡映射为逻辑卡 0,1。
# export ASCEND_RT_VISIBLE_DEVICES="${device_ids[0]},${device_ids[1]}"
rank_ids="0,1"
# echo "[INFO] UT devices: $rank_ids (ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES)"

set +e
bash run_all_examples.sh "$rank_ids"
run_ret=$?
set -e

echo "[INFO] run_all_examples exit code = $run_ret"

if [ -f execution_summary.log ] && grep -q "failed" execution_summary.log; then
    echo "ut 用例失败"
    exit 1
fi

exit $run_ret
