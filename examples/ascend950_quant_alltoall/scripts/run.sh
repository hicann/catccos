#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

source $PROJECT_ROOT/3rdparty/shmem/install/set_env.sh || {
    echo "[ERROR] Running set_env.sh in $PROJECT_ROOT/3rdparty/shmem/install failed."
    exit 1
}

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/ascend950_quant_alltoall/
DATA_DIR=`realpath ./out`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_quant_alltoall
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
set -euo pipefail          # 严格模式（推荐在调试完成后再打开）
IFS=$'\n\t' # 防止 read 受空格干扰

# ---------- 1️⃣ 把 CSV 转成 Unix 换行 ----------
if command -v dos2unix > /dev/null; then
    dos2unix -q "$CSV_FILE"
fi

# ---------- 2️⃣ 使用 process substitution 读取 ----------
# 创建一个临时文件来记录验证失败的标志
ERROR_FLAG="/tmp/verify_failed_$$_$(date +%s)"
rm -f "$ERROR_FLAG"
CASE_IDX=0
BASE_PORT=$((20000 + ($$ % 10000)))
# 这里只能单次测试一个shape 如果循环测试多个shape会有问题
while IFS=',' read -r M K N; do
    N=${N//$'\r'/}
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # --------- 生成 golden data ----------
    rm -f ./out/*_output.bin 2>/dev/null
    python3 "${UTILS_PATH}/gen_quant_alltoall_hif8_data.py" \
        --rankSize "${RANK_SIZE}" -m "${M}" -n "${N}" -k "${K}" \
        --data_dir "${DATA_DIR}"

    # --------- 启动 MPI 进程 ----------
    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:$((BASE_PORT + CASE_IDX * 10))"
    echo "Using SHMEM IPPORT: ${IPPORT}"

    # Start Process
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    if ! wait; then # 等待 mpirun 完成
        echo "[ERROR] rank process failed for M=${M}, K=${K}, N=${N}"
        exit 1
    fi

    sleep 1 

    # --------- 验证每个 rank 的输出 ----------
    echo "Verifying results..."
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        (
            verify_log=$(python3 ${UTILS_PATH}/verify_result.py ./out/rank_${idx}_output.bin ./out/rank_${idx}_golden.bin hif8 ${M} ${N} ${K} 2>&1 || true)
            echo "$verify_log"
            if ! echo "$verify_log" | grep -q "PASS"; then
                touch "$ERROR_FLAG"
            fi
        ) &
    done
    wait

    if [ -f "$ERROR_FLAG" ]; then
        echo "error case！！"
        rm -f "$ERROR_FLAG"
        cd "${CURRENT_DIR}"
        exit 1
    fi
    CASE_IDX=$((CASE_IDX + 1))

done < <(tail -n +2 "$CSV_FILE")

echo "All test cases passed successfully!"

cd ${CURRENT_DIR}