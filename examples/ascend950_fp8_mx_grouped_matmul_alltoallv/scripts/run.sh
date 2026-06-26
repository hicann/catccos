#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

source $PROJECT_ROOT/examples/utils/setup.sh -soc_type Ascend950 || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

EP_SIZE=${2:-2}
EXPERT_NUM=${3:-8}

cd ${PROJECT_ROOT}/examples/ascend950_fp8_mx_grouped_matmul_alltoallv/
EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_fp8_mx_grouped_matmul_alltoallv

mkdir -p output
tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf output/*.bin
    python3 ${UTILS_PATH}/gen_data_gmm_alltoallv_fp8_mx.py 5 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0 --expert $EXPERT_NUM --ep $EP_SIZE

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:27028"

    # Start Process
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" $EP_SIZE $EXPERT_NUM "$1" &
    done

    # Wait until all process exit
    wait

    # Verify output
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        python3 ${UTILS_PATH}/verify_result.py ./output/output_${idx}.bin ./output/golden_${idx}.bin 1 ${M} ${N} ${K} &
    done

    wait
done

cd ${CURRENT_DIR}
