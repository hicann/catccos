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

# eg. bash run.sh 0,1      # 在 0/1 卡上运行，rank size = 2
# eg. bash run.sh 1,3,5,7  # 在 1/3/5/6 卡上运行，rank size = 4

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils

CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

DATA_TYPE=1
EXPERT_NUM=${2:-4}
EP_SIZE=$RANK_SIZE
EP_PRE_RANK=$((EXPERT_NUM / $EP_SIZE))

cd ${PROJECT_ROOT}/examples/dispatch_gmm/
EXEC_BIN=${PROJECT_ROOT}/build/bin/dispatch_gmm

mkdir -p output
tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf output/*.bin
    python3 ${UTILS_PATH}/gen_data_moe.py "dispatch_gmm" ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} --top_k 4 --expert $EXPERT_NUM --ep $EP_SIZE

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:27008"

    FIRST_NPU="0"

    # Start Process
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" $EP_PRE_RANK $DATA_TYPE 1 0 &
    done

    # Wait until all process exit
    wait

    # Verify output
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        python3 ${UTILS_PATH}/verify_result.py ./output/output_${idx}.bin ./output/out_gmm_matrix_c_${idx}.bin $DATA_TYPE ${M} ${N} ${K} &
    done

    wait
done

cd ${CURRENT_DIR}