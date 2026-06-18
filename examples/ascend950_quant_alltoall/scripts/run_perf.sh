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
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
RANK_START=0

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/ascend950_quant_alltoall/
DATA_DIR=$(realpath ./out)
EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_quant_alltoall
source $PROJECT_ROOT/examples/ascend950_quant_alltoall/scripts/env.sh

# Performance test shapes (M values)
PERF_SHAPES=(1048576 4194304 8388608 18432000 36864000)

IPPORT="tcp://127.0.0.1:8788"

echo "======================================================"
echo " QuantAllGather Performance Test (rankSize=${RANK_SIZE})"
echo "======================================================"

for M in "${PERF_SHAPES[@]}"; do
    echo ""
    echo "--- M=${M} ---"
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" 1 1 "${DATA_DIR}" "$1" &
    done
done

echo ""
echo "======================================================"
echo " Performance test finished."
echo "======================================================"

cd ${CURRENT_DIR}
