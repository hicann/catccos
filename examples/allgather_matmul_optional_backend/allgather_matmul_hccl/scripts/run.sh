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
set -euo pipefail

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $( dirname $(dirname "$SCRIPT_DIR"))))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 16 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/allgather_matmul_optional_backend/allgather_matmul_hccl/

mkdir -p ./out
DATA_DIR=`realpath ./out`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=./build/allgather_matmul_hccl
CUSTOM_OPP_DIR=./build/custom_opp
if [ ! -x "${EXEC_BIN}" ]; then
    echo "Executable ${EXEC_BIN} not found. Please run scripts/build.sh first."
    exit 1
fi
if [ ! -f "${CUSTOM_OPP_DIR}/vendors/customize/bin/set_env.bash" ]; then
    echo "Custom OPP set_env.bash not found. Please run scripts/build.sh first."
    exit 1
fi
source "${CUSTOM_OPP_DIR}/vendors/customize/bin/set_env.bash"
export LD_LIBRARY_PATH="$(realpath "${CUSTOM_OPP_DIR}")/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

while IFS=',' read -r M K N || [ -n "$M$K$N" ]; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    #Generate golden data
    rm -rf ./out/*.bin
    python3 ${UTILS_PATH}/gen_data.py "agmm" 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}

    # Start Process
    if ! ${EXEC_BIN} "$RANK_SIZE" "$M" "$N" "$K" "${DATA_DIR}" "$1"; then
        echo "[ERROR] allgather_matmul_hccl failed for M=${M}, K=${K}, N=${N}"
        exit 1
    fi

    # Verify output
    python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin 1 ${M} ${N} ${K}
done < <(tail -n +2 "$CSV_FILE")

cd ${CURRENT_DIR}
