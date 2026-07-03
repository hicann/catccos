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
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${LD_LIBRARY_PATH}"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -eq 0 ]; then
    echo "Error: Device list is empty. Usage: bash run_python.sh \"0,1,2,3\""
    exit 1
fi
if [ "$RANK_SIZE" -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${CURRENT_DIR}
mkdir -p ./out
DATA_DIR=$(realpath ./out)
echo "DATA_DIR: $DATA_DIR"
PYTHON_SCRIPT=${SCRIPT_DIR}/allgather_matmul.py

TOTAL_CASES=0
PASSED_CASES=0
FAILED_CASES=0

# Read MNK from CSV
while IFS=',' read -r M K N; do
    if [ -z "$M" ]; then
        continue
    fi

    TOTAL_CASES=$((TOTAL_CASES + 1))
    CASE_PASSED=1
    echo "Processing test case #${TOTAL_CASES}: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf "${DATA_DIR}"/*.bin
    if ! python3 ${UTILS_PATH}/gen_data.py "agmm" 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}; then
        echo "[FAIL] Generate data failed for M=${M}, K=${K}, N=${N}"
        CASE_PASSED=0
    fi

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:8735"

    if [ ${CASE_PASSED} -eq 1 ]; then
        # Generate output file (m * rankSize * n * 2 bytes for fp16)
        FILE_SIZE=$((M * RANK_SIZE * N * 2))
        if ! dd if=/dev/zero of="${DATA_DIR}/output.bin" bs=${FILE_SIZE} count=1; then
            echo "[FAIL] Create output file failed for M=${M}, K=${K}, N=${N}"
            CASE_PASSED=0
        else
            echo "Output File Created!"
        fi
    fi

    if [ ${CASE_PASSED} -eq 1 ]; then
        # Start Python processes
        PIDS=()
        for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
            echo "Rank ${idx} started!"
            python3 ${PYTHON_SCRIPT} \
                --rank_size ${RANK_SIZE} \
                --rank_id ${idx} \
                --m ${M} \
                --n ${N} \
                --k ${K} \
                --data_dir "${DATA_DIR}" \
                --ip_port "${IPPORT}" &
            PIDS+=($!)
        done

        # Wait until all process exit
        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then
                CASE_PASSED=0
            fi
        done
    fi

    if [ ${CASE_PASSED} -eq 1 ]; then
        # Verify output
        if ! python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin 1 $((M * RANK_SIZE)) ${N} ${K}; then
            CASE_PASSED=0
        fi
    fi

    if [ ${CASE_PASSED} -eq 1 ]; then
        PASSED_CASES=$((PASSED_CASES + 1))
        echo "[PASS] M=${M}, K=${K}, N=${N}"
    else
        FAILED_CASES=$((FAILED_CASES + 1))
        echo "[FAIL] M=${M}, K=${K}, N=${N}"
    fi
done < <(tail -n +2 "$CSV_FILE")

echo "AGMM Python test summary: total=${TOTAL_CASES}, pass=${PASSED_CASES}, fail=${FAILED_CASES}"

cd ${CURRENT_DIR}
if [ ${FAILED_CASES} -ne 0 ]; then
    exit 1
fi
