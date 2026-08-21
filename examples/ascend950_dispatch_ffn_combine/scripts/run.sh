#!/bin/bash
set -e
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

CSV_FILE="${CSV_FILE:-${SCRIPT_DIR}/test_shapes.csv}"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <device_id_list> [expert_num]"
    exit 1
fi

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

DATA_TYPE=27
TOPK=${TOPK:-6}
GMM1_M_TILE=${GMM1_M_TILE:-128}
ROUTING_MODE=${ROUTING_MODE:-random}
EXPERT_NUM=${2:-16}

if ! [[ "$EXPERT_NUM" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] expert_num must be a positive integer."
    exit 1
fi
if ! [[ "$TOPK" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] TOPK must be a positive integer."
    exit 1
fi
if [ "$GMM1_M_TILE" != "128" ] && [ "$GMM1_M_TILE" != "256" ]; then
    echo "[ERROR] GMM1_M_TILE must be 128 or 256, got: ${GMM1_M_TILE}."
    exit 1
fi
if [ "$ROUTING_MODE" != "random" ] && [ "$ROUTING_MODE" != "balanced" ]; then
    echo "[ERROR] ROUTING_MODE must be random or balanced, got: ${ROUTING_MODE}."
    exit 1
fi
if [ "$EXPERT_NUM" -lt "$TOPK" ]; then
    echo "[ERROR] expert_num (${EXPERT_NUM}) must be greater than or equal to TOPK (${TOPK})."
    exit 1
fi

EP_SIZE=$RANK_SIZE
EP_PRE_RANK=$((EXPERT_NUM / $EP_SIZE))
if [ "$EP_PRE_RANK" -lt 1 ] || [ $((EXPERT_NUM % EP_SIZE)) -ne 0 ]; then
    echo "[ERROR] expert_num (${EXPERT_NUM}) must be divisible by rank size (${RANK_SIZE})."
    exit 1
fi

echo "[CONFIG] io=bf16 ffn=mxfp8_e4m3_e8m0 data_type=${DATA_TYPE} rank_size=${RANK_SIZE} ep_size=${EP_SIZE} topk=${TOPK} expert_num=${EXPERT_NUM} expert_per_rank=${EP_PRE_RANK} gmm1_m_tile=${GMM1_M_TILE} routing_mode=${ROUTING_MODE}"

cd ${PROJECT_ROOT}/examples/ascend950_dispatch_ffn_combine/
EXEC_BIN=${EXEC_BIN:-${PROJECT_ROOT}/build/bin/ascend950_dispatch_ffn_combine}
FIRST_NPU="${DEVICE_ID_LIST[0]}"

mkdir -p output
tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf output/*.bin
    ASCEND_RT_VISIBLE_DEVICES="${FIRST_NPU}" \
        python3 ${UTILS_PATH}/gen_data_moe.py "dispatch_ffn_combine" ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} --top_k ${TOPK} --expert $EXPERT_NUM --ep $EP_SIZE --routing-mode "$ROUTING_MODE"

    # Set necessary parameters
    IPPORT="${IPPORT:-tcp://127.0.0.1:27008}"

    # Start Process
    rank_pids=()
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" $EP_PRE_RANK $DATA_TYPE 1 0 "$1" "$TOPK" "$GMM1_M_TILE" &
        rank_pids+=("$!")
    done

    rank_failed=0
    for idx in "${!rank_pids[@]}"; do
        if ! wait "${rank_pids[$idx]}"; then
            echo "[ERROR] Rank ${idx} execution failed."
            rank_failed=1
        fi
    done
    if [ "$rank_failed" -ne 0 ]; then
        exit 1
    fi

    # Wait until every asynchronously flushed rank output has a stable, non-zero size.
    OUTPUT_FILE_WAIT_RETRIES="${OUTPUT_FILE_WAIT_RETRIES:-600}"
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        output_file="./output/output_${idx}.bin"
        previous_size=-1
        stable_checks=0
        for (( retry =0; retry < OUTPUT_FILE_WAIT_RETRIES; retry = retry + 1 )); do
            if [ -s "$output_file" ]; then
                current_size=$(wc -c <"$output_file")
                if [ "$current_size" -eq "$previous_size" ]; then
                    stable_checks=$((stable_checks + 1))
                    if [ "$stable_checks" -ge 2 ]; then
                        break
                    fi
                else
                    previous_size="$current_size"
                    stable_checks=0
                fi
            fi
            sleep 0.1
        done
        if [ "$stable_checks" -lt 2 ]; then
            echo "[ERROR] Timed out waiting for stable output file: ${output_file}"
            exit 1
        fi
    done

    # Verify output
    verify_pids=()
    verify_logs=()
    for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        verify_log="./output/verify_${idx}.log"
        verify_logs+=("$verify_log")
        python3 ${UTILS_PATH}/verify_result.py ./output/output_${idx}.bin ./output/out_gather_combine_${idx}.bin $DATA_TYPE ${M} ${N} ${K} >"$verify_log" 2>&1 &
        verify_pids+=("$!")
    done

    verify_failed=0
    for idx in "${!verify_pids[@]}"; do
        if ! wait "${verify_pids[$idx]}"; then
            echo "[ERROR] Rank ${idx} verification process failed."
            verify_failed=1
        fi
    done

    for idx in "${!verify_logs[@]}"; do
        verify_log="${verify_logs[$idx]}"
        echo "[VERIFY] Rank ${idx}"
        cat "$verify_log"
        if ! grep -q "PASS" "$verify_log" || grep -q "ERROR" "$verify_log"; then
            echo "[ERROR] Rank ${idx} numerical verification failed."
            verify_failed=1
        fi
    done

    if [ "$verify_failed" -ne 0 ]; then
        exit 1
    fi

done

cd ${CURRENT_DIR}
