#!/bin/bash
# eg. bash run.sh 0,1      # 在 0/1 卡上运行，rank size = 2
# eg. bash run.sh 1,3,5,7  # 在 1/3/5/6 卡上运行，rank size = 4
export SMEM_CONF_STORE_TLS_ENABLE=0
export debug=0

out_type=1

gen_data_file="gen_double_golden_data_mmrs.py"

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

PUBLIC_UTILS_PATH=${PROJECT_ROOT}/examples/utils
export PYTHONPATH=$PUBLIC_UTILS_PATH:$PYTHONPATH

CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/matmul_reduce_scatter/
DATA_DIR=`realpath ./output`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/matmul_reduce_scatter

tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf ./output/*.bin
    python3 ${PUBLIC_UTILS_PATH}/${gen_data_file} "mmrs" ${out_type} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:8734"

    # Start Process
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    # Wait until all process exit
    wait

    # Verify output
    python3 ${PUBLIC_UTILS_PATH}/verify_result.py \
        ${DATA_DIR}/output.bin \
        ${DATA_DIR}/golden_fp32.bin \
        ${out_type} \
        ${M} ${N} ${K} \
        --golden_low=${DATA_DIR}/golden_aclnn.bin
done

cd ${CURRENT_DIR}