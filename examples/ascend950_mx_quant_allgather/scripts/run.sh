#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

source $PROJECT_ROOT/3rdparty/shmem/install/set_env.sh || {
    echo "[ERROR] Running set_env.sh failed."
    exit 1
}

# Usage: run.sh <device_id_list> [dtype]
# dtype: fp8_e4m3 (default), fp8_e5m2, fp4_e2m1, fp4_e1m2

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

DTYPE=${2:-fp8_e4m3}
echo "[CONFIG] dtype=${DTYPE}"

cd ${PROJECT_ROOT}/examples/ascend950_mx_quant_allgather/
DATA_DIR=`realpath ./out`
echo "DATA_DIR: $DATA_DIR"

# Select binary based on dtype
case "${DTYPE}" in
    fp8_e4m3) EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_mx_quant_allgather ;;
    fp8_e5m2) EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_mx_quant_allgather_e5m2 ;;
    fp4_e2m1) EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_mx_quant_allgather_fp4_e2m1 ;;
    fp4_e1m2) EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_mx_quant_allgather_fp4_e1m2 ;;
    *) echo "[ERROR] Unknown dtype: ${DTYPE}"; exit 1 ;;
esac
echo "[CONFIG] EXEC_BIN=${EXEC_BIN}"

set -euo pipefail
IFS=$'\n\t'

if command -v dos2unix > /dev/null; then
    dos2unix -q "$CSV_FILE"
fi

ERROR_FLAG="/tmp/verify_mx_failed_$(date +%s)"
rm -f "$ERROR_FLAG"

# Determine verify dtype for fp4 (packed bytes)
if [ "$DTYPE" = "fp4_e2m1" ] || [ "$DTYPE" = "fp4_e1m2" ]; then
    VERIFY_DTYPE="uint8"
    PACK_RATIO=2
else
    VERIFY_DTYPE="${DTYPE}"
    PACK_RATIO=1
fi

while IFS=',' read -r M K N; do
    N=${N//$'\r'/}
    echo "Processing test case: M=${M}, K=${K}, N=${N}, dtype=${DTYPE}"

    # Generate golden data
    rm -f ./out/*.bin 2>/dev/null
    python3 "${UTILS_PATH}/gen_mx_quant_allgather_data.py" \
        --rankSize "${RANK_SIZE}" -m "${M}" -n "${N}" -k "${K}" \
        --block_size 32 --dtype "${DTYPE}" --data_dir "${DATA_DIR}"

    # Launch MPI processes
    IPPORT="tcp://127.0.0.1:8789"

    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done
    wait

    # Verify quantized output (byte-level comparison for all types)
    echo "Verifying quantized data..."
    QUANT_N=$((N / PACK_RATIO))
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        (
            verify_log=$(python3 -c "
import numpy as np, sys
a = np.fromfile('./out/rank_${idx}_output.bin', dtype=np.uint8)
b = np.fromfile('./out/rank_${idx}_golden.bin', dtype=np.uint8)
if len(a) != len(b):
    print(f'Rank ${idx} quant: FAIL (size mismatch: actual={len(a)} vs golden={len(b)})')
    sys.exit(1)
diff = np.where(a != b)[0]
if len(diff) == 0:
    print(f'Rank ${idx} quant: PASS (matched={len(a)}/{len(a)})')
else:
    print(f'Rank ${idx} quant: FAIL (mismatched={len(diff)}/{len(a)})')
    for i in diff[:20]:
        print(f'  [{i}]: actual=0x{a[i]:02x} golden=0x{b[i]:02x}')
    if len(diff) > 20:
        print(f'  ... and {len(diff)-20} more mismatches')
    sys.exit(1)
" 2>&1 || true)
            echo "$verify_log"
            if ! echo "$verify_log" | grep -q "PASS"; then
                touch "$ERROR_FLAG"
            fi
        ) &
    done
    wait

    # Verify mxscale output
    echo "Verifying mxscale data..."
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        (
            actual="./out/rank_${idx}_mxscale.bin"
            golden="./out/rank_${idx}_golden_mxscale.bin"
            if python3 -c "
import numpy as np, sys
a = np.fromfile('${actual}', dtype=np.uint8)
b = np.fromfile('${golden}', dtype=np.uint8)
if len(a) != len(b):
    print(f'Rank ${idx} mxscale: FAIL (size mismatch: actual={len(a)} vs golden={len(b)})')
    sys.exit(1)
diff = np.where(a != b)[0]
if len(diff) == 0:
    print(f'Rank ${idx} mxscale: PASS (matched={len(a)}/{len(a)})')
else:
    print(f'Rank ${idx} mxscale: FAIL (mismatched={len(diff)}/{len(a)})')
    num_scales_per_row = ${N} // 32
    for i in diff[:20]:
        row = i // num_scales_per_row
        col = i % num_scales_per_row
        print(f'  [{i}] row={row} col={col}: actual=0x{a[i]:02x} golden=0x{b[i]:02x}')
    if len(diff) > 20:
        print(f'  ... and {len(diff)-20} more mismatches')
    sys.exit(1)
" 2>&1; then
                :
            else
                touch "$ERROR_FLAG"
            fi
        ) &
    done
    wait

    if [ -f "$ERROR_FLAG" ]; then
        echo "ERROR: Test case failed!"
        rm -f "$ERROR_FLAG"
        cd "${CURRENT_DIR}"
        exit 1
    fi

done < <(tail -n +2 "$CSV_FILE")

echo "All test cases passed successfully!"
cd ${CURRENT_DIR}
