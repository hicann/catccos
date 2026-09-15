#!/usr/bin/env bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
if (( $# > 0 )) && ! [[ $1 =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "Expected a comma-separated device list without spaces or empty entries" >&2
    exit 1
fi
if (( $# >= 1 && $# <= 3 )); then
    IFS=',' read -ra csv_devices <<< "$1"
    case ${#csv_devices[@]} in
        2|4|8|16) ;;
        *) echo "CSV regression requires 2, 4, 8 or 16 devices" >&2; exit 1 ;;
    esac
    bin_dir=$(realpath "${2:-/tmp/catccos-build/bin}")
    csv_file=$script_dir/test_shapes.csv
    IFS= read -r header < "$csv_file"
    [[ ${header%$'\r'} == "M,K,N,transB,local_experts,seed" ]] || { echo "Invalid CSV header" >&2; exit 1; }
    if (( $# == 3 )); then
        output_root=$(realpath -m "$3")
        mkdir -p "$(dirname "$output_root")"
        mkdir "$output_root"
    else
        output_root=$(mktemp -d "${TMPDIR:-/tmp}/grouped_matmul_alltoallv_small_m.XXXXXX")
    fi
    echo "CSV results: $output_root"
    case_id=0
    while IFS=',' read -r m k n trans_b groups seed || [[ -n $m ]]; do
        seed=${seed%$'\r'}
        [[ $seed =~ ^[0-9]+$ ]] || { echo "Missing or invalid CSV seed" >&2; exit 1; }
        case_id=$((case_id + 1))
        echo "Case $case_id: M=$m K=$k N=$n transB=$trans_b seed=$seed"
        CATCCOS_SEED=$seed bash "$script_dir/run.sh" "$1" "$m" "$n" "$k" \
            "$bin_dir" "$output_root/case-$case_id" "$groups" "$trans_b"
    done < <(tail -n +2 "$csv_file")
    (( case_id > 0 )) || { echo "CSV has no cases" >&2; exit 1; }
    exit 0
fi
if (( $# < 6 || $# > 8 )); then
    echo "CSV: $0 device_id_list [bin_dir [new_output_dir]]" >&2
    echo "Usage: $0 device_id_list M N K bin_dir new_output_dir [local_experts [transB]]" >&2
    exit 1
fi
utils=$repo/examples/utils
export ASCEND_RT_VISIBLE_DEVICES=$1
IFS=',' read -ra devices <<< "$1"
ranks=${#devices[@]}
m=$2 n=$3 k=$4
groups=${7:-2}
trans_b=${8:-0}
if ! [[ $m =~ ^[1-9][0-9]*$ && $n =~ ^[1-9][0-9]*$ && $k =~ ^[1-9][0-9]*$ && $trans_b =~ ^[01]$ ]] ||
    (( ranks < 2 || ranks > 16 )); then
    echo "Expected positive M/N/K, transB 0/1 and 2 through 16 devices" >&2
    exit 1
fi
for dimension in "$m" "$n" "$k"; do
    if (( ${#dimension} > 10 )) || (( dimension > 4294967295 )); then
        echo "M/N/K must fit uint32_t" >&2; exit 1
    fi
done
if (( m > 1073741823 / 2 / ranks / n )); then
    echo "GMM requires 2 * ranks * M * N < 1 GiB" >&2; exit 1
fi
# The public double-golden helper calls ACLNN with non-transposed FP16 inputs.
if (( k > 65535 || n > 65535 )); then
    echo "Public ACLNN double-golden generation requires K/N <= 65535 on A2" >&2; exit 1
fi
local_devices=""
previous_device=-1
for ((rank = 0; rank < ranks; ++rank)); do
    if ! [[ ${devices[rank]} =~ ^(0|[1-9][0-9]*)$ ]] || (( devices[rank] <= previous_device )); then
        echo "Device IDs must be distinct nonnegative integers in ascending order" >&2
        exit 1
    fi
    previous_device=${devices[rank]}
    local_devices+="${local_devices:+,}$rank"
done
if ! [[ $groups =~ ^[1-9][0-9]*$ ]] || (( m % ranks != 0 )); then
    echo "The public GMM generator requires positive local_experts and M divisible by ranks" >&2
    exit 1
fi
if (( ${#groups} > 10 )) || (( groups > 4294967295 / ranks )); then
    echo "Total expert count must fit uint32_t" >&2; exit 1
fi
binary=$(realpath "$5/grouped_matmul_alltoallv_small_m")
[[ -x $binary ]] || { echo "Binary is not executable: $binary" >&2; exit 1; }
output_dir=$(realpath -m "$6")
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
cd "$output_dir"
mkdir output
data_dir=$PWD/output
kernel_m=$m
seed_args=()
if [[ -n ${CATCCOS_SEED:-} ]]; then
    seed_args=(--seed "$CATCCOS_SEED")
fi
python "$utils/gen_data_gmm_alltoallv.py" gmmata 1 "$ranks" "$m" "$n" "$k" 0 "$trans_b" --expert "$((groups * ranks))" --ep "$ranks" --double-golden "${seed_args[@]}" > gen-data.log 2>&1

pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((rank = 0; rank < ranks; ++rank)); do
    timeout --kill-after=5s "${CATCCOS_TIMEOUT:-120}" "$binary" "$ranks" "$rank" \
        "tcp://127.0.0.1:${CATCCOS_PORT:-18871}" "$kernel_m" "$n" "$k" "$data_dir" \
        "$local_devices" "$groups" "$trans_b" > "rank-$rank.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
pids=()

status=0
for ((rank = 0; rank < ranks; ++rank)); do
    if python "$utils/verify_result.py" "$data_dir/output_${rank}.bin" "$data_dir/golden_${rank}.bin" 1 \
        "$((m * ranks))" "$n" "$k" --golden_low "$data_dir/golden_low_${rank}.bin" > "verify-$rank.log" 2>&1; then
        echo "Rank $rank: PASS"
    else
        cat "verify-$rank.log"
        status=1
    fi
done
exit "$status"
