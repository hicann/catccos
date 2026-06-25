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

set -o pipefail

# modify example running list here
EXAMPLES=(
    "allgather_matmul"
    "allgather_matmul_dequant"
    "allgather_matmul_dequant_bias"
    "alltoallv_gmm_v2"
    "alltoallv_grouped_matmul"
    "grouped_matmul_alltoallv"
    "matmul_allreduce"
    "matmul_reduce_scatter"
    "matmul_dequant_reduce_scatter_v2"
)

RANK="${1:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DYNAMIC_TILING_DIR="$PROJECT_ROOT/tests/dynamic_tiling"

SUCCESS_LOG="$EXAMPLES_DIR/success.log"
FAILURE_LOG="$EXAMPLES_DIR/failure.log"
SUMMARY_LOG="$EXAMPLES_DIR/execution_summary.log"

function check_log_has_errors() {
    local log_file="$1"

    if [ ! -f "$log_file" ]; then
        return 0
    fi

    if grep -qE "error num: [1-9][0-9]*" "$log_file" 2>/dev/null; then
        return 0
    fi

    if grep -qE 'aclError:[0-9]+' "$log_file" 2>/dev/null; then
        return 0
    fi

    if grep -qE '(^|[[:space:]])ERROR([[:space:]]|$|\[0m)' "$log_file" 2>/dev/null; then
        return 0
    fi

    if grep -qE '\[ERROR\]' "$log_file" 2>/dev/null; then
        return 0
    fi

    if grep -qE 'Traceback \(most recent call last\):' "$log_file" 2>/dev/null; then
        return 0
    fi

    if grep -qE 'ModuleNotFoundError|ImportError|FileNotFoundError' "$log_file" 2>/dev/null; then
        return 0
    fi

    return 1
}

function get_example_log_dir() {
    local example_name="$1"

    if [ "$example_name" = "dynamic_tiling" ]; then
        echo "$DYNAMIC_TILING_DIR"
    else
        echo "$EXAMPLES_DIR/$example_name"
    fi
}

function run_single_example() {
    local example_dir="$1"
    local example_path="$EXAMPLES_DIR/$example_dir"

    echo "Entering directory: $example_dir"

    if [ ! -d "$example_path" ]; then
        echo "Directory $example_path does not exist. Skipping."
        return 1
    fi

    cd "$example_path" || { echo "Failed to enter directory $example_path"; return 1; }

    echo "Running build and run scripts for $example_dir"

    bash scripts/build.sh > building_log.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error running build.sh in $example_dir, skipping to next directory."
        echo "$example_dir: build failed" >> "$FAILURE_LOG"
        cd "$EXAMPLES_DIR"
        return 1
    fi

    bash scripts/run.sh "$RANK" > running_log.log 2>&1
    local run_exit_code=$?

    local has_log_error=false
    if check_log_has_errors "running_log.log"; then
        has_log_error=true
    fi

    if [ $run_exit_code -ne 0 ] || [ "$has_log_error" = true ]; then
        echo "Error running run.sh in $example_dir, skipping to next directory."
        echo "$example_dir: failed" >> "$FAILURE_LOG"
        cd "$EXAMPLES_DIR"
        return 1
    fi

    echo "$example_dir: succeeded" >> "$SUCCESS_LOG"
    cd "$EXAMPLES_DIR"
    return 0
}

function fn_run_examples() {
    local failed=0

    for EXAMPLE_DIR in "${EXAMPLES[@]}"; do
        if ! run_single_example "$EXAMPLE_DIR"; then
            ((failed++)) || true
        fi
    done

    return $failed
}

function fn_run_cases_in_dynamic_tiling() {
    if [ ! -d "$DYNAMIC_TILING_DIR" ]; then
        echo "Directory $DYNAMIC_TILING_DIR does not exist. Skipping dynamic_tiling."
        echo "dynamic_tiling: directory missing" >> "$FAILURE_LOG"
        return 1
    fi

    cd "$DYNAMIC_TILING_DIR" || return 1

    echo "Running cases for dynamic_tiling"

    bash scripts/build.sh > building_log.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error running build.sh in dynamic_tiling."
        echo "dynamic_tiling: build failed" >> "$FAILURE_LOG"
        cd "$EXAMPLES_DIR"
        return 1
    fi

    bash scripts/run.sh "mmar" 1 "$RANK" > running_log.log 2>&1
    local run_exit_code=$?

    local has_log_error=false
    if check_log_has_errors "running_log.log"; then
        has_log_error=true
    fi

    if [ $run_exit_code -ne 0 ] || [ "$has_log_error" = true ]; then
        echo "Error running run.sh in dynamic_tiling."
        echo "dynamic_tiling: failed" >> "$FAILURE_LOG"
        cd "$EXAMPLES_DIR"
        return 1
    fi

    echo "dynamic_tiling: succeeded" >> "$SUCCESS_LOG"
    cd "$EXAMPLES_DIR"
    return 0
}

function show_failed_logs() {
    if [ ! -f "$FAILURE_LOG" ] || [ ! -s "$FAILURE_LOG" ]; then
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Showing logs for failed examples:"
    echo "=========================================="

    while IFS= read -r line; do
        local failed_dir
        local failed_type
        local log_dir

        failed_dir=$(echo "$line" | cut -d: -f1)
        failed_type=$(echo "$line" | cut -d: -f2- | xargs)
        log_dir=$(get_example_log_dir "$failed_dir")

        echo ""
        echo ">>> [$failed_dir] ($failed_type)"
        echo "-------------------------------------------"

        if [ "$failed_type" = "build failed" ]; then
            if [ -f "$log_dir/building_log.log" ]; then
                echo "=== building_log.log ==="
                cat "$log_dir/building_log.log"
            else
                echo "building_log.log not found at $log_dir/building_log.log"
            fi
        else
            if [ -f "$log_dir/running_log.log" ]; then
                echo "=== running_log.log ==="
                cat "$log_dir/running_log.log"
            else
                echo "running_log.log not found at $log_dir/running_log.log"
            fi
        fi
        echo "-------------------------------------------"
    done < "$FAILURE_LOG"
}

function generate_summary() {
    local total_failed=${1:-0}

    echo "Summary of execution:" >> "$SUMMARY_LOG"
    echo "Total failures: $total_failed" >> "$SUMMARY_LOG"

    if [ -f "$SUCCESS_LOG" ] && [ -s "$SUCCESS_LOG" ]; then
        echo "Successful runs:" >> "$SUMMARY_LOG"
        cat "$SUCCESS_LOG" >> "$SUMMARY_LOG"
    fi

    if [ -f "$FAILURE_LOG" ] && [ -s "$FAILURE_LOG" ]; then
        echo "Failed runs:" >> "$SUMMARY_LOG"
        cat "$FAILURE_LOG" >> "$SUMMARY_LOG"
    fi
}

function cleanup() {
    [ -f "$SUCCESS_LOG" ] && rm "$SUCCESS_LOG"
    [ -f "$FAILURE_LOG" ] && rm "$FAILURE_LOG"
}

function main() {
    local failed=0
    local example_failed=0
    local dynamic_failed=0

    : > "$SUCCESS_LOG"
    : > "$FAILURE_LOG"
    echo "Execution Summary" > "$SUMMARY_LOG"
    echo "-----------------" >> "$SUMMARY_LOG"

    fn_run_examples
    example_failed=$?
    failed=$((failed + example_failed))

    fn_run_cases_in_dynamic_tiling
    dynamic_failed=$?
    failed=$((failed + dynamic_failed))

    generate_summary "$failed"
    show_failed_logs
    cleanup

    echo "Execution summary saved in $SUMMARY_LOG."
    if [ "$failed" -gt 0 ]; then
        echo "$failed test group(s) failed (examples: $example_failed, dynamic_tiling: $dynamic_failed)."
        exit 1
    fi

    echo "All test groups passed."
    exit 0
}

main
