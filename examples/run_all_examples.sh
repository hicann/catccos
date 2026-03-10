#!/bin/bash

# modify example running list here
EXAMPLES=(
    "allgather_matmul" 
    "allgather_matmul_dequant" 
    "allgather_matmul_dequant_bias" 
    "alltoallv_gmm_v2" 
    # "alltoallv_grouped_matmul"
    # "grouped_matmul_alltoallv"
    "matmul_allreduce" 
    "matmul_reduce_scatter"
)

RANK="${1:-0,1}"

CURRENT_DIR="$(pwd)"
SUCCESS_LOG="$CURRENT_DIR/success.log"
FAILURE_LOG="$CURRENT_DIR/failure.log"
SUMMARY_LOG="$CURRENT_DIR/execution_summary.log"

function fn_run_examples() {
    for EXAMPLE_DIR in "${EXAMPLES[@]}"; do
        echo "Entering directory: $EXAMPLE_DIR"

        if [ -d "$EXAMPLE_DIR" ]; then
            cd "$EXAMPLE_DIR" || { echo "Failed to enter directory $EXAMPLE_DIR"; continue; }

            echo "Running build and run scripts for $EXAMPLE_DIR"
            bash scripts/build.sh > building_log.log 2>&1
            if [ $? -ne 0 ]; then
                echo "Error running build.sh in $EXAMPLE_DIR, skipping to next directory."
                echo "$EXAMPLE_DIR: build failed" >> $FAILURE_LOG
                cd "$CURRENT_DIR"
                continue
            fi

            bash scripts/run.sh "$RANK" > running_log.log 2>&1
            if [ $? -ne 0 ]; then
                echo "Error running run.sh in $EXAMPLE_DIR, skipping to next directory."
                echo "$EXAMPLE_DIR: failed" >> $FAILURE_LOG
            else
                echo "$EXAMPLE_DIR: succeeded" >> $SUCCESS_LOG
            fi

            cd "$CURRENT_DIR"
        else
            echo "Directory $EXAMPLE_DIR does not exist. Skipping."
        fi
    done
}

function fn_run_cases_in_dynamic_tiling() {
    cd "dynamic_tiling"

    echo "Running cases for dynamic_tiling"
    bash scripts/build.sh > building_log.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error running build.sh in dynamic_tiling."
        echo "dynamic_tiling: build failed" >> $FAILURE_LOG
        cd "$CURRENT_DIR"
        continue
    fi

    bash scripts/run.sh "mmar" 1 "$RANK" > running_log.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error running run.sh in dynamic_tiling."
        echo "dynamic_tiling: failed" >> $FAILURE_LOG
    else
        echo "dynamic_tiling: succeeded" >> $SUCCESS_LOG
    fi

    cd "$CURRENT_DIR"

}

function main() {
    echo "Execution Summary" > $SUMMARY_LOG
    echo "-----------------" >> $SUMMARY_LOG

    fn_run_examples
    fn_run_cases_in_dynamic_tiling

    echo "Summary of execution:" >> $SUMMARY_LOG
    if [ -f "$SUCCESS_LOG" ] && [ -s "$SUCCESS_LOG" ]; then
        echo "Successful runs:" >> $SUMMARY_LOG
        cat $SUCCESS_LOG >> $SUMMARY_LOG
    fi

    if [ -f "$FAILURE_LOG" ] && [ -s "$FAILURE_LOG" ]; then
        echo "Failed runs:" >> $SUMMARY_LOG
        cat $FAILURE_LOG >> $SUMMARY_LOG
    fi

    [ -f "$SUCCESS_LOG" ] && rm "$SUCCESS_LOG"
    [ -f "$FAILURE_LOG" ] && rm "$FAILURE_LOG"

    echo "Execution summary saved in $SUMMARY_LOG."
}

main
