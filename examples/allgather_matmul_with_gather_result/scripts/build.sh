#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASE_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

SOURCE_DIR=$PROJECT_ROOT
BUILD_DIR=$PROJECT_ROOT/build
mkdir -p $BUILD_DIR
cmake -B $BUILD_DIR -S $SOURCE_DIR
cmake --build $BUILD_DIR --target allgather_matmul_with_gather_result -j