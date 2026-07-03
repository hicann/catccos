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
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

source "${PROJECT_ROOT}/examples/utils/setup.sh" || {
    echo "[ERROR] Running setup.sh in ${PROJECT_ROOT}/examples/utils failed."
    exit 1
}

SOURCE_DIR=$PROJECT_ROOT
BUILD_DIR=$PROJECT_ROOT/build

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is required but not found in PATH"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake is required but not found in PATH"
    exit 1
fi

# Get Torch's CMake prefix path
TORCH_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"

echo "Using TORCH_PREFIX_PATH=${TORCH_PREFIX_PATH}"
mkdir -p $BUILD_DIR
cmake -B $BUILD_DIR -S $SOURCE_DIR \
-DCATCCOS_TORCH_EXTENSION=ON \
-DCMAKE_PREFIX_PATH="${TORCH_PREFIX_PATH}" \
-DPython3_EXECUTABLE=$(which python3)


echo "== Build target allgather_matmul =="
cmake --build $BUILD_DIR --target allgather_matmul_kernel_build catccos_torch -j
