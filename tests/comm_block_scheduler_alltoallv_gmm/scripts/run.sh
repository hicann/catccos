#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$(dirname "$TEST_DIR")")
BUILD_DIR=${PROJECT_ROOT}/build_scheduler_ut
DEVICE_ID=${1:-0}

source "${PROJECT_ROOT}/examples/utils/setup.sh" -soc_type Ascend950

cmake -B "$BUILD_DIR" -S "$PROJECT_ROOT" \
    -DCATCCOS_BUILD_TESTS=ON \
    -DCATLASS_BISHENG_ARCH=a5
cmake --build "$BUILD_DIR" --target test_comm_block_scheduler_alltoallv_gmm -j

"${BUILD_DIR}/bin/test_comm_block_scheduler_alltoallv_gmm" "$DEVICE_ID"
