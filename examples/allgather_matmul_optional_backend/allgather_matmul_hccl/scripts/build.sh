#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
EXAMPLE_ROOT=$(dirname "$SCRIPT_DIR")

SOURCE_DIR=$EXAMPLE_ROOT
BUILD_DIR=$EXAMPLE_ROOT/build
CUSTOM_OPP_DIR=$BUILD_DIR/custom_opp
rm -rf $BUILD_DIR
cmake -B $BUILD_DIR -S $SOURCE_DIR -DCATCCOS_BUILD_TESTS=OFF
cmake --build $BUILD_DIR --target allgather_matmul_hccl -j

rm -rf "$CUSTOM_OPP_DIR"
mkdir -p "$CUSTOM_OPP_DIR"

OPP_PACKAGE=$(find "$BUILD_DIR/msopgen/build_out" -maxdepth 2 -name 'custom_opp_*.run' -type f | head -n 1)
if [ -z "$OPP_PACKAGE" ]; then
    echo "[ERROR] Cannot find custom_opp package under $BUILD_DIR/msopgen/build_out"
    echo "[ERROR] msOpGen documentation says the package should be generated as build_out/custom_opp_<target_os>_<target_architecture>.run."
    exit 1
fi

chmod +x "$OPP_PACKAGE"
"$OPP_PACKAGE" --quiet --install-path="$CUSTOM_OPP_DIR"
