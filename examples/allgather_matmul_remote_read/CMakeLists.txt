# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT "0" CACHE STRING
    "Use remote-read local-mm-opt kernel variant. 1: local-mm-opt, 0: base remote-read.")

catccos_example_add_executable(
    allgather_matmul_remote_read
    SOURCES main.cpp
    COMPILE_DEFINITIONS
        CATLASS_ARCH=2201
        ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT=${ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT}
)
