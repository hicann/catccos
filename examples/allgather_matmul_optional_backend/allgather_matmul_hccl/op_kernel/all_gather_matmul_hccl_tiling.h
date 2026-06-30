/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_MATMUL_TILING_H
#define ALLGATHER_MATMUL_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint64_t HCCL_WINDOW_SIZE = 16UL * 1024UL * 1024UL;

struct CocTilingParams {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
    uint32_t rankSize = 0;
    uint32_t epSize = 0;
    uint32_t expertNum = 0;
    uint64_t segmentSize = 0;
};

struct AllGatherMatmulHcclTiling {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CocTilingParams params;
};

#endif  // ALLGATHER_MATMUL_TILING_H
