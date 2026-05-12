/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_matmul_device.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

namespace CatccosKernel {

void catccos_allgather_matmul_wrapper(
    uint32_t blockDim,
    aclrtStream stream,
    uint64_t fftsAddr,
    uint8_t* aPtr,
    uint8_t* bPtr,
    uint8_t* cPtr,
    uint8_t* gmSymmetric,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    int rankId,
    int rankSize)
{
    using Config = AllGatherMatmulConfig_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using DeviceOp = Config::Device;

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 20;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        aPtr, bPtr, cPtr, gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    deviceOp.Run(stream, blockDim, fftsAddr);
}

} // namespace CatccosKernel