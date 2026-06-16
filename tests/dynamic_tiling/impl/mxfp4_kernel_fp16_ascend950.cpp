/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_fp4_mx_allgather_matmul/ascend950_fp4_mx_allgather_matmul_device.h"

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutMxScaleA0 = Catlass::layout::RowMajor;
using LayoutMxScaleB0 = Catlass::layout::RowMajor;

using LayoutMxScaleA1 = Catlass::layout::ColumnMajor;
using LayoutMxScaleB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;

using ElementA = float4_e2m1x2_t;
using ElementB = float4_e2m1x2_t;
using ElementMxScaleA = float8_e8m0_t;
using ElementMxScaleB = float8_e8m0_t;
using ElementC = half;

template <template <class, class, class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950Fp4MxAllGatherMatmulWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            kernelParams.ptrA, kernelParams.ptrB,
            kernelParams.customPtrs[0], kernelParams.customPtrs[1], kernelParams.ptrC,
            symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementMxScaleA, LayoutMxScaleA0, ElementMxScaleB, LayoutMxScaleB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementMxScaleA, LayoutMxScaleA0, ElementMxScaleB, LayoutMxScaleB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAscend950Fp4MxAllGatherMatmulFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950Fp4MxAllGatherMatmulWithConfig<Ascend950Fp4MxAllGatherMatmulConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950Fp4MxAllGatherMatmulWithConfig<Ascend950Fp4MxAllGatherMatmulConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}