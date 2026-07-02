/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend950_allgather_matmul/ascend950_allgather_matmul_device.h"
#include "ascend950_matmul_reduce_scatter/ascend950_matmul_reduce_scatter_device.h"
#include "ascend950_fp4_mx_matmul_reduce_scatter/ascend950_fp4_mx_matmul_reduce_scatter_device.h"
#include "ascend950_grouped_matmul_alltoallv/ascend950_grouped_matmul_alltoallv_device.h"
#include "ascend950_fp8_mx_grouped_matmul_alltoallv/ascend950_fp8_mx_grouped_matmul_alltoallv_device.h"
#include "ascend950_fp4_mx_grouped_matmul_alltoallv/ascend950_fp4_mx_grouped_matmul_alltoallv_device.h"
#include "ascend950_alltoallv_grouped_matmul/ascend950_alltoallv_grouped_matmul_device.h"
#include "ascend950_fp8_mx_allgather_matmul/ascend950_fp8_mx_allgather_matmul_device.h"
#include "ascend950_fp4_mx_allgather_matmul/ascend950_fp4_mx_allgather_matmul_device.h"
#include "ascend950_fp8_mx_alltoallv_grouped_matmul/ascend950_fp8_mx_alltoallv_grouped_matmul_device.h"
#include "ascend950_fp4_mx_alltoallv_grouped_matmul/ascend950_fp4_mx_alltoallv_grouped_matmul_device.h"

using namespace AscendC;

using ElementA = half;
using ElementB = half;
using ElementC = half;

using ElementFp8Mx = float8_e4m3_t;
using ElementFp4Mx = float4_e2m1x2_t;
using ElementMxScale = float8_e8m0_t;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutMxScaleA0 = Catlass::layout::RowMajor;
using LayoutMxScaleB0 = Catlass::layout::RowMajor;

using LayoutMxScaleA1 = Catlass::layout::ColumnMajor;
using LayoutMxScaleB1 = Catlass::layout::ColumnMajor;

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950AllGatherMatmulWithConfig(void *stream, uint32_t blockNum, uint64_t fftsAddr,
                                                     KernelParams &kernelParams, uint8_t *symmetricPtr,
                                                     CocTilingParams &cocTiling, uint32_t transA, uint32_t transB)
{
    (void)blockNum;
    (void)transA;
    (void)transB;
    auto launch = [&](auto &&deviceOp)
    {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{problemShape,
                                          static_cast<uint32_t>(shmem_my_pe()),
                                          static_cast<uint32_t>(shmem_n_pes()),
                                          cocTiling.commInterval,
                                          kernelParams.ptrA,
                                          kernelParams.ptrB,
                                          kernelParams.ptrC,
                                          symmetricPtr,
                                          commCoreSplit,
                                          commBlockShape,
                                          commTileShape};
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    launch(typename ConfigAlias<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>::Device{});
}

void LaunchAscend950AllGatherMatmulFP16(void *stream, uint32_t blockNum, uint64_t fftsAddr, KernelParams &kernelParams,
                                        uint8_t *workSpace, uint8_t *symmetricPtr, CocTilingParams &cocTiling,
                                        uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128)
    {
        LaunchAscend950AllGatherMatmulWithConfig<Ascend950AllGatherMatmulConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
    else
    {
        LaunchAscend950AllGatherMatmulWithConfig<Ascend950AllGatherMatmulConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

using ElementD = half;
using LayoutD = Catlass::layout::RowMajor;

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MatmulReduceScatterWithConfig(void *stream, uint32_t blockNum, uint64_t fftsAddr,
                                                         KernelParams &kernelParams, uint8_t *symmetricPtr,
                                                         CocTilingParams &cocTiling, uint32_t transA, uint32_t transB)
{
    (void)blockNum;
    (void)transA;
    (void)transB;
    auto launch = [&](auto &&deviceOp)
    {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{problemShape,
                                          static_cast<uint32_t>(shmem_my_pe()),
                                          static_cast<uint32_t>(shmem_n_pes()),
                                          cocTiling.commInterval,
                                          kernelParams.ptrA,
                                          kernelParams.ptrB,
                                          kernelParams.ptrC,
                                          symmetricPtr,
                                          commCoreSplit,
                                          commBlockShape,
                                          commTileShape};
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    launch(typename ConfigAlias<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>::Device{});
}

void LaunchAscend950MatmulReduceScatterFP16(void *stream, uint32_t blockNum, uint64_t fftsAddr,
                                            KernelParams &kernelParams, uint8_t *workSpace, uint8_t *symmetricPtr,
                                            CocTilingParams &cocTiling, uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128)
    {
        LaunchAscend950MatmulReduceScatterWithConfig<Ascend950MatmulReduceScatterConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
    else
    {
        LaunchAscend950MatmulReduceScatterWithConfig<Ascend950MatmulReduceScatterConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// fp4_mm_rs /////////////////////////////

template <typename ElementMxData_, typename ElementMxScale_,
    template <class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MxMatmulReduceScatterWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *aMxScalePtr = kernelParams.customPtrs[0];
    uint8_t *bMxScalePtr = kernelParams.customPtrs[1];
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            kernelParams.ptrA, kernelParams.ptrB, aMxScalePtr, bMxScalePtr, kernelParams.ptrC, symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB0, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB1, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else if (transA && !transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA1, ElementMxData_, LayoutB0, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else {
        launch(typename ConfigAlias<ElementMxData_, LayoutA1, ElementMxData_, LayoutB1, ElementC, LayoutC, ElementMxScale_>::Device{});
    }
}

void LaunchAscend950Fp4MxMatmulReduceScatterFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxMatmulReduceScatterWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxMatmulReduceScatterConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxMatmulReduceScatterWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxMatmulReduceScatterConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// fp8/fp4_mx_allgather_matmul /////////////////////////////

template <typename ElementMxData_, typename ElementMxScale_,
    template <class, class, class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MxAllGatherMatmulWithConfig(
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
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB0, ElementMxScale_, LayoutMxScaleA0, ElementMxScale_, LayoutMxScaleB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB1, ElementMxScale_, LayoutMxScaleA0, ElementMxScale_, LayoutMxScaleB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAscend950Fp8MxAllGatherMatmulFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxAllGatherMatmulWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxAllGatherMatmulConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxAllGatherMatmulWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxAllGatherMatmulConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
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
        LaunchAscend950MxAllGatherMatmulWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxAllGatherMatmulConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxAllGatherMatmulWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxAllGatherMatmulConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// gmm_alltoallv /////////////////////////////

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950GroupedMatmulAllToAllVWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *localExpertPtr = kernelParams.customPtrs[0];
    uint8_t *globalExpertPtr = kernelParams.customPtrs[1];
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            cocTiling.epSize, cocTiling.expertNum,
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
            localExpertPtr, globalExpertPtr,
            symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    } else if (transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else {
        launch(typename ConfigAlias<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAscend950GroupedMatmulAllToAllVFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950GroupedMatmulAllToAllVWithConfig<Ascend950GroupedMatmulAllToAllVConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950GroupedMatmulAllToAllVWithConfig<Ascend950GroupedMatmulAllToAllVConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// fp8/fp4_gmm_alltoallv /////////////////////////////

template <typename ElementMxData_, typename ElementMxScale_,
    template <class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MxGroupedMatmulAllToAllVWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *aMxScalePtr = kernelParams.customPtrs[0];
    uint8_t *bMxScalePtr = kernelParams.customPtrs[1];
    uint8_t *localExpertPtr = kernelParams.customPtrs[2];
    uint8_t *globalExpertPtr = kernelParams.customPtrs[3];
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};
        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            cocTiling.epSize, cocTiling.expertNum,
            kernelParams.ptrA, kernelParams.ptrB, aMxScalePtr, bMxScalePtr, kernelParams.ptrC,
            localExpertPtr, globalExpertPtr,
            symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB0, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB1, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else if (transA && !transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA1, ElementMxData_, LayoutB0, ElementC, LayoutC, ElementMxScale_>::Device{});
    } else {
        launch(typename ConfigAlias<ElementMxData_, LayoutA1, ElementMxData_, LayoutB1, ElementC, LayoutC, ElementMxScale_>::Device{});
    }
}

void LaunchAscend950Fp8MxGroupedMatmulAllToAllVFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxGroupedMatmulAllToAllVWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxGroupedMatmulAllToAllVConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxGroupedMatmulAllToAllVWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxGroupedMatmulAllToAllVConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

void LaunchAscend950Fp4MxGroupedMatmulAllToAllVFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxGroupedMatmulAllToAllVWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxGroupedMatmulAllToAllVConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxGroupedMatmulAllToAllVWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxGroupedMatmulAllToAllVConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// alltoallv_gmm /////////////////////////////

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950AllToAllVGroupedMatmulWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord gemmShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, RoundUp(cocTiling.k, cocTiling.k0)};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};
        typename DeviceOp::Arguments args{
            gemmShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            cocTiling.epSize, cocTiling.expertNum,
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
            kernelParams.customPtrs[0], kernelParams.customPtrs[1],
            symmetricPtr,
            commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAscend950AllToAllVGroupedMatmulFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950AllToAllVGroupedMatmulWithConfig<Ascend950AllToAllVGroupedMatmulConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950AllToAllVGroupedMatmulWithConfig<Ascend950AllToAllVGroupedMatmulConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

///////////////////////////// fp8/fp4_mx_alltoallv_grouped_matmul /////////////////////////////

template <typename ElementMxData_, typename ElementMxScale_,
    template <class, class, class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MxAllToAllVGroupedMatmulWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord gemmShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, RoundUp(cocTiling.k, cocTiling.k0)};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};
        typename DeviceOp::Arguments args{
            gemmShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.commInterval,
            cocTiling.epSize, cocTiling.expertNum,
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
            kernelParams.customPtrs[0], kernelParams.customPtrs[1],
            kernelParams.customPtrs[2], kernelParams.customPtrs[3],
            symmetricPtr,
            commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB0, ElementMxScale_, LayoutMxScaleA0, ElementMxScale_, LayoutMxScaleB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementMxData_, LayoutA0, ElementMxData_, LayoutB1, ElementMxScale_, LayoutMxScaleA0, ElementMxScale_, LayoutMxScaleB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAscend950Fp8MxAllToAllVGroupedMatmulFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxAllToAllVGroupedMatmulWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxAllToAllVGroupedMatmulConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxAllToAllVGroupedMatmulWithConfig<ElementFp8Mx, ElementMxScale,
            Ascend950Fp8MxAllToAllVGroupedMatmulConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

void LaunchAscend950Fp4MxAllToAllVGroupedMatmulFP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAscend950MxAllToAllVGroupedMatmulWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxAllToAllVGroupedMatmulConfig_M0_128>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAscend950MxAllToAllVGroupedMatmulWithConfig<ElementFp4Mx, ElementMxScale,
            Ascend950Fp4MxAllToAllVGroupedMatmulConfig_M0_256>(
                stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}
