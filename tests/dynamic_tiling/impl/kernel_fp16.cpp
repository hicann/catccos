/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_allreduce/matmul_allreduce_device.h"
#include "allgather_matmul/allgather_matmul_device.h"
#include "matmul_reduce_scatter/matmul_reduce_scatter_device.h"
#include "allgather_matmul_with_gather_result/allgather_matmul_with_gather_result_device.h"
#include "grouped_matmul_alltoallv/grouped_matmul_alltoallv_device.h"
#include "alltoallv_grouped_matmul/alltoallv_grouped_matmul_device.h"
#include "alltoallv_gmm_v2/alltoallv_gmm_v2_device.h"

#ifdef RDMA_TRANSPORT
#include "allgather_matmul_rdma/allgather_matmul_rdma_device.h"
#endif

using namespace AscendC;

using ElementA = half;
using ElementB = half;
using ElementC = half;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchMatmulAllReduceWithConfig(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
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
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC, symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
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

void LaunchMatmulAllReduceFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchMatmulAllReduceWithConfig<MatmulAllReduceConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchMatmulAllReduceWithConfig<MatmulAllReduceConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulWithConfig(
    void *stream, uint64_t fftsAddr,
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
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC, symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAllGatherMatmulFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulWithConfig<AllGatherMatmulConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAllGatherMatmulWithConfig<AllGatherMatmulConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchMatmulReduceScatterWithConfig(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
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
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC, symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
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

void LaunchMatmulReduceScatterFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchMatmulReduceScatterWithConfig<MatmulReduceScatterConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchMatmulReduceScatterWithConfig<MatmulReduceScatterConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulWithGatherResultWithConfig(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, RoundUp(cocTiling.k, cocTiling.k0)};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};

        uint32_t commCoreNum = cocTiling.commDataSplit * cocTiling.commNpuSplit;
        uint32_t copyCoreNum = BLOCK_NUM - commCoreNum;
        uint32_t rankSize = static_cast<uint32_t>(shmem_n_pes());
        uint32_t copyBlockM = CeilDiv((cocTiling.commInterval * cocTiling.m0), CeilDiv(copyCoreNum, rankSize));
        uint32_t copyBlockN = RoundUp<32 / sizeof(ElementA)>(cocTiling.k);
        Catlass::MatrixCoord copyGatherABlockShape{copyBlockM, copyBlockN};

        constexpr uint32_t UB_STAGES_VAL = 2;
        uint32_t maxCopyLength = Catlass::Arch::AtlasA2::UB_SIZE / UB_STAGES_VAL / sizeof(ElementA);
        uint32_t copyTileN = Min<uint32_t>(maxCopyLength, copyBlockN);
        uint32_t copyTileM = maxCopyLength / copyTileN;
        Catlass::MatrixCoord copyGatherATileShape{copyTileM, copyTileN};

        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), rankSize,
            cocTiling.commInterval,
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
            kernelParams.customPtrs[0],
            symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape,
            copyGatherABlockShape, copyGatherATileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAllGatherMatmulWithGatherResultFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulWithGatherResultWithConfig<AllGatherMatmulWithGatherResultConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAllGatherMatmulWithGatherResultWithConfig<AllGatherMatmulWithGatherResultConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchGroupedMatmulAllToAllVWithConfig(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
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
            kernelParams.customPtrs[0], kernelParams.customPtrs[1],
            symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
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

void LaunchGroupedMatmulAllToAllVFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchGroupedMatmulAllToAllVWithConfig<GroupedMatmulAllToAllVConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchGroupedMatmulAllToAllVWithConfig<GroupedMatmulAllToAllVConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllToAllVGroupedMatmulWithConfig(
    void *stream, uint64_t fftsAddr,
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
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAllToAllVGroupedMatmulFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAllToAllVGroupedMatmulWithConfig<AllToAllVGroupedMatmulConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAllToAllVGroupedMatmulWithConfig<AllToAllVGroupedMatmulConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}

#ifdef RDMA_TRANSPORT
template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulRdmaWithConfig(
    void *stream, uint64_t fftsAddr,
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
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC, symmetricPtr,
            commCoreSplit, commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAllGatherMatmulRdmaFP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulRdmaWithConfig<AllGatherMatmulRdmaConfig_M0_128>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAllGatherMatmulRdmaWithConfig<AllGatherMatmulRdmaConfig_M0_256>(
            stream, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}
#endif

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllToAllVGMMV2WithConfig(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
        Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
        Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};
        typename DeviceOp::Arguments args{
            problemShape,
            static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
            cocTiling.epSize, cocTiling.expertNum,
            kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
            kernelParams.customPtrs[0],
            workSpace,
            symmetricPtr,
            commBlockShape, commTileShape
        };
        DeviceOp op;
        op.Initialize(args);
        op.Run((aclrtStream)stream, BLOCK_NUM, fftsAddr);
    };
    if (!transA && !transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>::Device{});
    }
}

void LaunchAllToAllVGMMV2FP16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    if (cocTiling.m0 == 128) {
        LaunchAllToAllVGMMV2WithConfig<AllToAllVGMMV2Config_M0_128>(
            stream, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling, transA, transB);
    } else {
        LaunchAllToAllVGMMV2WithConfig<AllToAllVGMMV2Config_M0_256>(
            stream, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling, transA, transB);
    }
}