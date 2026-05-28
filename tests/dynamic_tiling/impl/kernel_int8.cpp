/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "allgather_matmul_dequant_bias/allgather_matmul_dequant_bias_device.h"
#include "allgather_matmul_dequant/allgather_matmul_dequant_device.h"

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = half;
using ElementScale = uint64_t;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;
using LayoutC0 = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;

// ============================================================================
// AllGatherMatmulDequantBias
// ============================================================================

template <template <class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulDequantBiasWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling)
{
    using DeviceOp = typename ConfigAlias<
        ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC0, ElementScale, LayoutScale>::Device;

    Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / WORKSPACE_STAGES};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / WORKSPACE_STAGES, cocTiling.k0};

    typename DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
        cocTiling.commInterval,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
        kernelParams.customPtrs[1],  // scalePtr
        kernelParams.customPtrs[0],  // biasPtr
        symmetricPtr,
        commCoreSplit, commBlockShape, commTileShape
    };
    DeviceOp op;
    op.Initialize(args);
    op.Run((aclrtStream)stream, blockNum, fftsAddr);
}

void LaunchAllGatherMatmulDequantBiasINT8(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    (void)transA;
    (void)transB;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulDequantBiasWithConfig<AllGatherMatmulDequantBiasConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling);
    } else {
        LaunchAllGatherMatmulDequantBiasWithConfig<AllGatherMatmulDequantBiasConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling);
    }
}

// ============================================================================
// AllGatherMatmulDequant (no padding)
// ============================================================================

template <template <class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulDequantWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling)
{
    using DeviceOp = typename ConfigAlias<
        ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC0, ElementScale, LayoutScale>::Device;

    Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / WORKSPACE_STAGES};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / WORKSPACE_STAGES, cocTiling.k0};

    typename DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
        cocTiling.commInterval,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
        kernelParams.customPtrs[0],  // scalePtr
        workSpace,                   // ptrWB (no padding, workspace unused)
        symmetricPtr,
        commCoreSplit, commBlockShape, commTileShape
    };
    DeviceOp op;
    op.Initialize(args);
    op.Run((aclrtStream)stream, blockNum, fftsAddr);
}

void LaunchAllGatherMatmulDequantINT8(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)transA;
    (void)transB;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulDequantWithConfig<AllGatherMatmulDequantConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling);
    } else {
        LaunchAllGatherMatmulDequantWithConfig<AllGatherMatmulDequantConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling);
    }
}

// ============================================================================
// AllGatherMatmulDequantPadding (with padding)
// ============================================================================

template <template <class, class, class, class, class, class, class, class> class ConfigAlias>
static void LaunchAllGatherMatmulDequantPaddingWithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling)
{
    using DeviceOp = typename ConfigAlias<
        ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC0, ElementScale, LayoutScale>::Device;

    Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / WORKSPACE_STAGES};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / WORKSPACE_STAGES, cocTiling.k0};

    typename DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(shmem_my_pe()), static_cast<uint32_t>(shmem_n_pes()),
        cocTiling.commInterval,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
        kernelParams.customPtrs[0],  // scalePtr
        workSpace,                   // ptrWB (padding workspace)
        symmetricPtr,
        commCoreSplit, commBlockShape, commTileShape
    };
    DeviceOp op;
    op.Initialize(args);
    op.Run((aclrtStream)stream, blockNum, fftsAddr);
}

void LaunchAllGatherMatmulDequantPaddingINT8(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)transA;
    (void)transB;
    if (cocTiling.m0 == 128) {
        LaunchAllGatherMatmulDequantPaddingWithConfig<AllGatherMatmulDequantPaddingConfig_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling);
    } else {
        LaunchAllGatherMatmulDequantPaddingWithConfig<AllGatherMatmulDequantPaddingConfig_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling);
    }
}