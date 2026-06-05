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

using namespace AscendC;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950AllGatherMatmulWithConfig(void *stream, uint32_t blockNum, uint64_t fftsAddr,
                                                     KernelParams &kernelParams, uint8_t *symmetricPtr,
                                                     CocTilingParams &cocTiling, uint32_t transA, uint32_t transB)
{
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

void LaunchAscend950AllGatherMatmulBF16(void *stream, uint32_t blockNum, uint64_t fftsAddr, KernelParams &kernelParams,
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

using ElementD = bfloat16_t;
using LayoutD = Catlass::layout::RowMajor;

template <template <class, class, class, class, class, class> class ConfigAlias>
static void LaunchAscend950MatmulReduceScatterWithConfig(void *stream, uint32_t blockNum, uint64_t fftsAddr,
                                                         KernelParams &kernelParams, uint8_t *symmetricPtr,
                                                         CocTilingParams &cocTiling, uint32_t transA, uint32_t transB)
{
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

void LaunchAscend950MatmulReduceScatterBF16(void *stream, uint32_t blockNum, uint64_t fftsAddr,
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
