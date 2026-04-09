/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMUL_REDUCE_SCATTER_KERNEL_H
#define MATMUL_REDUCE_SCATTER_KERNEL_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_scheduler_reduce_scatter.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/matmul_reduce_scatter.hpp"
#include "catccos/dgemm/kernel/matmul_reduce_scatter.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void MatmulReduceScatterImpl(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr
)
{
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using SymmetricType = DType;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, SymmetricType>;

    constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerReduceScatter<IS_DYNAMIC, void, 0, true, 7, 1>;

    using RemoteSrcType = SymmetricType;
    using RemoteDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasA2CommRemoteCopy<UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatter<
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    typename TileRemoteCopy::Params tileParams {
        commTileShape
    };

    typename BlockComm::Params blockParams{
        commBlockShape,
        tileParams
    };

    typename BlockCommScheduler::Params swizzleParams {
        commCoreSplit
    };

    // Prepare params
    DistGemmCoord distProblemShape{problemShape.m() / rankSize, problemShape.n(), problemShape.k(), rankSize};
    typename MatmulReduceScatterKernel::Params params{
        distProblemShape,
        rank, rankSize,
        commInterval,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        symmetricPtr,
        blockParams,
        swizzleParams
    };

    // Call kernel
    MatmulReduceScatterKernel matmulCommKernel;
    matmulCommKernel(params);
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_DEVICE
void MatmulReduceScatterImpl_M0_256(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr
)
{
    MatmulReduceScatterImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 256, 128, 256>(
        problemShape, l1TileShape,
        gmA, layoutA, gmB, layoutB, gmD, layoutD,
        rank, rankSize, commInterval,
        commCoreSplit, commBlockShape, commTileShape,
        symmetricPtr
    );
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_DEVICE
void MatmulReduceScatterImpl_M0_128(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr
)
{
    MatmulReduceScatterImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 128, 256, 256>(
        problemShape, l1TileShape,
        gmA, layoutA, gmB, layoutB, gmD, layoutD,
        rank, rankSize, commInterval,
        commCoreSplit, commBlockShape, commTileShape,
        symmetricPtr
    );
}

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_GLOBAL
void MatmulReduceScatter(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR symmetricPtr, CocTilingParams cocTiling, GM_ADDR dump
)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_GLOBAL
void MatmulReduceScatter(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR symmetricPtr, CocTilingParams cocTiling
)
{
#endif
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t n0 = cocTiling.n0;
    uint32_t k0 = cocTiling.k0;
    uint32_t commInterval = cocTiling.commInterval;
    uint32_t commTileM = cocTiling.commTileM;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commBlockM = cocTiling.commBlockM;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, n0};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};

    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m / rankSize, n};
    
    if (m0 == 128){
        MatmulReduceScatterImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rank, rankSize, commInterval,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr
        );
    } else {
        MatmulReduceScatterImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rank, rankSize, commInterval,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr
        );
    }
}

#endif // MATMUL_REDUCE_SCATTER_KERNEL_H