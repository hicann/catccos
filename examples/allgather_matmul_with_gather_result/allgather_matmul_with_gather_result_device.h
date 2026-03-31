/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_WITH_GATHER_RESULT_H
#define ALLGATHER_MATMUL_WITH_GATHER_RESULT_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

// utils
#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_with_gather_result.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_DEVICE
void AllGatherMatmulWithGatherResultImpl(
    Catlass::GemmCoord const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmC, LayoutC const &layoutC,
    GM_ADDR gmGatherA, LayoutA const &layoutGatherA,
    uint32_t commInterval,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    GM_ADDR symmetricPtr
)
{
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using GatherAType = AType;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, CType
    >;

    constexpr bool IS_DYNAMIC = true;

    using BlockScheduler = typename Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockCommScheduler = Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;
    using BlockCopyGatherAScheduler = Comm::Block::BlockSchedulerCopyGatherA;

    using RemoteSrcType = AType;
    using RemoteDstType = GatherAType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using GatherATileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t UB_STAGES = 2;
    using AllGatherDispatch = Comm::AtlasA2CommRemoteCopy<UB_STAGES, IS_DYNAMIC>;
    using BlockAllGather = Comm::Block::CommBlock<
        AllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    using CopyGatherADispatchPolicy = Comm::AtlasA2CommLocalCopy<UB_STAGES, IS_DYNAMIC>;
    using BlockCopyGatherA = Comm::Block::CommBlock<
        CopyGatherADispatchPolicy,
        AType, GatherAType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    using AllGatherMatmulWithGatherResultKernel = DGemm::Kernel::AllGatherMatmulWithGatherResult<
        BlockMmad,
        BlockAllGather,
        BlockCopyGatherA,
        BlockScheduler,
        BlockCommScheduler,
        BlockCopyGatherAScheduler,
        WORKSPACE_STAGES
    >;

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    typename TileRemoteCopy::Params tileParams {
        commTileShape
    };

    typename BlockAllGather::Params blockParams{
        commBlockShape,
        tileParams
    };

    typename BlockCommScheduler::Params swizzleParams {
        commCoreSplit
    };

    uint32_t commCoreNum = commCoreSplit.row() * commCoreSplit.column();
    uint32_t copyCoreNum = AscendC::GetBlockNum() - commCoreNum;

    uint32_t copyBlockM = CeilDiv((commInterval * M0), CeilDiv(copyCoreNum, rankSize));
    uint32_t copyBlockN = RoundUp<32 / sizeof(ElementA)>(problemShape.k());
    Catlass::MatrixCoord copyBlockShape{copyBlockM, copyBlockN};

    uint32_t maxCopyLength = ArchTag::UB_SIZE / UB_STAGES / sizeof(ElementA);
    uint32_t copyTileN = Min<uint32_t>(maxCopyLength, copyBlockN);
    uint32_t copyTileM = maxCopyLength / copyTileN;
    Catlass::MatrixCoord copyTileShape{copyTileM, copyTileN};

    typename GatherATileRemoteCopy::Params copyGatherATileParams{
        copyTileShape
    };

    typename BlockCopyGatherA::Params copyGatherAParams{
        copyBlockShape,
        copyGatherATileParams
    };

    // Prepare params
    typename AllGatherMatmulWithGatherResultKernel::Params params{
        problemShape,
        rank, rankSize,
        gmA, layoutA,
        gmB, layoutB,
        symmetricPtr,
        blockParams,
        swizzleParams,
        copyGatherAParams,
        gmGatherA, layoutGatherA,
        gmC, layoutC,
        commInterval
    };

    AllGatherMatmulWithGatherResultKernel matmulCommKernel;
    matmulCommKernel(params);
}

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_GLOBAL
void AllGatherMatmulWithGatherResult(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmGatherA, GM_ADDR symmetricPtr,
    CocTilingParams cocTiling, GM_ADDR dump
)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_GLOBAL
void AllGatherMatmulWithGatherResult(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmGatherA, GM_ADDR symmetricPtr,
    CocTilingParams cocTiling
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
    uint32_t rankSize = cocTiling.rankSize;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, RoundUp(k, k0)};
    Catlass::MatrixCoord commTileShape{commTileM / 2, k0};

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n};
    LayoutA layoutGatherA{m * rankSize, k};

    AllGatherMatmulWithGatherResultImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmGatherA, layoutGatherA,
        commInterval, commCoreSplit, commBlockShape, commTileShape, symmetricPtr
    );
}

#endif
