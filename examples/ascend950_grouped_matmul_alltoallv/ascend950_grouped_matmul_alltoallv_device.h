/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_GROUPED_MATMUL_ALLTOALLV_KERNEL_H
#define ASCEND950_GROUPED_MATMUL_ALLTOALLV_KERNEL_H

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
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_tla.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_reducescatter_alltoallv.hpp"
#include "catccos/comm/block/comm_block_scheduler_reducescatter_alltoallv.hpp"
 
using namespace AscendC;
using namespace Catccos;
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ProblemShape,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void Ascend950GroupedMatmulAllToAllVImpl(
    ProblemShape const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &tagA,
    GM_ADDR gmB, LayoutB const &tagB,
    GM_ADDR gmD, LayoutD const &tagD,
    GM_ADDR symmetricPtr, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t l1AStages = 2;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
        ArchTag,
        enableUnitFlag, useHF32, l0CStages, enableL1Resident,
        l1AStages, l1BStages, l0AStages, l0BStages
    >;

    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;
    
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutD = tla::MakeLayoutFromTag(tagD);
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy,
        L1TileShape, L0TileShape,
        ElementA, ElementB, ElementD, void, TileCopy
    >;

    constexpr uint32_t TP_SIZE_LIMITS = 1;
    constexpr uint32_t EP_SIZE_LIMITS = 8;
    constexpr uint32_t LOCAL_EXPERT_NUM_LIMITS = 64;
    using MoeConstraints = DGemm::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, LOCAL_EXPERT_NUM_LIMITS>;
    
    using BlockMmadScheduler = DGemm::Block::BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;

    constexpr bool IS_DYNAMIC = true;
    
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using RemoteSrcType = DType;
    using RemoteDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
 
    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;
    
    using BlockCommScheduler = CommEpilogue::Block::BlockCommSchedulerReduceScatterAllToAllV<MoeConstraints, IS_DYNAMIC, void>;
    
    constexpr uint32_t WORKSPACE_STAGES = 2;
    using GroupedMatmulAllToAllVKernel = DGemm::Kernel::GroupedMatmulAllToAllVTla<
        ProblemShape,
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

    typename GroupedMatmulAllToAllVKernel::Params params{
        problemShape,
        commInterval,
        tagD,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        symmetricPtr,
        syncMmadFinish, syncCommFinish,
        blockParams,
        swizzleParams,
    };
 
    // // Call kernel
    GroupedMatmulAllToAllVKernel matmulCommKernel;
    matmulCommKernel(params);
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ProblemShape
>
CATLASS_DEVICE
void Ascend950GroupedMatmulAllToAllVImpl_M0_256(
    ProblemShape const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmD, LayoutD const &layoutD,
    GM_ADDR symmetricPtr, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    Ascend950GroupedMatmulAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ProblemShape, 256, 128, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, symmetricPtr, syncMmadFinish, syncCommFinish,  
        commCoreSplit, commBlockShape, commTileShape, commInterval
    );
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ProblemShape
>
CATLASS_DEVICE
void Ascend950GroupedMatmulAllToAllVImpl_M0_128(
    ProblemShape const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmD, LayoutD const &layoutD,
    GM_ADDR symmetricPtr, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    Ascend950GroupedMatmulAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ProblemShape, 128, 256, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, symmetricPtr, syncMmadFinish, syncCommFinish,  
        commCoreSplit, commBlockShape, commTileShape, commInterval
    );
}

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_GLOBAL
void Ascend950GroupedMatmulAllToAllV(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetric, CocTilingParams cocTiling, GM_ADDR dump
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
void Ascend950GroupedMatmulAllToAllV(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetricPtr, CocTilingParams cocTiling
)
{
#endif
    using ArchTag = Catlass::Arch::Ascend950;
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
    uint32_t epSize = cocTiling.epSize;
    uint32_t expertNum = cocTiling.expertNum;

    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
 
    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;

    ProblemShape problemShape{
        {m, n, k}, rankSize, rankIdx, epSize, expertNum,
        gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert
    };;
    Catlass::GemmCoord l1TileShape{m0, n0, k0};
 
    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, n0};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};
 
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m, n};

    uint64_t symmetricOffset = 0;
    auto gmSymmetric = symmetricPtr + symmetricOffset;
    symmetricOffset += IPC_BUFF_MAX_SIZE;
    auto syncMmadFinish = symmetricPtr + symmetricOffset;
    symmetricOffset += SYNC_UNIT_SIZE;
    auto syncCommFinish = symmetricPtr + symmetricOffset;

    if (m0 == 128) {
        Ascend950GroupedMatmulAllToAllVImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ProblemShape>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            commCoreSplit, commBlockShape, commTileShape, commInterval
        );
    } else {
        Ascend950GroupedMatmulAllToAllVImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ProblemShape>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            commCoreSplit, commBlockShape, commTileShape, commInterval
        );
    } 
}
 
#endif // ASCEND950_GROUPED_MATMUL_ALLTOALLV_KERNEL_H