/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTOALLV_GROUPED_MATMUL_KERNEL_H
#define ALLTOALLV_GROUPED_MATMUL_KERNEL_H
 
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
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/kernel/alltoallv_grouped_matmul.hpp"
 
using namespace AscendC;
using namespace Catccos;
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ProblemShape,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void AllToAllVGroupedMatmulImpl(
    ProblemShape const &problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmC, LayoutC const &layoutC,
    GM_ADDR gmLocalTokensPerExpert,
    GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR gmSymmetric, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    // Declare BlockMmad
    constexpr uint32_t PRELOAD_STAGES = 1;
    constexpr uint32_t L1_STAGES = 2;
    constexpr uint32_t L0A_STAGES = 2;
    constexpr uint32_t L0B_STAGES = 2;
    constexpr uint32_t L0C_STAGES = 1;
    constexpr bool ENABLE_UNIT_FLAG = true;
    constexpr bool ENABLE_SHUFFLE_K = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
 
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, CType
    >;

    constexpr bool IS_DYNAMIC = true;
 
    // Declare BlockEpilogue
    constexpr uint32_t UB_STAGES = 2;
    constexpr Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using AllToAllVDispatch = Comm::AtlasA2CommRemoteCopy<UB_STAGES, IS_DYNAMIC>;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, COPY_DIRECT, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
 
    using BlockAllToAllV = Comm::Block::CommBlock<
        AllToAllVDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy,
        TileScheduler
    >;
 
    // Declare Scheduler
    constexpr uint32_t TP_SIZE_LIMITS = 1;
    constexpr uint32_t EP_SIZE_LIMITS = 8;
    constexpr uint32_t LOCAL_EXPERT_NUM_LIMITS = 64;
    using MoeConstraints = DGemm::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, LOCAL_EXPERT_NUM_LIMITS>;
 
    using BlockMmadScheduler = DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
    using BlockScheduler = Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;
 
    // Declare Kernel
    constexpr uint32_t WORKSPACE_STAGES = 2;
    using AllToAllVGroupedMatmulKernel = DGemm::Kernel::AllToAllVGroupedMatmul<
        ProblemShape,
        BlockMmad,
        BlockAllToAllV,
        BlockMmadScheduler,
        BlockScheduler,
        WORKSPACE_STAGES
    >;

    typename TileRemoteCopy::Params tileParams {
        commTileShape
    };
 
    typename AllToAllVGroupedMatmulKernel::Params params{
        problemShape,
        commInterval,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC,
        gmSymmetric,
        syncMmadFinish, syncCommFinish,
        {commBlockShape, tileParams}
    };
 
    // Call kernel
    AllToAllVGroupedMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ProblemShape
>
CATLASS_DEVICE
void AllToAllVGroupedMatmulImpl_M0_256(
    ProblemShape const &problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmC, LayoutC const &layoutC,
    GM_ADDR gmLocalTokensPerExpert,
    GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR gmSymmetric, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    AllToAllVGroupedMatmulImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ProblemShape, 256, 128, 256>(
        problemShape, l1TileShape,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC,
        gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
        gmSymmetric, syncMmadFinish, syncCommFinish,
        commCoreSplit,
        commBlockShape,
        commTileShape,
        commInterval
    );
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ProblemShape
>
CATLASS_DEVICE
void AllToAllVGroupedMatmulImpl_M0_128(
    ProblemShape const &problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmC, LayoutC const &layoutC,
    GM_ADDR gmLocalTokensPerExpert,
    GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR gmSymmetric, GM_ADDR syncMmadFinish, GM_ADDR syncCommFinish,
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    uint32_t commInterval
)
{
    AllToAllVGroupedMatmulImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ProblemShape, 128, 256, 256>(
        problemShape, l1TileShape,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC,
        gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
        gmSymmetric, syncMmadFinish, syncCommFinish,
        commCoreSplit,
        commBlockShape,
        commTileShape,
        commInterval
    );
}

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_GLOBAL
void AllToAllVGroupedMatmul(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetric, CocTilingParams cocTiling, GM_ADDR dump
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
void AllToAllVGroupedMatmul(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetric, CocTilingParams cocTiling
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
    uint32_t epSize = cocTiling.epSize;
    uint32_t expertNum = cocTiling.expertNum;
 
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;
 
    ProblemShape problemShape{
        {m, n, k}, rankSize, rankIdx, epSize, expertNum,
        gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert
    };
    Catlass::GemmCoord l1TileShape{m0, n0, k0};
 
    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, RoundUp(k, k0)};
    Catlass::MatrixCoord commTileShape{commTileM / 2, k0};
 
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
 
    uint64_t symmetricOffset = 0;
    auto gmSymmetric = symmetric + symmetricOffset;
    symmetricOffset += IPC_BUFF_MAX_SIZE;
    auto syncMmadFinish = symmetric + symmetricOffset;
    symmetricOffset += SYNC_UNIT_SIZE;
    auto syncCommFinish = symmetric + symmetricOffset;
 
    if (m0 == 128) {
        AllToAllVGroupedMatmulImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ProblemShape>(
            problemShape, l1TileShape,
            gmA, layoutA,
            gmB, layoutB,
            gmC, layoutC,
            gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            commCoreSplit, commBlockShape, commTileShape,
            commInterval
        );
    } else {
        AllToAllVGroupedMatmulImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ProblemShape>(
            problemShape, l1TileShape,
            gmA, layoutA,
            gmB, layoutB,
            gmC, layoutC,
            gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            commCoreSplit, commBlockShape, commTileShape,
            commInterval
        );
    }
}
 
#endif