/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUPED_MATMUL_ALLTOALLV_KERNEL_H
#define GROUPED_MATMUL_ALLTOALLV_KERNEL_H
 
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
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv.hpp"
 
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
void GroupedMatmulAllToAllVImpl(
    Catlass::GemmCoord const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmD, LayoutD const &layoutD,
    uint32_t rankIdx, uint32_t rankSize, uint32_t commInterval,
    uint32_t epSize, uint32_t expertNum,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert, 
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    GM_ADDR symmetricPtr
)
{
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
 
    // TODO: 当前catlass中MmadAtlasA2Pingpong没有动态版本, 预留了l1TileShape
    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;
 
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using SymmetricType = DType;

    constexpr bool IS_DYNAMIC = true;
 
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, SymmetricType
    >;
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
 
    using RemoteSrcType = SymmetricType;
    using RemoteDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
 
    using AllToAllVDispatch = Comm::AtlasA2CommRemoteCopy<UB_STAGES, IS_DYNAMIC>;
    using BlockAllToAllV = Comm::Block::CommBlock<
        AllToAllVDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, 
        TileScheduler
    >;
 
    using CommBlockScheduler = Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;
    
    constexpr uint32_t WORKSPACE_STAGES = 2;
 
    constexpr uint32_t TP_SIZE_LIMITS = 1;
    constexpr uint32_t EP_SIZE_LIMITS = 8;
    constexpr uint32_t EXPERT_NUM_LIMITS = 64;
    using MoeConstraints = DGemm::Kernel::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, EXPERT_NUM_LIMITS>;
 
    using GroupedMatmulAllToAllVKernel = DGemm::Kernel::GroupedMatmulAllToAllV<
        BlockMmad,
        BlockAllToAllV,
        BlockScheduler,
        CommBlockScheduler,
        WORKSPACE_STAGES,
        MoeConstraints
    >;

    typename TileRemoteCopy::Params tileParams {
        commTileShape
    };
 
    typename BlockAllToAllV::Params blockParams{
        commBlockShape,
        tileParams
    };

    typename CommBlockScheduler::Params swizzleParams {
        commCoreSplit
    };
 
    typename GroupedMatmulAllToAllVKernel::Params params{
        problemShape,
        commInterval,
        rankIdx, 
        rankSize,
        epSize,
        expertNum / epSize,
        gmA, layoutA,
        gmB, layoutB,
        gmLocalTokensPerExpert,
        gmGlobalTokensPerLocalExpert,
        symmetricPtr,
        blockParams,
        swizzleParams,
        gmD, layoutD,
    };
 
    // Call kernel
    GroupedMatmulAllToAllVKernel matmulKernel;
    matmulKernel(params);
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_DEVICE
void GroupedMatmulAllToAllVImpl_M0_256(
    Catlass::GemmCoord const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmD, LayoutD const &layoutD,
    uint32_t rankIdx, uint32_t rankSize, uint32_t commInterval,
    uint32_t epSize, uint32_t expertNum,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert, 
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    GM_ADDR symmetricPtr
)
{
    GroupedMatmulAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 256, 128, 256> 
        (problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, rankIdx, rankSize, commInterval, 
        epSize, expertNum, gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert, 
        commCoreSplit, commBlockShape, commTileShape, symmetricPtr);
}
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_DEVICE
void GroupedMatmulAllToAllVImpl_M0_128(
    Catlass::GemmCoord const &problemShape,
    Catlass::GemmCoord const &l1TileShape,
    GM_ADDR gmA, LayoutA const &layoutA,
    GM_ADDR gmB, LayoutB const &layoutB,
    GM_ADDR gmD, LayoutD const &layoutD,
    uint32_t rankIdx, uint32_t rankSize, uint32_t commInterval,
    uint32_t epSize, uint32_t expertNum,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert, 
    Catlass::MatrixCoord const &commCoreSplit,
    Catlass::MatrixCoord const &commBlockShape,
    Catlass::MatrixCoord const &commTileShape,
    GM_ADDR symmetricPtr
)
{
    GroupedMatmulAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 128, 256, 256> 
        (problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, rankIdx, rankSize, commInterval, 
        epSize, expertNum, gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert, 
        commCoreSplit, commBlockShape, commTileShape, symmetricPtr);
}

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD
>
CATLASS_GLOBAL
void GroupedMatmulAllToAllV(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetricPtr, CocTilingParams cocTiling, GM_ADDR dump
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
void GroupedMatmulAllToAllV(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmLocalTokensPerExpert, GM_ADDR gmGlobalTokensPerLocalExpert,
    GM_ADDR symmetricPtr, CocTilingParams cocTiling
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
 
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};
 
    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, n0};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};
 
    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
 
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m, n};
 
    if (m0 == 128) {
        GroupedMatmulAllToAllVImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rankIdx, rankSize, commInterval,
            epSize, expertNum, gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr
        );
    } else {
        GroupedMatmulAllToAllVImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rankIdx, rankSize, commInterval,
            epSize, expertNum, gmLocalTokensPerExpert, gmGlobalTokensPerLocalExpert,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr
        );
    } 
}
 
#endif // GROUPED_MATMUL_ALLTOALLV_KERNEL_H