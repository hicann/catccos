/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_KERNEL_ASCEND950_H
#define ALLGATHER_MATMUL_KERNEL_ASCEND950_H

#include "info.h"

// from catlass
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.hpp"
#endif

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_ascend950.hpp"

using namespace AscendC;
using namespace Catccos;

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
          uint32_t M0, uint32_t N0, uint32_t K0>
CATLASS_DEVICE void AllGatherMatmulImpl(Catlass::GemmCoord& problemShape, Catlass::GemmCoord& l1TileShape, GM_ADDR gmA,
                                        LayoutA& tagA, GM_ADDR gmB, LayoutB& tagB, GM_ADDR gmC, LayoutC& tagC,
                                        uint32_t commInterval, Catlass::MatrixCoord& commCoreSplit,
                                        Catlass::MatrixCoord& commBlockShape, Catlass::MatrixCoord& commTileShape,
                                        GM_ADDR gmSymmetric)
{
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG>;

    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void,
                                                      CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using AllGatherDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockAllGather =
        Comm::Block::CommBlock<AllGatherDispatch, RemoteSrcType, RemoteDstType, void, TileRemoteCopy, TileScheduler>;

    using AllGatherMatmulKernel =
        DGemm::Kernel::AllGatherMatmul<BlockMmad, BlockAllGather, BlockMmadScheduler, BlockScheduler, WORKSPACE_STAGES>;

    typename TileRemoteCopy::Params tileParams{commTileShape};

    typename BlockAllGather::Params blockParams{commBlockShape, tileParams};

    typename BlockScheduler::Params swizzleParams{commCoreSplit};

    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    typename AllGatherMatmulKernel::Params params{problemShape, rank,        rankSize,    commInterval,
                                                  tagA,  // layoutGatherSrc
                                                  gmA,          layoutA,     gmB,         layoutB,      gmC,
                                                  layoutC,      gmSymmetric, blockParams, swizzleParams};

    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void AllGatherMatmulImpl_M0_256(Catlass::GemmCoord& problemShape, Catlass::GemmCoord& l1TileShape,
                                               GM_ADDR gmA, LayoutA& layoutA, GM_ADDR gmB, LayoutB& layoutB,
                                               GM_ADDR gmC, LayoutC& layoutC, uint32_t commInterval,
                                               Catlass::MatrixCoord& commCoreSplit,
                                               Catlass::MatrixCoord& commBlockShape,
                                               Catlass::MatrixCoord& commTileShape, GM_ADDR gmSymmetric)
{
    AllGatherMatmulImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, commInterval, commCoreSplit,
        commBlockShape, commTileShape, gmSymmetric);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void AllGatherMatmulImpl_M0_128(Catlass::GemmCoord& problemShape, Catlass::GemmCoord& l1TileShape,
                                               GM_ADDR gmA, LayoutA& layoutA, GM_ADDR gmB, LayoutB& layoutB,
                                               GM_ADDR gmC, LayoutC& layoutC, uint32_t commInterval,
                                               Catlass::MatrixCoord& commCoreSplit,
                                               Catlass::MatrixCoord& commBlockShape,
                                               Catlass::MatrixCoord& commTileShape, GM_ADDR gmSymmetric)
{
    AllGatherMatmulImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, commInterval, commCoreSplit,
        commBlockShape, commTileShape, gmSymmetric);
}

#if defined(ENABLE_ASCENDC_DUMP)
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void AllGatherMatmul(uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR symmetricPtr,
                                    CocTilingParams cocTiling, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void AllGatherMatmul(uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR symmetricPtr,
                                    CocTilingParams cocTiling)
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
    uint32_t rankSize = cocTiling.rankSize;

    shmemx_set_ffts_config(fftsAddr);

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};

    uint32_t strideA;
    if constexpr (std::is_same_v<LayoutA, Catlass::layout::RowMajor>)
    {
        strideA = k;
    }
    else if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>)
    {
        strideA = m;
    }

    uint32_t strideB;
    if constexpr (std::is_same_v<LayoutB, Catlass::layout::RowMajor>)
    {
        strideB = n;
    }
    else if constexpr (std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>)
    {
        strideB = k;
    }

    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m * rankSize, n, n};

    // Allocate sync signal memory from symmetricPtr
    uint64_t symmetricOffset = 0;
    auto gmSymmetric = symmetricPtr + symmetricOffset;
    symmetricOffset += IPC_BUFF_MAX_SIZE;

    if (m0 == 128)
    {
        AllGatherMatmulImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, commInterval, commCoreSplit,
            commBlockShape, commTileShape, gmSymmetric);
    }
    else
    {
        AllGatherMatmulImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, commInterval, commCoreSplit,
            commBlockShape, commTileShape, gmSymmetric);
    }
}

#endif  // ALLGATHER_MATMUL_KERNEL_ASCEND950_H
