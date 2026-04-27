/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTOALLV_GROUPED_MATMUL_KERNEL_v2_H
#define ALLTOALLV_GROUPED_MATMUL_KERNEL_v2_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/block/comm_block_local_copy.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"

#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_gmm.hpp"
#include "catccos/dgemm/kernel/alltoallv_gmm_v2.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct AllToAllVGMMV2Config {
    using ArchTag = Catlass::Arch::AtlasA2;

    static constexpr bool enableUnitFlag = true;
    static constexpr bool enableShuffleK = true;

    static constexpr uint32_t preloadStages = 1;
    static constexpr uint32_t l1Stages = 2;
    static constexpr uint32_t l0AStages = 2;
    static constexpr uint32_t l0BStages = 2;
    static constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsync<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK
    >;
    using L1TileShape = GemmShape<M0_, N0_, K0_>;
    using L0TileShape = GemmShape<M0_, N0_, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy,
        L1TileShape, L0TileShape, 
        AType, BType, CType
    >;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;

    static constexpr bool IS_DYNAMIC = true;

    // remote comm
    static constexpr uint32_t UB_STAGES_VAL = 2;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemoteCommDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using RemoteCommBlock = Catccos::Comm::Block::CommBlock<
        RemoteCommDispatch,
        RemoteSrcType,
        RemoteDstType,
        void,
        TileRemoteCopy,
        TileScheduler
    >;

    // local copy
    using CopySrcType = AType;
    using CopyDstType = AType;
    using LocalCopyBlockShape = Catlass::MatrixShape<48, UINT_MAX / 2>;
    using LocalCopyTileShape = Catlass::MatrixShape<48, 1024>;
    using TileLocalCopy = Catccos::Comm::Tile::TileRemoteCopy<ArchTag, false, RemoteSrcType, RemoteDstType, LocalCopyTileShape, CopyDirect::Get, CopyTransport::Mte>;
    using LocalCopyDispatch = Comm::AtlasA2CommLocalCopy<UB_STAGES_VAL>;
    using LocalCopyBlock = Catccos::Comm::Block::CommBlock<
        LocalCopyDispatch,
        CopySrcType,
        CopyDstType,
        LocalCopyBlockShape,
        TileLocalCopy,
        TileScheduler
    >;

    using BlockCommScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerAllToAllVGmm;

    // kernel level
    using ElementGroupList = int64_t;
    using Kernel = Catccos::DGemm::Kernel::AlltoallvGMMKernel<
        BlockMmad,
        BlockScheduler,
        ElementGroupList,
        LocalCopyBlock,
        RemoteCommBlock,
        BlockCommScheduler
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllToAllVGMMV2Config_M0_128 = AllToAllVGMMV2Config<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllToAllVGMMV2Config_M0_256 = AllToAllVGMMV2Config<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>;

#endif