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
 
#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_reducescatter_alltoallv.hpp"
#include "catccos/comm/block/comm_block_scheduler_reducescatter_alltoallv.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
 
using namespace AscendC;
using namespace Catccos;
 
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct GroupedMatmulAllToAllVConfig {
    using ArchTag = Catlass::Arch::AtlasA2;

    static constexpr uint32_t preloadStages = 1;
    static constexpr uint32_t l1Stages = 2;
    static constexpr uint32_t l0AStages = 2;
    static constexpr uint32_t l0BStages = 2;
    static constexpr uint32_t l0CStages = 1;
    static constexpr bool enableUnitFlag = true;
    static constexpr bool enableShuffleK = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsync<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;

    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using SymmetricType = DType;
    static constexpr bool IS_DYNAMIC = true;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, SymmetricType
    >;
    static constexpr uint32_t TP_SIZE_LIMITS = 1;
    static constexpr uint32_t EP_SIZE_LIMITS = 8;
    static constexpr uint32_t LOCAL_EXPERT_NUM_LIMITS = 16;
    using MoeConstraints = DGemm::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, LOCAL_EXPERT_NUM_LIMITS>;
    
    using BlockMmadScheduler = DGemm::Block::BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;
 
    using RemoteSrcType = SymmetricType;
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
    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;
    static constexpr uint32_t WORKSPACE_STAGES = 2;
    
    using Kernel = DGemm::Kernel::GroupedMatmulAllToAllV<
        ProblemShape,
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using GroupedMatmulAllToAllVConfig_M0_128 = GroupedMatmulAllToAllVConfig<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using GroupedMatmulAllToAllVConfig_M0_256 = GroupedMatmulAllToAllVConfig<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 256, 128, 256>;

#endif // GROUPED_MATMUL_ALLTOALLV_KERNEL_H