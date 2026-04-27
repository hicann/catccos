/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
 
#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/kernel/alltoallv_grouped_matmul.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
 
using namespace AscendC;
using namespace Catccos;
 
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct AllToAllVGroupedMatmulConfig {
    using ArchTag = Catlass::Arch::AtlasA2;

    constexpr static uint32_t PRELOAD_STAGES = 1;
    constexpr static uint32_t L1_STAGES = 2;
    constexpr static uint32_t L0A_STAGES = 2;
    constexpr static uint32_t L0B_STAGES = 2;
    constexpr static uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0_, N0_, K0_>;
    using L0TileShape = Catlass::GemmShape<M0_, N0_, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    static constexpr bool IS_DYNAMIC = true;

    constexpr static uint32_t UB_STAGES_VAL = 2;
    constexpr static Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES_VAL, IS_DYNAMIC>;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, COPY_DIRECT, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy,
        TileScheduler
    >;

    constexpr static uint32_t TP_SIZE_LIMITS = 1;
    constexpr static uint32_t EP_SIZE_LIMITS = 8;
    constexpr static uint32_t LOCAL_EXPERT_NUM_LIMITS = 64;
    using MoeConstraints = DGemm::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, LOCAL_EXPERT_NUM_LIMITS>;

    using BlockMmadScheduler = DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
    using BlockScheduler = Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;

    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;

    using Kernel = DGemm::Kernel::AllToAllVGroupedMatmul<
        ProblemShape,
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllToAllVGroupedMatmulConfig_M0_128 = AllToAllVGroupedMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllToAllVGroupedMatmulConfig_M0_256 = AllToAllVGroupedMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>;

#endif