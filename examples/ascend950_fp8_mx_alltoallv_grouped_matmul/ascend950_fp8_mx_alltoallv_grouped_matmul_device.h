/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL_KERNEL_H
#define ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL_KERNEL_H

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
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/kernel/mx_alltoallv_group_matmul.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementMxScaleA, class LayoutMxScaleA,
    class ElementMxScaleB, class LayoutMxScaleB,
    class ElementC, class LayoutC,
    uint32_t M0, uint32_t N0, uint32_t K0
>
struct Ascend950Fp8MxAllToAllVGroupedMatmulConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    constexpr static uint32_t PRELOAD_STAGES = 1;
    constexpr static uint32_t L1_STAGES = 2;
    constexpr static uint32_t L0A_STAGES = 2;
    constexpr static uint32_t L0B_STAGES = 2;
    constexpr static uint32_t L0C_STAGES = 1;
    constexpr static bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, ENABLE_UNIT_FLAG>;

    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<128>>;

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
            ArchTag, ElementA, LayoutA, ElementB, LayoutB, 
            ElementMxScaleA, decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutA, false>(0U, 0U)),
            ElementMxScaleB, decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutB, true>(0U, 0U)),
            ElementC, LayoutC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy
    >;

    constexpr static bool IS_DYNAMIC = true;

    constexpr static uint32_t UB_STAGES_VAL = 2;
    constexpr static Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES_VAL, IS_DYNAMIC>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using ScaleAType = Catlass::Gemm::GemmType<ElementMxScaleA, LayoutMxScaleA>;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemoteScaleSrcType = ScaleAType;
    using RemoteScaleDstType = ScaleAType;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, COPY_DIRECT, CopyTransport::Mte>;
    using TileRemoteCopyScale = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteScaleSrcType, RemoteScaleDstType, void, COPY_DIRECT, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr static uint32_t TP_SIZE_LIMITS = 1;
    constexpr static uint32_t EP_SIZE_LIMITS = 8;
    constexpr static uint32_t LOCAL_EXPERT_NUM_LIMITS = 64;
    using MoeConstraints = DGemm::MoeConstraints<TP_SIZE_LIMITS, EP_SIZE_LIMITS, LOCAL_EXPERT_NUM_LIMITS>;

    using BlockMmadScheduler = Catccos::DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;

    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;

    using BlockCommAllGather = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;
    using BlockCommAllGatherScale = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteScaleSrcType, RemoteScaleDstType,
        void,
        TileRemoteCopyScale, TileScheduler
    >;

    using Kernel = DGemm::Kernel::MxAllToAllVGroupedMatmulTla<
        ProblemShape,
        BlockMmad,
        BlockCommAllGather,
        BlockCommAllGatherScale,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementMxScaleA, class LayoutMxScaleA, 
    class ElementMxScaleB, class LayoutMxScaleB, class ElementC, class LayoutC>
using Ascend950Fp8MxAllToAllVGroupedMatmulConfig_M0_128 = Ascend950Fp8MxAllToAllVGroupedMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA,
    ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutC, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementMxScaleA, class LayoutMxScaleA, 
    class ElementMxScaleB, class LayoutMxScaleB, class ElementC, class LayoutC>
using Ascend950Fp8MxAllToAllVGroupedMatmulConfig_M0_256 = Ascend950Fp8MxAllToAllVGroupedMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA,
    ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutC, 256, 128, 256>;

#endif // ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL_KERNEL_H