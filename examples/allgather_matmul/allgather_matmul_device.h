/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_KERNEL_H
#define ALLGATHER_MATMUL_KERNEL_H

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
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_with_local_mm_opt.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "padding.h"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    bool EnablePadding_,
    uint32_t M0_ = 128, uint32_t N0_ = 256, uint32_t K0_ = 256
>
struct AllGatherMatmulConfig {
    using ArchTag = Catlass::Arch::AtlasA2;

    static constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0_, N0_, K0_>;
    using L0TileShape = Catlass::GemmShape<M0_, N0_, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    static constexpr bool IS_DYNAMIC = true;

    using PaddingHelperB = typename Catccos::Padding::PaddingHelper<BType, EnablePadding_>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, ActualTypeB, CType
    >;

    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    using Kernel = DGemm::Kernel::AllGatherMatmul<
        GlobalPaddingB,
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Without padding
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllGatherMatmulConfig_M0_128 = AllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, false, 128, 256, 256>;
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllGatherMatmulConfig_M0_256 = AllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, false, 256, 128, 256>;

// With padding
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllGatherMatmulPaddingConfig_M0_128 = AllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, true, 128, 256, 256>;
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using AllGatherMatmulPaddingConfig_M0_256 = AllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, true, 256, 128, 256>;

#endif // ALLGATHER_MATMUL_KERNEL_H
