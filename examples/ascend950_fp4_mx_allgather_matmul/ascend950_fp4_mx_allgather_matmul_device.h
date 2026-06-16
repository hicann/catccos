/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_FP4_MX_ALLGATHER_MATMUL_KERNEL_H
#define ASCEND950_FP4_MX_ALLGATHER_MATMUL_KERNEL_H

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
#include "catccos/dgemm/kernel/mx_allgather_matmul.hpp"
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
struct Ascend950Fp4MxAllGatherMatmulConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    constexpr static bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, ENABLE_UNIT_FLAG>;

    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<256>>;

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
            ArchTag, ElementA, LayoutA, ElementB, LayoutB, 
            ElementMxScaleA, decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutA, false>(0U, 0U)),
            ElementMxScaleB, decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutB, true>(0U, 0U)),
            ElementC, LayoutC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy
    >;

    constexpr static bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using ScaleAType = Catlass::Gemm::GemmType<ElementMxScaleA, LayoutMxScaleA>;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemoteScaleSrcType = ScaleAType;
    using RemoteScaleDstType = ScaleAType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileRemoteCopyScale = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteScaleSrcType, RemoteScaleDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
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

    using Kernel = DGemm::Kernel::MxAllGatherMatmulTla<
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
    class ElementMxScaleB, class LayoutMxScaleB,class ElementC, class LayoutC>
using Ascend950Fp4MxAllGatherMatmulConfig_M0_128 = Ascend950Fp4MxAllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA,
    ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutC, 128, 256, 512>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementMxScaleA, class LayoutMxScaleA, 
    class ElementMxScaleB, class LayoutMxScaleB,class ElementC, class LayoutC>
using Ascend950Fp4MxAllGatherMatmulConfig_M0_256 = Ascend950Fp4MxAllGatherMatmulConfig<ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA,
    ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutC, 256, 128, 512>;

#endif // ASCEND950_FP4_MX_ALLGATHER_MATMUL_KERNEL_H