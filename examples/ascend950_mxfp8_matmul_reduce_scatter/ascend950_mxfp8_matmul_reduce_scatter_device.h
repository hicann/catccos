/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_KERNEL_H
#define ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_KERNEL_H

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
#include "catccos/comm/block/comm_block_scheduler_reduce_scatter.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/matmul_reduce_scatter_mx_tla.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;
using namespace Catlass;
using namespace tla;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementAScale, class LayoutAScale,
    class ElementBScale, class LayoutBScale,
    class ElementC, class LayoutC,
    class ElementD, class LayoutD,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct Ascend950MxFp8MatmulReduceScatterConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool enableUnitFlag = true;
    static constexpr uint32_t l0CStages = 1;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;

    using L1TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<K0_>>;
    using L0TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<128>>;

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB,
        ElementAScale, LayoutAScale, ElementBScale, LayoutBScale,
        ElementC, LayoutC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape,
        ElementA, ElementB, ElementC, void, TileCopy
    >;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerReduceScatter<IS_DYNAMIC, void, 0, true, 7, 1>;

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

    using Kernel = DGemm::Kernel::MatmulReduceScatterMxTla<
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementAScale, class LayoutAScale, class ElementBScale, class LayoutBScale,
          class ElementC, class LayoutC, class ElementD, class LayoutD>
using Ascend950MxFp8MatmulReduceScatterConfig_M0_128 =
    Ascend950MxFp8MatmulReduceScatterConfig<ElementA, LayoutA, ElementB, LayoutB,
        ElementAScale, LayoutAScale, ElementBScale, LayoutBScale,
        ElementC, LayoutC, ElementD, LayoutD, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementAScale, class LayoutAScale, class ElementBScale, class LayoutBScale,
          class ElementC, class LayoutC, class ElementD, class LayoutD>
using Ascend950MxFp8MatmulReduceScatterConfig_M0_256 =
    Ascend950MxFp8MatmulReduceScatterConfig<ElementA, LayoutA, ElementB, LayoutB,
        ElementAScale, LayoutAScale, ElementBScale, LayoutBScale,
        ElementC, LayoutC, ElementD, LayoutD, 256, 256, 256>;

#endif // ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_KERNEL_H
