/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMUL_DEQUANT_REDUCE_SCATTER_V2_KERNEL_H
#define MATMUL_DEQUANT_REDUCE_SCATTER_V2_KERNEL_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
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
#include "catccos/dgemm/dispatch_policy.hpp"
#include "catccos/dgemm/block/block_mmad_pingpong_optional_bias.hpp"
#include "catccos/epilogue/dispatch_policy.hpp"
#include "catccos/epilogue/block/block_epilogue_per_token_dequant.hpp"
#include "catccos/dgemm/kernel/matmul_dequant_reduce_scatter_v2.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementD, class LayoutD,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct MatmulDequantReduceScatterV2Config {
    using ArchTag = Catlass::Arch::AtlasA2;

    static constexpr bool ENABLE_UNIT_FLAG = false;
    using MmadDispatchPolicy = Catccos::Gemm::MmadAtlasA2PingpongOptionalBias<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0_, N0_, K0_>;
    using L0TileShape = Catlass::GemmShape<M0_, N0_, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using BiasType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::VectorLayout>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType>;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerReduceScatter<IS_DYNAMIC, void, 0, true, 7, 1>;

    using ReduceScatterSrcType = DType;
    using ReduceScatterDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, ReduceScatterSrcType, ReduceScatterDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        ReduceScatterSrcType, ReduceScatterDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    using LocalCopyTileShape = Catlass::MatrixShape<48, 1024>;
    using TileLocalCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, ReduceScatterSrcType, ReduceScatterDstType,
        LocalCopyTileShape, CopyDirect::Put, CopyTransport::Mte>;
    using LocalCopyDispatchPolicy = Catccos::Comm::AtlasA2CommLocalCopy<UB_STAGES, IS_DYNAMIC>;
    using LocalCopyBlock = Comm::Block::CommBlock<
        LocalCopyDispatchPolicy,
        ReduceScatterSrcType, ReduceScatterDstType,
        void,
        TileLocalCopy, TileScheduler
    >;

    using DequantScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using DequantPerTokenScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using DequantDispatchPolicy = Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantWithDst<2>;
    using EpilogueTileShape = Catlass::MatrixShape<4, 256>;
    using ComputeType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;

    using TileRowBroadcastMul = Catlass::Epilogue::Tile::TileRowBroadcastMul<
        ArchTag, ComputeType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Catlass::Epilogue::Tile::TileBroadcastOneBlk<ArchTag, ComputeType, 4>;
    using TileOneBlkColumnBroadcastMul = Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<
        ArchTag, ComputeType, EpilogueTileShape>;
    using TileCopy = Catlass::Epilogue::Tile::TileCopy<
        ArchTag, CType, DequantScaleType, DequantPerTokenScaleType, DType>;
    using EpilogueTileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using BlockEpilogueDequant = Catlass::Epilogue::Block::BlockEpilogue<
        DequantDispatchPolicy,
        CType,
        DequantScaleType,
        DequantPerTokenScaleType,
        DType,
        TileRowBroadcastMul,
        TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul,
        TileCopy,
        EpilogueTileSwizzle>;

    using Kernel = DGemm::Kernel::MatmulDequantReduceScatterV2<
        BlockMmad,
        BlockComm,
        LocalCopyBlock,
        BlockEpilogueDequant,
        BlockMmadScheduler,
        BlockCommScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementC, class LayoutC, class ElementD, class LayoutD>
using MatmulDequantReduceScatterV2Config_M0_128 =
    MatmulDequantReduceScatterV2Config<ElementA, LayoutA, ElementB, LayoutB,
        ElementC, LayoutC, ElementD, LayoutD, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementC, class LayoutC, class ElementD, class LayoutD>
using MatmulDequantReduceScatterV2Config_M0_256 =
    MatmulDequantReduceScatterV2Config<ElementA, LayoutA, ElementB, LayoutB,
        ElementC, LayoutC, ElementD, LayoutD, 256, 128, 256>;

#endif // MATMUL_DEQUANT_REDUCE_SCATTER_V2_KERNEL_H
