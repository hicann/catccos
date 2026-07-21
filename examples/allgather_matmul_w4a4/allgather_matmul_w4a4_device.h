/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_W4A4_KERNEL_H
#define ALLGATHER_MATMUL_W4A4_KERNEL_H

#include "info.h"

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
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_w4a4.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};

template <
    class ElementA_, class LayoutA_,
    class ElementB_, class LayoutB_,
    class ElementC_, class LayoutC_,
    class ElementD_, class LayoutD_,
    class ElementScale_, class LayoutScale_,
    class ElementPerTokenScale_, class LayoutPerTokenScale_
>
struct AllGatherMatmulW4A4Config {
    using ArchTag = Catlass::Arch::AtlasA2;

    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using ElementD = ElementD_;
    using LayoutD = LayoutD_;
    using ElementScale = ElementScale_;
    using LayoutScale = LayoutScale_;
    using ElementPerTokenScale = ElementPerTokenScale_;
    using LayoutPerTokenScale = LayoutPerTokenScale_;

    constexpr static uint32_t PRELOAD_STAGES = 1;
    constexpr static uint32_t L1_STAGES = 2;
    constexpr static uint32_t L0A_STAGES = 2;
    constexpr static uint32_t L0B_STAGES = 2;
    constexpr static uint32_t L0C_STAGES = 1;
    constexpr static bool ENABLE_UNIT_FLAG = false;
    constexpr static bool ENABLE_SHUFFLE_K = true;

    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2W4A4MatmulPerTokenPerChannelDequant<
        PRELOAD_STAGES, L1_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

    using L1TileShape = Catlass::GemmShape<128, 256, 1024>;
    using L0TileShape = Catlass::GemmShape<128, 256, 256>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;

    using TileCopyMmad =
        Catlass::Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, void, ScaleGranularity::PER_CHANNEL>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>;

    constexpr static bool IS_DYNAMIC = true;

    using BlockSchedulerForAllgather = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockSchedulerForDequantInAiv = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using CommBlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void,
        CopyDirect::Get, CopyTransport::Mte>;
    using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileSchedulerForAllgather
    >;

    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2W4A4PerTokenPerChannelDequant;
    using PerTokenScaleType = Catlass::Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using BroadcastOneBlkType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;

    using EpilogueTileShape = Catlass::MatrixShape<48, 256>;
    using TileBroadcastOneBlk =
        Catlass::Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Catlass::Epilogue::Tile::TileCopyW4A4Gemm<ArchTag, CType, PerTokenScaleType, DType>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, PerTokenScaleType, DType,
        TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
        TileCopy, TileScheduler>;

    using Kernel = DGemm::Kernel::AllGatherW4A4Matmul<
        BlockMmad,
        BlockEpilogue,
        BlockComm,
        BlockSchedulerForAllgather,
        BlockSchedulerForDequantInAiv,
        CommBlockScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <
    class EA, class LA, class EB, class LB,
    class EC, class LC, class ED, class LD,
    class ES, class LS, class EPS, class LPS
>
using AllGatherMatmulW4A4Config_M0_128 = AllGatherMatmulW4A4Config<EA, LA, EB, LB, EC, LC, ED, LD, ES, LS, EPS, LPS>;

#endif // ALLGATHER_MATMUL_W4A4_KERNEL_H
