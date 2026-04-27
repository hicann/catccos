/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_DEQUANT_KERNEL_H
#define ALLGATHER_MATMUL_DEQUANT_KERNEL_H

#include "info.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
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
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_dequant.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA_, class LayoutA_,
    class ElementB_, class LayoutB_,
    class ElementD_, class LayoutD_,
    class ElementScale_, class LayoutScale_,
    bool EnablePadding_,
    uint32_t M0_ = 128, uint32_t N0_ = 256, uint32_t K0_ = 256
>
struct AllGatherMatmulDequantConfig {
    using ArchTag = Catlass::Arch::AtlasA2;

    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementD = ElementD_;
    using LayoutD = LayoutD_;
    using ElementScale = ElementScale_;
    using LayoutScale = LayoutScale_;

    static constexpr uint32_t M0 = M0_;
    static constexpr uint32_t N0 = N0_;
    static constexpr uint32_t K0 = K0_;

    constexpr static bool ENABLE_UNIT_FLAG = false;
    constexpr static bool ENABLE_SHUFFLE_K = true;
    constexpr static bool IS_INT8 = std::is_same<ElementA, int8_t>::value;
    constexpr static int32_t L1TileShapeK = IS_INT8 ? TILE_SHAPE_512 : TILE_SHAPE_256;
    constexpr static int32_t L0TileShapeK = IS_INT8 ? TILE_SHAPE_128 : TILE_SHAPE_64;

    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;
    using L1TileShape = Catlass::GemmShape<M0, N0, L1TileShapeK>;
    using L0TileShape = Catlass::GemmShape<M0, N0, L0TileShapeK>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    constexpr static bool IS_DYNAMIC = true;

    using PaddingHelperB = typename Catccos::Padding::PaddingHelper<BType, EnablePadding_>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using BlockMmad = DGemm::Block::FixpipeBlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape,
        AType, ActualTypeB, DType>;

    using BlockSchedulerForAllgather = typename Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using CommBlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileSchedulerForAllgather
    >;

    using Kernel = DGemm::Kernel::AllGatherDequantMatmul<
        GlobalPaddingB,
        BlockMmad,
        BlockComm,
        BlockSchedulerForAllgather,
        CommBlockScheduler,
        WORKSPACE_STAGES
    >;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Without padding
template <class EA, class LA, class EB, class LB, class ED, class LD, class ES, class LS>
using AllGatherMatmulDequantConfig_M0_128 = AllGatherMatmulDequantConfig<EA, LA, EB, LB, ED, LD, ES, LS, false, 128, 256, 256>;
template <class EA, class LA, class EB, class LB, class ED, class LD, class ES, class LS>
using AllGatherMatmulDequantConfig_M0_256 = AllGatherMatmulDequantConfig<EA, LA, EB, LB, ED, LD, ES, LS, false, 256, 128, 256>;

// With padding
template <class EA, class LA, class EB, class LB, class ED, class LD, class ES, class LS>
using AllGatherMatmulDequantPaddingConfig_M0_128 = AllGatherMatmulDequantConfig<EA, LA, EB, LB, ED, LD, ES, LS, true, 128, 256, 256>;
template <class EA, class LA, class EB, class LB, class ED, class LD, class ES, class LS>
using AllGatherMatmulDequantPaddingConfig_M0_256 = AllGatherMatmulDequantConfig<EA, LA, EB, LB, ED, LD, ES, LS, true, 256, 128, 256>;

#endif // ALLGATHER_MATMUL_DEQUANT_KERNEL_H