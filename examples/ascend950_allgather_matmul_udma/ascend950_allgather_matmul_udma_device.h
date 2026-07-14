/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_ALLGATHER_MATMUL_UDMA_DEVICE_H
#define ASCEND950_ALLGATHER_MATMUL_UDMA_DEVICE_H

#include "info.h"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/ascend950_allgather_matmul_with_udma.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    uint32_t M0_, uint32_t N0_, uint32_t K0_
>
struct Ascend950AllGatherMatmulUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG>;

    using L1TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<K0_>>;
    using L0TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<64>>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileLocalCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void,
        CopyDirect::Put, CopyTransport::Mte>;
     using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void,
        CopyDirect::Put, CopyTransport::Udma>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using AllGatherDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using UdmaAllGatherDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES>;
    using BlockLocalComm = Comm::Block::CommBlock<
        AllGatherDispatch, RemoteSrcType, RemoteDstType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Comm::Block::CommBlock<
        UdmaAllGatherDispatch, RemoteSrcType, RemoteDstType, TileRemoteCopy>;

    using Kernel = DGemm::Kernel::Ascend950AllGatherMatmulWithUdma<
        BlockMmad, BlockLocalComm, BlockRemoteComm, BlockMmadScheduler, BlockCommScheduler, WORKSPACE_STAGES>;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using Ascend950AllGatherMatmulUdmaConfig_M0_128 =
    Ascend950AllGatherMatmulUdmaConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using Ascend950AllGatherMatmulUdmaConfig_M0_256 =
    Ascend950AllGatherMatmulUdmaConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>;

#endif  // ASCEND950_ALLGATHER_MATMUL_UDMA_DEVICE_H
