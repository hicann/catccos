/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "catccos/arch/ascend950_hcomm_comm.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/kernel/ascend950_allgather_matmul_with_local_optional_backend.hpp"

#include "lib/matmul_intf.h"
#include "all_gather_matmul_hccl_tiling.h"

using namespace AscendC;
using namespace Catccos;

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void AllGatherMatmulHcclImpl(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmC, LayoutC& layoutC,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    AllGatherMatmulHcclTiling *tilingData,
    GM_ADDR workspace,
    uint32_t rankSize
)
{
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG>;

    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using AllGatherDispatch = Comm::AtlasCommLocalCopy<ArchTag, 2, IS_DYNAMIC>;
    using BlockAllGather = Comm::Block::CommBlock<
        AllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileScheduler
    >;

    using CommBackend = Arch::Ascend950HcommComm<AllGatherMatmulHcclTiling>;

    using AllGatherMatmulKernel = DGemm::Kernel::Ascend950AllGatherMatmulWithLocalOptionalBackend<
        BlockMmad, BlockAllGather, BlockMmadScheduler, BlockScheduler, 2, CommBackend>;

    typename TileRemoteCopy::Params tileParams{
        commTileShape
    };

    typename BlockAllGather::Params blockParams {
        commBlockShape,
        tileParams
    };

    typename BlockScheduler::Params swizzleParams {
        commCoreSplit
    };

    Catlass::layout::RowMajor layoutGatherSrc{problemShape.m(), problemShape.k(), problemShape.k()};
    auto layoutTlaA = tla::MakeLayoutFromTag(layoutA);
    auto layoutTlaB = tla::MakeLayoutFromTag(layoutB);
    auto layoutTlaC = tla::MakeLayoutFromTag(layoutC);

    // Ascend950 does not provide GetRankDim/GetRankId for the CCU backend, so
    // rankSize comes from host tiling and is also used to describe the output.
    typename AllGatherMatmulKernel::Params params {
        problemShape,
        0,
        static_cast<uint32_t>(layoutC.shape(0) / problemShape.m()),
        commInterval,
        layoutGatherSrc,
        gmA, layoutTlaA,
        gmB, layoutTlaB,
        gmC, layoutTlaC,
        blockParams,
        swizzleParams,
        typename CommBackend::Params{
            tilingData, offsetof(AllGatherMatmulHcclTiling, mc2CcTiling), workspace, rankSize}
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_DEVICE
void AllGatherMatmulHcclImpl_M0_256(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmC, LayoutC& layoutC,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    AllGatherMatmulHcclTiling *tilingData,
    GM_ADDR workspace,
    uint32_t rankSize
)
{
    AllGatherMatmulHcclImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
        commInterval, commCoreSplit, commBlockShape, commTileShape, tilingData, workspace, rankSize
    );
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_DEVICE
void AllGatherMatmulHcclImpl_M0_128(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmC, LayoutC& layoutC,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    AllGatherMatmulHcclTiling *tilingData,
    GM_ADDR workspace,
    uint32_t rankSize
)
{
    AllGatherMatmulHcclImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
        commInterval, commCoreSplit, commBlockShape, commTileShape, tilingData, workspace, rankSize
    );
}

extern "C" __aicore__ __global__
void all_gather_matmul_hccl(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AllGatherMatmulHcclTiling);
    GET_TILING_DATA_WITH_STRUCT(AllGatherMatmulHcclTiling, tilingData, tiling);

    using ArchTag = Catlass::Arch::Ascend950;

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;

    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::RowMajor;

    uint32_t m = tilingData.params.m;
    uint32_t n = tilingData.params.n;
    uint32_t k = tilingData.params.k;
    uint32_t m0 = tilingData.params.m0;
    uint32_t n0 = tilingData.params.n0;
    uint32_t k0 = tilingData.params.k0;
    uint32_t commInterval = tilingData.params.commInterval;
    uint32_t commTileM = tilingData.params.commTileM;
    uint32_t commNpuSplit = tilingData.params.commNpuSplit;
    uint32_t commDataSplit = tilingData.params.commDataSplit;
    uint32_t commBlockM = tilingData.params.commBlockM;
    uint32_t rankSize = tilingData.params.rankSize;
    uint64_t systemWorkspaceSize = tilingData.params.segmentSize;
    GM_ADDR gatherWorkspace = workspace + systemWorkspaceSize;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n, n};

    if(m0 == 128){
        AllGatherMatmulHcclImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, l1TileShape, a, layoutA, b, layoutB, c, layoutC,
            commInterval, commCoreSplit, commBlockShape, commTileShape, &tilingData, gatherWorkspace, rankSize
        );
    } else {
        AllGatherMatmulHcclImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, l1TileShape, a, layoutA, b, layoutB, c, layoutC,
            commInterval, commCoreSplit, commBlockShape, commTileShape, &tilingData, gatherWorkspace, rankSize
        );
    }
}
