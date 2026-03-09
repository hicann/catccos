#ifndef ALLGATHER_MATMUL_DEQUANT_PADDING_KERNEL_H
#define ALLGATHER_MATMUL_DEQUANT_PADDING_KERNEL_H

#include "info.h"

// from catlass
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

using namespace AscendC;
using namespace Catccos;

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementScale, class LayoutScale,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void AllGatherMatmulDequantPaddingImpl(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    GM_ADDR gmScale, LayoutScale& layoutScale,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR gmWorkSpace, GM_ADDR symmetricPtr
)
{
    // Block level, define BlockMmad
    constexpr bool ENABLE_UNIT_FLAG = false;
    constexpr bool ENABLE_SHUFFLE_K = true;
    constexpr bool IS_INT8 = std::is_same<ElementA, int8_t>::value;
    constexpr int32_t L1TileShapeK = IS_INT8 ? TILE_SHAPE_512 : TILE_SHAPE_256;
    constexpr int32_t L0TileShapeK = IS_INT8 ? TILE_SHAPE_128 : TILE_SHAPE_64;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

    using L1TileShape = Catlass::GemmShape<M0, N0, L1TileShapeK>;
    using L0TileShape = Catlass::GemmShape<M0, N0, L0TileShapeK>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;

    constexpr bool IS_DYNAMIC = true;

    using BlockSchedulerForAllgather = typename Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using CommBlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using AllGatherDispatch = Comm::AtlasA2CommRemoteCopy<UB_STAGES, IS_DYNAMIC>;
    using BlockAllGather = Comm::Block::CommBlock<
        AllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopy, TileSchedulerForAllgather
    >;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Catccos::Comm::AtlasA2PerTensorDequant<ubStages>;
    using ScaleType = Catlass::Gemm::GemmType<ElementScale, LayoutScale>;
 
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using PaddingHelperB = typename Catccos::Padding::PaddingHelper<BType, true>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    LayoutWB layoutWB = PaddingHelperB::GetLayoutW(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using BlockMmad = DGemm::Block::FixpipeBlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape,
        AType, ActualTypeB, DType>;

    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherDequantMatmul<
        GlobalPaddingB,
        BlockMmad,
        BlockAllGather,
        BlockSchedulerForAllgather,
        CommBlockScheduler,
        WORKSPACE_STAGES
    >;

    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    typename TileRemoteCopy::Params tileParams{
        commTileShape
    };

    typename BlockAllGather::Params blockParams {
        commBlockShape,
        tileParams
    };

    typename CommBlockScheduler::Params swizzleParams {
        commCoreSplit
    };

    // Prepare params
    typename AllGatherMatmulKernel::Params params {
        problemShape,
        rank, rankSize,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        gmScale, layoutScale,
        gmWorkSpace, layoutWB,
        symmetricPtr,
        blockParams,
        swizzleParams,
        commInterval
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementScale, class LayoutScale
>
CATLASS_DEVICE
void AllGatherMatmulDequantPaddingImpl_M0_256(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    GM_ADDR gmScale, LayoutScale& layoutScale,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR gmWorkSpace, GM_ADDR symmetricPtr
)
{
    AllGatherMatmulDequantPaddingImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB,
        ElementD, LayoutD, ElementScale, LayoutScale, 256, 128, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, gmScale, layoutScale,
        commInterval, commCoreSplit, commBlockShape, commTileShape, gmWorkSpace, symmetricPtr
    );
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementScale, class LayoutScale
>
CATLASS_DEVICE
void AllGatherMatmulDequantPaddingImpl_M0_128(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    GM_ADDR gmScale, LayoutScale& layoutScale,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR gmWorkSpace, GM_ADDR symmetricPtr
)
{
    AllGatherMatmulDequantPaddingImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB,
        ElementD, LayoutD, ElementScale, LayoutScale, 128, 256, 256>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
        gmD, layoutD, gmScale, layoutScale,
        commInterval, commCoreSplit, commBlockShape, commTileShape, gmWorkSpace, symmetricPtr
    );
}

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementScale, class LayoutScale
>
CATLASS_GLOBAL
void AllGatherMatmulDequantPadding(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmScale, GM_ADDR gmWorkSpace, GM_ADDR symmetricPtr, CocTilingParams cocTiling
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t n0 = cocTiling.n0;
    uint32_t k0 = cocTiling.k0;
    uint32_t commInterval = cocTiling.commInterval;
    uint32_t commTileM = cocTiling.commTileM;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commBlockM = cocTiling.commBlockM;
    uint32_t rankSize = cocTiling.rankSize;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{commTileM / 2, k0};

    uint32_t strideA;
    if constexpr (std::is_same_v<LayoutA, Catlass::layout::RowMajor>) {
        strideA = k;
    } else if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {
        strideA = m;
    }

    uint32_t strideB;
    if constexpr (std::is_same_v<LayoutB, Catlass::layout::RowMajor>) {
        strideB = n;
    } else if constexpr (std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>) {
        strideB = k;
    }

    // Block level, Define the layout of each input matrix
    LayoutA layoutA{m, k};                      //->AG(rank_sz)-> {m*rank_sz, k}
    LayoutB layoutB{k, n};                      // weight
    LayoutD layoutD{m * rankSize, n};           // c->dequant(s1{m*rank_sz}, s2{n}, {m*rank_sz, n})-> half({m*rank_sz, n})
    LayoutScale layoutScale{n};                 // perChannleScale

    if(m0 == 128){
        AllGatherMatmulDequantPaddingImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementScale, LayoutScale>(
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
            gmD, layoutD, gmScale, layoutScale,
            commInterval, commCoreSplit, commBlockShape, commTileShape, gmWorkSpace, symmetricPtr
        );
    } else {
        AllGatherMatmulDequantPaddingImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementScale, LayoutScale>(
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB,
            gmD, layoutD, gmScale, layoutScale,
            commInterval, commCoreSplit, commBlockShape, commTileShape, gmWorkSpace, symmetricPtr
        );
    }
}

#endif // ALLGATHER_MATMUL_DEQUANT_PADDING_KERNEL_H