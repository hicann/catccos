#ifndef MX_QUANT_ALLGATHER_DEVICE_H
#define MX_QUANT_ALLGATHER_DEVICE_H

#include "info.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/block/comm_block_mx_quant.hpp"
#include "catccos/comm/kernel/mx_quant_allgather.hpp"

using namespace AscendC;
using namespace Catccos;

template <class LayoutType>
CATLASS_DEVICE
uint32_t GetLayoutStride(uint32_t m, uint32_t n)
{
    if constexpr (std::is_same_v<LayoutType, Catlass::layout::RowMajor>) {
        return n;
    } else {
        return m;
    }
}

template <
    class ArchTag,
    class ElementInput,
    class LayoutInput,
    class ElementOutput,
    class LayoutOutput
>
CATLASS_DEVICE
void MxQuantAllGatherImpl(
    Catlass::GemmCoord& problemShape,
    GM_ADDR gmInput,
    LayoutInput &layoutInput,
    GM_ADDR gmOutput,
    LayoutOutput &layoutOutput,
    GM_ADDR gmMxScale,
    uint32_t commInterval,
    Catlass::MatrixCoord &commCoreSplit,
    uint32_t &commBlockShape,
    Catlass::MatrixCoord &commTileShape,
    GM_ADDR symmetricPtr,
    uint32_t rank,
    uint32_t rankSize
)
{
    constexpr bool IS_DYNAMIC = true;
    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr int64_t ROUND_MODE = 2;  // CAST_RINT
    constexpr uint32_t KERNEL_WORKSPACE_STAGES = 4;

    using InputType = Catlass::Gemm::GemmType<ElementInput, LayoutInput>;
    using OutputType = Catlass::Gemm::GemmType<ElementOutput, LayoutOutput>;

    // MX quant block
    using BlockMxQuant = Catccos::Comm::Block::CommBlockMxQuant<
        UB_STAGES, BLOCK_SIZE, ROUND_MODE,
        InputType, OutputType
    >;

    // AllGather comm block (CommBlock wrapping TileRemoteCopy)
    using RemoteSrcType = OutputType;
    using RemoteDstType = OutputType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopyQuant = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void,
        CopyDirect::Get, CopyTransport::Mte
    >;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommDispatchPolicy = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockAllGather = Catccos::Comm::Block::CommBlock<
        CommDispatchPolicy,
        RemoteSrcType, RemoteDstType,
        void,
        TileRemoteCopyQuant, TileScheduler
    >;

    // Scale gather comm block (uint8_t element type for E8M0 scales)
    using ScaleSrcType = Catlass::Gemm::GemmType<float8_e8m0_t, LayoutOutput>;
    using ScaleDstType = ScaleSrcType;
    using TileRemoteCopyScale = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, ScaleSrcType, ScaleDstType, void,
        CopyDirect::Get, CopyTransport::Mte
    >;
    using BlockScaleGather = Catccos::Comm::Block::CommBlock<
        CommDispatchPolicy,
        ScaleSrcType, ScaleDstType,
        void,
        TileRemoteCopyScale, TileScheduler
    >;

    using QuantScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;
    using AllToAllScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using MxQuantAllGatherKernel = Comm::Kernel::MxQuantAllGather<
        BlockMxQuant, BlockAllGather, BlockScaleGather,
        QuantScheduler, AllToAllScheduler,
        KERNEL_WORKSPACE_STAGES
    >;

    typename BlockAllGather::TileRemoteCopy::Params tileParams{commTileShape};
    typename BlockAllGather::Params blockParams{Catlass::MatrixCoord{commBlockShape, 1}, tileParams};
    typename BlockScaleGather::TileRemoteCopy::Params scaleTileParams{commTileShape};
    typename BlockScaleGather::Params scaleBlockParams{Catlass::MatrixCoord{commBlockShape, 1}, scaleTileParams};
    typename QuantScheduler::Params quantSwizzleParams{commCoreSplit};
    typename AllToAllScheduler::Params allToAllSwizzleParams{commCoreSplit};

    typename MxQuantAllGatherKernel::Params params{};
    params.problemShape = problemShape;
    params.rankIdx = rank;
    params.rankSize = rankSize;
    params.commInterval = commInterval;
    params.ptrInput = reinterpret_cast<__gm__ ElementInput *>(gmInput);
    params.layoutInput = layoutInput;
    params.ptrOutput = reinterpret_cast<__gm__ ElementOutput *>(gmOutput);
    params.layoutOutput = layoutOutput;
    params.ptrMxScale = reinterpret_cast<__gm__ float8_e8m0_t *>(gmMxScale);
    params.ptrSymmetric = symmetricPtr;
    params.AllGatherParams = blockParams;
    params.ScaleGatherParams = scaleBlockParams;
    params.quantParams = quantSwizzleParams;
    params.commParams = allToAllSwizzleParams;

    MxQuantAllGatherKernel kernel;
    kernel(params);
}

template <
    class ElementInput,
    class LayoutInput,
    class ElementOutput,
    class LayoutOutput
>
CATLASS_GLOBAL
void MxQuantAllGather(
    uint64_t fftsAddr,
    GM_ADDR gmInput,
    GM_ADDR gmOutput,
    GM_ADDR gmMxScale,
    GM_ADDR commArgsPtr,
    CocTilingParams cocTiling,
    uint32_t magic
)
{
    using ArchTag = Catlass::Arch::Ascend950;
    Catlass::Arch::Resource<ArchTag> resource;

    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t commInterval = cocTiling.commInterval;
    uint32_t commTileM = cocTiling.commTileM;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commBlockM = cocTiling.commBlockM;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    uint32_t commBlockShape = commBlockM;
    Catlass::MatrixCoord commTileShape{commTileM / 2, TILE_SHAPE_256};

    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t strideInput = GetLayoutStride<LayoutInput>(m, n);
    uint32_t strideOutput = GetLayoutStride<LayoutOutput>(m, n);
    LayoutInput layoutInput{m, n, strideInput};
    LayoutOutput layoutOutput{m, n, strideOutput};

    MxQuantAllGatherImpl<ArchTag, ElementInput, LayoutInput, ElementOutput, LayoutOutput>(
        problemShape,
        gmInput, layoutInput,
        gmOutput, layoutOutput,
        gmMxScale,
        commInterval,
        commCoreSplit, commBlockShape, commTileShape,
        commArgsPtr, rank, rankSize
    );
}

#endif // MX_QUANT_ALLGATHER_DEVICE_H
