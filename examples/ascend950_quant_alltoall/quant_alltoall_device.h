#ifndef QUANT_ALLTOALL_DEVICE_H
#define QUANT_ALLTOALL_DEVICE_H

#include "info.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/epilogue/block/block_epilogue_per_tensor_quant.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/kernel/quant_alltoall.hpp"

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

template <class ElementOutput>
CATLASS_DEVICE
void SetupPeerMemory(
    GM_ADDR symmetricPtr,
    uint32_t rank,
    uint32_t rankSize,
    uint32_t magic,
    __gm__ ElementOutput *peerMems[MAX_RANK_SIZE],
    __gm__ ElementOutput *anotherPeerMems[MAX_RANK_SIZE]
)
{
    constexpr uint32_t IPC_BUFF_STRIDE = IPC_BUFF_MAX_SIZE / PING_PONG_SIZE + IPC_DATA_OFFSET;
    uint32_t pingOffset = (magic % PING_PONG_SIZE) * IPC_BUFF_STRIDE;
    uint32_t pongOffset = ((magic + 1) % PING_PONG_SIZE) * IPC_BUFF_STRIDE;
    for (int i = 0; i < rankSize; ++i) {
        if (i == rank) {
            peerMems[i] = reinterpret_cast<__gm__ ElementOutput *>(symmetricPtr + pingOffset);
            anotherPeerMems[i] = reinterpret_cast<__gm__ ElementOutput *>(symmetricPtr + pongOffset);
        } else {
            auto remoteBase = reinterpret_cast<GM_ADDR>(aclshmem_ptr(reinterpret_cast<__gm__ void *>(symmetricPtr), i));
            peerMems[i] = reinterpret_cast<__gm__ ElementOutput *>(remoteBase + pingOffset);
            anotherPeerMems[i] = reinterpret_cast<__gm__ ElementOutput *>(remoteBase + pongOffset);
        }
    }
}

template <
    class ArchTag,
    class ElementInput,
    class LayoutInput,
    class ElementOutput,
    class LayoutOutput,
    class ElementScale
>
CATLASS_DEVICE
void QuantAllToAllImpl(
    Catlass::GemmCoord& problemShape,
    GM_ADDR gmInput,
    LayoutInput &layoutInput,
    GM_ADDR gmScales,
    GM_ADDR gmOutput,
    LayoutOutput &layoutOutput,
    uint32_t commInterval,
    Catlass::MatrixCoord &commCoreSplit,
    uint32_t &commBlockShape,
    Catlass::MatrixCoord &commTileShape,
    GM_ADDR symmetricPtr, 
    uint32_t magic,
    uint32_t rank,
    uint32_t rankSize
)
{
    constexpr uint32_t KERNEL_WORKSPACE_STAGES = 4;
    constexpr bool IS_DYNAMIC = true;
    constexpr uint32_t UB_STAGES = 2;

    using InputType = Catlass::Gemm::GemmType<ElementInput, LayoutInput>;
    using OutputType = Catlass::Gemm::GemmType<ElementOutput, LayoutOutput>;

    using QuantDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA5PerTensorQuant<UB_STAGES, IS_DYNAMIC>;
    using TileSchedulerForQuant = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockQuant = Catlass::Epilogue::Block::BlockEpilogue<
        QuantDispatchPolicy,
        InputType,
        OutputType,
        ElementScale,
        TileSchedulerForQuant
    >;
    using RemoteSrcType = OutputType;
    using RemoteDstType = OutputType;

    using AllToAllDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockAllToAll = Comm::Block::CommBlock<
        AllToAllDispatch,
        RemoteSrcType,
        RemoteDstType,
        void
    >;
    using QuantScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;
    using AllToAllScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

    using QuantAllToAllKernel = Comm::Kernel::QuantAllToAll<
        BlockQuant,
        BlockAllToAll,
        QuantScheduler,
        AllToAllScheduler,
        KERNEL_WORKSPACE_STAGES
    >;

    typename BlockAllToAll::Params blockParams {
        commBlockShape
    };
    typename QuantScheduler::Params quantSwizzleParams {
        commCoreSplit
    };

    typename AllToAllScheduler::Params allToAllSwizzleParams {
        commCoreSplit
    };

    __gm__ ElementOutput *peerMems[MAX_RANK_SIZE] = {};
    __gm__ ElementOutput *anotherPeerMems[MAX_RANK_SIZE] = {};
    SetupPeerMemory<ElementOutput>(symmetricPtr, rank, rankSize, magic, peerMems, anotherPeerMems);
    typename QuantAllToAllKernel::Params params {
        problemShape,
        rank,
        rankSize,
        magic,
        commInterval,
        gmInput,
        layoutInput,
        gmScales,
        gmOutput,
        layoutOutput,
        reinterpret_cast<GM_ADDR>(peerMems[rank]),
        (__gm__ ElementOutput **)(peerMems),
        (__gm__ ElementOutput **)(anotherPeerMems),
        blockParams,
        quantSwizzleParams,
        allToAllSwizzleParams
    };
    QuantAllToAllKernel quantAllToAllKernel;
    quantAllToAllKernel(params);
}

template <
    class ElementInput,
    class LayoutInput,
    class ElementOutput,
    class LayoutOutput,
    class ElementScale
>
CATLASS_GLOBAL
void QuantAllToAll(
    uint64_t fftsAddr,
    GM_ADDR gmInput,
    GM_ADDR gmScales,
    GM_ADDR gmOutput,
    GM_ADDR symmetricPtr, 
    CocTilingParams cocTiling,
    uint32_t magic
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
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
    // commTileM/2: each tile handles half the comm tile rows; 256: fixed tile column width in bytes
    Catlass::MatrixCoord commTileShape{commTileM / 2, 256};
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t strideInput = GetLayoutStride<LayoutInput>(m, n);
    uint32_t strideOutput = GetLayoutStride<LayoutOutput>(m, n);
    LayoutInput layoutInput{m, n, strideInput};
    LayoutOutput layoutOutput{m, n, strideOutput};
    QuantAllToAllImpl<ArchTag, ElementInput, LayoutInput, ElementOutput, LayoutOutput, ElementScale>(
        problemShape,
        gmInput,
        layoutInput,
        gmScales,
        gmOutput,
        layoutOutput,
        commInterval,
        commCoreSplit,
        commBlockShape,
        commTileShape,
        symmetricPtr,
        magic,
        rank,
        rankSize
    );
}

#endif // QUANT_ALLTOALL_DEVICE_H
