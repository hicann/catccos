#ifndef CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_SHALLOW_FUSION_HPP
#define CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_SHALLOW_FUSION_HPP

#include "catccos/comm/tile/tile_remote_copy_shmem_to_local_ub.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {
using namespace Catlass;

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class PaddingA, class PaddingB>
class MatmulReduceScatterShallowFusion {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;

    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using BlockScheduler = BlockScheduler_;

    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        GM_ADDR ptrSymmetric;
        GM_ADDR ptrSyncFlags;
        uint32_t rankSize;
        uint32_t syncValue;
        uint32_t needDoneSignal;
        uint32_t nBlocksPerChunk;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GemmCoord const& L1Shape_, MatrixCoord const& commBlockShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_, GM_ADDR ptrSymmetric_,
            GM_ADDR ptrSyncFlags_, uint32_t rankSize_, uint32_t syncValue_, uint32_t needDoneSignal_,
            uint32_t nBlocksPerChunk_)
            : problemShape(problemShape_),
              L1Shape(L1Shape_),
              commBlockShape(commBlockShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrWA(ptrWA_),
              layoutWA(layoutWA_),
              ptrWB(ptrWB_),
              layoutWB(layoutWB_),
              ptrSymmetric(ptrSymmetric_),
              ptrSyncFlags(ptrSyncFlags_),
              rankSize(rankSize_),
              syncValue(syncValue_),
              needDoneSignal(needDoneSignal_),
              nBlocksPerChunk(nBlocksPerChunk_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrC;
        LayoutC layoutC;
        uint8_t* ptrWA;
        LayoutWA layoutWA;
        uint8_t* ptrWB;
        LayoutWB layoutWB;
        uint8_t* ptrSymmetric;
        uint8_t* ptrSyncFlags;
        uint32_t rankSize;
        uint32_t syncValue{1};
        uint32_t needDoneSignal{0};
        uint32_t nBlocksPerChunk{1};
    };

    static bool CanImplement(const Arguments& args) { return true; }

    static size_t GetWorkspaceSize(const Arguments& args) { return 0; }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape, args.L1Shape,   args.commBlockShape, args.ptrA,           args.layoutA,
                      args.ptrB,         args.layoutB,   args.ptrC,           args.layoutC,        args.ptrWA,
                      args.layoutWA,     args.ptrWB,     args.layoutWB,       args.ptrSymmetric,   args.ptrSyncFlags,
                      args.rankSize,     args.syncValue, args.needDoneSignal, args.nBlocksPerChunk};
        return params;
    }

    CATLASS_DEVICE
    MatmulReduceScatterShallowFusion() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    // AIC: write matmul partial results and publish readiness signals.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t rankSize = aclshmem_n_pes();
        int64_t rankId = aclshmem_my_pe();
        __gm__ int32_t* sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSyncFlags);
        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        GemmCoord globalProblemShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(params.L1Shape, resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrSymmetric);
        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        GemmCoord actualCocShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};
        BlockScheduler matmulBlockScheduler(actualCocShape, MakeCoord(params.L1Shape.m(), params.L1Shape.n()));

        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        if (params.nBlocksPerChunk == 0) { // 不使用切N机制
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
                if (loopIdx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                    continue;
                }

                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.k() * params.L1Shape.k()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockCoord.k() * params.L1Shape.k(), blockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                auto tensorBlockC = GetTile(
                    tensorC, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                bool isFirstBlock =
                    (loopIdx == (AscendC::GetBlockIdx() + AscendC::GetBlockNum()) % AscendC::GetBlockNum());
                bool hasNextBlock = false;
                GemmCoord nextBlockCoord;
                GemmCoord nextActualBlockShape;
                if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                    hasNextBlock = true;
                    nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                    nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
                }
                auto nextTensorBlockA = GetTile(
                    tensorA,
                    tla::MakeCoord(nextBlockCoord.m() * params.L1Shape.m(), nextBlockCoord.k() * params.L1Shape.k()),
                    tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k()));
                auto nextTensorBlockB = GetTile(
                    tensorB,
                    tla::MakeCoord(nextBlockCoord.k() * params.L1Shape.k(), nextBlockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n()));

                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, actualBlockShape,
                    nextActualBlockShape, isFirstBlock, hasNextBlock);
            }

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishMatmul);
            return;
        }

        uint32_t nBlockCount = (params.problemShape.n() + params.L1Shape.n() - 1) / params.L1Shape.n();
        uint64_t touchedShardMask = 0;
        uint64_t waitedDoneShardMask = 0;                // 掩码记录哪些分块已经等待过通信完成
        bool useChunkReady = params.nBlocksPerChunk > 1; // Group multiple N blocks per readiness signal.
        uint32_t nBlocksPerChunk = useChunkReady ? params.nBlocksPerChunk : 1;
        uint32_t nChunkCount = (nBlockCount + nBlocksPerChunk - 1) / nBlocksPerChunk;
        __gm__ int32_t* doneSigAddr = sigAddr + rankSize * rankSize * nChunkCount * 512;

        if (useChunkReady && params.syncValue > 1) { // 重用工作区前，等待上一轮所有输出分片完成通信。
            if (AscendC::GetBlockIdx() == 0) {
                for (uint32_t shardIdx = 0; shardIdx < rankSize; shardIdx++) {
                    aclshmem_signal_wait_until(doneSigAddr + shardIdx * 512, ACLSHMEM_CMP_GE, params.syncValue - 1);
                }
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();
        }

        for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if (loopIdx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }

            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            uint32_t firstShard = 0; // 当前 block 覆盖的输出 rank 分片编号。
            uint32_t lastShard = 0;
            uint32_t rowStart = blockCoord.m() * params.L1Shape.m(); // 当前block的起始行坐标
            uint32_t rowEnd = rowStart + actualBlockShape.m();       // 当前 block 的结束行坐标（不含）。
            firstShard =
                rowStart / params.commBlockShape.row(); // 根据通信划分的行块大小，计算当前block涉及到的第一个rank
            lastShard =
                (rowEnd - 1) / params.commBlockShape.row(); // 根据通信划分的行块大小，计算当前block涉及到的最后一个rank
            if (lastShard >= rankSize) { // 避免越界
                lastShard = rankSize - 1;
            }
            if (!useChunkReady && params.syncValue > 1) {
                for (uint32_t shardIdx = firstShard; shardIdx <= lastShard; shardIdx++) {
                    uint64_t shardMask = static_cast<uint64_t>(1) << shardIdx;
                    if ((waitedDoneShardMask & shardMask) == 0) { // 如果这个shard还没有等待过通信完成
                        aclshmem_signal_wait_until(doneSigAddr + shardIdx * 512, ACLSHMEM_CMP_GE, params.syncValue - 1);
                        waitedDoneShardMask |= shardMask; // 按位或操作标记已经等待过的shard，避免重复等待
                    }
                }
            }
            for (uint32_t shardIdx = firstShard; shardIdx <= lastShard; shardIdx++) {
                touchedShardMask |= static_cast<uint64_t>(1) << shardIdx;
            }

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * params.L1Shape.k(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() + AscendC::GetBlockNum()) % AscendC::GetBlockNum());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }
            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * params.L1Shape.m(), nextBlockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k()));
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * params.L1Shape.k(), nextBlockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n()));

            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, actualBlockShape,
                nextActualBlockShape, isFirstBlock, hasNextBlock);

            if (useChunkReady) {
                uint32_t nChunkIdx = blockCoord.n() / nBlocksPerChunk;
                AscendC::PipeBarrier<PIPE_FIX>();
                for (uint32_t shardIdx = firstShard; shardIdx <= lastShard; shardIdx++) {
                    aclshmemx_signal_op(
                        sigAddr + ((rankId * rankSize + shardIdx) * nChunkCount + nChunkIdx) *
                                      512, // sigAddr[rankId][shardIdx][nChunkIdx]
                        1, ACLSHMEM_SIGNAL_ADD, shardIdx);
                }
            }
        }
    }

    // AIV: prepare padded inputs and reduce the output shard for this rank.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        if (params.nBlocksPerChunk == 0) {
            aclshmemx_barrier_all_vec();
        }
        int64_t rankId = aclshmem_my_pe();
        uint32_t rankSize = aclshmem_n_pes();

        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));

            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorWA = tla::MakeTensor(gmWA, params.layoutWA, Arch::PositionGM{});

            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }
        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorWB = tla::MakeTensor(gmWB, params.layoutWB, Arch::PositionGM{});
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
        }
        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmWC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        gmWC.SetGlobalBuffer((__gm__ ElementC*)params.ptrSymmetric);

        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorSymmtric = tla::MakeTensor(gmWC, params.layoutC, Arch::PositionGM{});
        __gm__ int32_t* sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSyncFlags);
        uint32_t nBlockCount = (params.problemShape.n() + params.L1Shape.n() - 1) / params.L1Shape.n();

        // Reuse communication buffers and event IDs across transfer steps.
        AscendC::LocalTensor<ElementC> commTmpBuffer[COMM_BUFFER_NUM];
        uint64_t commBufferOffset = 0;
        for (uint32_t i = 0; i < COMM_BUFFER_NUM; ++i) {
            commTmpBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementC>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::LocalTensor<ElementC> commAddBuffer[1];
        for (uint32_t i = 0; i < 1; ++i) {
            commAddBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementC>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::TEventID commEventIds[COMM_BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
        bool useFullSync = params.nBlocksPerChunk == 0; // GEMM completes on all ranks before communication.
        bool useChunkReady = params.nBlocksPerChunk > 1;
        uint32_t nBlocksPerChunk = useChunkReady ? params.nBlocksPerChunk : 1;
        uint32_t nChunkCount = (nBlockCount + nBlocksPerChunk - 1) / nBlocksPerChunk;
        __gm__ int32_t* doneSigAddr = sigAddr + rankSize * rankSize * nChunkCount * 512; // 通信完成信号

        MatrixCoord actualCommBlockShape{params.commBlockShape.row(), params.commBlockShape.column()};

        if (useFullSync) {
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishMatmul);
            aclshmemx_barrier_all_vec();

            auto tensorBlockC = GetTile(
                tensorSymmtric, tla::MakeCoord(rankId * params.commBlockShape.row(), 0),
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column()));
            auto tensorBlockGm = GetTile(
                tensorC, tla::MakeCoord(0, 0),
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column()));
            ReduceScatterTile(tensorBlockGm, tensorBlockC, commTmpBuffer, commAddBuffer, commEventIds);
        } else if (useChunkReady) {
            for (uint32_t nChunkIdx = 0; nChunkIdx < nChunkCount; nChunkIdx++) {
                uint32_t firstNBlock = nChunkIdx * nBlocksPerChunk;
                uint32_t lastNBlock = firstNBlock + nBlocksPerChunk;
                if (lastNBlock > nBlockCount) {
                    lastNBlock = nBlockCount;
                }
                uint32_t colStart = firstNBlock * params.L1Shape.n();
                uint32_t actualColumns = (lastNBlock - firstNBlock) * params.L1Shape.n();
                if (colStart + actualColumns > params.problemShape.n()) {
                    actualColumns = params.problemShape.n() - colStart;
                }
                if (AscendC::GetBlockIdx() == 0) {
                    uint32_t shardRows = params.commBlockShape.row(); // 每个 rank 的输出行数。
                    uint32_t shardStart = rankId * shardRows;         // 当前rank结果范围
                    uint32_t shardEnd = shardStart + shardRows;
                    GemmCoord actualCocShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};
                    BlockScheduler matmulBlockScheduler(
                        actualCocShape, MakeCoord(params.L1Shape.m(), params.L1Shape.n()));
                    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
                    uint32_t chunkReadyThreshold = 0;
                    for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
                        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                        if (blockCoord.n() < firstNBlock || blockCoord.n() >= lastNBlock) {
                            continue;
                        }
                        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                        uint32_t rowStart = blockCoord.m() * params.L1Shape.m();
                        uint32_t rowEnd = rowStart + actualBlockShape.m();
                        if (rowStart < shardEnd && rowEnd > shardStart) {
                            chunkReadyThreshold++; // 计算当前rank当前通信段应收到的aic信号次数
                        }
                    }
                    uint32_t chunkReadyValue = params.syncValue * chunkReadyThreshold; // 轮次累加
                    for (uint32_t srcRank = 0; srcRank < rankSize; srcRank++) {
                        aclshmem_signal_wait_until(
                            sigAddr + ((srcRank * rankSize + rankId) * nChunkCount + nChunkIdx) * 512, ACLSHMEM_CMP_GE,
                            chunkReadyValue);
                    }
                }
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

                auto tensorBlockC = GetTile(
                    tensorSymmtric, tla::MakeCoord(rankId * params.commBlockShape.row(), colStart),
                    tla::MakeShape(actualCommBlockShape.row(), actualColumns));
                auto tensorBlockGm = GetTile(
                    tensorC, tla::MakeCoord(0, colStart), tla::MakeShape(actualCommBlockShape.row(), actualColumns));
                ReduceScatterTile(tensorBlockGm, tensorBlockC, commTmpBuffer, commAddBuffer, commEventIds);
            }
        }
        if (!useFullSync && params.needDoneSignal != 0) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        }
        if (!useFullSync && params.needDoneSignal != 0 && AscendC::GetBlockIdx() == 0) {
            for (uint32_t pe = 0; pe < rankSize; pe++) { // 告诉其他rank，当前rank需要的数据已经完成通信
                aclshmemx_signal_op(doneSigAddr + rankId * 512, params.syncValue, ACLSHMEM_SIGNAL_SET, pe);
            }
        }
    }

private:
    static constexpr uint32_t COMM_BUFFER_NUM = 2;
    static constexpr uint32_t COMM_BUFFER_BYTES = 64 * 1024;
    static constexpr uint32_t COMM_COMPUTE_LENGTH = COMM_BUFFER_BYTES / sizeof(ElementC);

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<
        AscendC::LocalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm =
        tla::Tensor<AscendC::GlobalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyUb2Gm = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerUb, TensorInnerDstGm>;
    using TileRemoteCopyShmemToLocalUb =
        Catccos::Comm::Tile::TileRemoteCopyShmemToLocalUb<ArchTag, TensorInnerUb, TensorInnerSrcGm>;

    template <typename T_Dst, typename T_Src>
    CATLASS_DEVICE void ReduceScatterTile(
        T_Dst& tensorDst, T_Src& tensorSrc, AscendC::LocalTensor<ElementC>* __restrict tmpBuffer,
        AscendC::LocalTensor<ElementC>* __restrict addBuffer, AscendC::TEventID const* __restrict eventIds)
    {
        CopyUb2Gm copyUb2Gm;
        TileRemoteCopyShmemToLocalUb copyShmemToLocalUb;
        int64_t rankId = aclshmem_my_pe();
        uint32_t rankSize = aclshmem_n_pes();

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t tilesNum = tla::get<0>(tensorSrc.shape());
        uint32_t tileLen = tla::get<1>(tensorSrc.shape());
        uint32_t roundTileLen = RoundUp<512 / sizeof(ElementC)>(tileLen);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv + (aivId >= tileRemain ? tileRemain : 0);
        uint32_t tilesPerLoop = COMM_COMPUTE_LENGTH / roundTileLen;
        uint32_t coreloops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);

        for (uint32_t loopIdx = 0; loopIdx < coreloops; ++loopIdx) {
            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = (tilesPerAiv - tileIdx < tilesPerLoop) ? (tilesPerAiv - tileIdx) : tilesPerLoop;
            const uint32_t computeLength = actualTilesNum * roundTileLen;
            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

            auto tensorTileSymmSrc = GetTile(tensorSrc, offset, tla::MakeShape(actualTilesNum, tileLen));
            auto layoutUb = tla::MakeLayout(
                tla::MakeShape(actualTilesNum, tileLen),
                tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));

            auto tensorAddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            auto tensorTmpUb0 = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
            auto tensorTmpUb1 = tla::MakeTensor(tmpBuffer[1], layoutUb, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            copyShmemToLocalUb(tensorAddUb, tensorTileSymmSrc, rankId, eventIds[0]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);

            uint32_t remoteCount = rankSize - 1;

            if (remoteCount > 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);

                uint32_t midx_0 = (rankId + 1) % rankSize;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                copyShmemToLocalUb(tensorTmpUb0, tensorTileSymmSrc, midx_0, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);

                uint32_t k = 0;
                for (; k < remoteCount - 1; k += 2) {
                    uint32_t midx_next = (rankId + 1 + (k + 1)) % rankSize;

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
                    copyShmemToLocalUb(tensorTmpUb1, tensorTileSymmSrc, midx_next, eventIds[1]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[1]);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], computeLength);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]); // Buffer 0 Free

                    if (k + 2 < remoteCount) {
                        uint32_t midx_next_next = (rankId + 1 + (k + 2)) % rankSize;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                        copyShmemToLocalUb(tensorTmpUb0, tensorTileSymmSrc, midx_next_next, eventIds[0]);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[1]);
                    AscendC::Add(addBuffer[0], tmpBuffer[1], addBuffer[0], computeLength);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]); // Buffer 1 Free
                }
                if (k < remoteCount) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], computeLength);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileGmDst = GetTile(tensorDst, offset, tla::MakeShape(actualTilesNum, tileLen));
            copyUb2Gm(tensorTileGmDst, tensorAddUb);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    }

    static constexpr Arch::FlagID FLAG_AIV_FINISH_REDUCESCATTER = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_REDUCESCATTER = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishMatmul{
        FLAG_AIV_FINISH_REDUCESCATTER, RV_FLAG_AIV_FINISH_REDUCESCATTER};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 2;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_STORE = 3;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishPadding{FLAG_AIV_FINISH_STORE, RV_FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_SHALLOW_FUSION_HPP
