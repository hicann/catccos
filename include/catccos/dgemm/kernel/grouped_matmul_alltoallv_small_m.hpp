#ifndef CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_SMALL_M_HPP
#define CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_SMALL_M_HPP

#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {
using namespace Catlass;

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class TensorC_>
class GroupedMatmulAlltoAllvSmallM {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementWC = typename BlockMmad::ElementC;
    using LayoutWC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using BlockScheduler = BlockScheduler_;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<
        AscendC::LocalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using LayoutInnerDstGm = tla::Layout<
        tla::Shape<tla::Shape<uint32_t, uint32_t>, tla::Shape<uint32_t, uint32_t>>,
        tla::Stride<tla::Stride<int64_t, int64_t>, tla::Stride<tla::Int<1>, int64_t>>>;
    using TensorInnerDstGm = tla::Tensor<
        AscendC::GlobalTensor<ElementC>, LayoutInnerDstGm, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;
    using CopyUb2Gm = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerUb, TensorInnerSrcGm>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t localExpertNum;
        GM_ADDR globalTokensPerExpert;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWC;
        LayoutWC layoutWC;
        GM_ADDR smPtr;
        LayoutC layoutC;
        uint32_t rankSize;
        GM_ADDR ptrC;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t localExpertNum_, GM_ADDR globalTokensPerExpert_, GM_ADDR ptrA_,
            LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrWC_, LayoutWC layoutWC_, GM_ADDR smPtr_,
            LayoutC layoutC_, uint32_t rankSize_, GM_ADDR ptrC_)
            : problemShape(problemShape_),
              localExpertNum(localExpertNum_),
              globalTokensPerExpert(globalTokensPerExpert_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrWC(ptrWC_),
              layoutWC(layoutWC_),
              smPtr(smPtr_),
              layoutC(layoutC_),
              rankSize(rankSize_),
              ptrC(ptrC_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t localExpertNum;
        uint8_t* globalTokensPerExpert;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrWC;
        LayoutWC layoutWC;
        uint8_t* smPtr;
        LayoutC layoutC;
        uint32_t rankSize;
        uint8_t* ptrC;
    };

    static bool CanImplement(const Arguments& args) { return true; }

    static size_t GetWorkspaceSize(const Arguments& args) { return 0; }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape, args.localExpertNum, args.globalTokensPerExpert,
                      args.ptrA,         args.layoutA,        args.ptrB,
                      args.layoutB,      args.ptrWC,          args.layoutWC,
                      args.smPtr,        args.layoutC,        args.rankSize,
                      args.ptrC};
        return params;
    }

    CATLASS_DEVICE
    GroupedMatmulAlltoAllvSmallM() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAivFinishShmemBarrier);

        int64_t rankId = aclshmem_my_pe();

        BlockScheduler matmulBlockScheduler;

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementWC> gmWC;
        gmWC.SetGlobalBuffer((__gm__ ElementWC*)params.ptrWC);

        __gm__ int32_t* globalTokensPerExpert = (__gm__ int32_t*)params.globalTokensPerExpert;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetWC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
            uint32_t currentTokensNum = 0;
            uint32_t globalExpertIdx = rankId * params.localExpertNum + localExpertIdx;
            for (int i = 0; i < params.rankSize; i++) {
                currentTokensNum +=
                    globalTokensPerExpert[i * params.rankSize * params.localExpertNum + globalExpertIdx];
            }

            GemmCoord inGroupProblemShape{currentTokensNum, params.problemShape.n(), params.problemShape.k()};
            matmulBlockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA + gmGroupOffsetA);
            gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB + gmGroupOffsetB);
            gmWC.SetGlobalBuffer((__gm__ ElementWC*)params.ptrWC + gmGroupOffsetWC);
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorWC = tla::MakeTensor(gmWC, params.layoutWC, Arch::PositionGM{});

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockWC = GetTile(
                    tensorWC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                bool isFirstBlock = (loopIdx == startLoopIdx);
                bool hasNextBlock = false;
                GemmCoord nextBlockCoord;
                GemmCoord nextActualBlockShape;
                if (loopIdx + coreNum < coreLoops) {
                    hasNextBlock = true;
                    nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + coreNum);
                    nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
                }

                auto nextTensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                    tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k()));
                auto nextTensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                    tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n()));

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockWC, nextTensorBlockA, nextTensorBlockB, actualBlockShape,
                    nextActualBlockShape, isFirstBlock, hasNextBlock);

                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishBlockMmad);
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetWC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        aclshmemx_barrier_all_vec();
        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishShmemBarrier);

        int64_t rankId = aclshmem_my_pe();
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        __gm__ int32_t* globalTokensPerExpert = (__gm__ int32_t*)params.globalTokensPerExpert;

        AscendC::GlobalTensor<ElementWC> gmWC;
        AscendC::GlobalTensor<ElementC> gmShmem;
        AscendC::GlobalTensor<ElementWC> gmExpert;
        AscendC::GlobalTensor<ElementC> gmC;
        gmWC.SetGlobalBuffer((__gm__ ElementWC*)params.ptrWC);
        gmShmem.SetGlobalBuffer((__gm__ ElementC*)params.smPtr);
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        auto tensorWC = tla::MakeTensor(gmWC, params.layoutWC, tla::Coord<tla::_0, tla::_0>(), Arch::PositionGM{});
        auto tensorShmem =
            tla::MakeTensor(gmShmem, params.layoutWC, tla::Coord<tla::_0, tla::_0>(), Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, tla::Coord<tla::_0, tla::_0>(), Arch::PositionGM{});

        BlockScheduler matmulBlockScheduler;
        // Reuse communication buffers and event IDs across transfer steps.
        AscendC::LocalTensor<ElementWC> commTmpBuffer[COMM_BUFFER_NUM];
        uint64_t commBufferOffset = 0;
        for (uint32_t i = 0; i < COMM_BUFFER_NUM; ++i) {
            commTmpBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementWC>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::TEventID commEventIds[COMM_BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
        uint32_t commBufferIndex = 0;

        int64_t expertOffsetWC = 0;
        uint32_t startCoreIdx = 0;

        for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
            uint32_t globalExpertIdx = rankId * params.localExpertNum + localExpertIdx;

            uint32_t currentTokensNum = 0;
            for (int i = 0; i < params.rankSize; i++) {
                currentTokensNum +=
                    globalTokensPerExpert[i * params.rankSize * params.localExpertNum + globalExpertIdx];
            }
            GemmCoord inGroupProblemShape{currentTokensNum, params.problemShape.n(), params.problemShape.k()};
            matmulBlockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            gmExpert.SetGlobalBuffer((__gm__ ElementWC*)params.ptrWC + expertOffsetWC);
            auto tensorExpert =
                tla::MakeTensor(gmExpert, params.layoutWC, tla::Coord<tla::_0, tla::_0>(), Arch::PositionGM{});

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                MatrixCoord coordOnExpert(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N);

                auto tensorBlockExpert = GetTile(
                    tensorExpert, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAicFinishBlockMmad);

                ScatterExpertTile(
                    tensorShmem, tensorBlockExpert, coordOnExpert, globalExpertIdx, params.localExpertNum,
                    params.globalTokensPerExpert, params.problemShape, commTmpBuffer, commEventIds, commBufferIndex);
            }
            expertOffsetWC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        aclshmemx_barrier_all_vec();

        auto tensorConSymmem = GetTile(
            tensorShmem, tla::MakeCoord(params.problemShape.m() * rankId, (uint32_t)0),
            tla::MakeShape(params.problemShape.m(), params.problemShape.n()));

        // Copy this rank's M-row symmetric slice to its output tensor through UB.
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementC>(bufferOffset * sizeof(ElementC));
            bufferOffset += COMPUTE_LENGTH;
        }
        uint32_t tilesNum = params.problemShape.m();
        uint32_t tileLen = params.problemShape.n();
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(ElementC)>(params.problemShape.n());
        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{0};
        if (roundTileLen > COMPUTE_LENGTH) {
            // Split each row across multiple UB transfers when it exceeds one buffer.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                auto offset = tla::MakeCoord(mIdx + tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc =
                    GetTile(tensorConSymmem, offset, tla::MakeShape(static_cast<uint32_t>(1), actualDataNum));

                auto tensorTileDst = GetTile(tensorC, offset, tla::MakeShape(static_cast<uint32_t>(1), actualDataNum));

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum),
                    tla::MakeStride(static_cast<int64_t>(COMPUTE_LENGTH), tla::Int<1>{}));
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto tensorSrcUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});
                copyUb2Gm(tensorTileDst, tensorSrcUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Pack as many complete rows as fit in each UB transfer.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(tensorConSymmem, offset, tla::MakeShape(actualTilesNum, tileLen));

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(actualTilesNum, tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    tla::MakeShape(static_cast<uint32_t>(1), tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));
                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    auto tensorTileDst = GetTile(
                        tensorC, tla::MakeCoord(mIdx + tileIdx + i, static_cast<uint32_t>(0)),
                        tla::MakeShape(static_cast<uint32_t>(1), tileLen));
                    auto tensorSrcUb =
                        tla::MakeTensor(inputBuffer[bufferIndex][i * roundTileLen], layoutSrcUb, Arch::PositionUB{});
                    copyUb2Gm(tensorTileDst, tensorSrcUb);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);

        aclshmemx_barrier_all_vec();
    }

private:
    static constexpr uint32_t COMM_BUFFER_NUM = 2;
    static constexpr uint32_t COMM_BUFFER_BYTES = 96 * 1024;

    using RemoteCopyType = Catlass::Gemm::GemmType<ElementWC, Catlass::layout::RowMajor>;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, true, RemoteCopyType, RemoteCopyType, void, Catccos::detail::CopyDirect::Put,
        Catccos::detail::CopyTransport::Mte>;

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE void ScatterExpertTile(
        TensorOut& tensorDst, TensorIn const& tensorSrc, MatrixCoord blockCoord, uint32_t globalExpertIdx,
        uint32_t localExpertNum, GM_ADDR globalTokensPerExpert_, GemmCoord problemShape,
        AscendC::LocalTensor<ElementWC> (&tmpBuffer)[COMM_BUFFER_NUM],
        AscendC::TEventID const (&eventIds)[COMM_BUFFER_NUM], uint32_t& bufferIndex)
    {
        TileRemoteCopy tileRemoteCopy;
        uint32_t aivNum = AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetSubBlockIdx();
        int64_t rankSize = aclshmem_n_pes();

        uint32_t tokenLen = tla::get<1>(tensorSrc.shape());
        uint32_t BlockstartTokenIdx = blockCoord.row();

        uint32_t BlockendTokenIdx = BlockstartTokenIdx + tla::get<0>(tensorSrc.shape()) - 1;

        __gm__ int32_t* globalTokensPerExpert = (__gm__ int32_t*)globalTokensPerExpert_;

        uint32_t tokensNumForPrevRanks = 0;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);

        for (uint32_t targetRank = 0; targetRank < rankSize; ++targetRank) {
            uint32_t subtokenNum = globalTokensPerExpert[localExpertNum * rankSize * targetRank + globalExpertIdx];

            if (subtokenNum > 0) {
                uint32_t RankstartTokenIdx = tokensNumForPrevRanks;
                uint32_t RankendTokenIdx = RankstartTokenIdx + subtokenNum - 1;

                uint32_t overlapStart = RankstartTokenIdx > BlockstartTokenIdx ? RankstartTokenIdx : BlockstartTokenIdx;
                uint32_t overlapEnd = RankendTokenIdx < BlockendTokenIdx ? RankendTokenIdx : BlockendTokenIdx;

                if (overlapStart <= overlapEnd) {
                    if ((targetRank % aivNum) == aivId) {
                        uint32_t tokensNumForRank = overlapEnd - overlapStart + 1;

                        uint32_t tokenIdxInTarget = 0;
                        for (uint32_t i = 0; i < globalExpertIdx; i++) {
                            tokenIdxInTarget += globalTokensPerExpert[localExpertNum * rankSize * targetRank + i];
                        }
                        tokenIdxInTarget += overlapStart - RankstartTokenIdx;

                        uint32_t localSrcStartOffset = overlapStart - BlockstartTokenIdx;

                        auto tileOffsetSrc = tla::MakeCoord(localSrcStartOffset, (uint32_t)0);
                        auto tileOffsetDst =
                            tla::MakeCoord(tokenIdxInTarget + targetRank * problemShape.m(), blockCoord.column());

                        auto tensorTileSrc =
                            GetTile(tensorSrc, tileOffsetSrc, tla::MakeShape(tokensNumForRank, tokenLen));
                        auto tensorTileDst =
                            GetTile(tensorDst, tileOffsetDst, tla::MakeShape(tokensNumForRank, tokenLen));

                        AscendC::TEventID eventId = eventIds[bufferIndex];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

                        MatrixCoord copyShape{tla::get<0>(tensorTileSrc.shape()), tla::get<1>(tensorTileSrc.shape())};
                        Catlass::layout::RowMajor srcLayout{
                            copyShape.row(), copyShape.column(), tla::get<0>(tensorTileSrc.stride())};
                        Catlass::layout::RowMajor dstLayout{
                            copyShape.row(), copyShape.column(), tla::get<0>(tensorTileDst.stride())};
                        auto srcOffset = tensorTileSrc.layout()(tensorTileSrc.coord());
                        auto dstOffset = tensorTileDst.layout()(tensorTileDst.coord());
                        tileRemoteCopy(
                            tensorTileDst.data()[dstOffset], dstLayout, tensorTileSrc.data()[srcOffset], srcLayout,
                            copyShape, tmpBuffer[bufferIndex], eventId, targetRank);

                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        bufferIndex = (bufferIndex + 1) % COMM_BUFFER_NUM;
                    }
                }
            }
            tokensNumForPrevRanks += subtokenNum;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    static constexpr Arch::FlagID FLAG_AIC_FINISH_BLOCK_MMAD = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_BLOCK_MMAD = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishBlockMmad{FLAG_AIC_FINISH_BLOCK_MMAD, RV_FLAG_AIC_FINISH_BLOCK_MMAD};

    static constexpr Arch::FlagID FLAG_AIV_FINISH_SHMEM_BARRIER = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_SHMEM_BARRIER = 1;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishShmemBarrier{
        FLAG_AIV_FINISH_SHMEM_BARRIER, RV_FLAG_AIV_FINISH_SHMEM_BARRIER};

    Arch::Resource<ArchTag> resource;

    static const uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(ElementC);

    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementC> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{0};
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementC) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_SMALL_M_HPP
