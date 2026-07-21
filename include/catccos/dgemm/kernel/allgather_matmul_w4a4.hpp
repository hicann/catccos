/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_W4A4_HPP
#define CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_W4A4_HPP

#include "catccos/catccos.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/copy_int4_rowmajor.hpp"
#include "catccos/detail/remote_copy_type.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockComm_,
    class BlockMmadSchedulerForAllgather_,
    class BlockSchedulerForDequantInAiv_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_>
class AllGatherW4A4Matmul {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockComm = BlockComm_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementScale = uint64_t;
    using LayoutScale = Catlass::layout::VectorLayout;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockCommParams = typename BlockComm::Params;
    using BlockMmadScheduler = BlockMmadSchedulerForAllgather_;
    using BlockDequantScheduler = BlockSchedulerForDequantInAiv_;
    using CommScheduler = BlockCommScheduler_;
    using CommSchedulerParams = typename CommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrD;
        LayoutD layoutD;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrSymmetric;
        BlockCommParams blockCommParams;
        CommSchedulerParams commSchedulerParams;
        uint32_t commInterval;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rankIdx_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrScale_, LayoutScale const &layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            GM_ADDR ptrSymmetric_,
            BlockCommParams const &blockCommParams_,
            CommSchedulerParams const &commSchedulerParams_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rankIdx_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrD(ptrD_), layoutD(layoutD_),
            ptrScale(ptrScale_), layoutScale(layoutScale_),
            ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            ptrSymmetric(ptrSymmetric_),
            blockCommParams(blockCommParams_),
            commSchedulerParams(commSchedulerParams_),
            commInterval(commInterval_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrD;
        GM_ADDR ptrScale;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t * /*workspace*/ = nullptr)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        LayoutA layoutA{m, k};
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
        LayoutC layoutC{m * args.rankSize, n};
        LayoutD layoutD{m * args.rankSize, n};
        LayoutScale layoutScale{n};
        LayoutPerTokenScale layoutPerTokenScale{m * args.rankSize};

        MatrixCoord commTileShape{args.commTileShape.row(), k};
        typename BlockComm::TileRemoteCopy::Params tileParams{commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        CommSchedulerParams commSchedulerParams{args.commCoreSplit};

        return Params(
            args.problemShape,
            args.rankIdx, args.rankSize,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            args.ptrD, layoutD,
            args.ptrScale, layoutScale,
            args.ptrPerTokenScale, layoutPerTokenScale,
            args.ptrSymmetric,
            blockCommParams,
            commSchedulerParams,
            args.commInterval
        );
    }

    static size_t GetWorkspaceSize(Arguments const &)
    {
        return 0;
    }

    CATLASS_DEVICE
    AllGatherW4A4Matmul()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrScale));

        auto layoutScale = params.layoutScale;
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        Callback callbackBeforeFixpipe{};
        Callback callbackAfterFixpipe{};

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize);
            BlockMmadScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                int64_t offsetScale = blockOffset.n();
                auto gmBlockScale = gmScale[offsetScale];
                auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(offsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(offsetB)];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];

                mmad(
                    gmBlockScale, layoutScale,
                    gmBlockA, layoutSymmetric,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape.GetCoordMNK(),
                    callbackBeforeFixpipe, callbackAfterFixpipe
                );
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                mmad.SynchronizeBlock();
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockEpilogue blockEpilogue(resource);
        BlockComm blockRemoteCopy(resource, params.blockCommParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        auto layoutC = params.layoutC;

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.commSchedulerParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);

        uint32_t dequantCommIdx = 0;
        uint32_t dequantStageIdx = 0;
        GemmCoord dequantActualShape;

        const uint32_t aivLoopCount = commLoops + 1;
        for (uint32_t commIdx = 0; commIdx < aivLoopCount; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);

            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};

            if (commIdx < commLoops) {
                if (commIdx >= WORKSPACE_STAGES) {
                    Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
                }

                aclshmemx_barrier_all_vec();
                blockRemoteCopy.InitBlockLoop();
                if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                    for (uint32_t commLoopIdx = aicoreIdx;
                         commLoopIdx < commCoreLoops;
                         commLoopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                        MatrixCoord blockOffsetInRank =
                            commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape =
                            commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                        uint32_t remoteRankIdx = commBlockCoord.rank();
                        uint32_t dstRankIdx = params.rankIdx;
                        if constexpr (BlockComm::RemoteCopyDirect == detail::CopyDirect::Get) {
                            dstRankIdx = remoteRankIdx;
                        }

                        auto offsetSrc = commSrcOffset + blockOffsetInRank;
                        auto dstBaseRow = layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, dstRankIdx, 0));
                        auto offsetDst = MatrixCoord{dstBaseRow, 0} + blockOffsetInRank;

                        auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);

                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                        blockRemoteCopy(
                            gmBlockSrc, layoutBlockSrc,
                            gmBlockDst, layoutBlockDst,
                            actualCommBlockShape,
                            remoteRankIdx % params.rankSize);
                    }
                }
                blockRemoteCopy.FinalizeBlockLoop();

                aclshmemx_barrier_all_vec();

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
            }

            if (commIdx > 0) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[dequantStageIdx]);
                aclshmemx_barrier_all_vec();

                GemmCoord blockShapeMNK = L1TileShape::ToCoord();
                for (uint32_t deviceIdx = 0; deviceIdx < params.rankSize; ++deviceIdx) {
                    BlockDequantScheduler blockSchedulerForDequantInAiv(
                        dequantActualShape, L1TileShape::ToCoordMN());
                    uint32_t coreLoops = blockSchedulerForDequantInAiv.GetCoreLoops();

                    uint32_t rowBase = deviceIdx * params.problemShape.m() + dequantCommIdx * commSizeM;
                    uint32_t dequantM = dequantActualShape.m();
                    LayoutPerTokenScale chunkPerTokenLayout{dequantM};
                    LayoutD chunkLayoutD{dequantM, params.problemShape.n()};

                    EpilogueParams epilogueParams{
                        reinterpret_cast<__gm__ ElementPerTokenScale *>(
                            params.ptrPerTokenScale + rowBase * sizeof(ElementPerTokenScale)),
                        chunkPerTokenLayout,
                        reinterpret_cast<__gm__ ElementD *>(
                            params.ptrD + static_cast<uint64_t>(rowBase) * params.problemShape.n() * sizeof(ElementD)),
                        chunkLayoutD
                    };
                    blockEpilogue.UpdateParams(epilogueParams);

                    for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                        GemmCoord blockCoordMNK = blockSchedulerForDequantInAiv.GetBlockCoord(loopIdx);
                        GemmCoord actualBlockShapeMNK =
                            blockSchedulerForDequantInAiv.GetActualBlockShape(blockCoordMNK);

                        MatrixCoord offsetCBase{
                            layoutCRow(Catlass::MakeCoord<int>(deviceIdx, dequantCommIdx, 0)), 0};
                        MatrixCoord offsetC =
                            offsetCBase + blockCoordMNK.GetCoordMN() * blockShapeMNK.GetCoordMN();
                        int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                        auto gmBlockC = gmC[gmOffsetC];
                        auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                        blockEpilogue(
                            blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                    }
                }
            }

            dequantCommIdx = commIdx;
            dequantStageIdx = stageId;
            dequantActualShape = GemmCoord(actualCommSizeM, params.problemShape.n(), params.problemShape.k());
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_W4A4_HPP
