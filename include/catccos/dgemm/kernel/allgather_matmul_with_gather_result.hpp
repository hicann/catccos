/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_HPP
#define CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_HPP

#include "catccos/catccos.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class BlockMmad_,
    class BlockComm_,
    class BlockGatherA_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    class BlockGatherAScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmulWithGatherResult {
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
    using ElementGatherA = ElementA;
    using LayoutGatherA = LayoutA;

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;

    using BlockGatherA = BlockGatherA_;
    using BlockGatherAParams = typename BlockGatherA::Params;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockGatherAScheduler = BlockGatherAScheduler_;

    using CommSchedulerParams = typename BlockCommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        GM_ADDR ptrSymmetric;
        BlockCommParams blockCommParams;
        CommSchedulerParams commSchedulerParams;
        BlockGatherAParams blockGatherAParams;

        __gm__ ElementGatherA *ptrGatherA;
        LayoutGatherA layoutGatherA;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        uint32_t commInterval;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrSymmetric_,
            BlockCommParams const &blockCommParams_,
            CommSchedulerParams const &commSchedulerParams_,
            BlockGatherAParams const &copyGatherAParams_,
            GM_ADDR ptrGatherA_, LayoutGatherA const &layoutGatherA_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrSymmetric(ptrSymmetric_),
            blockCommParams(blockCommParams_),
            commSchedulerParams(commSchedulerParams_),
            blockGatherAParams(copyGatherAParams_),
            ptrGatherA(reinterpret_cast<__gm__ ElementGatherA *>(ptrGatherA_)), layoutGatherA(layoutGatherA_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            commInterval(commInterval_)
        {
        }
    };

    /// User-facing arguments
    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrGatherA;
        GM_ADDR ptrSymmetric;
        Catlass::MatrixCoord commCoreSplit;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
        Catlass::MatrixCoord copyGatherABlockShape;
        Catlass::MatrixCoord copyGatherATileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m() * args.rankSize, args.problemShape.n()};
        LayoutGatherA layoutGatherA{args.problemShape.m() * args.rankSize, args.problemShape.k()};

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        CommSchedulerParams commSchedulerParams{args.commCoreSplit};

        typename BlockGatherA::TileRemoteCopy::Params copyGatherATileParams{args.copyGatherATileShape};
        BlockGatherAParams copyGatherAParams{args.copyGatherABlockShape, copyGatherATileParams};

        return Params(
            args.problemShape,
            args.rankIdx, args.rankSize,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrSymmetric,
            blockCommParams,
            commSchedulerParams,
            copyGatherAParams,
            args.ptrGatherA, layoutGatherA,
            args.ptrC, layoutC,
            args.commInterval
        );
    }

    // Methods
    CATLASS_DEVICE
    AllGatherMatmulWithGatherResult()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx< WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~AllGatherMatmulWithGatherResult()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementGatherA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGatherA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize
            );
            BlockMmadScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(offsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(offsetB)];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];

                // Compute block-scoped matrix multiply-add
                mmad(
                    gmBlockA, layoutSymmetric,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape.GetCoordMNK()
                );
            }

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMK = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockComm blockRemoteCopy(resource, params.blockCommParams);
        BlockGatherA copyGatherA(resource, params.blockGatherAParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementGatherA> gmGatherA;
        gmGatherA.SetGlobalBuffer(params.ptrGatherA);
        AscendC::GlobalTensor<ElementGatherA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGatherA *>(params.ptrSymmetric));

        auto layoutGatherARowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        layout::AffineRankN<3> layoutGatherARow{layoutGatherARowLogicStride};
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.commSchedulerParams.CoreSplit();
        BlockCommScheduler commScheduler(commBlockShape, commCoreSplit);

        MatrixCoord copyGatherABlockShape = params.blockGatherAParams.BlockShape();

        uint32_t copyCommIdx;
        uint32_t copyStageId;
        DistMatrixCoord copyActualShape;
        for (uint32_t commIdx = 0; commIdx < commLoops + 1; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            // wait aic
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            aclshmemx_barrier_all_vec();

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif

            if (commIdx < commLoops) {
                if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                    blockRemoteCopy.InitBlockLoop();
                    for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                        MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                        uint32_t remoteRankIdx = commBlockCoord.rank();

                        auto offsetSrc = commSrcOffset + blockOffsetInRank;
                        auto offsetDst = commDstOffset + blockOffsetInRank;

                        auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);

                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                        blockRemoteCopy(
                            gmBlockSrc, layoutBlockSrc,
                            gmBlockDst, layoutBlockDst,
                            actualCommBlockShape, remoteRankIdx % params.rankSize
                        );
                    }
                    blockRemoteCopy.FinalizeBlockLoop();
                }
            }

            if (commIdx > 0) {
                if (subcoreIdx == 1 && aicoreIdx >= commAicoreNum) {
                    BlockGatherAScheduler copyGatherAScheduler{copyActualShape, copyGatherABlockShape};
                    uint32_t copyCoreLoops = copyGatherAScheduler.GetCoreLoops();
                    uint32_t copyAicoreNum = aicoreNum - commAicoreNum;
                    uint32_t copyAicoreIdx = aicoreIdx - commAicoreNum;

                    copyGatherA.InitBlockLoop();
                    for (uint32_t loopIdx = copyAicoreIdx; loopIdx < copyCoreLoops; loopIdx += copyAicoreNum) {
                        auto blockOffset = copyGatherAScheduler.GetBlockOffset(loopIdx);
                        auto actualBlockShape = copyGatherAScheduler.GetActualBlockShapeByOffset(blockOffset).GetCoordInRank();

                        auto rowOffsetSrc = Catlass::MakeCoord<int>(copyStageId, blockOffset.rank(), blockOffset.row());
                        auto rowOffsetDst = Catlass::MakeCoord<int>(blockOffset.rank(), copyCommIdx, blockOffset.row());
                        MatrixCoord offsetSrc{
                            static_cast<uint32_t>(layoutSymmetricRow(rowOffsetSrc)), blockOffset.column()
                        };
                        MatrixCoord offsetDst{
                            static_cast<uint32_t>(layoutGatherARow(rowOffsetDst)), blockOffset.column()
                        };

                        auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualBlockShape);

                        auto gmBlockDst = gmGatherA[params.layoutGatherA.GetOffset(offsetDst)];
                        auto layoutBlockDst = params.layoutGatherA.GetTileLayout(actualBlockShape);

                        copyGatherA(
                            gmBlockSrc, layoutBlockSrc,
                            gmBlockDst, layoutBlockDst,
                            actualBlockShape
                        );
                    }
                    copyGatherA.FinalizeBlockLoop();
                }
            }

            // BlockAllGather is completed, waiting until tasks on all devices are complete.
            aclshmemx_barrier_all_vec();

            // set aic
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);

            copyCommIdx = commIdx;
            copyStageId = stageId;
            copyActualShape = actualCommShape;
        }
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

} // namespace Catccos::Gemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_HPP
