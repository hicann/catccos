/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_BIAS_HPP
#define CATCCOS_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_BIAS_HPP

#include "catccos/catccos.hpp"
#include "catccos/dgemm/block/block_mmad_pingpong_bias_fixpipe.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadSchedulerForAllgather_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_>
class AllGatherDequantMatmulBias {
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

    using ElementBias = typename BlockMmad::ElementBias; // Get Bias type from BlockMmad
    using LayoutBias = typename BlockMmad::LayoutBias;
	
    using BlockComm = BlockComm_;

    using ElementScale = uint64_t;
    using LayoutScale = Catlass::layout::VectorLayout;

    using BlockCommParams = typename BlockComm::Params;

    using BlockMmadScheduler = BlockMmadSchedulerForAllgather_;
    using CommScheduler = BlockCommScheduler_;
    using CommSchedulerParams = typename CommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutB layoutC;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
		GM_ADDR ptrBias; LayoutBias layoutBias; // Global memory address and layout of Bias
		
        GM_ADDR ptrSymmetric;
        BlockCommParams blockCommParams;
        CommSchedulerParams commSchedulerParams;
        uint32_t commInterval;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrScale_, LayoutScale const &layoutScale_,
			GM_ADDR ptrBias_, LayoutBias const &layoutBias_, 
            GM_ADDR ptrSymmetric_,
            BlockCommParams const &blockCommParams_,
            CommSchedulerParams const &commSchedulerParams_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrScale(ptrScale_), layoutScale(layoutScale_),
			ptrBias(ptrBias_), layoutBias(layoutBias_), 
            ptrSymmetric(ptrSymmetric_),
            blockCommParams(blockCommParams_),
            commSchedulerParams(commSchedulerParams_),
            commInterval(commInterval_)
        {}
    };

    /// User-facing arguments (host-constructible)
    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrScale;
        GM_ADDR ptrBias;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutC layoutC{m * args.rankSize, n};
        LayoutScale layoutScale{n};
        LayoutBias layoutBias{n};

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        CommSchedulerParams commSchedulerParams{args.commCoreSplit};

        return Params(
            args.problemShape,
            args.rankIdx, args.rankSize,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            args.ptrScale, layoutScale,
            args.ptrBias, layoutBias,
            args.ptrSymmetric,
            blockCommParams,
            commSchedulerParams,
            args.commInterval
        );
    };

    // Methods
    CATLASS_DEVICE
    AllGatherDequantMatmulBias()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);    // 将id设置为0,1... (WORKSPACE_STAGES-1)
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i);
        }
    }

    CATLASS_DEVICE
    ~AllGatherDequantMatmulBias()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
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

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrScale));

        AscendC::GlobalTensor<ElementBias> gmBias; 
        gmBias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias *>(params.ptrBias));
		
        auto layoutSymmetric = Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize);
            BlockMmadScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

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
                auto tmpOffsetB = params.layoutB.GetOffset(offsetB);
                auto gmBlockB = gmB[tmpOffsetB];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];
                int64_t offsetScale = blockOffset.n();

                int64_t offsetBias = params.layoutBias.GetOffset(Catlass::MakeCoord(offsetB[1]));
                mmad(gmBlockA,
                    layoutSymmetric,
                    gmBlockB,
                    params.layoutB,
                    gmBlockC,
                    layoutC,
                    gmScale[offsetScale], params.layoutScale,
                    gmBias[offsetBias],
                    actualBlockShape.GetCoordMNK());
            }
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMK = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockComm blockRemoteCopy(resource, params.blockCommParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));

        auto layoutSymmetric = Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.commSchedulerParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
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

            blockRemoteCopy.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
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

                    blockRemoteCopy(gmBlockSrc,
                        layoutBlockSrc,
                        gmBlockDst,
                        layoutBlockDst,
                        actualCommBlockShape,
                        remoteRankIdx % params.rankSize);
                }
            }
            blockRemoteCopy.FinalizeBlockLoop();
            // AllGather is completed, waiting until tasks on all devices are complete.
            aclshmemx_barrier_all_vec();

            // set aic
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
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

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCCOS_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_BIAS_HPP
