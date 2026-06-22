/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_REMOTE_READ_LOCAL_MM_OPT_HPP
#define CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_REMOTE_READ_LOCAL_MM_OPT_HPP

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
    class BlockLocalCopy_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmulWithRemoteReadLocalMmOpt {
public:
    using BlockMmad = BlockMmad_;
    using BlockLocalCopy = BlockLocalCopy_;
    using BlockLocalCopyParams = typename BlockLocalCopy::Params;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using CommScheduler = BlockCommScheduler_;
    using CommSchedulerParams = typename CommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    struct Params {
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;

        __gm__ ElementB *ptrB;
        LayoutB layoutB;

        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        GM_ADDR ptrSymmetric;

        BlockLocalCopyParams localCopyParams;
        BlockCommParams blockCommParams;
        CommSchedulerParams commSchedulerParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_,
            BlockLocalCopyParams const &localCopyParams_,
            BlockCommParams const &blockCommParams_,
            CommSchedulerParams const &commSchedulerParams_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_),
            localCopyParams(localCopyParams_),
            blockCommParams(blockCommParams_),
            commSchedulerParams(commSchedulerParams_)
        {
        }
    };

    /// User API arguments
    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static size_t GetWorkspaceSize(Arguments const &args) { return 0; }

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m() * args.rankSize, args.problemShape.n(), args.problemShape.n()};

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        typename BlockLocalCopy::TileRemoteCopy::Params localTileParams{args.commTileShape};
        BlockLocalCopyParams localCopyParams{args.commBlockShape, localTileParams};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        CommSchedulerParams commSchedulerParams{args.commCoreSplit};

        return Params{
            args.problemShape,
            args.rankIdx, args.rankSize,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            args.ptrSymmetric,
            localCopyParams,
            blockCommParams,
            commSchedulerParams
        };
    }

    CATLASS_DEVICE
    AllGatherMatmulWithRemoteReadLocalMmOpt()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~AllGatherMatmulWithRemoteReadLocalMmOpt()
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

        auto layoutAStage = Catlass::layout::RowMajor(
            params.problemShape.m(), params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        GM_ADDR ptrGatheredA = params.ptrSymmetric + layout::Capacity(layoutAStage) * sizeof(ElementA);

        AscendC::GlobalTensor<ElementA> gmALocal;
        if (params.ptrA != nullptr) {
            gmALocal.SetGlobalBuffer(params.ptrA);
        } else {
            gmALocal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        }
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(ptrGatheredA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        //// Local matmul
        auto localProblemShape = Catlass::MakeCoord<uint32_t>(
            params.problemShape.m(), params.problemShape.n(), params.problemShape.k(), 1
        );
        BlockMmadScheduler localMmadScheduler(localProblemShape, blockShape.GetCoordMN());
        uint32_t localCoreLoops = localMmadScheduler.GetCoreLoops();

        for (uint32_t loopIdx = aicoreIdx; loopIdx < localCoreLoops; loopIdx += aicoreNum) {
            auto blockOffset = localMmadScheduler.GetBlockOffset(loopIdx);
            auto actualBlockShape = localMmadScheduler.GetActualBlockShapeByOffset(blockOffset);

            MatrixCoord blockOffsetA = blockOffset.GetCoordMK();
            MatrixCoord blockOffsetB = blockOffset.GetCoordKN();
            MatrixCoord rankOffsetC =
                params.problemShape.GetCoordMN() * Catlass::MakeCoord<uint32_t>(params.rankIdx, 0);
            MatrixCoord blockOffsetC = rankOffsetC + blockOffset.GetCoordMN();

            auto gmBlockB = gmB[params.layoutB.GetOffset(blockOffsetB)];
            auto gmBlockC = gmC[params.layoutC.GetOffset(blockOffsetC)];

            if (params.ptrA != nullptr) {
                auto gmBlockA = gmALocal[params.layoutA.GetOffset(blockOffsetA)];

                mmad(
                    gmBlockA, params.layoutA,
                    gmBlockB, params.layoutB,
                    gmBlockC, params.layoutC,
                    actualBlockShape.GetCoordMNK());
            } else {
                auto gmBlockA = gmALocal[layoutAStage.GetOffset(blockOffsetA)];
                mmad(
                    gmBlockA, layoutAStage,
                    gmBlockB, params.layoutB,
                    gmBlockC, params.layoutC,
                    actualBlockShape.GetCoordMNK());
            }
        }

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        //// Remote matmul
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize - 1
            );
            BlockMmadScheduler mmScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmScheduler.GetCoreLoops();

            // Wait until AIV has gathered remote-rank A windows into gmSymmetric.
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                if (srcRankIdx >= params.rankIdx) {
                    srcRankIdx += 1;
                }

                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                auto offsetA = commOffsetA + blockOffset.GetCoordMK();
                auto offsetB = blockOffset.GetCoordKN();
                auto offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(offsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(offsetB)];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];

                mmad(
                    gmBlockA, layoutSymmetric,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape.GetCoordMNK());
            }

            // Notify AIV that this workspace stage can be reused.
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
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMK = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockLocalCopy localCopy(resource, params.localCopyParams);
        BlockComm blockRemoteGet(resource, params.blockCommParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementA> gmAStage;
        AscendC::GlobalTensor<ElementA> gmSymmetric;

        auto layoutAStage = Catlass::layout::RowMajor(
            params.problemShape.m(), params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        GM_ADDR ptrGatheredA = params.ptrSymmetric + layout::Capacity(layoutAStage) * sizeof(ElementA);
        gmAStage.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(ptrGatheredA));

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.commSchedulerParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);

        // Wait before touching the shmem A-stage region. Without this cross-rank
        // synchronization, a fast rank may overwrite its A stage for the next
        // launch while a slow rank is still remote-reading the previous launch.
        aclshmemx_barrier_all_vec();

        // When ptrA != nullptr, stage full A into the first region of ptrSymmetric.
        // When ptrA == nullptr, the caller has already placed A in that shmem A-stage region.
        if (params.ptrA != nullptr) {
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV_LOCAL_COPY);
#endif
            uint32_t copyCoreIdx = get_block_idx() + get_subblockid() * get_block_num();
            uint32_t copyCoreNum = get_block_num() * get_subblockdim();
            copyCoreNum = (copyCoreNum == 0) ? 1 : copyCoreNum;

            uint32_t rowsToCopy = params.problemShape.m() / copyCoreNum;
            uint32_t rowOffset = rowsToCopy * copyCoreIdx;
            if (copyCoreIdx == copyCoreNum - 1) {
                rowsToCopy += params.problemShape.m() % copyCoreNum;
            }

            if (rowsToCopy > 0) {
                MatrixCoord copyOffset{rowOffset, 0};
                auto actualCopyShape = Catlass::MakeCoord<uint32_t>(rowsToCopy, params.problemShape.k());

                auto gmBlockSrc = gmA[params.layoutA.GetOffset(copyOffset)];
                auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCopyShape);

                auto gmBlockDst = gmAStage[layoutAStage.GetOffset(copyOffset)];
                auto layoutBlockDst = layoutAStage.GetTileLayout(actualCopyShape);

                localCopy.InitBlockLoop();
                localCopy(
                    gmBlockSrc, layoutBlockSrc,
                    gmBlockDst, layoutBlockDst,
                    actualCopyShape
                );
                localCopy.FinalizeBlockLoop();
            }

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV_LOCAL_COPY);
#endif

            // Publish the newly staged A to all ranks before remote get reads it.
            // The ptrA == nullptr path does not write A stage in this kernel, so
            // it only needs the entry barrier above.
            aclshmemx_barrier_all_vec();
        }

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            // Wait AIC before reusing the same workspace stage.
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            //// Remote get A from shmem staging
            // Only synchronize AIV cores within this NPU before issuing remote gets.
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            blockRemoteGet.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (
                    uint32_t commLoopIdx = aicoreIdx;
                    commLoopIdx < commCoreLoops;
                    commLoopIdx += commAicoreNum
                ) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t srcRankIdx = commBlockCoord.rank();
                    // Self rank has already been handled by Local matmul.
                    if (srcRankIdx == params.rankIdx) {
                        continue;
                    }

                    auto offsetSrc = MatrixCoord{commIdx * commSizeM, 0} + blockOffsetInRank;
                    MatrixCoord commDstOffset{
                        layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0
                    };
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmAStage[layoutAStage.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = layoutAStage.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    blockRemoteGet(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualCommBlockShape,
                        srcRankIdx % params.rankSize
                    );
                }
            }
            blockRemoteGet.FinalizeBlockLoop();

            // Wait until all local AIV cores finish filling their parts of gmSymmetric.
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

            // Notify AIC that gmSymmetric holds the current remote-rank A windows.
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_REMOTE_READ_LOCAL_MM_OPT_HPP
