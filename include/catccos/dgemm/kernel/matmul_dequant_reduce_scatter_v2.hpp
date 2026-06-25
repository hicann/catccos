/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_MATMUL_DEQUANT_REDUCE_SCATTER_V2_HPP
#define CATCCOS_DGEMM_KERNEL_MATMUL_DEQUANT_REDUCE_SCATTER_V2_HPP

#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"

#include "catlass/arch/resource.hpp"
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
    class LocalCopyBlock_,
    class BlockEpilogueDequant_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MatmulDequantReduceScatterV2 {
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
    using ElementBias = typename BlockMmad::ElementBias;
    using LayoutBias = typename BlockMmad::LayoutBias;

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;
    using LocalCopyBlock = LocalCopyBlock_;
    using LocalCopyParams = typename LocalCopyBlock::Params;

    using Dequant = BlockEpilogueDequant_;
    using DequantParams = typename Dequant::Params;
    using ElementD = typename Dequant::ElementD;
    using LayoutD = typename Dequant::LayoutD;
    using LayoutWorkspace = layout::DistRowMajor;
    using LayoutSymmetric = Catlass::layout::RowMajor;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockCommSchedulerParams = typename BlockCommScheduler::Params;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    struct Params {
        DistGemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementBias *ptrBias;
        LayoutBias layoutBias;
        __gm__ ElementC *ptrWorkspace;
        LayoutWorkspace layoutWorkspace;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrSymmetric;
        LayoutSymmetric layoutSymmetric;

        BlockCommParams blockCommParams;
        LocalCopyParams localCopyParams;
        DequantParams dequantParams;
        BlockCommSchedulerParams blockCommSchedulerParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            DistGemmCoord const &problemShape_, uint32_t rankIdx_, uint32_t rankSize_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrBias_, LayoutBias const &layoutBias_,
            GM_ADDR ptrWorkspace_, LayoutWorkspace const &layoutWorkspace_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_, LayoutSymmetric const &layoutSymmetric_,
            BlockCommParams const &blockCommParams_,
            LocalCopyParams const &localCopyParams_,
            DequantParams const &dequantParams_,
            BlockCommSchedulerParams const &blockCommSchedulerParams_
        ) : problemShape(problemShape_), rankIdx(rankIdx_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrBias(reinterpret_cast<__gm__ ElementBias *>(ptrBias_)), layoutBias(layoutBias_),
            ptrWorkspace(reinterpret_cast<__gm__ ElementC *>(ptrWorkspace_)), layoutWorkspace(layoutWorkspace_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_), layoutSymmetric(layoutSymmetric_),
            blockCommParams(blockCommParams_),
            localCopyParams(localCopyParams_),
            dequantParams(dequantParams_),
            blockCommSchedulerParams(blockCommSchedulerParams_)
        {
        }
    };

    struct Arguments {
        GemmCoord problemShape;  // m (total, not per-rank), n, k
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        uint32_t blockNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrScaleX1;
        GM_ADDR ptrScaleX2;
        GM_ADDR ptrBias;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrD;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static size_t GetWorkspaceSize(Arguments const &args)
    {
        uint32_t blockPerCommInRank = args.blockNum * args.commInterval / args.rankSize;
        LayoutWorkspace layoutWorkspace{
            blockPerCommInRank * L1TileShape::M, L1TileShape::N, args.rankSize};
        return WORKSPACE_STAGES * layout::Capacity(layoutWorkspace) * sizeof(ElementC);
    }

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutBias layoutBias{args.problemShape.n()};
        LayoutD layoutD{args.problemShape.m() / args.rankSize, args.problemShape.n()};
        uint32_t blockPerCommInRank = args.blockNum * args.commInterval / args.rankSize;
        LayoutWorkspace layoutWorkspace{blockPerCommInRank * L1TileShape::M, L1TileShape::N, args.rankSize};
        LayoutSymmetric layoutSymmetric{args.problemShape.m() / args.rankSize, args.problemShape.n()};

        DistGemmCoord distProblemShape{
            args.problemShape.m() / args.rankSize,
            args.problemShape.n(),
            args.problemShape.k(),
            args.rankSize
        };

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        typename LocalCopyBlock::TileRemoteCopy::Params localCopyTileParams{args.commTileShape};
        LocalCopyParams localCopyParams{args.commBlockShape, localCopyTileParams};
        DequantParams dequantParams{
            reinterpret_cast<__gm__ float *>(args.ptrScaleX2), Catlass::layout::VectorLayout(args.problemShape.n()),
            reinterpret_cast<__gm__ float *>(args.ptrScaleX1), Catlass::layout::VectorLayout(args.problemShape.m())
        };
        BlockCommSchedulerParams blockCommSchedulerParams{args.commCoreSplit};

        return Params{
            distProblemShape,
            args.rankIdx, args.rankSize,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrBias, layoutBias,
            args.ptrWorkspace, layoutWorkspace,
            args.ptrD, layoutD,
            args.ptrSymmetric, layoutSymmetric,
            blockCommParams,
            localCopyParams,
            dequantParams,
            blockCommSchedulerParams
        };
    }

    CATLASS_DEVICE
    MatmulDequantReduceScatterV2()
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
    ~MatmulDequantReduceScatterV2()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        GemmCoord blockShape = L1TileShape::ToCoord();
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK();
        BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShape.GetCoordMN());
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer(params.ptrBias);

        AscendC::GlobalTensor<ElementC> gmWorkspaceList[WORKSPACE_STAGES];
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmWorkspaceList[stageIdx].SetGlobalBuffer(
                params.ptrWorkspace + stageIdx * layout::Capacity(params.layoutWorkspace));
        }

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmWorkspace = gmWorkspaceList[stageIdx];

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ?
                (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            for (uint32_t blockIdxInComm = aicoreIdx;
                 blockIdxInComm < actualBlockPerComm;
                 blockIdxInComm += aicoreNum) {
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                uint32_t blockIdxInRank = blockIdxInComm - targetRankIdx * actualBlockPerCommInRank;
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInRank;

                GemmCoord blockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);

                GemmCoord offsetCoord = blockCoord * blockShape;
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();

                auto gmBlockA = gmA[params.layoutA.GetOffset(blockOffsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(blockOffsetB)];
                int64_t offsetBias = params.layoutBias.GetOffset(Catlass::MakeCoord(blockOffsetB[1]));
                auto gmBlockBias = (gmBias.GetPhyAddr() != nullptr) ? gmBias[offsetBias] : gmBias;

                auto blockOffsetWorkspace = DistMatrixCoord(blockIdxInRank * L1TileShape::M, 0, targetRankIdx);
                auto gmBlockC = gmWorkspace[params.layoutWorkspace.GetOffset(blockOffsetWorkspace)];
                auto layoutBlockC = params.layoutWorkspace.GetTileLayout(actualBlockShape.GetCoordMN());

                blockMmad(
                    gmBlockA, params.layoutA,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutBlockC,
                    gmBlockBias,
                    actualBlockShape
                );
            }
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t blockPerCommInRank = AscendC::GetBlockNum() * params.commInterval / params.rankSize;

        Dequant dequantEpilogue(resource, params.dequantParams);

        AscendC::GlobalTensor<ElementC> gmWorkspaceList[WORKSPACE_STAGES];
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmWorkspaceList[stageIdx].SetGlobalBuffer(
                params.ptrWorkspace + stageIdx * layout::Capacity(params.layoutWorkspace));
        }

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.blockCommSchedulerParams.CoreSplit();

        BlockCommScheduler commScheduler(commBlockShape, commCoreSplit,
            params.rankSize, params.rankIdx, blockPerCommInRank,
            {params.problemShape, L1TileShape::ToCoordMN()});

        uint32_t commLoops = commScheduler.GetCommLoops();

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmWorkspace = gmWorkspaceList[stageIdx];
            auto remapperDst = commScheduler.GetRemapperDst(commIdx);

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            for (auto iter = commScheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = commScheduler.GetBlockOffset(iter.taskIdx);
                uint32_t targetRankIdx = blockOffset.remote();

                remapperDst.UpdateMmadContext(blockOffset);

                auto blockOffsetSrc = DistMatrixCoord(blockOffset[0], blockOffset[1], targetRankIdx);
                auto blockOffsetDst = remapperDst(blockOffset);
                auto actualCommBlockShape = commScheduler.RemapActualBlockShape(blockOffset, remapperDst);

                auto gmBlockC = gmWorkspace[params.layoutWorkspace.GetOffset(blockOffsetSrc)];
                auto layoutBlockC = params.layoutWorkspace.GetTileLayout(actualCommBlockShape);

                auto ptrRemoteSymmetric = aclshmem_ptr(params.ptrSymmetric, targetRankIdx);
                AscendC::GlobalTensor<ElementD> gmSymmetric;
                gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(ptrRemoteSymmetric));

                auto gmBlockSymmetric = gmSymmetric[params.layoutSymmetric.GetOffset(blockOffsetDst)];
                auto layoutBlockSymmetric = params.layoutSymmetric.GetTileLayout(actualCommBlockShape);

                MatrixCoord targetRankOffset{params.problemShape.m() * targetRankIdx, 0};
                dequantEpilogue(
                    blockOffsetDst + targetRankOffset,
                    actualCommBlockShape,
                    gmBlockC, layoutBlockC,
                    gmBlockSymmetric, layoutBlockSymmetric
                );
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmemx_barrier_all_vec();

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            if (commIdx < commLoops - WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }

        aclshmemx_barrier_all_vec();

        // ========= 搬运shmem到C矩阵 start =========
        if (params.ptrD != nullptr) {
            AscendC::GlobalTensor<ElementD> gmD;
            gmD.SetGlobalBuffer(params.ptrD);
            AscendC::GlobalTensor<ElementD> gmSymmetric;
            gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrSymmetric));

            uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            uint32_t coreNum = get_block_num() * get_subblockdim();
            uint32_t rowsToCopy = params.problemShape.m() / coreNum;
            uint32_t rowOffset = rowsToCopy * coreIdx;
            if (coreIdx == coreNum - 1) {
                rowsToCopy += params.problemShape.m() % coreNum;
            }
            if (rowsToCopy > 0) {
                MatrixCoord blockOffsetSrc{rowOffset, 0};

                auto blockSrc = gmSymmetric[params.layoutSymmetric.GetOffset(blockOffsetSrc)];
                auto blockDst = gmD[params.layoutD.GetOffset(blockOffsetSrc)];

                auto actualBlockShape = Catlass::MakeCoord<uint32_t>(rowsToCopy, params.problemShape.n());

                auto layoutBlockSrc = params.layoutSymmetric.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = params.layoutD.GetTileLayout(actualBlockShape);

                LocalCopyBlock localCopyBlockEpilogue(resource, params.localCopyParams);
                localCopyBlockEpilogue.InitBlockLoop();
                localCopyBlockEpilogue(
                    blockSrc,
                    layoutBlockSrc,
                    blockDst,
                    layoutBlockDst,
                    actualBlockShape
                );
                localCopyBlockEpilogue.FinalizeBlockLoop();
            }
            aclshmemx_barrier_all_vec();
        }
        // ========= 搬运shmem到C矩阵 end =========
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCCOS_DGEMM_KERNEL_MATMUL_DEQUANT_REDUCE_SCATTER_V2_HPP
