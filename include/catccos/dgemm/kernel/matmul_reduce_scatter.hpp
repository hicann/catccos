/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_HPP
#define CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_HPP

#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class BlockMmad_,
    class BlockReduceScatter_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MatmulReduceScatter {
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

    using BlockReduceScatter = BlockReduceScatter_;
    using BlockReduceScatterParams = typename BlockReduceScatter::Params;

    using ElementD = typename BlockReduceScatter::ElementDst;
    using LayoutD = typename BlockReduceScatter::LayoutDst;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockCommParams = typename BlockCommScheduler::Params;

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
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrSymmetric;

        BlockReduceScatterParams reduceScatterParams;
        BlockCommParams commParams;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            DistGemmCoord const &problemShape_, uint32_t rankIdx_, uint32_t rankSize_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_,
            BlockReduceScatterParams const &reduceScatterParams_,
            BlockCommParams const &commParams_
        ) : problemShape(problemShape_), rankIdx(rankIdx_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_),
            commParams(commParams_)
        {
        }
    };

    CATLASS_DEVICE
    MatmulReduceScatter()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
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
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto layoutSymmetric = layout::DistRowMajor(blockPerCommInRank * L1TileShape::M, L1TileShape::N, params.rankSize);
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }

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

                AscendC::GlobalTensor<ElementC> gmBlockC;
                Catlass::layout::RowMajor layoutC;
                if (targetRankIdx == params.rankIdx) {
                    MatrixCoord blockOffsetD = offsetCoord.GetCoordMN();
                    gmBlockC = gmD[params.layoutD.GetOffset(blockOffsetD)];
                    layoutC = params.layoutD;
                } else {
                    auto blockOffsetSymm = DistMatrixCoord(blockIdxInRank * L1TileShape::M, 0, targetRankIdx);
                    gmBlockC = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetSymm)];
                    layoutC = layoutSymmetric.GetTileLayout(actualBlockShape.GetCoordMN());
                }

                blockMmad(
                    gmBlockA, params.layoutA,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape
                );
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t blockPerCommInRank = aicoreNum * params.commInterval / params.rankSize;
        
        BlockReduceScatter reduceScatter(resource, params.reduceScatterParams);

        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto layoutSymmetric = layout::DistRowMajor(
            blockPerCommInRank * L1TileShape::M, L1TileShape::N, params.rankSize);
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        
        BlockCommScheduler commScheduler(commBlockShape, commCoreSplit, 
            params.rankSize, params.rankIdx, blockPerCommInRank,
            {params.problemShape, L1TileShape::ToCoordMN()});
        
        uint32_t commLoops = commScheduler.GetCommLoops();
        
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];
            auto remapperDst = commScheduler.GetRemapperDst(commIdx);

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            reduceScatter.InitBlockLoop();
            for (auto iter = commScheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = commScheduler.GetBlockOffset(iter.taskIdx);

                uint32_t remoteRankIdx = blockOffset.remote();
                if (remoteRankIdx == params.rankIdx) {
                    continue;
                }

                remapperDst.UpdateMmadContext(blockOffset);

                auto blockOffsetSrc = blockOffset.GetLocalCoord();
                auto blockOffsetDst = remapperDst(blockOffset);

                auto actualCommBlockShape = commScheduler.RemapActualBlockShape(blockOffset, remapperDst);
                
                auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetSrc)];
                auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                auto gmBlockDst = gmD[params.layoutD.GetOffset(blockOffsetDst)];
                auto layoutBlockDst = params.layoutD.GetTileLayout(actualCommBlockShape);

                reduceScatter(
                    gmBlockSrc, layoutBlockSrc,
                    gmBlockDst, layoutBlockDst,
                    actualCommBlockShape, remoteRankIdx % params.rankSize
                );
            }
            reduceScatter.FinalizeBlockLoop();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmemx_barrier_all_vec();

            if (commIdx < commLoops - WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_HPP
