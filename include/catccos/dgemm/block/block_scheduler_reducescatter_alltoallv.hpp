/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::DGemm::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <class MoeConstraints_ = DefaultMoeConstraints>
struct BlockMmadSchedulerReduceScatterAllToAllV {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;

    struct CommContext {
        uint32_t blockNumList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t blockOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t inGroupOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
    };

    struct LocalExpertContext {
        GemmCoord *ptrProblemShape;
        GemmCoord *ptrBlockGrid;
        uint64_t *ptrFinishedLoops;
        
        uint32_t localExpertIdx;
        uint32_t dstRankIdx;
        uint32_t blockAccumList[MoeConstraints::RANK_SIZE_LIMIT + 1] = {0};
        uint32_t blockNum;
        
        const GemmCoord& GetProblemShape() const { return ptrProblemShape[dstRankIdx]; }
        const GemmCoord& GetBlockGrid() const { return ptrBlockGrid[dstRankIdx]; }
        uint32_t GetFinishedLoops() const { return ptrFinishedLoops[dstRankIdx]; }

        void UpdateDstRank(uint32_t taskIdx) {
            while (taskIdx >= blockAccumList[dstRankIdx + 1]) {
                dstRankIdx++;
            }
        }
        
        uint32_t GetBlockIdxInRank(uint32_t taskIdx) const {
            return taskIdx - blockAccumList[dstRankIdx];
        }
    };

    ProblemShape problemShape;
    
    uint32_t commLoops;
    GemmCoord blockShape;
    uint32_t blockPerCommInRank;

    uint32_t outputSplitBlock[MoeConstraints::RANK_SIZE_LIMIT] = {0};
    uint32_t outputSplitBlockInLocalExpert[MoeConstraints::RANK_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT] = {0};
    uint32_t outputOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT] = {0};

    GemmCoord mmadProblemShapes[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT];
    GemmCoord mmadBlockGrids[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT];

    CommContext commContext{};
    LocalExpertContext localExpertCtx{};

    CATLASS_DEVICE
    BlockMmadSchedulerReduceScatterAllToAllV() = default;

    CATLASS_DEVICE
    BlockMmadSchedulerReduceScatterAllToAllV(
        ProblemShape const &problemShape_,
        uint32_t blockPerCommInRank_,
        MatrixCoord const &blockShapeMN_
    ) : problemShape(problemShape_), blockPerCommInRank(blockPerCommInRank_)
    {
        blockShape = {blockShapeMN_.row(), blockShapeMN_.column(), problemShape.k()};
        commLoops = 0;

        uint32_t nLoops = CeilDiv(problemShape.n(), blockShape.n());

        uint32_t maxInputSplit = 0;
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            uint32_t inputSplit = 0;
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                inputSplit += CeilDiv(problemShape.localTokensPerExpert(epIdx, localExpertIdx), blockShape.m());
            }
            maxInputSplit = Max(maxInputSplit, inputSplit);
        }
        commLoops = Max(commLoops, CeilDiv(maxInputSplit * nLoops, blockPerCommInRank));

        uint32_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
                uint32_t tokens = problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx);
                outputSplitBlockInLocalExpert[rankIdx][localExpertIdx] = CeilDiv(tokens, blockShape.m()) * nLoops;
                outputSplitBlock[rankIdx] += outputSplitBlockInLocalExpert[rankIdx][localExpertIdx];
                outputOffsetList[localExpertIdx][rankIdx] = outputOffset; // offset in matrix A
                outputOffset += tokens;

                mmadProblemShapes[localExpertIdx][rankIdx] = {tokens, problemShape.n(), problemShape.k()};
                mmadBlockGrids[localExpertIdx][rankIdx] = CeilDiv(mmadProblemShapes[localExpertIdx][rankIdx], blockShape);
            }
        }

        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            auto sendLoops = CeilDiv(outputSplitBlock[rankIdx], blockPerCommInRank);
            commLoops = Max(commLoops, sendLoops);
        }
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const
    {
        return commLoops;
    }

    CATLASS_DEVICE
    void UpdateCommContext(uint32_t commIdx)
    {
        auto commOffset = commIdx * blockPerCommInRank;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            auto actualCommBlocks = Min(blockPerCommInRank, ClipSub(outputSplitBlock[dstRankIdx], commOffset));

            uint32_t totalBlocks{0};
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                auto blockNumInLastGroup = commContext.blockNumList[localExpertIdx][dstRankIdx];
                commContext.inGroupOffsetList[localExpertIdx][dstRankIdx] += blockNumInLastGroup;
                
                auto outputSplitBlocks = outputSplitBlockInLocalExpert[dstRankIdx][localExpertIdx];
                auto residueBlocks = outputSplitBlocks - commContext.inGroupOffsetList[localExpertIdx][dstRankIdx];
                
                uint32_t blocks = Min<uint64_t>(actualCommBlocks - totalBlocks, residueBlocks);
                commContext.blockNumList[localExpertIdx][dstRankIdx] = blocks;
                commContext.blockOffsetList[localExpertIdx][dstRankIdx] = totalBlocks * blockShape.m();
                totalBlocks += blocks;
            }
        }
    }

    CATLASS_DEVICE
    void UpdateLocalExpertContext(uint32_t localExpertIdx)
    {
        localExpertCtx.ptrProblemShape = mmadProblemShapes[localExpertIdx];
        localExpertCtx.ptrBlockGrid = mmadBlockGrids[localExpertIdx];
        localExpertCtx.ptrFinishedLoops = commContext.inGroupOffsetList[localExpertIdx];
        localExpertCtx.localExpertIdx = localExpertIdx;
        localExpertCtx.dstRankIdx = 0;
    
        uint32_t blockAccum = 0;
        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            blockAccum += commContext.blockNumList[localExpertIdx][rankIdx];
            localExpertCtx.blockAccumList[rankIdx + 1] = blockAccum;
        }
        localExpertCtx.blockNum = blockAccum;
    }

    uint32_t startTask = AscendC::GetBlockIdx();
    uint32_t blockIdxInRank;

    struct Iter {
        using Scheduler = BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;
        Scheduler *const scheduler;
        uint32_t taskIdx;
        uint32_t coreLoops;

        CATLASS_DEVICE
        void Next() {
            taskIdx += AscendC::GetBlockNum();
        }

        CATLASS_DEVICE
        bool End() {
            if (taskIdx >= coreLoops) {
                scheduler->startTask = taskIdx - coreLoops;
            }
            return taskIdx >= coreLoops;
        }
    };

    CATLASS_DEVICE
    Iter Begin()
    {
        uint32_t taskIdx = startTask;
        uint32_t coreLoops = localExpertCtx.blockNum;
        return Iter{this, taskIdx, coreLoops};
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockOffset(Iter const &iter)
    {
        localExpertCtx.UpdateDstRank(iter.taskIdx);
        blockIdxInRank = localExpertCtx.GetBlockIdxInRank(iter.taskIdx);
        auto blockCoordMN = MatrixSwizzle<7, 1>::GetCoord(
            localExpertCtx.GetBlockGrid().GetCoordMN(), blockIdxInRank + localExpertCtx.GetFinishedLoops());
        return Catlass::MakeCoord(blockCoordMN.row() * blockShape.m(), 
                                  blockCoordMN.column() * blockShape.n(), 0U, localExpertCtx.dstRankIdx);
    }

    struct RemapperA {
        using Scheduler = BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;
        const Scheduler *scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(DistGemmCoord const &blockOffset) const
        {
            auto tokenOffset = scheduler->outputOffsetList[localExpertIdx][blockOffset.rank()];
            return blockOffset.GetCoordMK() + Catlass::MakeCoord<uint32_t>(tokenOffset, 0);
        }

        CATLASS_DEVICE
        GemmCoord GetResidueShape(GemmCoord const &blockOffset) const
        {
            return ClipSub(scheduler->localExpertCtx.GetProblemShape(), blockOffset);
        }
    };

    CATLASS_DEVICE
    const RemapperA GetRemapperA(uint32_t commIdx, uint32_t localExpertIdx) const
    {
        return {this, commIdx, localExpertIdx};
    }

    struct RemapperC {
        using Scheduler = BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;
        const Scheduler *scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;

        CATLASS_DEVICE
        DistMatrixCoord operator()(DistGemmCoord const &blockOffset) const
        {
            auto outputOffset = scheduler->commContext.blockOffsetList[localExpertIdx][blockOffset.rank()];
            MatrixCoord localOffset = Catlass::MakeCoord<uint32_t>(
                scheduler->blockIdxInRank * scheduler->blockShape.m() + outputOffset, 0);
            return {localOffset, blockOffset.rank()};
        }
    };

    CATLASS_DEVICE
    const RemapperC GetRemapperC(uint32_t commIdx, uint32_t localExpertIdx) const
    {
        return {this, commIdx, localExpertIdx};
    }

    CATLASS_DEVICE
    GemmCoord RemapActualBlockShape(GemmCoord const &blockOffset,
        RemapperA const &remapperA, RemapperC const &remapperC) const
    {
        (void)remapperC;
        return Min(blockShape, remapperA.GetResidueShape(blockOffset));
    }
};

}  // namespace Catccos::DGemm::Block

#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV_HPP
