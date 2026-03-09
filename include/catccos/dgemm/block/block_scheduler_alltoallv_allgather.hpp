/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP
#define CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP
 
#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
 
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dist_coord.hpp"
 
namespace Catccos::DGemm::Block {
 
using Catlass::MatrixCoord;
using Catlass::GemmCoord;
 
template <class MoeConstraints_ = DefaultMoeConstraints>
struct BlockMmadSchedulerAllToAllVAllGather {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = AllToAllVAllGatherProblemShape;
 
    struct CommContext {
        uint32_t tokenNumList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t tokenOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t inGroupOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
    };
 
    struct MmadContext {
        GemmCoord problemShape;
        GemmCoord blockGrid;
    };
 
    ProblemShape problemShape;
    uint32_t commShapeM;
    GemmCoord blockShape;
    uint32_t commLoops;
 
    uint32_t outputSplitList[MoeConstraints::RANK_SIZE_LIMIT]{0};
    uint64_t outputOffsetList[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT][MoeConstraints::RANK_SIZE_LIMIT]{0};
 
    CommContext commContext{};
    MmadContext mmadContext{};
 
    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGather() = default;
 
    CATLASS_DEVICE
    BlockMmadSchedulerAllToAllVAllGather(
        ProblemShape const &problemShape_,
        uint32_t commShapeM_,
        MatrixCoord const &blockShapeMN_
    ) : problemShape(problemShape_), commShapeM(commShapeM_)
    {
        blockShape = {blockShapeMN_.row(), blockShapeMN_.column(), problemShape.k()};
        commLoops = 0;
 
        uint64_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
                uint32_t tokens = problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx);
                outputSplitList[rankIdx] += tokens;
                outputOffsetList[localExpertIdx][rankIdx] = outputOffset;
                outputOffset += tokens;
            }
        }
 
        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            // 统计从每个 rank 接收所需的通信轮次
            auto receiveLoops = CeilDiv(outputSplitList[rankIdx], commShapeM_);
            commLoops = Max(commLoops, receiveLoops);
        }
 
 
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            uint32_t inputSplit = 0;
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                inputSplit += problemShape.localTokensPerExpert(epIdx, localExpertIdx);
            }
            // 统计发送到每个 EP 的通信轮次
            auto sendLoops = CeilDiv(inputSplit, commShapeM_);
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
        for (uint32_t srcRankIdx = 0; srcRankIdx < problemShape.rankSize(); ++srcRankIdx) {
            auto actualCommTokens = Min(commShapeM, ClipSub(outputSplitList[srcRankIdx], commIdx * commShapeM));
 
            uint64_t tokenOffset{0};
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                auto tokenNumInLastGroup = commContext.tokenNumList[localExpertIdx][srcRankIdx];
                commContext.inGroupOffsetList[localExpertIdx][srcRankIdx] += tokenNumInLastGroup;
 
                auto globalTokens = problemShape.globalTokensPerLocalExpert(srcRankIdx, localExpertIdx);
                auto residueTokens = globalTokens - commContext.inGroupOffsetList[localExpertIdx][srcRankIdx];
                auto tokens = Min<uint64_t>(actualCommTokens - tokenOffset, residueTokens);
                commContext.tokenNumList[localExpertIdx][srcRankIdx] = tokens;
                commContext.tokenOffsetList[localExpertIdx][srcRankIdx] = tokenOffset;
                tokenOffset += tokens;
            }
        }
    }
 
    CATLASS_DEVICE
    void UpdateMmadContext(uint32_t localExpertIdx, uint32_t srcRankIdx)
    {
        mmadContext.problemShape = {
            commContext.tokenNumList[localExpertIdx][srcRankIdx],
            problemShape.n(),
            problemShape.k()
        };
        mmadContext.blockGrid = CeilDiv(mmadContext.problemShape, blockShape);
    }
 
    uint32_t startTask = AscendC::GetBlockIdx();
 
    struct Iter {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
        Scheduler *const scheduler;
        uint32_t taskIdx;
        uint32_t coreLoops;
 
        CATLASS_DEVICE
        void Next()
        {
            taskIdx += AscendC::GetBlockNum();
        }
 
        CATLASS_DEVICE
        bool End()
        {
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
        uint32_t coreLoops = Numel(mmadContext.blockGrid);
        return Iter{this, taskIdx, coreLoops};
    }
 
    CATLASS_DEVICE
    GemmCoord GetBlockOffset(Iter const &iter) const
    {
        auto blockCoordMN = MatrixSwizzle<1, 0>::GetCoord(mmadContext.blockGrid.GetCoordMN(), iter.taskIdx);
        return Catlass::MakeCoord<uint32_t>(blockCoordMN.row(), blockCoordMN.column(), 0) * blockShape;
    }
 
    struct RemapperA {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
        const Scheduler *scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;
        uint32_t srcRankIdx;
 
        CATLASS_DEVICE
        DistMatrixCoord operator()(GemmCoord const &blockOffset) const
        {
            auto tokenOffset = scheduler->commContext.tokenOffsetList[localExpertIdx][srcRankIdx];
            return {blockOffset.GetCoordMK() + Catlass::MakeCoord<uint32_t>(tokenOffset, 0), srcRankIdx};
        }
 
        CATLASS_DEVICE
        GemmCoord GetResidueShape(GemmCoord const &blockOffset) const
        {
            return ClipSub(scheduler->mmadContext.problemShape, blockOffset);
        }
    };
 
    CATLASS_DEVICE
    const RemapperA GetRemapperA(uint32_t commIdx, uint32_t localExpertIdx, uint32_t srcRankIdx) const
    {
        return {this, commIdx, localExpertIdx, srcRankIdx};
    }
 
    struct RemapperC {
        using Scheduler = BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
        const Scheduler *scheduler;
        uint32_t commIdx;
        uint32_t localExpertIdx;
        uint32_t srcRankIdx;
 
        CATLASS_DEVICE
        MatrixCoord operator()(GemmCoord const &blockOffset) const
        {
            auto outputOffset = scheduler->outputOffsetList[localExpertIdx][srcRankIdx];
            auto inGroupOffset = scheduler->commContext.inGroupOffsetList[localExpertIdx][srcRankIdx];
            return blockOffset.GetCoordMN() + Catlass::MakeCoord<uint32_t>(outputOffset + inGroupOffset, 0);
        }
 
        CATLASS_DEVICE
        GemmCoord GetResidueShape(GemmCoord const &blockOffset) const
        {
            return ClipSub(scheduler->mmadContext.problemShape, blockOffset);
        }
    };
 
    CATLASS_DEVICE
    const RemapperC GetRemapperC(uint32_t commIdx, uint32_t localExpertIdx, uint32_t srcRankIdx) const
    {
        return {this, commIdx, localExpertIdx, srcRankIdx};
    }
 
    CATLASS_DEVICE
    GemmCoord RemapActualBlockShape(GemmCoord const &blockOffset,
        RemapperA const &remapperA, RemapperC const &remapperC) const
    {
        auto remapABlockShape = Min(blockShape, remapperA.GetResidueShape(blockOffset));
        return Min(remapABlockShape, remapperC.GetResidueShape(blockOffset));
    }
};
 
}  // namespace Catccos::DGemm::Block
 
#endif  // CATCCOS_DGEMM_BLOCK_SCHEDULER_ALLTOALLV_ALLGATHER_HPP