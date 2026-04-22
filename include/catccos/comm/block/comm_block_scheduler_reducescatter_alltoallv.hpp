/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV
#define CATCCOS_COMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/symm_coord.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"

namespace Catccos::CommEpilogue::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <class MoeConstraints_ = DGemm::DefaultMoeConstraints,
    bool IsDynamic_ = true, class CoreSplit_ = void, uint32_t SWIZZLE_DIRECTION_ = 0, bool IS_DETERMINISTIC_ = false>
struct BlockCommSchedulerReduceScatterAllToAllV {
    using MoeConstraints = MoeConstraints_;
    using ProblemShape = DGemm::AllToAllVAllGatherProblemShape;
    static constexpr uint32_t SWIZZLE_DIRECTION = SWIZZLE_DIRECTION_;
    static constexpr uint32_t IS_DETERMINISTIC = IS_DETERMINISTIC_;
    static constexpr bool IsDynamic = IsDynamic_;
    using CoreSplit = CoreSplit_;

    static_assert((IS_DETERMINISTIC && SWIZZLE_DIRECTION == 0) || !IS_DETERMINISTIC,
        "Deterministic calculation requires that the swizzle direction be 0.");

    struct CommContext {
        uint32_t commOffset;
        MatrixCoord actualCommShape;
    };

    struct LocalExpertContext {
        uint32_t startLocalExpert{0};
        uint32_t accumBlockNum{0};
        uint32_t mmadIdx;
    };

    ProblemShape problemShape;
    MatrixCoord mmadBlockShape;
    MatrixCoord commBlockShape;
    MatrixCoord coreSplit;
    MatrixCoord blockGrid;

    uint32_t blockPerCommInRank;
    uint32_t srcEpIdx;
    uint32_t commLoops{0};
    uint32_t coreLoops;

    uint32_t inputSplitBlock[MoeConstraints::EP_SIZE_LIMIT] = {0};
    uint32_t inputSplitBlockInLocalExpert[MoeConstraints::EP_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT] = {0};
    uint32_t inputOffsetList[MoeConstraints::EP_SIZE_LIMIT][MoeConstraints::LOCAL_EXPERT_NUM_LIMIT] = {0};

    uint32_t outputSplitBlock[MoeConstraints::RANK_SIZE_LIMIT] = {0};

    MatrixCoord actualMmadShapes[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT];
    MatrixCoord mmadBlockGrids[MoeConstraints::LOCAL_EXPERT_NUM_LIMIT];

    CommContext commContext{};
    LocalExpertContext localExpertCtx{};

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord CoreSplit() { return CoreSplit::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord coreSplit;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord coreSplit_) 
            : coreSplit(coreSplit_) {}

        CATLASS_DEVICE
        MatrixCoord CoreSplit() const { return coreSplit; }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;

    CATLASS_DEVICE
    BlockCommSchedulerReduceScatterAllToAllV() = default;

    CATLASS_DEVICE
    BlockCommSchedulerReduceScatterAllToAllV(
        ProblemShape const &problemShape_,
        uint32_t blockPerCommInRank_,
        uint32_t srcEpIdx_,
        MatrixCoord const &mmadBlockShape_,
        MatrixCoord const &commBlockShape_,
        MatrixCoord const &coreSplit_
    ) : problemShape(problemShape_),
        blockPerCommInRank(blockPerCommInRank_),
        srcEpIdx(srcEpIdx_),
        mmadBlockShape(mmadBlockShape_),
        commBlockShape(commBlockShape_),
        coreSplit(coreSplit_)
    {
        MatrixCoord commShape = Catlass::MakeCoord<uint32_t>(blockPerCommInRank * mmadBlockShape.row(), mmadBlockShape.column());
        blockGrid = CeilDiv(commShape, commBlockShape);
        uint32_t tileCount = blockGrid.row() * blockGrid.column();
        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
            coreLoops = RoundUp<uint32_t>(tileCount, coreSplit.row());
        } else {
            coreLoops = tileCount;
        }

        auto rankSize = problemShape.rankSize();
        uint32_t nLoops = CeilDiv(problemShape.n(), mmadBlockShape.column());

        uint32_t inputOffset = 0;
        for (uint32_t epIdx = 0; epIdx < problemShape.epSize(); ++epIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                uint32_t tokens = problemShape.localTokensPerExpert(epIdx, localExpertIdx);
                inputSplitBlockInLocalExpert[epIdx][localExpertIdx] = CeilDiv(tokens, mmadBlockShape.row()) * nLoops;
                inputSplitBlock[epIdx] += inputSplitBlockInLocalExpert[epIdx][localExpertIdx];
                inputOffsetList[epIdx][localExpertIdx] = inputOffset;
                inputOffset += tokens;
            }
            uint32_t receiveLoops = CeilDiv(inputSplitBlock[epIdx], blockPerCommInRank);
            commLoops = Max(commLoops, receiveLoops);
        }

        for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
            uint32_t tokens = problemShape.localTokensPerExpert(srcEpIdx, localExpertIdx);
            actualMmadShapes[localExpertIdx] = Catlass::MakeCoord(tokens, problemShape.n());
            mmadBlockGrids[localExpertIdx] = CeilDiv(actualMmadShapes[localExpertIdx], mmadBlockShape);
        }

        for (uint32_t rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < problemShape.localExpertNum(); ++localExpertIdx) {
                outputSplitBlock[rankIdx] 
                    += CeilDiv(problemShape.globalTokensPerLocalExpert(rankIdx, localExpertIdx), mmadBlockShape.row());
            }
            outputSplitBlock[rankIdx] *=  nLoops;

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
        commContext.commOffset = commIdx * blockPerCommInRank;
        uint32_t actualblockPerCommInRank = Min(
            blockPerCommInRank, ClipSub(inputSplitBlock[srcEpIdx],commContext.commOffset));
        commContext.actualCommShape = MatrixCoord(
            actualblockPerCommInRank * mmadBlockShape.row() , mmadBlockShape.column());
    }

    CATLASS_DEVICE
    void UpdateLocalExpertContext(MatrixCoord const &blockOffset)
    {
        localExpertCtx.mmadIdx = commContext.commOffset + blockOffset.row() / mmadBlockShape.row();
        for (; localExpertCtx.startLocalExpert < problemShape.localExpertNum(); 
            localExpertCtx.startLocalExpert++) {
            auto localExpertIdx = localExpertCtx.startLocalExpert;
            if ((localExpertCtx.accumBlockNum + 
                    inputSplitBlockInLocalExpert[srcEpIdx][localExpertIdx]) > localExpertCtx.mmadIdx) {
                break;
            }
            localExpertCtx.accumBlockNum += inputSplitBlockInLocalExpert[srcEpIdx][localExpertIdx];
        }
    }

    CATLASS_DEVICE
    uint32_t GetActualReceiveAccum() const
    {
        uint32_t receiveCount = 0;
        for (uint32_t rankIdx = 0; rankIdx < problemShape.rankSize(); ++rankIdx) {
            if (commContext.commOffset < outputSplitBlock[rankIdx]) {
                receiveCount++;
            }
        }
        return receiveCount;
    }

    struct Iter {
        uint32_t taskIdx{0};
        uint32_t coreLoops{0};

        CATLASS_DEVICE
        void Next()
        {
            taskIdx++;
        }

        CATLASS_DEVICE
        bool End() const
        {
            return taskIdx >= coreLoops;
        }
    };

    CATLASS_DEVICE
    Iter Begin() const
    {
        return {/*taskIdx*/0, coreLoops};
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(Iter const &iter) const
    {
        uint32_t m = iter.taskIdx / blockGrid.column();
        uint32_t n = iter.taskIdx - m * blockGrid.column();
        auto blockCoord = Catlass::MakeCoord(m, n);
        return blockCoord * commBlockShape;
    }

    CATLASS_DEVICE
    MatrixCoord GetSwizzleBlockOffset(Iter const &iter) const {
        auto loops = DistMatrixCoord(blockGrid, 1);
        auto blockCoord = CommSwizzle<SWIZZLE_DIRECTION, IS_DETERMINISTIC>::GetCoord(
            loops, coreSplit, iter.taskIdx);
        if (blockCoord.IsOverflow(loops)) {
            return MatrixCoord{UINT_MAX, UINT_MAX};
        }
        return blockCoord.GetCoordInRank() * commBlockShape;
    }

    struct RemapperDst {
        using Scheduler = BlockCommSchedulerReduceScatterAllToAllV<MoeConstraints,
            IsDynamic, CoreSplit, SWIZZLE_DIRECTION, IS_DETERMINISTIC>;

        Scheduler *scheduler;
        MatrixCoord residueInMmadBlock;

        CATLASS_DEVICE
        MatrixCoord operator()(MatrixCoord const &blockOffset)
        {
            uint32_t accumBlockNum = scheduler->localExpertCtx.accumBlockNum;
            uint32_t localExpertIdx = scheduler->localExpertCtx.startLocalExpert;
            uint32_t mmadIdxInLocalExpert = scheduler->localExpertCtx.mmadIdx - accumBlockNum;
            
            auto srcEpIdx = scheduler->srcEpIdx;
            auto &actualMmadShape = scheduler->actualMmadShapes[localExpertIdx];
            auto &mmadBlockGrid = scheduler->mmadBlockGrids[localExpertIdx];

            auto mmadBlockCoordMN = MatrixSwizzle<7, 1>::GetCoord(mmadBlockGrid, mmadIdxInLocalExpert);
            auto mmadBlockOffset = mmadBlockCoordMN * scheduler->mmadBlockShape;
            auto actualMmadBlockShape = Min<uint32_t, 2>(scheduler->mmadBlockShape, 
                    ClipSub<MatrixCoord>(actualMmadShape, mmadBlockOffset));

            auto offsetInMmadBlock = blockOffset % scheduler->mmadBlockShape;
            residueInMmadBlock = actualMmadBlockShape - Min<uint32_t, 2>(actualMmadBlockShape, offsetInMmadBlock);
            MatrixCoord offsetDst{scheduler->inputOffsetList[srcEpIdx][localExpertIdx], 0};
            offsetDst += mmadBlockOffset + offsetInMmadBlock;
            return offsetDst;
        }

        CATLASS_DEVICE
        MatrixCoord GetResidueShape(MatrixCoord const &blockOffset) const
        {
            return ClipSub(scheduler->commContext.actualCommShape, blockOffset);
        }
    };

    CATLASS_DEVICE
    RemapperDst GetRemapperDst()
    {
        return {this};
    }

    CATLASS_DEVICE
    MatrixCoord RemapActualBlockShape(MatrixCoord const &blockOffset, RemapperDst &remapperDst) const
    {
        return Min(commBlockShape, remapperDst.GetResidueShape(blockOffset));
    }
};

}

#endif // CATCCOS_COMM_BLOCK_SCHEDULER_REDUCESCATTER_ALLTOALLV
