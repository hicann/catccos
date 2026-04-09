/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_SCHEDULER_REDUCE_SCATTER_HPP
#define CATCCOS_COMM_BLOCK_SCHEDULER_REDUCE_SCATTER_HPP

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dist_coord.hpp"
#include "catccos/symm_coord.hpp"

// from catlass
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Block {

using Catlass::MatrixCoord;

template<bool IsDynamic_, class CoreSplit_, 
    uint32_t SWIZZLE_DIRECTION_ = 0, bool IS_DETERMINISTIC_ = false,
    uint32_t MMAD_SWIZZLE_OFFSET_ = 1, uint32_t MMAD_SWIZZLE_DIRECTION_ = 0>
struct BlockCommSchedulerReduceScatter {
    static constexpr uint32_t SWIZZLE_DIRECTION = SWIZZLE_DIRECTION_;
    static constexpr bool IS_DETERMINISTIC = IS_DETERMINISTIC_;
    static constexpr uint32_t MMAD_SWIZZLE_OFFSET = MMAD_SWIZZLE_OFFSET_;
    static constexpr uint32_t MMAD_SWIZZLE_DIRECTION = MMAD_SWIZZLE_DIRECTION_;
    static constexpr bool IsDynamic = IsDynamic_;
    using CoreSplit = CoreSplit_;

    static_assert((IS_DETERMINISTIC && SWIZZLE_DIRECTION == 0) || !IS_DETERMINISTIC,
        "Deterministic calculation requires that the swizzle direction be 0.");

    struct MmadInfo {
        DistGemmCoord mmadProblemShape;
        MatrixCoord mmadBlockShape;
        MatrixCoord mmadLoops;

        CATLASS_DEVICE
        MmadInfo() = default;

        CATLASS_DEVICE
        MmadInfo(DistGemmCoord const &mmadProblemShape_, MatrixCoord const &mmadBlockShape_)
            : mmadProblemShape(mmadProblemShape_), mmadBlockShape(mmadBlockShape_)
        {
            mmadLoops = CeilDiv(MatrixCoord(mmadProblemShape.GetCoordMN()), mmadBlockShape);
        }
    };
    
    DistMatrixCoord loops;
    MatrixCoord actualCommShape;

    MatrixCoord coreSplit;
    DistMatrixCoord blockShape;

    MmadInfo mmadInfo;

    uint32_t blockPerCommInRank;

    uint32_t commLoops;
    uint32_t coreLoops;

    uint32_t rankIdx;
    uint32_t rankSize;

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
    BlockCommSchedulerReduceScatter() {}

    CATLASS_DEVICE
    BlockCommSchedulerReduceScatter(MatrixCoord const &blockShapeInRank_, MatrixCoord const &coreSplit_, 
        uint32_t const rankSize_, uint32_t const rankIdx_, uint32_t blockPerCommInRank_,
        MmadInfo const &mmadInfo_)
        : coreSplit(coreSplit_), rankSize(rankSize_), rankIdx(rankIdx_), 
          blockPerCommInRank(blockPerCommInRank_),
          mmadInfo(mmadInfo_)
    {
        blockShape = Catlass::MakeCoord<uint32_t>(blockShapeInRank_.row(), blockShapeInRank_.column(), 1);
        commLoops = CeilDiv<uint32_t>(Numel(mmadInfo.mmadLoops), blockPerCommInRank_);
        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
        }
    }

    CATLASS_DEVICE
    uint32_t GetRealCores() const
    {
        return coreSplit.row() * coreSplit.column();
    }

    struct Iter {
        uint32_t taskIdx;
        uint32_t iterStep;
        uint32_t coreLoops;
 
        CATLASS_DEVICE
        void Next() {
            taskIdx += iterStep;
        }
 
        CATLASS_DEVICE
        bool End() {
            return taskIdx >= coreLoops;
        }
    };
 
    CATLASS_DEVICE
    Iter Begin(uint32_t commIdx) {
        uint32_t taskIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t iterStep = GetRealCores();
        if (AscendC::GetSubBlockIdx() != 0 || taskIdx >= iterStep) {
            return {0, 0, 0}; 
        }

        uint32_t actualBlockInRank = Min<uint32_t>(
            blockPerCommInRank, Numel(mmadInfo.mmadLoops) - commIdx * blockPerCommInRank);
        actualCommShape = MatrixCoord{actualBlockInRank, 1} * mmadInfo.mmadBlockShape;
        loops = DistMatrixCoord(CeilDiv<MatrixCoord>(actualCommShape, blockShape.GetCoordInRank()), rankSize);
        
        uint32_t tileCount = loops.row() * loops.column();
        coreLoops = (IS_DETERMINISTIC ? RoundUp<uint32_t>(tileCount, coreSplit.row()) : tileCount) * loops.rank();
        return {taskIdx, iterStep, coreLoops};
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const
    {
        return commLoops;
    }

    CATLASS_DEVICE
    SymmMatrixCoord GetBlockOffset(uint32_t taskIdx) const {
        auto blockCoord = CommSwizzle<SWIZZLE_DIRECTION, IS_DETERMINISTIC>::GetCoord(
            loops, coreSplit, taskIdx);
        
        if (blockCoord.IsOverflow(loops)) {
            return SymmMatrixCoord{UINT_MAX, UINT_MAX, rankIdx, rankSize};
        }
        return SymmMatrixCoord{blockCoord.GetCoordInRank() * blockShape.GetCoordInRank(), rankIdx, blockCoord.rank()};
    }

    struct RemapperDst {
        using Scheduler = BlockCommSchedulerReduceScatter<IsDynamic, CoreSplit,
            SWIZZLE_DIRECTION, IS_DETERMINISTIC, MMAD_SWIZZLE_OFFSET, MMAD_SWIZZLE_DIRECTION>;

        Scheduler *scheduler;
        uint32_t commIdx;

        struct MmadContext {
            MatrixCoord mmadBlockOffset;
            MatrixCoord offsetInMmadBlock;
        };

        MmadContext mmadCtx;

        CATLASS_DEVICE
        void UpdateMmadContext(SymmMatrixCoord const &blockOffset)
        {
            MatrixCoord blockOffsetInRank = blockOffset.GetMatrixCoord();
            uint32_t mmadLoopIdx = commIdx * scheduler->blockPerCommInRank 
                + blockOffsetInRank.row() / scheduler->mmadInfo.mmadBlockShape.row(); // 昂贵的除法

            MatrixCoord mmadBlockCoord = MatrixSwizzle<MMAD_SWIZZLE_OFFSET, MMAD_SWIZZLE_DIRECTION>::GetCoord(
                scheduler->mmadInfo.mmadLoops, mmadLoopIdx);

            mmadCtx.mmadBlockOffset = mmadBlockCoord * scheduler->mmadInfo.mmadBlockShape;
            mmadCtx.offsetInMmadBlock = blockOffsetInRank % scheduler->mmadInfo.mmadBlockShape; // 昂贵的取模
        }

        CATLASS_DEVICE
        MatrixCoord operator()(SymmMatrixCoord const &blockOffset) const
        {
            return mmadCtx.mmadBlockOffset + mmadCtx.offsetInMmadBlock;
        }

        CATLASS_DEVICE
        MatrixCoord GetResidueShape(SymmMatrixCoord const &blockOffset) const 
        {
            auto actualMmadBlockShape = Min(scheduler->mmadInfo.mmadBlockShape, 
                ClipSub<MatrixCoord>(scheduler->mmadInfo.mmadProblemShape.GetCoordMN(), mmadCtx.mmadBlockOffset));
            return actualMmadBlockShape - Min<uint32_t, 2>(actualMmadBlockShape, mmadCtx.offsetInMmadBlock);
        }
    };

    CATLASS_DEVICE
    const RemapperDst GetRemapperDst(uint32_t commIdx) {
        return {this, commIdx};
    }

    CATLASS_DEVICE
    MatrixCoord RemapActualBlockShape(SymmMatrixCoord const &blockOffset, RemapperDst const &remapperDst) const
    {
        MatrixCoord blockOffsetInRank = blockOffset.GetMatrixCoord();
        auto actualBlockShape = Min(blockShape.GetCoordInRank(), 
            ClipSub<MatrixCoord>(actualCommShape, blockOffsetInRank));
        return Min<uint32_t, 2>(actualBlockShape, remapperDst.GetResidueShape(blockOffset));
    }

};

}  // namespace Catccos::Comm::Block

#endif // CATCCOS_COMM_BLOCK_SCHEDULER_REDUCE_SCATTER_HPP