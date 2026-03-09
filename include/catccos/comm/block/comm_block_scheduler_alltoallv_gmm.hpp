/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_EPILOGUE_BLOCK_SCHEDULER_ALLTOALLV_GMM_HPP
#define CATCCOS_COMM_EPILOGUE_BLOCK_SCHEDULER_ALLTOALLV_GMM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "shmem.h"
using namespace AscendC;

namespace Catlass::Gemm::Block {
struct BlockCommSchedulerAllToAllVGmm {
    using ArchTag = Arch::AtlasA2;

    uint32_t rank;
    uint32_t rankSize;
    int32_t expertPerRank;
    int32_t EP;
    GemmCoord problemShape;
    uint32_t coreIdx;
    uint32_t coreNum;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;

    struct CommContext {
        int32_t localExpertIdx = 0;
        int32_t prevGroupSum = 0;
        int32_t prevSum = 0;
    };

    CommContext commContext{};

    CATLASS_DEVICE
    BlockCommSchedulerAllToAllVGmm() = default;

    CATLASS_DEVICE
    BlockCommSchedulerAllToAllVGmm(
            uint32_t rank_,
            uint32_t rankSize_,
            int32_t expertPerRank_,
            int32_t EP_,
            GemmCoord problemShape_,
            uint32_t coreIdx_,
            uint32_t coreNum_,
            AscendC::GlobalTensor<int32_t> &tokenPerExpert_,
            AscendC::GlobalTensor<int32_t> &cumsumMM_,
            Arch::Resource<ArchTag> const &resource
    ) : rank(rank_),
        rankSize(rankSize_),
        expertPerRank(expertPerRank_),
        EP(EP_),
        problemShape(problemShape_),
        coreIdx(coreIdx_),
        coreNum(coreNum_),
        cumsumMM(cumsumMM_),
        tokenPerExpert(tokenPerExpert_)
    {
        for (int32_t i = 0; i < rank * expertPerRank; i++) {
            commContext.prevSum += tokenPerExpert(coreIdx * EP * expertPerRank + i);
        }

        if (coreIdx == coreNum - 1) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, resource);
        }

        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }

    // Calculate the number of EP (Execution Packets) assigned to a specific core
    // Note: In the current implementation, each coreIdx will be assigned exactly one EP.
    CATLASS_DEVICE
    uint32_t GetCommLoops()
    {
        if (coreNum == 0) {
            return 1;
        }

        uint32_t res = EP / coreNum;
        if (coreIdx < (EP % coreNum)) {
            res++;
        }

        return res;
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(uint32_t commIdx) {
        uint32_t dstEpIdx = coreIdx + commIdx * coreNum;
        return {dstEpIdx, 0};
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> &tokenPerExpert, AscendC::GlobalTensor<int32_t> &result,
                           Arch::Resource<ArchTag> const &resource)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);

        AscendC::DataCopyPad(
            tmpBuffer,
            tokenPerExpert[rank * expertPerRank],
            {
                static_cast<uint16_t>(EP),
                static_cast<uint16_t>(expertPerRank * sizeof(int32_t)),
                static_cast<uint16_t>((EP - 1) * expertPerRank * sizeof(int32_t)),
                0
            },
            {}
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer[i * expertPerRankAligned], tmpBuffer[i * expertPerRankAligned], tmpBuffer[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(
            result,
            tmpBuffer,
            {static_cast<uint16_t>(EP), static_cast<uint16_t>((expertPerRank) * sizeof(int32_t)), 0, 0}
        );
    }

    CATLASS_DEVICE
    void UpdateLocalExpertIdx(int32_t localExpertIdx)
    {
        commContext.localExpertIdx = localExpertIdx;
    }

    CATLASS_DEVICE
    int32_t GetLocalExpertIdx()
    {
        return commContext.localExpertIdx;
    }

    CATLASS_DEVICE
    void UpdatePrevGroupSum()
    {
        commContext.prevGroupSum += cumsumMM((EP - 1) * expertPerRank + commContext.localExpertIdx);
    }

    CATLASS_DEVICE
    void UpdateSrcPrevSum(uint32_t rows)
    {
        commContext.prevSum += rows;
    }

    CATLASS_DEVICE
    int32_t GetPrevGroupSum()
    {
        return commContext.prevGroupSum;
    }

    CATLASS_DEVICE
    int32_t GetSrcPrevSum()
    {
        return commContext.prevSum;
    }

    struct RemapperDst {
        using Scheduler = BlockCommSchedulerAllToAllVGmm;
        Scheduler *scheduler;

        CATLASS_DEVICE
        MatrixCoord operator()(MatrixCoord const &blockOffset) const
        {
            if (blockOffset.row() == 0) {
                return MatrixCoord{static_cast<uint32_t>(scheduler->GetPrevGroupSum()), 0};
            }

            int32_t offsetInRank = (blockOffset.row() - 1) * scheduler->expertPerRank + scheduler->GetLocalExpertIdx();
            uint32_t rowStartIdx = (scheduler->cumsumMM)(offsetInRank) + scheduler->GetPrevGroupSum();
            return MatrixCoord{rowStartIdx, 0};
        }

        CATLASS_DEVICE
        MatrixCoord GetResidueShape(MatrixCoord const &blockOffset) const
        {
            int32_t rowInExpertPerRank = blockOffset.row() * scheduler->EP * scheduler->expertPerRank;
            int32_t tokenOffset = rowInExpertPerRank + scheduler->rank * scheduler->expertPerRank + scheduler->GetLocalExpertIdx();
            uint32_t tokenNum = scheduler->tokenPerExpert(tokenOffset);
            return Catlass::MakeCoord<uint32_t>(tokenNum, scheduler->problemShape.k());
        }
    };

    CATLASS_DEVICE
    const RemapperDst GetRemapperDst()
    {
        return {this};
    }

    CATLASS_DEVICE
    MatrixCoord RemapActualBlockShape(MatrixCoord const &blockOffset, RemapperDst const &remapperDst) const
    {
        return remapperDst.GetResidueShape(blockOffset);
    }
};
}

#endif