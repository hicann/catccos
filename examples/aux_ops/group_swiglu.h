/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUP_SWIGLU_H 
#define GROUP_SWIGLU_H

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/detail/callback.hpp"

#include "kernel_operator.h"

using namespace AscendC;
using namespace Catlass;

namespace Catccos::DGemm::Kernel {

template <
    class SwigluKernel
>
struct AivFinishSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ptr->flagAivFinishStore);
    }

    SwigluKernel *ptr;
};

template <
    class SwigluKernel
>
struct AicWaitSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreWaitFlag(ptr->flagAivFinishStore);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }

    SwigluKernel *ptr;
};

template <
    class ArchTag_,
    class SwigluBlock_
>
class SwigluKernel {
public:
    using ArchTag = ArchTag_;
    using SwigluBlock = SwigluBlock_;
    using ElementC = typename SwigluBlock::ElementC;
    using LayoutC = typename SwigluBlock::LayoutC;
    using ElementD = typename SwigluBlock::ElementD;
    using LayoutD = typename SwigluBlock::LayoutD;

    using SwigluParams = typename SwigluBlock::Params;

    friend class AivFinishSync<SwigluKernel>;
    friend class AicWaitSync<SwigluKernel>;

    struct Params {
        uint32_t problemCount;
        MatrixCoord problemShape;
        __gm__ ElementC *ptrC;
        __gm__ ElementD *ptrD;
        LayoutC layoutC;
        LayoutD layoutD;
        GM_ADDR ptrGroupList;
        Callback waitCallback;
        Callback notifyCallback;
        int32_t syncInterval;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            uint32_t problemCount_,
            MatrixCoord problemShape_,
            GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrD_, LayoutD layoutD_,
            GM_ADDR ptrGroupList_,
            const Callback &waitCallback_ = Callback{},
            const Callback &notifyCallback_ = Callback{},
            int32_t syncInterval_ = INT_MAX
        ) : problemCount(problemCount_),
            problemShape(problemShape_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrGroupList(ptrGroupList_),
            waitCallback(waitCallback_),
            notifyCallback(notifyCallback_),
            syncInterval(syncInterval_)
        {
        }
    };

    CATLASS_DEVICE
    SwigluKernel()
    {
        flagAivFinishStore = Catlass::Arch::CrossCoreFlag(4);
    }


    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);
        AscendC::GlobalTensor<int32_t> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrGroupList));

        // ======== swiglu begin =========
        uint32_t nOut = params.problemShape.column() >> 1;
        int64_t gmGroupOffsetD = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        
        SwigluParams swigluParams{
            {1, params.problemShape.column()}
        };
        SwigluBlock swigluBlockEpilogue(resource, swigluParams);
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList(groupIdx);

            LayoutD layoutD = params.layoutD.GetTileLayout(MakeCoord(currentM, nOut));
            
            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == 0) {
                params.waitCallback();
            }
            
            uint32_t rows = currentM;
            uint32_t coreLoops = (currentM + rows - 1) / rows;

            MatrixCoord cBlockShape{rows, params.problemShape.column()};
            MatrixCoord dBlockShape{rows, nOut};
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            LayoutC layoutC = LayoutC(currentM, params.problemShape.column());
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                MatrixCoord blockCoord{loopIdx, 0};
                auto actualRow = currentM - loopIdx * rows > rows ? rows : currentM - loopIdx * rows;
                MatrixCoord actualBlockShape{actualRow, params.problemShape.column()};
                MatrixCoord actualChunkBlockShape{actualRow, nOut};

                MatrixCoord offsetC = blockCoord * cBlockShape;
                int64_t gmOffsetC = gmGroupOffsetC + layoutC.GetOffset(offsetC);
                auto gmBlockC = gmC[gmOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape);

                MatrixCoord offsetD = blockCoord * dBlockShape;
                int64_t gmOffsetD = gmGroupOffsetD + layoutD.GetOffset(offsetD);
                auto gmBlockD = gmD[gmOffsetD];
                auto layoutBlockD = layoutD.GetTileLayout(actualChunkBlockShape);

                swigluBlockEpilogue(gmBlockC, layoutBlockC, gmBlockD, layoutBlockD, actualBlockShape);
            }

            gmGroupOffsetD += currentM * nOut;
            gmGroupOffsetC += currentM * params.problemShape.column();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;

            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == params.problemCount - 1) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                params.notifyCallback();
            }

        }   
    }

private:
    Catlass::Arch::CrossCoreFlag flagAivFinishStore;
};
}

#endif