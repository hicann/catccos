/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_AUX_OPS_GROUP_SWIGLU_DEQUANT_H
#define CATCCOS_AUX_OPS_GROUP_SWIGLU_DEQUANT_H

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

template <
    class ArchTag_,
    class DequantSwigluBlock_
>
class DequantSwigluKernel {
public:
    using ArchTag = ArchTag_;
    using DequantSwigluBlock = DequantSwigluBlock_;
    using ElementC = typename DequantSwigluBlock::ElementC;
    using LayoutC = typename DequantSwigluBlock::LayoutC;
    using ElementD = typename DequantSwigluBlock::ElementD;
    using LayoutD = typename DequantSwigluBlock::LayoutD;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = Catlass::layout::VectorLayout;
    using TensorCoord = Catlass::layout::VectorLayout::TensorCoord;

    using DequantSwigluParams = typename DequantSwigluBlock::Params;

    struct Params {
        uint32_t problemCount;
        MatrixCoord problemShape;
        __gm__ ElementC *ptrC;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        __gm__ ElementD *ptrD;
        LayoutC layoutC;
        LayoutPerTokenScale layoutPerTokenScale;
        LayoutD layoutD;
        GM_ADDR ptrGroupList;
        Callback callback;
        int32_t syncInterval;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            uint32_t problemCount_,
            MatrixCoord problemShape_,
            GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
            GM_ADDR ptrD_, LayoutD layoutD_,
            GM_ADDR ptrGroupList_,
            const Callback &callback_ = Callback{},
            int32_t syncInterval_ = INT_MAX
        ) : problemCount(problemCount_),
            problemShape(problemShape_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrGroupList(ptrGroupList_),
            callback(callback_),
            syncInterval(syncInterval_)
        {
        }
    };

    CATLASS_DEVICE
    DequantSwigluKernel()
    {
    }

    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);
        AscendC::GlobalTensor<int32_t> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrGroupList));

        // ======== swiglu begin =========
        uint32_t nOut = params.problemShape.column() >> 1;
        int64_t gmGroupOffsetD = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        
        DequantSwigluParams dequantSwigluParams{
            {1, params.problemShape.column()}
        };
        DequantSwigluBlock dequantSwigluBlockEpilogue(resource, dequantSwigluParams);
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList(groupIdx);
            
            LayoutC layoutC = LayoutC(currentM, params.problemShape.column());
            LayoutPerTokenScale layoutPerTokenScale = LayoutPerTokenScale{currentM};
            LayoutD layoutD = params.layoutD.GetTileLayout(MakeCoord(currentM, nOut));

            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == 0) {
                params.callback();
            }
            
            uint32_t rows = currentM;
            uint32_t coreLoops = (currentM + rows - 1) / rows;

            MatrixCoord cBlockShape{rows, params.problemShape.column()};
            MatrixCoord dBlockShape{rows, nOut};
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                MatrixCoord blockCoord{loopIdx, 0};
                auto actualRow = currentM - loopIdx * rows > rows ? rows : currentM - loopIdx * rows;
                MatrixCoord actualBlockShape{actualRow, params.problemShape.column()};
                MatrixCoord actualChunkBlockShape{actualRow, nOut};

                MatrixCoord offsetC = blockCoord * cBlockShape;
                int64_t gmOffsetC = gmGroupOffsetC + layoutC.GetOffset(offsetC);
                auto gmBlockC = gmC[gmOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape);
                
                TensorCoord offsetPerTokenScale{loopIdx * rows};
                int64_t gmOffsetPerTokenScale = gmGroupOffsetPerTokenScale
                    + layoutPerTokenScale.GetOffset(offsetPerTokenScale);
                auto gmBlockPerTokenScale = gmPerTokenScale[gmOffsetPerTokenScale];
                auto layoutBlockPerTokenScale = layoutPerTokenScale.GetTileLayout(TensorCoord{actualRow});

                MatrixCoord offsetD = blockCoord * dBlockShape;
                int64_t gmOffsetD = gmGroupOffsetD + layoutD.GetOffset(offsetD);
                auto gmBlockD = gmD[gmOffsetD];
                auto layoutBlockD = layoutD.GetTileLayout(actualChunkBlockShape);

                dequantSwigluBlockEpilogue(
                    gmBlockC, layoutBlockC, 
                    gmBlockPerTokenScale, layoutBlockPerTokenScale,
                    gmBlockD, layoutBlockD,
                    actualBlockShape);
            }
            gmGroupOffsetD += currentM * nOut;
            gmGroupOffsetPerTokenScale += currentM;
            gmGroupOffsetC += currentM * params.problemShape.column();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }
};

#endif // CATCCOS_AUX_OPS_GROUP_SWIGLU_DEQUANT_H