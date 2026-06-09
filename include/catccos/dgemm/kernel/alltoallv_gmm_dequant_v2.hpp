/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ALL_TO_ALL_V_GMM_DEQUANT_V2_HPP
#define CATCCOS_DGEMM_KERNEL_ALL_TO_ALL_V_GMM_DEQUANT_V2_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/detail/callback.hpp"

#include "shmem.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace Catlass;

namespace Catccos::DGemm::Kernel {

template <
    class MatmulKernel
>
struct AicFinishSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStore);
    }

    MatmulKernel *ptr;
};

template <
    class MatmulKernel
>
struct AivWaitSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreWaitFlag(ptr->flagAicFinishStore);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }

    MatmulKernel *ptr;
};

template <
    class BlockMmad_,
    class BlockScheduler_,
    class ElementGroupList_,
    class RemoteCommBlockEpilogue_,
    class BlockEpilogueScheduler_
>
class AlltoallvGmmDequantKernel {
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
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = Catlass::layout::VectorLayout;
    using ElementScale = uint64_t;
    using LayoutScale = Catlass::layout::VectorLayout;

    constexpr static uint32_t ALIGN_512 = 512;
    using TensorCoord = Catlass::layout::VectorLayout::TensorCoord;

    using BlockScheduler = BlockScheduler_;
    using RemoteCommBlockEpilogue = RemoteCommBlockEpilogue_;

    using BlockEpilogueScheduler = BlockEpilogueScheduler_;
    using RemoteCommParams = typename RemoteCommBlockEpilogue::Params;

    friend class AicFinishSync<AlltoallvGmmDequantKernel>;
    friend class AivWaitSync<AlltoallvGmmDequantKernel>;

    struct Params {
        GemmCoord problemShape;
        int32_t EP;
        int32_t expertPerRank;
        uint32_t rank;
        uint32_t rankSize;

        __gm__ ElementA *ptrA; // int8_t
        __gm__ ElementB *ptrB; // int8_t
        __gm__ ElementC *ptrC; // int32_t
        __gm__ ElementScale *ptrScale; // uint64_t
        LayoutA layoutA;
        LayoutB layoutB;
        LayoutC layoutC;
        LayoutScale layoutScale;
        GM_ADDR ptrPerTokenScale; // output
        GM_ADDR ptrTokenPerExpert; // input
        GM_ADDR ptrCumsumMM; // output
        GM_ADDR symmetricPtr;

        RemoteCommParams remoteCommParams;
        Callback callback;
        int32_t syncInterval;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            GemmCoord problemShape_,
            uint32_t EP_, uint32_t expertPerRank_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrScale_, LayoutScale layoutScale_,
            GM_ADDR ptrPerTokenScale_, GM_ADDR ptrTokenPerExpert_, GM_ADDR ptrCumsumMM_,
            GM_ADDR symmetricPtr_, 
            RemoteCommParams remoteCommParams_,
            const Callback &callback_ = Callback{},
            int32_t syncInterval_ = INT_MAX
        ) : problemShape(problemShape_),
            EP(EP_), expertPerRank(expertPerRank_),
            rank(rank_), rankSize(rankSize_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)), layoutScale(layoutScale_),
            ptrPerTokenScale(ptrPerTokenScale_),
            ptrTokenPerExpert(ptrTokenPerExpert_), ptrCumsumMM(ptrCumsumMM_),
            symmetricPtr(symmetricPtr_),
            remoteCommParams(remoteCommParams_),
            callback(callback_),
            syncInterval(syncInterval_)
        {
        }
    };

    CATLASS_DEVICE
    AlltoallvGmmDequantKernel()
    {   
        flagAivFinishCumsum = Catlass::Arch::CrossCoreFlag(0);
        flagAicFinishStore = Catlass::Arch::CrossCoreFlag(4);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> resource);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);

        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrCumsumMM));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;

        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCumsum);

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
            LayoutScale layoutScale = params.layoutScale;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            AscendC::CrossCoreWaitFlag<0x2>(1);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = gmGroupOffsetA + layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = gmGroupOffsetB + layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = gmGroupOffsetC + layoutC.GetOffset(offsetC);
                int64_t gmOffsetS =
                    groupIdx * params.problemShape.n() + blockCoord.n() * L1TileShape::N;   // 每个expert一组scale
                
                blockMmad(
                    gmA[gmOffsetA], layoutA,
                    gmB[gmOffsetB], layoutB,
                    gmC[gmOffsetC], layoutC,
                    gmScale[gmOffsetS], layoutScale,
                    actualBlockShape
                );
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            
            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == params.expertPerRank - 1) {
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                params.callback();
            }
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num(); 
        uint32_t coreNum = get_block_num() * get_subblockdim();
        coreNum = (coreNum == 0) ? 1 : coreNum;

        RemoteCommBlockEpilogue remoteCommBlockEpilogue(resource, params.remoteCommParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<float> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ptrPerTokenScale));
        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrCumsumMM));
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.symmetricPtr));

        auto layoutRemoteA = LayoutA{params.problemShape.m(), params.problemShape.k() + ALIGN_512};
        auto layoutPerTokenScale = Catlass::layout::VectorLayout{params.problemShape.m()};

        BlockEpilogueScheduler blockEpilogueScheduler(
            params.rank,
            params.rankSize,
            params.expertPerRank,
            params.EP,
            params.problemShape,
            coreIdx,
            coreNum,
            tokenPerExpert,
            cumsumMM,
            resource
        );

        uint32_t commLoops = blockEpilogueScheduler.GetCommLoops();
        auto remapperDst = blockEpilogueScheduler.GetRemapperDst();

        for (int32_t localExpertIdx = 0; localExpertIdx < params.expertPerRank; ++localExpertIdx) {
            blockEpilogueScheduler.UpdateLocalExpertIdx(localExpertIdx);

            for (int32_t commIdx = 0; commIdx < commLoops; commIdx++) {
                auto blockOffset = blockEpilogueScheduler.GetBlockOffset(commIdx);
                auto blockOffsetDst = remapperDst(blockOffset); // {rowStartIdx, 0};
                int32_t dstEpIdx = blockOffset.row();

                __gm__ void *peermemPtr = shmem_ptr(params.symmetricPtr, dstEpIdx);

                AscendC::GlobalTensor<ElementA> gmRemoteA;
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA * >(peermemPtr));

                uint32_t srcRowStartIdx = blockEpilogueScheduler.GetSrcPrevSum();
                MatrixCoord blockOffsetSrc{srcRowStartIdx, 0};
                TensorCoord blockOffsetPerTokenScale{blockOffsetDst[0]};

                auto actualDstBlockShape = blockEpilogueScheduler.RemapActualBlockShape(blockOffset, remapperDst);
                auto actualSrcBlockShape = MatrixCoord{actualDstBlockShape[0], actualDstBlockShape[1] + ALIGN_512};
                auto actualPerTokenScaleBlockShape = TensorCoord{actualDstBlockShape[0]};

                auto gmBlockSrc = gmRemoteA[layoutRemoteA.GetOffset(blockOffsetSrc)];
                auto gmBlockDst = gmA[params.layoutA.GetOffset(blockOffsetDst)];
                auto gmBlockPerTokenScale = gmPerTokenScale[layoutPerTokenScale.GetOffset(blockOffsetPerTokenScale)];
                
                auto layoutBlockSrc = layoutRemoteA.GetTileLayout(actualSrcBlockShape);
                auto layoutBlockDst = params.layoutA.GetTileLayout(actualDstBlockShape);
                auto layoutBlockPerTokenScale = layoutPerTokenScale.GetTileLayout(actualPerTokenScaleBlockShape);

                remoteCommBlockEpilogue.InitBlockLoop();
                remoteCommBlockEpilogue(
                    gmBlockSrc, layoutBlockSrc,
                    gmBlockDst, layoutBlockDst,
                    gmBlockPerTokenScale, layoutBlockPerTokenScale,
                    actualSrcBlockShape,
                    dstEpIdx
                );
                remoteCommBlockEpilogue.FinalizeBlockLoop();

                blockEpilogueScheduler.UpdateSrcPrevSum(actualDstBlockShape.row());
            }
            blockEpilogueScheduler.UpdatePrevGroupSum();
            aclshmemx_barrier_all_vec();

            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAivFinishCumsum;
    Catlass::Arch::CrossCoreFlag flagAicFinishStore;
    Catlass::Arch::CrossCoreFlag flagAivFinishComm;
};
}

#endif // CATCCOS_DGEMM_KERNEL_ALL_TO_ALL_V_GMM_DEQUANT_V2_HPP