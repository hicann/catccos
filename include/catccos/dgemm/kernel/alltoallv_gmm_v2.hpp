/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ALLTOALLV_GMM_V2_HPP
#define CATCCOS_DGEMM_KERNEL_ALLTOALLV_GMM_V2_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
// #include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/detail/callback.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

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
    class MatmulKernel
>
struct AllToAllVGmmEmptyCallBack {
    CATLASS_DEVICE
    void operator()() const
    {
    }

    MatmulKernel *ptr;
};

template <
    class BlockMmad_,
    class BlockScheduler_,
    class ElementGroupList_,
    class LocalCopyBlockEpilogue_,
    class RemoteCommBlockEpilogue_,
    class BlockEpilogueScheduler_
>
class AlltoallvGMMKernel {
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using BlockScheduler = BlockScheduler_;

    using LocalCopyBlockEpilogue = LocalCopyBlockEpilogue_;
    using RemoteCommBlockEpilogue = RemoteCommBlockEpilogue_;
    
    using BlockEpilogueScheduler = BlockEpilogueScheduler_;

    using LocalCopyParams = typename LocalCopyBlockEpilogue::Params;
    using RemoteCommParams = typename RemoteCommBlockEpilogue::Params;

    friend class AicFinishSync<AlltoallvGMMKernel>;
    friend class AivWaitSync<AlltoallvGMMKernel>;
    friend class AllToAllVGmmEmptyCallBack<AlltoallvGMMKernel>;

    struct Params {
        GemmCoord problemShape;
        __gm__ ElementA *ptrA;
        __gm__ ElementB *ptrB;
        __gm__ ElementC *ptrC;
        LayoutA layoutA;
        LayoutB layoutB;
        LayoutC layoutC;
        GM_ADDR ptrWorkspace;
        int32_t EP;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        __gm__ int32_t* ptrTokenPerExpert;
        GM_ADDR symmetricPtr;
        int32_t ubMoveNum;
        LocalCopyParams localCopyParams;
        RemoteCommParams remoteCommParams;
        Callback callback;
        int32_t syncInterval;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_,
            uint32_t EP_, uint32_t expertPerRank_, uint32_t maxOutputSize_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrTokenPerExpert_,
            GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrWorkspace_, GM_ADDR symmetricPtr_, 
            LocalCopyParams localCopyParams_, RemoteCommParams remoteCommParams_,
            const Callback &callback_ = Callback{},
            int32_t syncInterval_ = INT_MAX
        ) : problemShape(problemShape_),
            EP(EP_), expertPerRank(expertPerRank_), maxOutputSize(maxOutputSize_),
            rank(rank_), rankSize(rankSize_),
            ptrTokenPerExpert(reinterpret_cast<__gm__ int32_t *>(ptrTokenPerExpert_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrWorkspace(ptrWorkspace_), symmetricPtr(symmetricPtr_),
            localCopyParams(localCopyParams_),
            remoteCommParams(remoteCommParams_),
            callback(callback_),
            syncInterval(syncInterval_)
        {
        }
    };

    /// User-facing arguments
    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t epSize;
        uint32_t expertNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrSymmetric;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};

        uint32_t expertPerRank = args.expertNum / args.epSize;
        uint32_t maxOutputSize = args.problemShape.m() * args.rankSize;

        typename RemoteCommBlockEpilogue::TileRemoteCopy::Params tileParams{args.commTileShape};
        RemoteCommParams remoteCommParams{args.commBlockShape, tileParams};
        LocalCopyParams localCopyParams{};

        return Params(
            args.problemShape,
            args.rankSize, expertPerRank, maxOutputSize,
            args.rankIdx, args.rankSize,
            args.ptrTokenPerExpert,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            args.ptrWorkspace, args.ptrSymmetric,
            localCopyParams, remoteCommParams
        );
    }

    struct WorkspaceInfo {
        GM_ADDR ptrTempOut;
        GM_ADDR ptrcumsumMM;

        CATLASS_DEVICE
        WorkspaceInfo(const Params & params) {
            ptrTempOut = params.ptrWorkspace;
            ptrcumsumMM = params.ptrWorkspace + params.maxOutputSize * params.problemShape.k() * sizeof(ElementA);
        }
    };

    CATLASS_DEVICE
    AlltoallvGMMKernel()
    {   
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        flagAivFinishCumsum = Catlass::Arch::CrossCoreFlag(0);
        flagAicFinishStore = Catlass::Arch::CrossCoreFlag(4);
        
    }

    CATLASS_DEVICE
    ~AlltoallvGMMKernel()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> resource = Catlass::Arch::Resource<ArchTag>());

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmWorkspace;
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWorkspace));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        WorkspaceInfo workspaceInfo(params);

        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(params.ptrTokenPerExpert);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;

        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCumsum);

        // ========= 矩阵乘法 start =========
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
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
                
                blockMmad(
                    gmWorkspace[gmOffsetA], layoutA,
                    gmB[gmOffsetB], layoutB,
                    gmC[gmOffsetC], layoutC,
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

        LocalCopyBlockEpilogue localCopyBlockEpilogue(resource, params.localCopyParams);
        RemoteCommBlockEpilogue remoteCommBlockEpilogue(resource, params.remoteCommParams);

        WorkspaceInfo workspaceInfo(params);
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementA> gmWorkspace;
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWorkspace));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(params.ptrTokenPerExpert);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.symmetricPtr));

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

        // ========= 搬运A矩阵到shmem start =========
        if (params.ptrA != nullptr) {
            uint32_t rowsToCopy = params.problemShape.m() / coreNum;
            uint32_t rowOffset = rowsToCopy * coreIdx;
            if (coreIdx == coreNum - 1) {
                rowsToCopy += params.problemShape.m() % coreNum;
            }
            if (rowsToCopy > 0) {
                MatrixCoord blockOffsetSrc{rowOffset, 0};

                auto blockSrc = gmA[params.layoutA.GetOffset(blockOffsetSrc)];
                auto blockDst = gmSymmetric[params.layoutA.GetOffset(blockOffsetSrc)];

                auto actualBlockShape = Catlass::MakeCoord<uint32_t>(rowsToCopy, params.problemShape.k());

                auto layoutBlockSrc = params.layoutA.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = params.layoutA.GetTileLayout(actualBlockShape);

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
        // ========= 搬运A矩阵到shmem end =========
        
        // ======== all to allv start ========
        uint32_t commLoops = blockEpilogueScheduler.GetCommLoops();
        auto remapperDst = blockEpilogueScheduler.GetRemapperDst();

        for (int32_t localExpertIdx = 0; localExpertIdx < params.expertPerRank; ++localExpertIdx) {
            blockEpilogueScheduler.UpdateLocalExpertIdx(localExpertIdx);

            for (int32_t commIdx = 0; commIdx < commLoops; commIdx++) {
                auto blockOffset = blockEpilogueScheduler.GetBlockOffset(commIdx);
                auto blockOffsetDst = remapperDst(blockOffset);

                int32_t dstEpIdx = blockOffset.row();

                uint32_t srcRowStartIndex = blockEpilogueScheduler.GetSrcPrevSum();
                MatrixCoord blockOffsetSrc{srcRowStartIndex, 0};

                auto gmBlockSrc = gmSymmetric[params.layoutA.GetOffset(blockOffsetSrc)];
                auto gmBlockDst = gmWorkspace[params.layoutA.GetOffset(blockOffsetDst)];

                auto actualBlockShape = blockEpilogueScheduler.RemapActualBlockShape(blockOffset, remapperDst);
                
                auto layoutBlockSrc = params.layoutA.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = params.layoutA.GetTileLayout(actualBlockShape);

                remoteCommBlockEpilogue.InitBlockLoop();
                remoteCommBlockEpilogue(
                    gmBlockSrc,
                    layoutBlockSrc,
                    gmBlockDst,
                    layoutBlockDst,
                    actualBlockShape,
                    dstEpIdx
                );
                remoteCommBlockEpilogue.FinalizeBlockLoop();

                blockEpilogueScheduler.UpdateSrcPrevSum(actualBlockShape.row());
            }
            blockEpilogueScheduler.UpdatePrevGroupSum();

            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
        }
        // ======== all to allv end ========
    }

private:
    Catlass::Arch::CrossCoreFlag flagAivFinishCumsum;
    Catlass::Arch::CrossCoreFlag flagAicFinishStore;
    Catlass::Arch::CrossCoreFlag flagAivFinishComm;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};
}

#endif
