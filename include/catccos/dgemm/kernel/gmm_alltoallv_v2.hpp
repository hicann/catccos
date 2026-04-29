/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_GMM_ALL_TO_ALL_V_V2_HPP
#define CATCCOS_DGEMM_KERNEL_GMM_ALL_TO_ALL_V_V2_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
// #include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "shmem.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace Catlass;

ACLSHMEM_DEVICE void cube_guard()
{
    using namespace AscendC;
 
#ifdef __DAV_C220_CUBE__
    LocalTensor<float> result;
    result.address_.logicPos = (uint8_t)TPosition::CO1;
    result.InitBuffer(0, 256);
    
    LocalTensor<half> left;
    left.address_.logicPos = (uint8_t)TPosition::A2;
    left.InitBuffer(0, 256);
 
    LocalTensor<half> right;
    right.address_.logicPos = (uint8_t)TPosition::B2;
    right.InitBuffer(0, 256);
 
    MmadParams param;
    param.m = 16;
    param.n = 16;
    param.k = 16;
 
    Mmad<float, half, half>(result, left, right, param);
#endif
}

namespace Catccos::DGemm::Kernel {

template <
    class MatmulKernel
>
struct EmptyCallBack {
    CATLASS_DEVICE
    void operator()() const
    {
    }

    MatmulKernel *ptr;
};

template <
    class MatmulKernel
>
struct AivFinishSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ptr->flagAivFinishStore);
    }

    MatmulKernel *ptr;
};

template <
    class MatmulKernel
>
struct AicWaitSync {
    CATLASS_DEVICE
    void operator()() const
    {
        Catlass::Arch::CrossCoreWaitFlag(ptr->flagAivFinishStore);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
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
class GMMAlltoallvKernel {
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

    friend class AivFinishSync<GMMAlltoallvKernel>;
    friend class AicWaitSync<GMMAlltoallvKernel>;
    friend class EmptyCallBack<GMMAlltoallvKernel>;

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

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
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

    struct WorkspaceInfo {
        GM_ADDR ptrTempOut;
        GM_ADDR ptrcumsumMM;

        CATLASS_DEVICE
        WorkspaceInfo(const Params & params) {
            ptrTempOut = params.ptrWorkspace;
            ptrcumsumMM = params.ptrWorkspace + params.maxOutputSize * params.problemShape.n() * sizeof(ElementC);
        }
    };

    CATLASS_DEVICE
    GMMAlltoallvKernel()
    {   
        flagAivFinishCumsum = Catlass::Arch::CrossCoreFlag(0);
        flagAivFinishStore = Catlass::Arch::CrossCoreFlag(4);
        
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
        WorkspaceInfo workspaceInfo(params);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));

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

        // AscendC::PipeBarrier<PIPE_ALL>();

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCumsum);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()}; // M N K

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == params.expertPerRank - 1) {
                params.callback();     // GMM2等swigluquant-2
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                if (currentM > 0) {
                    blockMmad(
                            gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                            gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                            gmWorkspace[gmGroupOffsetC + gmOffsetC], layoutC,
                            actualBlockShape
                        );
                }
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1); // AIC通知AIV开始Combine

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num(); 
        uint32_t coreNum = get_block_num() * get_subblockdim();
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;

        LocalCopyBlockEpilogue localCopyBlockEpilogue(resource, params.localCopyParams);
        RemoteCommBlockEpilogue remoteCommBlockEpilogue(resource, params.remoteCommParams); // 用远端写
        WorkspaceInfo workspaceInfo(params);

        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(params.ptrTokenPerExpert);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        AscendC::GlobalTensor<ElementC> gmSymmetric;
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

        uint32_t commLoops = blockEpilogueScheduler.GetCommLoops();
        auto remapperSrc = blockEpilogueScheduler.GetRemapperSrc();
        for (int32_t localExpertIdx = 0; localExpertIdx < params.expertPerRank; ++localExpertIdx) {
            blockEpilogueScheduler.UpdateLocalExpertIdx(localExpertIdx);
            AscendC::CrossCoreWaitFlag<0x2>(1); // AIV等AIC，等待GMM2完成
            AscendC::SyncAll<true>(); // 等所有的AIC核都执行完
            for (int32_t commIdx = 0; commIdx < commLoops; commIdx++) { // 每个core负责一个dstEP
                auto blockOffset = blockEpilogueScheduler.GetBlockOffset(commIdx);
                auto blockOffsetSrc = remapperSrc(blockOffset); // 获取src地址

                int32_t dstEpIdx = blockOffset.row();
                uint32_t dstRowStartIndex = blockEpilogueScheduler.GetSrcPrevSum(); // 获取dst地址
                MatrixCoord blockOffsetDst{dstRowStartIndex, 0};

                auto gmBlockSrc = gmWorkspace[params.layoutC.GetOffset(blockOffsetSrc)];
                auto gmBlockDst = gmSymmetric[params.layoutC.GetOffset(blockOffsetDst)];

                auto actualBlockShape = blockEpilogueScheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                
                auto layoutBlockSrc = params.layoutC.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = params.layoutC.GetTileLayout(actualBlockShape);

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

                blockEpilogueScheduler.UpdateSrcPrevSum(actualBlockShape.row()); // 维护当前`dst`的起始位置
            }
            blockEpilogueScheduler.UpdatePrevGroupSum(); // 加上本expert搬运的数据量，计算下个expert的src起始位置
        }

        shmemx_barrier_all_vec(); // 等所有的AIV核都执行完再进行搬运
        
        // ========= 搬运shmem到C矩阵 start =========
        if (params.ptrC != nullptr) {
            uint32_t rowsToCopy = params.problemShape.m() / coreNum;
            uint32_t rowOffset = rowsToCopy * coreIdx;
            if (coreIdx == coreNum - 1) {
                rowsToCopy += params.problemShape.m() % coreNum;
            }
            if (rowsToCopy > 0) {
                MatrixCoord blockOffsetSrc{rowOffset, 0};

                auto blockSrc = gmSymmetric[params.layoutC.GetOffset(blockOffsetSrc)];
                auto blockDst = gmC[params.layoutC.GetOffset(blockOffsetSrc)];

                auto actualBlockShape = Catlass::MakeCoord<uint32_t>(rowsToCopy, params.problemShape.n());

                auto layoutBlockSrc = params.layoutC.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = params.layoutC.GetTileLayout(actualBlockShape);

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
            shmemx_barrier_all_vec();
        }
        // ========= 搬运shmem到C矩阵 end =========
    }

private:
    Catlass::Arch::CrossCoreFlag flagAivFinishCumsum;
    Catlass::Arch::CrossCoreFlag flagAivFinishStore;
};
}

#endif