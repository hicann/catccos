/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ALLTOALLV_GROUPED_MATMUL_HPP
#define CATCCOS_DGEMM_KERNEL_ALLTOALLV_GROUPED_MATMUL_HPP
 
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
 
#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"
 
namespace Catccos::DGemm::Kernel {
 
using Catlass::MatrixCoord;
using Catlass::GemmCoord;
 
template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockAllToAllVGmm_,
    class BlockMmadScheduler_,
    class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllToAllVGroupedMatmul {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
 
    using BlockAllToAllVGmm = BlockAllToAllVGmm_;
    using AllToAllVGmmParams = typename BlockAllToAllVGmm::Params;
 
    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockScheduler = BlockScheduler_;
 
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
 
    struct Params {
        ProblemShape problemShape;
        uint32_t commInterval;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        GM_ADDR ptrSymmetric;
        GM_ADDR syncMmadFinish;
        GM_ADDR syncCommFinish;
        AllToAllVGmmParams allToAllVGmmParams;
 
        CATLASS_DEVICE
        Params() = default;
 
        CATLASS_DEVICE
        Params(
            ProblemShape const &problemShape_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            AllToAllVGmmParams const &allToAllVGmmParams_
        ) : problemShape(problemShape_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            allToAllVGmmParams(allToAllVGmmParams_)
        {
        }
    };
 
    CATLASS_DEVICE
    AllToAllVGroupedMatmul()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishComm[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }
 
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);
 
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockMmad mmad{resource};
 
        uint32_t commShapeM = params.commInterval * L1TileShape::M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        BlockMmadScheduler scheduler{params.problemShape, commShapeM, L1TileShape::ToCoordMN()};
 
        auto rankSize = params.problemShape.rankSize();
        auto layoutSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }
 
        auto const &layoutC = params.layoutC;
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
 
        auto commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];
 
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);
 
            for (uint32_t localExpertIdx = 0; localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                auto const &layoutB = params.layoutB;
                AscendC::GlobalTensor<ElementB> gmB;
                gmB.SetGlobalBuffer(params.ptrB + localExpertIdx * layout::Capacity(layoutB));
 
                for (uint32_t srcRankIdx = 0; srcRankIdx < params.problemShape.rankSize(); ++srcRankIdx) {
                    scheduler.UpdateMmadContext(localExpertIdx, srcRankIdx);
                    auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx, srcRankIdx);
                    auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx, srcRankIdx);
 
                    for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                        auto blockOffset = scheduler.GetBlockOffset(iter);
 
                        auto blockOffsetA = remapperA(blockOffset);
                        auto blockOffsetB = blockOffset.GetCoordKN();
                        auto blockOffsetC = remapperC(blockOffset);
                        auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperA, remapperC);
 
                        auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetA)];
                        auto gmBlockB = gmB[layoutB.GetOffset(blockOffsetB)];
                        auto gmBlockC = gmC[layoutC.GetOffset(blockOffsetC)];
 
                        auto layoutBlockA = layoutSymmetric.GetTileLayout(actualBlockShape.GetCoordMK());
                        auto layoutBlockB = layoutB.GetTileLayout(actualBlockShape.GetCoordKN());
                        auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());
 
                        mmad(
                            gmBlockA, layoutBlockA,
                            gmBlockB, layoutBlockB,
                            gmBlockC, layoutBlockC,
                            actualBlockShape
                        );
                    }
                }
            }
 
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
    }
 
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockAllToAllVGmm allToAllVGmm(resource, params.allToAllVGmmParams);
 
        uint32_t commShapeM = params.commInterval * L1TileShape::M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        MatrixCoord blockShape = params.allToAllVGmmParams.BlockShape();
        BlockScheduler scheduler{params.problemShape, commShapeM, blockShape};
 
        auto const &layoutA = params.layoutA;
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
 
        auto rankSize = params.problemShape.rankSize();
        auto layoutSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }
 
        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);
 
        // 选择唯一的 aicore，对当前 rank 的信号地址进行初始化
        bool isRootCore = (AscendC::GetBlockIdx() == 0);
        auto rankIdx = params.problemShape.rankIdx();
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
        }
        aclshmemx_barrier_all_vec();
 
        uint32_t receiveAccum = 0;
        auto commLoops = scheduler.GetCommLoops();
        allToAllVGmm.InitBlockLoop();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];
 
            // 预计本轮过后，当前 rank 将累计收到多少 token
            receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            auto remapperSrc = scheduler.GetRemapperSrc(commIdx);
 
            // 等待上一个计算轮次所有 aicore 的计算完成
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
            // 选择唯一的 aicore，将当前 rank 的 syncMmadFinish 设置为 commIdx + 1，表示上一个通信轮次计算任务已完成
            if (isRootCore) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }
 
            for (auto iter = scheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = scheduler.GetBlockOffset(iter);
 
                auto blockOffsetSrc = remapperSrc(blockOffset);
                auto blockOffsetDst = blockOffset.GetLocalCoord();
                auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                if (Numel(actualBlockShape) == 0) {
                    continue;
                }
 
                auto gmBlockSrc = gmA[layoutA.GetOffset(blockOffsetSrc)];
                auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetDst)];
 
                auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualBlockShape);
 
                auto dstRankIdx = blockOffset.remote();
                // 等待目标 rank 上一次通信的计算流程完成，确保 symmetric 空闲
                auto remoteSyncMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstRankIdx));
                aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                allToAllVGmm(
                    gmBlockSrc, layoutBlockSrc,
                    gmBlockDst, layoutBlockDst,
                    actualBlockShape, dstRankIdx
                );
 
                // 通知目标 rank 又有新的 tokens 通信完成了
                aclshmem_fence();
                aclshmemx_signal_op(syncCommFinish, 1, ACLSHMEM_SIGNAL_ADD, dstRankIdx);
            }
 
            // 等待当前 commIdx 下，所有写到当前 rank symmetric 上的数据都写完
            if (isRootCore) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            // 等待当前通信轮次下数据接收完成后，所有的 aic 通知 aiv 开始计算本轮的
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
        allToAllVGmm.FinalizeBlockLoop();
    }
 
private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};
 
}  // namespace Catccos::DGemm::Kernel
 
#endif