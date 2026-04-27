/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_HPP
#define CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/symm_coord.hpp"
#include "catccos/layout/dist_matrix.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class GroupedMatmulAllToAllV {
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

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockCommSchedulerParams = typename BlockCommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    using MoeConstraints = typename BlockMmadScheduler::MoeConstraints;

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
        BlockCommParams blockCommParams;
        BlockCommSchedulerParams blockCommSchedulerParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            ProblemShape const &problemShape_, uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            BlockCommParams const &blockCommParams_,
            BlockCommSchedulerParams const &blockCommSchedulerParams_
        ) : problemShape(problemShape_), commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            blockCommParams(blockCommParams_),
            blockCommSchedulerParams(blockCommSchedulerParams_)
        {
        }
    };

    /// User-facing arguments
    struct Arguments {
        Catlass::GemmCoord gemmShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        uint32_t epSize;
        uint32_t expertNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrLocalTokensPerExpert;
        GM_ADDR ptrGlobalTokensPerLocalExpert;
        GM_ADDR ptrSymmetric;
        Catlass::MatrixCoord commCoreSplit;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        
        LayoutA layoutA{args.gemmShape.m(), args.gemmShape.k()};
        LayoutB layoutB{args.gemmShape.k(), args.gemmShape.n()};
        LayoutC layoutC{args.gemmShape.m(), args.gemmShape.n()};

        ProblemShape problemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.epSize, args.expertNum,
            args.ptrLocalTokensPerExpert, args.ptrGlobalTokensPerLocalExpert
        };

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        BlockCommSchedulerParams blockCommSchedulerParams{args.commCoreSplit};

        constexpr size_t IPC_BUFF_MAX_SIZE = 200 * 1024 * 1024 * sizeof(half);
        constexpr size_t SYNC_UNIT_SIZE = 4 * sizeof(int64_t);
        uint64_t symmetricOffset = 0;
        auto gmSymmetric = args.ptrSymmetric + symmetricOffset;
        symmetricOffset += IPC_BUFF_MAX_SIZE;
        auto syncMmadFinish = args.ptrSymmetric + symmetricOffset;
        symmetricOffset += SYNC_UNIT_SIZE;
        auto syncCommFinish = args.ptrSymmetric + symmetricOffset;

        return Params(
            problemShape,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            blockCommParams,
            blockCommSchedulerParams
        );
    }

    CATLASS_DEVICE
    GroupedMatmulAllToAllV()
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
        auto rankSize = params.problemShape.rankSize();
        uint32_t blockPerComm = AscendC::GetBlockNum() * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / rankSize;
        BlockMmadScheduler scheduler{params.problemShape, blockPerCommInRank, L1TileShape::ToCoordMN()};

        auto layoutSymmetric = layout::DistRowMajor(blockPerCommInRank * L1TileShape::M, L1TileShape::N, rankSize);
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }

        BlockMmad mmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);

        for (uint32_t commIdx = 0; commIdx < scheduler.GetCommLoops(); ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);
            }

            for (uint32_t localExpertIdx = 0; localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                scheduler.UpdateLocalExpertContext(localExpertIdx);
                auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx);
                auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx);
                size_t localExpertOffsetB = localExpertIdx * params.problemShape.k() * params.problemShape.n();

                for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                    DistGemmCoord blockOffset = scheduler.GetBlockOffset(iter);
                    
                    auto offsetA = remapperA(blockOffset);
                    auto offsetB = blockOffset.GetCoordKN();
                    auto offsetC = remapperC(blockOffset);
                    auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset.GetCoordMNK(), remapperA, remapperC);

                    auto gmBlockA = gmA[params.layoutA.GetOffset(offsetA)];
                    auto layoutBlockA = params.layoutA.GetTileLayout(actualBlockShape.GetCoordMK());

                    auto gmBlockB = gmB[localExpertOffsetB + params.layoutB.GetOffset(offsetB)];
                    auto layoutBlockB = params.layoutB.GetTileLayout(actualBlockShape.GetCoordKN());

                    auto gmBlockC = gmSymmetric[layoutSymmetric.GetOffset(offsetC)];
                    auto layoutBlockC = layoutSymmetric.GetTileLayout(actualBlockShape.GetCoordMN());

                    mmad(
                        gmBlockA, layoutBlockA,
                        gmBlockB, layoutBlockB,
                        gmBlockC, layoutBlockC,
                        actualBlockShape
                    );
                }
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                mmad.SynchronizeBlock();
            }
            
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.blockCommSchedulerParams.CoreSplit();

        auto rankSize = params.problemShape.rankSize();
        auto rankIdx = params.problemShape.rankIdx();
        auto srcRankIdx = aicoreIdx;
        auto srcEpIdx = srcRankIdx % params.problemShape.epSize();

        uint32_t blockPerComm = AscendC::GetBlockNum() * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / rankSize;
        BlockCommScheduler scheduler{params.problemShape, blockPerCommInRank, srcEpIdx,
                                     L1TileShape::ToCoordMN(), commBlockShape, commCoreSplit};

        BlockComm blockRemoteCopy(resource, params.blockCommParams);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        auto layoutSymmetric = Catlass::layout::RowMajor(blockPerCommInRank * L1TileShape::M, L1TileShape::N);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            uint32_t srcBlockOffset = stageIdx * rankSize + rankIdx;
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + srcBlockOffset * layout::Capacity(layoutSymmetric));
        }
        
        auto syncMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(params.syncMmadFinish, srcRankIdx));
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);

        bool isSyncCore = (subcoreIdx == 0) && (srcRankIdx == rankIdx);
        bool isCommCore = (subcoreIdx == 0) && (srcRankIdx < rankSize);
        if (isSyncCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
        }
        aclshmemx_barrier_all_vec();

        uint32_t receiveAccum = 0;
        for (uint32_t commIdx = 0; commIdx < scheduler.GetCommLoops(); ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

            scheduler.UpdateCommContext(commIdx);
            receiveAccum += scheduler.GetActualReceiveAccum();

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

            if (isSyncCore) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }

            if (isCommCore && (scheduler.inputSplitBlock[srcEpIdx] > commIdx * blockPerCommInRank)) {
                auto remapperDst = scheduler.GetRemapperDst();
                auto const &gmSymmetric = gmSymmetricList[stageIdx];

                aclshmem_signal_wait_until(syncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);

                blockRemoteCopy.InitBlockLoop();
                for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                    auto blockOffset = scheduler.GetSwizzleBlockOffset(iter);
                    scheduler.UpdateLocalExpertContext(blockOffset);

                    auto blockOffsetDst = remapperDst(blockOffset);

                    auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperDst);
                    if (Numel(actualBlockShape) == 0) {
                        continue;
                    }
                    actualBlockShape = Min(remapperDst.residueInMmadBlock, actualBlockShape);

                    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(blockOffset)];
                    auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualBlockShape);

                    auto gmBlockDst = gmC[params.layoutC.GetOffset(blockOffsetDst)];
                    auto layoutBlockDst = params.layoutC.GetTileLayout(actualBlockShape);

                    blockRemoteCopy(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualBlockShape, srcRankIdx
                    );
                }
                blockRemoteCopy.FinalizeBlockLoop();

                aclshmem_fence();
                aclshmemx_signal_op(syncCommFinish, 1, ACLSHMEM_SIGNAL_ADD, srcRankIdx);
            }

            if (isSyncCore) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_HPP