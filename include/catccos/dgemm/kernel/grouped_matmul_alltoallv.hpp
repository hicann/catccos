/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_HPP
#define CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_HPP
 
#include "catccos/catccos.hpp"
 
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
 
namespace Catccos::DGemm::Kernel {
 
using Catlass::MatrixCoord;
using Catlass::GemmCoord;
 
template <uint32_t TP_SIZE_LIMIT_, uint32_t EP_SIZE_LIMIT_, uint32_t EXPERT_NUM_LIMIT_>
struct MoeConstraints {
    static constexpr uint32_t TP_SIZE_LIMIT = TP_SIZE_LIMIT_;
    static constexpr uint32_t EP_SIZE_LIMIT = EP_SIZE_LIMIT_;
    static constexpr uint32_t EXPERT_NUM_LIMIT = EXPERT_NUM_LIMIT_;
 
    static_assert(EXPERT_NUM_LIMIT % EP_SIZE_LIMIT == 0, "The number of experts must be a multiple of ep_size.");
    static constexpr uint32_t LOCAL_EXPERT_NUM_LIMIT = EXPERT_NUM_LIMIT / EP_SIZE_LIMIT;
    static constexpr uint32_t RANK_SIZE_LIMIT = TP_SIZE_LIMIT * EP_SIZE_LIMIT;
};
 
template <
    class BlockMmad_,
    class BlockGmmAllToAllV_,
    class BlockScheduler_,
    class BlockGmmAllToAllVScheduler_,
    uint32_t WORKSPACE_STAGES_,
    class MoeConstraint_
>
class GroupedMatmulAllToAllV {
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
 
    using BlockGmmAllToAllV = BlockGmmAllToAllV_;
    using GmmAllToAllVParams = typename BlockGmmAllToAllV::Params;
 
    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockGmmAllToAllVScheduler_;
    using BlockCommParams = typename CommScheduler::Params;
 
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
 
    using MoeConstraint = MoeConstraint_;
    static constexpr uint32_t TP_SIZE_LIMIT = MoeConstraint::TP_SIZE_LIMIT;
    static constexpr uint32_t EP_SIZE_LIMIT = MoeConstraint::EP_SIZE_LIMIT;
    static constexpr uint32_t EXPERT_NUM_LIMIT = MoeConstraint::EXPERT_NUM_LIMIT;
    static constexpr uint32_t LOCAL_EXPERT_NUM_LIMIT = MoeConstraint::LOCAL_EXPERT_NUM_LIMIT;
    static constexpr uint32_t RANK_SIZE_LIMIT = MoeConstraint::RANK_SIZE_LIMIT;
 
    struct Params {
        GemmCoord problemShape;
 
        uint32_t commInterval;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t epSize;
        uint32_t localExpertNum;
 
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
 
        __gm__ uint32_t *ptrLocalTokensPerExpert;
        __gm__ uint32_t *ptrGlobalTokensPerLocalExpert;
 
        GM_ADDR ptrSymmetric;
 
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
 
        GmmAllToAllVParams gmmAllToAllVParams;
        BlockCommParams commParams;
 
        CATLASS_DEVICE
        Params() = default;
 
        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t commInterval_, uint32_t rankIdx_, uint32_t rankSize_,
            uint32_t epSize_, uint32_t localExpertNum_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrLocalTokensPerExpert_, GM_ADDR ptrGlobalTokensPerLocalExpert_,
            GM_ADDR ptrSymmetric_,
            GmmAllToAllVParams const &gmmAllToAllVParams_,
            BlockCommParams const &commParams_,
            GM_ADDR ptrC_, LayoutC const &layoutC_
        ) : problemShape(problemShape_), commInterval(commInterval_), rankIdx(rankIdx_), rankSize(rankSize_),
            epSize(epSize_), localExpertNum(localExpertNum_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrLocalTokensPerExpert(reinterpret_cast<__gm__ uint32_t *>(ptrLocalTokensPerExpert_)),
            ptrGlobalTokensPerLocalExpert(reinterpret_cast<__gm__ uint32_t *>(ptrGlobalTokensPerLocalExpert_)),
            ptrSymmetric(ptrSymmetric_),
            gmmAllToAllVParams(gmmAllToAllVParams_),
            commParams(commParams_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_) {}
    };
 
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
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
 
        Init(params);
 
        BlockMmad mmad(resource);
 
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
 
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * blockPerLoopPerRank * L1TileShape::M, 
            L1TileShape::N, 
            L1TileShape::N
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, blockPerLoopPerRank);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);
 
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
 
            uint32_t actualBlockNumPerRankPerLocalExpert[RANK_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};
            uint32_t offsetBlockNumPerRankPerLocalExpert[RANK_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};
            for (uint32_t dstRankIdx = 0; dstRankIdx < params.rankSize; ++dstRankIdx) {
                uint32_t residueBlockNum{0};
                if (outputSplitPerRank[dstRankIdx] > commIdx * blockPerLoopPerRank) {
                    residueBlockNum = Min(blockPerLoopPerRank, outputSplitPerRank[dstRankIdx] - commIdx * blockPerLoopPerRank);
                }
 
                uint32_t localExpertOffsetBlockNum{0};
                for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
                    uint32_t actualBlockNum = Min(residueBlockNum, outputSplitPerRankPerLocalExpert[dstRankIdx][localExpertIdx]);
 
                    actualBlockNumPerRankPerLocalExpert[dstRankIdx][localExpertIdx] = actualBlockNum;
                    residueBlockNum -= actualBlockNum;
 
                    offsetBlockNumPerRankPerLocalExpert[dstRankIdx][localExpertIdx] = localExpertOffsetBlockNum;
                    localExpertOffsetBlockNum += actualBlockNum;
                }
            }
 
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);
            }
 
            uint32_t startCoreIdx = 0;
            for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
                size_t localExpertOffsetB = localExpertIdx * params.problemShape.k() * params.problemShape.n();
 
                for (uint32_t dstRankIdx = 0; dstRankIdx < params.rankSize; ++dstRankIdx) {
                    uint32_t m = params.ptrGlobalTokensPerLocalExpert[dstRankIdx * params.localExpertNum + localExpertIdx];
                    GemmCoord actualProblemShape{m, params.problemShape.n(), params.problemShape.k()};
                    BlockScheduler mmadScheduler(actualProblemShape, L1TileShape::ToCoordMN());
                    uint32_t coreLoops = mmadScheduler.GetCoreLoops();
 
                    uint32_t offsetBlockNum = offsetBlockNumPerRankPerLocalExpert[dstRankIdx][localExpertIdx];
                    MatrixCoord commOffsetA{outputOffsetPerRankPerLocalExpert[dstRankIdx][localExpertIdx], 0};
                    MatrixCoord commOffsetC{layoutSymmetricRow(Catlass::MakeCoord<int>(stageIdx, dstRankIdx, offsetBlockNum)) * L1TileShape::M, 0};
 
                    uint32_t actualBlockNum = actualBlockNumPerRankPerLocalExpert[dstRankIdx][localExpertIdx];
                    uint32_t curBlock = aicoreIdx >= startCoreIdx ? aicoreIdx : aicoreIdx + aicoreNum; 
                    while (curBlock < startCoreIdx + actualBlockNum) {
                        uint32_t loopIdx = curBlock - startCoreIdx + (coreLoops - outputSplitPerRankPerLocalExpert[dstRankIdx][localExpertIdx]);
                        
                        GemmCoord blockCoord = mmadScheduler.GetBlockCoord(loopIdx);
                        GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);
 
                        GemmCoord blockOffset = L1TileShape::ToCoord() * blockCoord;
                        MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                        MatrixCoord offsetB = blockOffset.GetCoordKN();
 
                        MatrixCoord dstBlockOffset{(curBlock - startCoreIdx) * L1TileShape::M, 0};
                        MatrixCoord offsetC = commOffsetC + dstBlockOffset;
 
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
 
                        curBlock += aicoreNum;
                    }
                    outputSplitPerRankPerLocalExpert[dstRankIdx][localExpertIdx] -= actualBlockNum;
                    startCoreIdx = (startCoreIdx + actualBlockNum) % aicoreNum;
                }
            }
 
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
    }
 
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
 
        Init(params);
 
        BlockGmmAllToAllV gmmAllToAllV(resource, params.gmmAllToAllVParams);
 
        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
 
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * blockPerLoopPerRank * L1TileShape::M, 
            L1TileShape::N, 
            L1TileShape::N
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, blockPerLoopPerRank);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);
 
        uint32_t epIdx = aicoreIdx % params.epSize;
 
        constexpr size_t IPC_BUFF_MAX_SIZE = 200 * 1024 * 1024 * sizeof(half);
        constexpr size_t SYNC_UNIT_SIZE = 4 * sizeof(int64_t);
        auto syncAddr0 = reinterpret_cast<__gm__ int32_t *>(
            shmem_ptr(params.ptrSymmetric + IPC_BUFF_MAX_SIZE, aicoreIdx));
        auto syncAddr1 = reinterpret_cast<__gm__ int32_t *>(params.ptrSymmetric + IPC_BUFF_MAX_SIZE + SYNC_UNIT_SIZE);
 
        if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
            shmemx_signal_op(syncAddr0, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            shmemx_signal_op(syncAddr1, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
        }
 
        MatrixCoord commBlockShape = params.gmmAllToAllVParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
 
        uint32_t receiveCount = 0;
        uint32_t localExpertIdx = 0;
        uint32_t accumBlockNum = 0;
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
 
            for (uint32_t srcRankIdx = 0; srcRankIdx < params.rankSize; ++srcRankIdx) {
                if (commIdx * blockPerLoopPerRank < inputSplitPerEp[srcRankIdx]) {
                    ++receiveCount;
                }
            }
 
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
 
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
                shmemx_signal_op(syncAddr0, commIdx + 1, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
 
            if (subcoreIdx == 0 && aicoreIdx < params.rankSize && inputSplitPerEp[epIdx] > commIdx * blockPerLoopPerRank) {
                uint32_t actualblockPerLoopPerRank = Min(blockPerLoopPerRank, inputSplitPerEp[epIdx] - commIdx * blockPerLoopPerRank);
                auto actualCommShape = DistMatrixCoord(actualblockPerLoopPerRank * L1TileShape::M, L1TileShape::N, 1);
                MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
 
 
                MatrixCoord commProblemOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageIdx, params.rankIdx, 0)) * L1TileShape::M, 0}; 
                commScheduler.UpdateProblem(actualCommShape, loopsInRank);
                auto commAicoreNum = commScheduler.GetRealCore();
                auto commCoreLoops = commScheduler.GetCoreLoop();
 
                shmem_signal_wait_until(syncAddr0, ACLSHMEM_CMP_EQ, commIdx + 1);
 
                for (uint32_t idx = 0; idx < commCoreLoops; idx++) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(idx);
                    MatrixCoord commBlockOffset = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(commBlockOffset);
 
                    uint32_t mmadIdx = commIdx * blockPerLoopPerRank + commBlockOffset.row() / L1TileShape::M;
 
                    for (; localExpertIdx < params.localExpertNum; localExpertIdx++) {
                        if ((accumBlockNum + inputSplitPerEpPerLocalExpert[epIdx][localExpertIdx]) > mmadIdx) {
                            break;
                        }
                        accumBlockNum += inputSplitPerEpPerLocalExpert[epIdx][localExpertIdx];
                    }
 
                    uint32_t m = params.ptrLocalTokensPerExpert[epIdx * params.localExpertNum + localExpertIdx];
                    GemmCoord actualMmadShape{m, params.problemShape.n(), params.problemShape.k()};
                    BlockScheduler mmadScheduler(actualMmadShape, L1TileShape::ToCoordMN());
 
                    // mmad swizzle
                    uint32_t mmadIdxInLocalExpert = mmadIdx - accumBlockNum;
                    GemmCoord mmadBlockCoord = mmadScheduler.GetBlockCoord(mmadIdxInLocalExpert);
                    MatrixCoord actualMmadBlockShape = mmadScheduler.GetActualBlockShape(mmadBlockCoord).GetCoordMN();
 
                    // block overlapping
                    MatrixCoord offsetInMmadBlock = commBlockOffset % L1TileShape::ToCoordMN();
                    MatrixCoord residueInMmadBlock = actualMmadBlockShape - Min<uint32_t, 2>(actualMmadBlockShape, offsetInMmadBlock);
                    actualCommBlockShape = Min<uint32_t, 2>(actualCommBlockShape, residueInMmadBlock);
                    
                    MatrixCoord offsetSrc = commProblemOffset + commBlockOffset;
                    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualCommBlockShape);
 
                    MatrixCoord offsetDst{inputOffsetPerEpPerLocalExpert[epIdx][localExpertIdx], 0};
                    offsetDst += mmadBlockCoord.GetCoordMN() * L1TileShape::ToCoordMN() + offsetInMmadBlock;
                    auto gmBlockDst = gmC[params.layoutC.GetOffset(offsetDst)];
                    auto layoutBlockDst = params.layoutC.GetTileLayout(actualCommBlockShape);
 
                    gmmAllToAllV.InitBlockLoop();
                    gmmAllToAllV(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualCommBlockShape, aicoreIdx
                    );
                    gmmAllToAllV.FinalizeBlockLoop();
                }
 
                shmem_fence();
                shmemx_signal_op(syncAddr1, 1, ACLSHMEM_SIGNAL_ADD, aicoreIdx);
            }
 
            if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
                shmem_signal_wait_until(syncAddr1, ACLSHMEM_CMP_EQ, receiveCount);
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
 
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
    }
 
private:
    CATLASS_DEVICE
    void Init(Params const &params)
    {
        blockPerLoopPerRank = params.commInterval * AscendC::GetBlockNum() / params.rankSize;
 
        uint32_t nLoops = CeilDiv(params.problemShape.n(), L1TileShape::N);
        for (uint32_t epIdx = 0; epIdx < params.epSize; ++epIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
                uint32_t expertIdx = epIdx * params.localExpertNum + localExpertIdx;
                inputSplitPerEpPerLocalExpert[epIdx][localExpertIdx] = CeilDiv(params.ptrLocalTokensPerExpert[expertIdx], L1TileShape::M) * nLoops;
                inputSplitPerEp[epIdx] += inputSplitPerEpPerLocalExpert[epIdx][localExpertIdx];
            }
            
            uint32_t inputSplitCommLoops = CeilDiv(inputSplitPerEp[epIdx], blockPerLoopPerRank);
            if (inputSplitCommLoops > commLoops) {
                commLoops = inputSplitCommLoops;
            }
        }
 
        for (uint32_t rankIdx = 0; rankIdx < params.rankSize; ++rankIdx) {
            uint32_t epIdx = rankIdx % params.epSize;
            uint32_t tpIdx = rankIdx / params.epSize;
 
            for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
                uint32_t expertIdx = epIdx * params.localExpertNum + localExpertIdx;
                outputSplitPerRankPerLocalExpert[rankIdx][localExpertIdx] = CeilDiv(params.ptrGlobalTokensPerLocalExpert[expertIdx], L1TileShape::M) * nLoops;
                outputSplitPerRank[rankIdx] += outputSplitPerRankPerLocalExpert[rankIdx][localExpertIdx];
            }
 
            uint32_t outputSplitCommLoops = CeilDiv(outputSplitPerRank[rankIdx], blockPerLoopPerRank);
            if (outputSplitCommLoops > commLoops) {
                commLoops = outputSplitCommLoops;
            }
        }
 
        
        uint32_t inputOffset = 0;
        for (uint32_t epIdx = 0; epIdx < params.epSize; ++epIdx) {
            for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
                uint32_t expertIdx = epIdx * params.localExpertNum + localExpertIdx;
                inputOffsetPerEpPerLocalExpert[epIdx][localExpertIdx] = inputOffset;
                inputOffset += params.ptrLocalTokensPerExpert[expertIdx];
            }
        }
 
        uint32_t outputOffset = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < params.localExpertNum; ++localExpertIdx) {
            for (uint32_t rankIdx = 0; rankIdx < params.rankSize; ++rankIdx) {
                uint32_t epIdx = rankIdx % params.epSize;
                uint32_t tpIdx = rankIdx / params.epSize;
 
                uint32_t expertIdx = epIdx * params.localExpertNum + localExpertIdx;
                outputOffsetPerRankPerLocalExpert[rankIdx][localExpertIdx] = outputOffset;
                outputOffset += params.ptrGlobalTokensPerLocalExpert[expertIdx];
            }
        }
    }
 
    uint32_t commLoops{0};
    uint32_t blockPerLoopPerRank{0};
 
 
    uint32_t inputSplitPerEp[EP_SIZE_LIMIT] = {0};                                              // by block
    uint32_t inputSplitPerEpPerLocalExpert[EP_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};        // by block
    uint32_t inputOffsetPerEpPerLocalExpert[EP_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};       // by token
 
    uint32_t outputSplitPerRank[RANK_SIZE_LIMIT] = {0};                                         // by block
    uint32_t outputSplitPerRankPerLocalExpert[RANK_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};   // by block
    uint32_t outputOffsetPerRankPerLocalExpert[RANK_SIZE_LIMIT][LOCAL_EXPERT_NUM_LIMIT] = {0};  // by token
 
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};
 
}  // namespace Catccos::DGemm::Kernel
 
#endif