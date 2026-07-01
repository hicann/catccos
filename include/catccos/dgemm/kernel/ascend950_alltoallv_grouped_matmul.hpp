/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_HPP
 
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
 
#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif
 
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
class Ascend950AllToAllVGroupedMatmul {
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

    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
 
    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
 
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
 
    struct Params {
        ProblemShape problemShape;
        uint32_t commInterval;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        LayoutTagA layoutTagA;
        LayoutTagB layoutTagB;
        GM_ADDR ptrSymmetric;
        GM_ADDR syncMmadFinish;
        GM_ADDR syncCommFinish;
        BlockCommParams blockCommParams;
 
        CATLASS_HOST_DEVICE
        Params() {};
 
        CATLASS_HOST_DEVICE
        Params(
            ProblemShape const &problemShape_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            LayoutTagA layoutTagA_, LayoutTagB layoutTagB_,
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            BlockCommParams const &allToAllVGmmParams_
        ) : problemShape(problemShape_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            layoutTagA(layoutTagA_), layoutTagB(layoutTagB_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            blockCommParams(allToAllVGmmParams_)
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
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        LayoutTagA tagA{args.gemmShape.m(), args.gemmShape.k()};
        LayoutTagB tagB{args.gemmShape.k(), args.gemmShape.n()};
        LayoutTagC tagC{args.gemmShape.m() * args.rankSize, args.gemmShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);

        ProblemShape problemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.epSize, args.expertNum,
            args.ptrLocalTokensPerExpert, args.ptrGlobalTokensPerLocalExpert
        };

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};

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
            tagA, tagB,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            blockCommParams
        );
    }
 
    CATLASS_DEVICE
    Ascend950AllToAllVGroupedMatmul()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishComm[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~Ascend950AllToAllVGroupedMatmul()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);
 
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockMmad mmad{resource};

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        BlockMmadScheduler scheduler{params.problemShape, commShapeM, blockShape.GetCoordMN()};
 
        auto rankSize = params.problemShape.rankSize();
        auto layoutTagSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * rankSize * commShapeM, params.problemShape.k());
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, rankSize, commShapeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);
 
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto tensorSymmetric = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});

        auto rankIdx = params.problemShape.rankIdx();
 
        auto commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            for (uint32_t localExpertIdx = 0; localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                auto const &layoutB = params.layoutTagB;
                size_t localExpertOffsetB = localExpertIdx * layout::Capacity(layoutB);
                auto tensorB = tla::MakeTensor(gmB[localExpertOffsetB], params.layoutB, Catlass::Arch::PositionGM{});
 
                for (uint32_t srcRankIdx = 0; srcRankIdx < params.problemShape.rankSize(); ++srcRankIdx) {
                    scheduler.UpdateMmadContext(localExpertIdx, srcRankIdx);
                    auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx, srcRankIdx);
                    auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx, srcRankIdx);
 
                    for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                        auto blockOffset = scheduler.GetBlockOffset(iter);
 
                        MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageIdx, srcRankIdx, 0)), 0};
                        auto blockOffsetA = remapperA(blockOffset);
                        MatrixCoord blockOffsetB = blockOffset.GetCoordKN();
                        auto blockOffsetC = remapperC(blockOffset);
                        auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperA, remapperC);
 
                        auto tensorBlockA = GetTile(tensorSymmetric,
                                                    tla::MakeCoord(blockOffsetA.row() + commOffsetA.row(), blockOffsetA.column()),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockB = GetTile(tensorB,
                                                    tla::MakeCoord(blockOffsetB.row(), blockOffsetB.column()),
                                                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockC = GetTile(tensorC,
                                                    tla::MakeCoord(blockOffsetC.row(), blockOffsetC.column()),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                        mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                    }
                }
            }
 
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
            }
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockComm blockRemoteCopy(resource, params.blockCommParams);
 
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        MatrixCoord blockShape = params.blockCommParams.BlockShape();
        BlockCommScheduler scheduler{params.problemShape, commShapeM, blockShape};
 
        auto const &layoutA = params.layoutTagA;
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
 
        auto rankSize = params.problemShape.rankSize();
        auto layoutTagSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutTagSymmetric));
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
        blockRemoteCopy.InitBlockLoop();
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
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
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
                auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetDst)];
 
                auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(actualBlockShape);
 
                auto dstRankIdx = blockOffset.remote();
                // 等待目标 rank 上一次通信的计算流程完成，确保 symmetric 空闲
                auto remoteSyncMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstRankIdx));
                aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                blockRemoteCopy(
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
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
        blockRemoteCopy.FinalizeBlockLoop();
    }
 
private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};
 
}  // namespace Catccos::DGemm::Kernel
 
#endif
