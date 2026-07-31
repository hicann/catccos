/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_LOCAL_OPTIONAL_BACKEND_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_LOCAL_OPTIONAL_BACKEND_HPP

#include <type_traits>

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"

// from catlass
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel
{

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <class DispatchPolicy>
struct IsAscend950OptionalBackendLocalCopyDispatch : std::false_type
{
};

template <class CommBackend, class = void>
struct IsAscend950CollectiveAllGatherBackend : std::false_type
{
};

template <class CommBackend>
struct IsAscend950CollectiveAllGatherBackend<
    CommBackend, std::void_t<decltype(CommBackend::IS_COLLECTIVE_ALL_GATHER)>>
    : std::bool_constant<CommBackend::IS_COLLECTIVE_ALL_GATHER>
{
};

template <class ArchTag, uint32_t UbStages, bool IsDynamic>
struct IsAscend950OptionalBackendLocalCopyDispatch<Comm::AtlasCommLocalCopy<ArchTag, UbStages, IsDynamic>> : std::true_type
{
};

template <class BlockMmad_, class BlockAllGather_, class BlockScheduler_, class BlockAllGatherScheduler_,
          uint32_t WORKSPACE_STAGES_, class CommBackend_>
class Ascend950AllGatherMatmulWithLocalOptionalBackend
{
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

    using BlockAllGather = BlockAllGather_;
    using BlockAllGatherParams = typename BlockAllGather::Params;
    using CommBackend = CommBackend_;
    using BackendParams = typename CommBackend::Params;

    using ElementGatherSrc = typename BlockAllGather::ElementSrc;
    using LayoutGatherSrc = typename BlockAllGather::LayoutSrc;

    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockAllGatherScheduler_;
    using BlockCommParams = typename CommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params
    {
        // Data members
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;

        uint32_t commInterval;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        LayoutGatherSrc layoutGatherSrc;
        GM_ADDR ptrSymmetric;
        BackendParams backendParams;

        BlockAllGatherParams allGatherParams;
        BlockCommParams commParams;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t rank_, uint32_t rankSize_, uint32_t commInterval_,
               LayoutGatherSrc layoutGatherSrc_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_,
               LayoutB const &layoutB_, GM_ADDR ptrC_, LayoutC const &layoutC_, GM_ADDR ptrSymmetric_,
               BlockAllGatherParams const &allGatherParams_, BlockCommParams const &commParams_)
            : problemShape(problemShape_),
              rankIdx(rank_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutGatherSrc(layoutGatherSrc_),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_),
              ptrSymmetric(ptrSymmetric_),
              backendParams(ptrSymmetric_),
              allGatherParams(allGatherParams_),
              commParams(commParams_)
        {
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t rank_, uint32_t rankSize_, uint32_t commInterval_,
               LayoutGatherSrc layoutGatherSrc_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_,
               LayoutB const &layoutB_, GM_ADDR ptrC_, LayoutC const &layoutC_,
               BlockAllGatherParams const &allGatherParams_, BlockCommParams const &commParams_,
               BackendParams const &backendParams_)
            : problemShape(problemShape_),
              rankIdx(rank_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutGatherSrc(layoutGatherSrc_),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_),
              ptrSymmetric(nullptr),
              backendParams(backendParams_),
              allGatherParams(allGatherParams_),
              commParams(commParams_)
        {
        }
    };

    struct Arguments
    {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static size_t GetWorkspaceSize(Arguments const &args) { return 0; }

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        (void)workspace;
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        LayoutGatherSrc layoutGatherSrc{m, k, k};
        Catlass::layout::RowMajor layoutTagB{k, n, n};
        Catlass::layout::RowMajor layoutTagC{m * args.rankSize, n, n};

        auto layoutA = tla::MakeLayoutFromTag(layoutGatherSrc);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);

        typename BlockAllGather::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockAllGatherParams allGatherParams{args.commBlockShape, tileParams};
        BlockCommParams commParams{args.commCoreSplit};

        return Params(args.problemShape, args.rankIdx, args.rankSize, args.commInterval, layoutGatherSrc, args.ptrA,
                      layoutA, args.ptrB, layoutB, args.ptrC, layoutC, args.ptrSymmetric, allGatherParams, commParams);
    }

    // Methods
    CATLASS_DEVICE
    Ascend950AllGatherMatmulWithLocalOptionalBackend()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t *timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr)
        {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx)
        {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~Ascend950AllGatherMatmulWithLocalOptionalBackend()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        CommBackend commBackend;
        commBackend.Init(params.backendParams);
        uint32_t rankSize = commBackend.GetRankSize();

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(commBackend.GetPeerMem()));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        int64_t symmetricStrideK = params.problemShape.k();
        if constexpr (!IsAscend950CollectiveAllGatherBackend<CommBackend>::value)
        {
            symmetricStrideK =
                RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA));
        }
        auto layoutTagSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * rankSize * commSizeM, params.problemShape.k(), symmetricStrideK);
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        auto tensorPeerMem = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx)
        {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(actualCommSizeM, params.problemShape.n(),
                                                                   params.problemShape.k(), rankSize);
            BlockScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum)
            {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto tensorBlockA = GetTile(tensorPeerMem, tla::MakeCoord(offsetA.row(), offsetA.column()),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(offsetC.row(), offsetC.column()),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape.GetCoordMNK());
            }
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif

            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES)
            {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        CommBackend commBackend;
        if constexpr (IsAscend950CollectiveAllGatherBackend<CommBackend>::value) {
            if (!commBackend.Init(params.backendParams)) {
                return;
            }
        } else {
            commBackend.Init(params.backendParams);
        }
        uint32_t rankSize = commBackend.GetRankSize();

        if constexpr (IsAscend950CollectiveAllGatherBackend<CommBackend>::value)
        {
            uint64_t rankStride = static_cast<uint64_t>(commSizeM) * params.problemShape.k();
            uint64_t stageStride = rankStride * rankSize;

            for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx)
            {
                uint32_t stageId = commIdx % WORKSPACE_STAGES;
                if (commIdx >= WORKSPACE_STAGES)
                {
                    if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                        AscendC::PRINTF("[AGMM_DIAG] AIV wait AIC workspace, seq=%u, stage=%u\n", commIdx,
                                        stageId);
                    }
                    Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
                    if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                        AscendC::PRINTF("[AGMM_DIAG] AIV workspace ready, seq=%u, stage=%u\n", commIdx,
                                        stageId);
                    }
                }

#ifdef ENABLE_TIMER
                timer.Tik(AscendTimer::AIV);
#endif
                uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
                uint64_t sendCount = static_cast<uint64_t>(actualCommSizeM) * params.problemShape.k();
                GM_ADDR sendBuffer = reinterpret_cast<GM_ADDR>(
                    params.ptrA + static_cast<uint64_t>(commIdx) * commSizeM * params.problemShape.k());
                GM_ADDR recvBuffer = commBackend.GetPeerMem() + stageId * stageStride * sizeof(ElementA);

                if (!commBackend.AllGather(sendBuffer, recvBuffer, sendCount, rankStride)) {
                    return;
                }

                // The HCCL collective is executed by AIV core 0. Synchronize all
                // AIV cores before any of them wakes its paired AIC, otherwise
                // an AIC may read the gather workspace while core 0 is still
                // completing the communication.
                AscendC::SyncAll<true>();

#ifdef ENABLE_TIMER
                timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
            }
            commBackend.Finalize();
            return;
        }

        uint32_t rankIdx = commBackend.GetRankIdx();
        BlockAllGather allGather(resource, params.allGatherParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(commBackend.GetPeerMem()));

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx)
        {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES)
            {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }
            commBackend.CrossRankSync();

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, rankIdx, 0)), 0};

            allGather.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum)
            {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum)
                {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank();

                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmA[params.layoutGatherSrc.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutGatherSrc.GetTileLayout(actualCommBlockShape);

                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    if constexpr (IsAscend950OptionalBackendLocalCopyDispatch<typename BlockAllGather::DispatchPolicy>::value)
                    {
                        AscendC::GlobalTensor<ElementA> gmRemoteSymmetric;
                        gmRemoteSymmetric.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementA *>(commBackend.GetPeerMem(remoteRankIdx)));
                        auto gmBlockDst = gmRemoteSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        allGather(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape);
                    }
                    else
                    {
                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        allGather(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape,
                                  remoteRankIdx % rankSize);
                    }
                }
            }
            allGather.FinalizeBlockLoop();

            commBackend.CrossRankSync();

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
    }

   private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_LOCAL_OPTIONAL_BACKEND_HPP
