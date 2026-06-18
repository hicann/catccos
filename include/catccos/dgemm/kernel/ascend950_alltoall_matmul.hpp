/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_HPP

#include "catccos/catccos.hpp"

// from catlass
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::DGemm::Kernel
{

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/// AllToAll + MatMul kernel for Ascend950.
///
/// Uniform AllToAll: each rank has A of shape (M x K), splits A into rankSize
/// equal chunks of (chunkM x K) along M dimension, and sends chunk[j] to rank j.
/// After AllToAll, each rank receives one chunk from every rank, then performs
/// MatMul: received_data (M x K) x B (K x N) = C (M x N).
///
/// Key differences from AllGather+MatMul:
///   - AIV: source offset = destRank * chunkM + commIdx * commSizeM (different data per dest)
///   - AIC: output C is M x N (not M*R x N), offset = srcRank * chunkM + commIdx * commSizeM
///   - commLoops = ceil(chunkM / commSizeM) instead of ceil(M / commSizeM)
template <class BlockMmad_, class BlockAllToAll_, class BlockScheduler_, class BlockAllToAllScheduler_,
          uint32_t WORKSPACE_STAGES_>
class AllToAllMatmul
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

    using BlockAllToAll = BlockAllToAll_;
    using BlockAllToAllParams = typename BlockAllToAll::Params;

    using ElementAllToAllSrc = typename BlockAllToAll::ElementSrc;
    using LayoutAllToAllSrc = typename BlockAllToAll::LayoutSrc;

    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockAllToAllScheduler_;
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
        LayoutAllToAllSrc layoutAllToAllSrc;
        GM_ADDR ptrSymmetric;

        BlockAllToAllParams allToAllParams;
        BlockCommParams commParams;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t rank_, uint32_t rankSize_, uint32_t commInterval_,
               LayoutAllToAllSrc layoutAllToAllSrc_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_,
               LayoutB const &layoutB_, GM_ADDR ptrC_, LayoutC const &layoutC_, GM_ADDR ptrSymmetric_,
               BlockAllToAllParams const &allToAllParams_, BlockCommParams const &commParams_)
            : problemShape(problemShape_),
              rankIdx(rank_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutAllToAllSrc(layoutAllToAllSrc_),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)),
              layoutC(layoutC_),
              ptrSymmetric(ptrSymmetric_),
              allToAllParams(allToAllParams_),
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

        LayoutAllToAllSrc layoutAllToAllSrc{m, k, k};
        Catlass::layout::RowMajor layoutTagB{k, n, n};
        // AllToAll output: M x N (not M*rankSize x N like AllGather)
        Catlass::layout::RowMajor layoutTagC{m, n, n};

        auto layoutA = tla::MakeLayoutFromTag(layoutAllToAllSrc);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);

        typename BlockAllToAll::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockAllToAllParams allToAllParams{args.commBlockShape, tileParams};
        BlockCommParams commParams{args.commCoreSplit};

        return Params(args.problemShape, args.rankIdx, args.rankSize, args.commInterval, layoutAllToAllSrc, args.ptrA,
                      layoutA, args.ptrB, layoutB, args.ptrC, layoutC, args.ptrSymmetric, allToAllParams, commParams);
    }

    // Methods
    CATLASS_DEVICE
    AllToAllMatmul()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx)
        {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    /// AIC (Cube cores): Read received data from symmetric memory, compute MatMul, write to C.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};

        // chunkM: rows per rank destination (M / rankSize)
        uint32_t chunkM = params.problemShape.m() / params.rankSize;
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(chunkM, commSizeM);

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        // Symmetric memory layout: [WORKSPACE_STAGES * rankSize * commSizeM] x K
        auto layoutTagSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        auto tensorPeerMem = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

        // C layout row logic: offset(srcRank, commIdx) = srcRank * chunkM + commIdx * commSizeM
        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(chunkM, commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx)
        {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, chunkM - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(actualCommSizeM, params.problemShape.n(),
                                                                   params.problemShape.k(), params.rankSize);
            BlockScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);

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

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// AIV (Vector cores): Perform AllToAll communication - put different chunks to different ranks.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        // chunkM: rows per rank destination (M / rankSize)
        uint32_t chunkM = params.problemShape.m() / params.rankSize;
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(chunkM, commSizeM);

        BlockAllToAll allToAll(resource, params.allToAllParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.allToAllParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx)
        {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES)
            {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }
            aclshmemx_barrier_all_vec();

            uint32_t actualCommSizeM = Min(commSizeM, chunkM - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            // Destination offset in symmetric memory: stageId, myRankIdx, 0
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            allToAll.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum)
            {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum)
                {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank();

                    // KEY DIFFERENCE from AllGather:
                    // AllGather src: commIdx * commSizeM + blockOffset  (same for all dest ranks)
                    // AllToAll src:  remoteRankIdx * chunkM + commIdx * commSizeM + blockOffset  (different per dest)
                    MatrixCoord commSrcOffset{remoteRankIdx * chunkM + commIdx * commSizeM, 0};
                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmA[params.layoutAllToAllSrc.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutAllToAllSrc.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    allToAll(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape,
                              remoteRankIdx % params.rankSize);
                }
            }
            allToAll.FinalizeBlockLoop();

            aclshmemx_barrier_all_vec();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
    }

   private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_HPP
