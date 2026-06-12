#ifndef CATCOC_DGEMM_KERNEL_MX_ALLGATHER_MATMUL_HPP
#define CATCOC_DGEMM_KERNEL_MX_ALLGATHER_MATMUL_HPP

#include "catccos/catccos.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

using namespace AscendC;
using namespace tla;

template <
    class BlockMmad_,
    class BlockAllGather_,
    class BlockAllGatherScale_,
    class BlockScheduler_,
    class BlockAllGatherScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MxAllGatherMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;

    using BlockAllGather = BlockAllGather_;
    using BlockAllGatherParams = typename BlockAllGather::Params;

    using BlockAllGatherScale = BlockAllGatherScale_;
    using BlockAllGatherScaleParams = typename BlockAllGatherScale::Params;

    using LayoutTagMxScaleA = typename BlockAllGatherScale::LayoutSrc;

    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockAllGatherScheduler_;
    using BlockCommParams = typename CommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
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
        __gm__ ElementMxScaleA *ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB *ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        LayoutTagA layoutTagA;
        LayoutTagMxScaleA layoutTagMxScaleA;
        GM_ADDR ptrSymmetric;

        BlockAllGatherParams allGatherParams;
        BlockAllGatherScaleParams allGatherScaleParams;
        BlockCommParams commParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            uint32_t commInterval_,
            LayoutTagA layoutGatherSrc_, LayoutTagMxScaleA layoutGatherMxScaleSrc_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrMxScaleA_, LayoutMxScaleA const &layoutMxScaleA_,
            GM_ADDR ptrMxScaleB_, LayoutMxScaleB const &layoutMxScaleB_,
            GM_ADDR ptrSymmetric_,
            BlockAllGatherParams const &allGatherParams_,
            BlockAllGatherScaleParams const &allGatherScaleParams_,
            BlockCommParams const &commParams_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            commInterval(commInterval_),
            layoutTagA(layoutGatherSrc_), layoutTagMxScaleA(layoutGatherMxScaleSrc_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA *>(ptrMxScaleA_)), layoutMxScaleA(layoutMxScaleA_),
            ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB *>(ptrMxScaleB_)), layoutMxScaleB(layoutMxScaleB_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            allGatherScaleParams(allGatherScaleParams_),
            commParams(commParams_)
        {
        }
    };

    /// User API arguments
    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrMxScaleA;
        GM_ADDR ptrMxScaleB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        LayoutTagA tagA{args.problemShape.m(), args.problemShape.k()};
        LayoutTagB tagB{args.problemShape.k(), args.problemShape.n()};
        LayoutTagC tagC{args.problemShape.m() * args.rankSize, args.problemShape.n(), args.problemShape.n()};
        LayoutTagA tagMxScaleA{args.problemShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(args.problemShape.k())};

        uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(args.problemShape.k());
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(args.problemShape.m(), mxScaleK);
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(mxScaleK, args.problemShape.n());

        typename BlockAllGather::TileRemoteCopy::Params tileParams{args.commTileShape};
        typename BlockAllGatherScale::TileRemoteCopy::Params tileScaleParams{args.commTileShape};
        BlockAllGatherParams blockCommParams{args.commBlockShape, tileParams};
        BlockAllGatherScaleParams blockCommScaleParams{args.commBlockShape, tileScaleParams};
        BlockCommParams commSchedulerParams{args.commCoreSplit};

        return Params{
            args.problemShape,
            args.rankIdx, args.rankSize,
            args.commInterval,
            tagA, tagMxScaleA,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            args.ptrMxScaleA, layoutMxScaleA,
            args.ptrMxScaleB, layoutMxScaleB,
            args.ptrSymmetric,
            blockCommParams,
            blockCommScaleParams,
            commSchedulerParams
        };
    }

    CATLASS_DEVICE
    MxAllGatherMatmulTla()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB);

        auto layoutTagSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<ElementA>::value)
        );
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        size_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(params.problemShape.k());
        size_t mxScalePeerMemM = WORKSPACE_STAGES * params.rankSize * commSizeM;
        auto layoutScaleSymmetric = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagMxScaleA, false>(mxScalePeerMemM, mxScaleK);

        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA *>(params.ptrSymmetric) + layoutTagSymmetric.Capacity());

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        auto tensorPeerMem = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutScaleSymmetric, Catlass::Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Catlass::Arch::PositionGM{});

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize
            );
            BlockScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto tensorBlockA = GetTile(tensorPeerMem,
                                            tla::MakeCoord(offsetA.row(), offsetA.column()),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(tensorB,
                                            tla::MakeCoord(offsetB.row(), offsetB.column()),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(offsetA.row(), offsetA.column() / Catlass::MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()))
                );

                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n())
                );

                auto tensorBlockC = GetTile(tensorC,
                                            tla::MakeCoord(offsetC.row(), offsetC.column()),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape.GetCoordMNK(), tensorBlockMxScaleA, tensorBlockMxScaleB);
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                mmad.SynchronizeBlock();
            }

            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockAllGather allGather(resource, params.allGatherParams);
        BlockAllGatherScale allGatherScale(resource, params.allGatherScaleParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA);

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<ElementA>::value)
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));

        AscendC::GlobalTensor<ElementMxScaleA> gmScaleSymmetric;
        gmScaleSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA *>(params.ptrSymmetric) + layoutSymmetric.Capacity());

        size_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(params.problemShape.k());
        auto layoutScaleSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, mxScaleK);

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            // wait aic
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            aclshmemx_barrier_all_vec();

            allGather.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);
                    
                    uint32_t remoteRankIdx = commBlockCoord.rank() % params.rankSize;

                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmA[params.layoutTagA.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutTagA.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    allGather(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualCommBlockShape, remoteRankIdx
                    );
                }
            }
            allGather.FinalizeBlockLoop();

            allGatherScale.InitBlockLoop();
            if (subcoreIdx == 1 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);
                    
                    uint32_t remoteRankIdx = commBlockCoord.rank() % params.rankSize;

                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    MatrixCoord offsetScaleSrc{offsetSrc.row(), offsetSrc.column() / Catlass::MX_SCALE_GROUP_NUM};
                    MatrixCoord offsetScaleDst{offsetDst.row(), offsetDst.column() / Catlass::MX_SCALE_GROUP_NUM};
                    MatrixCoord actualScaleCommBlockShape{actualCommBlockShape.row(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualCommBlockShape.column())};

                    auto gmBlockSrc = gmMxScaleA[params.layoutTagMxScaleA.GetOffset(offsetScaleSrc)];
                    auto layoutBlockSrc = params.layoutTagMxScaleA.GetTileLayout(actualScaleCommBlockShape);

                    auto gmBlockDst = gmScaleSymmetric[layoutScaleSymmetric.GetOffset(offsetScaleDst)];
                    auto layoutBlockDst = layoutScaleSymmetric.GetTileLayout(actualScaleCommBlockShape);

                    allGatherScale(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualScaleCommBlockShape, remoteRankIdx
                    );
                }
            }
            allGatherScale.FinalizeBlockLoop();
            
            // AllGather is completed, waiting until tasks on all devices are complete.
            aclshmemx_barrier_all_vec();

            // set aic
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif  // CATCOC_DGEMM_KERNEL_MX_ALLGATHER_MATMUL_HPP
