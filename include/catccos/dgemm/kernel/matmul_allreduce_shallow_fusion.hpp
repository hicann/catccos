#ifndef CATCCOS_DGEMM_KERNEL_MATMUL_ALLREDUCE_SHALLOW_FUSION_HPP
#define CATCCOS_DGEMM_KERNEL_MATMUL_ALLREDUCE_SHALLOW_FUSION_HPP

#include "catccos/comm/tile/tile_remote_copy_local_ub_to_shmem.hpp"
#include "catccos/comm/tile/tile_remote_copy_shmem_to_local_ub.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {
using namespace Catlass;

enum class PaddingTag { NO_PADDING, PADDING_ND, PADDING_BLOCK_ND, PADDING_NZ };

template <class ArchTag_, class TensorIn_, class TensorOut_, uint32_t COMPUTE_LENGTH>
struct PaddingMatrixNZ {
    using ArchTag = ArchTag_;
    using TensorIn = TensorIn_;
    using TensorOut = TensorOut_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;
    using LayoutOut = typename TensorOut::Layout;

    using ComputeLayoutSrc = Catlass::layout::RowMajor;
    using ComputeLayoutDst = Catlass::layout::zN;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<
        AscendC::LocalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using LayoutInnerDstGm = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<16>, uint32_t>, tla::Shape<tla::Int<16>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<16>, tla::Int<256>>, tla::Stride<tla::Int<1>, int64_t>>>;
    using TensorInnerDstGm = tla::Tensor<
        AscendC::GlobalTensor<Element>, LayoutInnerDstGm, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;

    constexpr static uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    constexpr static uint32_t VNCHW_SIZE = 16;

    static const PaddingTag paddingTag = PaddingTag::PADDING_NZ;
    CATLASS_HOST_DEVICE static LayoutOut GetWorkspaceLayout(const LayoutIn& layout)
    {
        return LayoutOut::template MakeLayout<Element>(layout.shape(0), layout.shape(1));
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols)
    {
        if constexpr (std::is_same_v<LayoutIn, Catlass::layout::RowMajor>) {
            return static_cast<size_t>(RoundUp(rows, Catlass::C0_NUM_PER_FRACTAL)) * RoundUp(cols, ELE_NUM_PER_C0) *
                   sizeof(Element);
        } else {
            return static_cast<size_t>(RoundUp(rows, ELE_NUM_PER_C0)) * RoundUp(cols, Catlass::C0_NUM_PER_FRACTAL) *
                   sizeof(Element);
        }
    }

    CopyGm2Ub copyGm2Ub;

    CATLASS_DEVICE
    PaddingMatrixNZ(Arch::Resource<ArchTag>& resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    void CopyUb2Ub(
        AscendC::LocalTensor<Element> const& dst, AscendC::LocalTensor<Element> const& src,
        ComputeLayoutDst const& layoutDst, ComputeLayoutSrc const& layoutSrc)
    {
        uint32_t loops = CeilDiv(layoutSrc.shape(0), BLK_NUM_PER_VECTOR_FRACTAL);
        for (uint32_t i = 0; i < loops; ++i) {
            uint64_t offsetSrc = i * BLK_NUM_PER_VECTOR_FRACTAL * layoutSrc.stride(0);
            uint64_t offsetDst = i * BLK_NUM_PER_VECTOR_FRACTAL * layoutDst.stride(0);
            uint32_t repeatTimes = CeilDiv(layoutSrc.shape(1), ELE_NUM_PER_C0);
            uint64_t mask = 256 / sizeof(Element);
            AscendC::CopyRepeatParams copyRepeatParams{
                1, static_cast<uint16_t>(layoutSrc.stride(0) / ELE_NUM_PER_C0),
                static_cast<uint16_t>(layoutDst.stride(3) / ELE_NUM_PER_C0), 1};
            AscendC::Copy(dst[offsetDst], src[offsetSrc], mask, repeatTimes, copyRepeatParams);
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void CopyUb2Gm(TensorDst& tensorDst, TensorSrc const& tensorSrc, int row, int col)
    {
        if (tla::get<1, 1>(tensorDst.stride()) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            AscendC::DataCopyParams dataCopyParams(
                tla::get<1>(tla::get<1>(tensorDst.shape())),
                tla::get<0>(tla::get<0>(tensorDst.shape())) * tla::get<1>(tla::get<0>(tensorDst.shape())),
                (tla::get<1>(tla::get<1>(tensorSrc.stride())) - tla::get<0>(tla::get<0>(tensorSrc.shape())) *
                                                                    tla::get<1>(tla::get<0>(tensorSrc.shape())) *
                                                                    tla::get<0>(tla::get<1>(tensorSrc.shape()))) /
                    ELE_NUM_PER_C0,
                (tla::get<1>(tla::get<1>(tensorDst.stride())) - tla::get<0>(tla::get<0>(tensorDst.shape())) *
                                                                    tla::get<1>(tla::get<0>(tensorDst.shape())) *
                                                                    tla::get<0>(tla::get<1>(tensorDst.shape()))) /
                    ELE_NUM_PER_C0);
            auto dstOffset = tensorDst.layout()(tensorDst.coord());
            auto srcOffset = tensorSrc.layout()(tensorSrc.coord());
            AscendC::DataCopy(tensorDst.data()[dstOffset], tensorSrc.data()[srcOffset], dataCopyParams);
        } else {
            uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(col);
            uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(row);
            AscendC::DataCopyParams dataCopyParams(1, blockLen, 0, 0);
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset =
                    tensorDst.layout()(tensorDst.coord()) + i * tla::get<1>(tla::get<1>(tensorDst.stride()));
                uint64_t srcOffset =
                    tensorSrc.layout()(tensorSrc.coord()) + i * tla::get<1>(tla::get<1>(tensorSrc.stride()));
                AscendC::DataCopy(tensorDst.data()[dstOffset], tensorSrc.data()[srcOffset], dataCopyParams);
            }
        }
    }

    template <class Tensor>
    CATLASS_DEVICE auto GetPaddingTensorSrc(Tensor const& tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), tla::MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE auto GetPaddingTensorDst(TensorDst& tensorDst, TensorSrc const& tensorSrc)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(Catlass::BYTE_PER_C0) / SizeOfBits<half>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(Catlass::BYTE_PER_FRACTAL) / SizeOfBits<half>::value;
        if constexpr (std::is_same_v<typename TensorDst::Layout, LayoutInnerDstGm>) {
            return tensorDst;
        } else {
            auto shape = tla::MakeShape(
                tla::MakeShape(
                    C0_NUM_PER_FRACTAL,
                    RoundUp<C0_NUM_PER_FRACTAL>((uint32_t)(tla::get<1>(tensorSrc.shape()))) / C0_NUM_PER_FRACTAL),
                tla::MakeShape(
                    ELE_NUM_PER_C0, RoundUp<ELE_NUM_PER_C0>(tla::get<0>(tensorSrc.shape())) / ELE_NUM_PER_C0));
            auto stride = tla::MakeStride(
                tla::MakeStride(ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL),
                tla::MakeStride(
                    Int<1>{}, RoundUp<C0_NUM_PER_FRACTAL>(tla::get<1>(tensorSrc.shape())) * ELE_NUM_PER_C0));
            return tla::MakeTensor(tensorDst.data(), tla::MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& tensorDst, TensorSrc const& tensorSrc)
    {
        auto PaddingtensorSrc = GetPaddingTensorSrc(tensorSrc);
        auto PaddingtensorDst = GetPaddingTensorDst(tensorDst, tensorSrc);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        uint32_t rows = tla::get<0>(PaddingtensorSrc.shape());
        uint32_t cols = tla::get<1>(PaddingtensorSrc.shape());

        uint32_t refTileRows = 16;
        uint32_t refTileCols = COMPUTE_LENGTH / refTileRows;
        uint32_t tileRows = refTileRows;
        uint32_t tileCols = RoundUp(cols / CeilDiv(cols, refTileCols), ELE_NUM_PER_C0);

        uint32_t rowTiles = CeilDiv(rows, tileRows);
        uint32_t colTiles = CeilDiv(cols, tileCols);
        uint32_t totalTiles = rowTiles * colTiles;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);

        for (uint32_t loopIdx = aivId; loopIdx < totalTiles; loopIdx += aivNum) {
            uint32_t rowTileIdx = loopIdx / colTiles;
            uint32_t colTileIdx = loopIdx % colTiles;
            uint32_t rowTileActual = (rowTileIdx == rowTiles - 1) ? (rows - rowTileIdx * tileRows) : tileRows;
            uint32_t colTileActual = (colTileIdx == colTiles - 1) ? (cols - colTileIdx * tileCols) : tileCols;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            auto offset = tla::MakeCoord(rowTileIdx * tileRows, colTileIdx * tileCols);
            ComputeLayoutSrc ubLayoutIn = ComputeLayoutSrc{rowTileActual, RoundUp(colTileActual, ELE_NUM_PER_C0)};
            auto tensorTileSrc = GetTile(PaddingtensorSrc, offset, tla::MakeShape(rowTileActual, colTileActual));

            auto layoutDstUb = tla::MakeLayout(
                tla::MakeShape(rowTileActual, colTileActual),
                tla::MakeStride(static_cast<int64_t>(RoundUp(colTileActual, ELE_NUM_PER_C0)), tla::Int<1>{}));
            auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

            if ((tileCols < cols) || (cols % ELE_NUM_PER_C0) || (cols != rows)) {
                copyGm2Ub(tensorDstUb, tensorTileSrc);
            } else {
                // A tile of the 2D source would crop the linear copy to one row.
                auto layoutConti = tla::MakeLayout(
                    tla::MakeShape(1, rowTileActual * colTileActual),
                    tla::MakeStride(rowTileActual * colTileActual, tla::Int<1>{}));
                auto tensorTileSrcConti = tla::MakeTensor(
                    tensorTileSrc.data()[tensorTileSrc.layout()(tensorTileSrc.coord())], layoutConti,
                    Arch::PositionGM{});
                auto tensorDstUbConti = tla::MakeTensor(inputBuffer[bufferIndex], layoutConti, Arch::PositionUB{});
                copyGm2Ub(tensorDstUbConti, tensorTileSrcConti);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);
            ComputeLayoutDst ubLayoutOut = ComputeLayoutDst::template MakeLayout<Element>(rowTileActual, colTileActual);
            CopyUb2Ub(outputBuffer[bufferIndex], inputBuffer[bufferIndex], ubLayoutOut, ubLayoutIn);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);

            auto tensorTileDst = GetTile(PaddingtensorDst, offset, tla::MakeShape(rowTileActual, colTileActual));
            constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(Catlass::BYTE_PER_C0) / SizeOfBits<Element>::value;
            constexpr uint32_t ELE_NUM_PER_FRACTAL =
                BytesToBits(Catlass::BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
            auto layoutSrcUb = tla::MakeLayout(
                tla::MakeShape(
                    tla::MakeShape(C0_NUM_PER_FRACTAL, RoundUp<C0_NUM_PER_FRACTAL>(rowTileActual) / C0_NUM_PER_FRACTAL),
                    tla::MakeShape(ELE_NUM_PER_C0, RoundUp<ELE_NUM_PER_C0>(colTileActual) / ELE_NUM_PER_C0)),
                tla::MakeStride(
                    tla::MakeStride(ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL),
                    tla::MakeStride(Int<1>{}, RoundUp<C0_NUM_PER_FRACTAL>(rowTileActual) * ELE_NUM_PER_C0)));
            auto tensorSrcUb = tla::MakeTensor(outputBuffer[bufferIndex], layoutSrcUb, Arch::PositionUB{});
            CopyUb2Gm(tensorTileDst, tensorSrcUb, rows, cols);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);
            bufferIndex = 1 - bufferIndex;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<Element> outputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{0};
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) * 2 <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

} // namespace Catccos::DGemm::Kernel

namespace Catccos::DGemm::Kernel {
using namespace Catlass;

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class PaddingA, class PaddingB>
class MatmulAllReduceShallowFusion {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;

    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using BlockScheduler = BlockScheduler_;

    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        GM_ADDR ptrSymmetric;
        uint32_t rankSize;
        GM_ADDR ptrSignal;
        uint32_t tag;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GemmCoord const& L1Shape_, MatrixCoord const& commBlockShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_, GM_ADDR ptrSymmetric_,
            uint32_t rankSize_, GM_ADDR ptrSignal_, uint32_t tag_)
            : problemShape(problemShape_),
              L1Shape(L1Shape_),
              commBlockShape(commBlockShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrWA(ptrWA_),
              layoutWA(layoutWA_),
              ptrWB(ptrWB_),
              layoutWB(layoutWB_),
              ptrSymmetric(ptrSymmetric_),
              rankSize(rankSize_),
              ptrSignal(ptrSignal_),
              tag(tag_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrC;
        LayoutC layoutC;
        uint8_t* ptrWA;
        LayoutWA layoutWA;
        uint8_t* ptrWB;
        LayoutWB layoutWB;
        uint8_t* ptrSymmetric;
        uint32_t rankSize;
        uint8_t* ptrSignal;
        uint32_t tag;
    };
    static bool CanImplement(const Arguments& args) { return true; }

    static size_t GetWorkspaceSize(const Arguments& args) { return 0; }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape, args.L1Shape, args.commBlockShape, args.ptrA,         args.layoutA,
                      args.ptrB,         args.layoutB, args.ptrC,           args.layoutC,      args.ptrWA,
                      args.layoutWA,     args.ptrWB,   args.layoutWB,       args.ptrSymmetric, args.rankSize,
                      args.ptrSignal,    args.tag};
        return params;
    }

    CATLASS_DEVICE
    MatmulAllReduceShallowFusion() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    // AIC: compute rank-local matmul results into symmetric memory.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t rankSize = aclshmem_n_pes();
        int64_t rankId = aclshmem_my_pe();
        uint32_t aicId = AscendC::GetBlockIdx();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        GemmCoord globalProblemShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(params.L1Shape, resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrSymmetric);

        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        BlockScheduler matmulBlockScheduler(globalProblemShape, MakeCoord(params.L1Shape.m(), params.L1Shape.n()));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if ((loopIdx) % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }

            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * params.L1Shape.k(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() + AscendC::GetBlockNum()) % AscendC::GetBlockNum());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * params.L1Shape.m(), nextBlockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k()));
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * params.L1Shape.k(), nextBlockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n()));

            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, actualBlockShape,
                nextActualBlockShape, isFirstBlock, hasNextBlock);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();

        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishAllReduce);
    }

    // AIV: prepare padded inputs and reduce rank-local matmul results.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        aclshmemx_barrier_all_vec();
        int signaltag = params.tag;

        __gm__ int32_t* sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSignal);

        uint32_t rankSize = aclshmem_n_pes();
        int64_t rankId = aclshmem_my_pe();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorWA = tla::MakeTensor(gmWA, params.layoutWA, Arch::PositionGM{});
            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }

        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorWB = tla::MakeTensor(gmWB, params.layoutWB, Arch::PositionGM{});
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
        }

        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::PipeBarrier<PIPE_ALL>();

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmWC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        gmWC.SetGlobalBuffer((__gm__ ElementC*)params.ptrSymmetric);

        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorSymmtric = tla::MakeTensor(gmWC, params.layoutC, Arch::PositionGM{});

        // Reuse communication buffers and event IDs across transfer steps.
        AscendC::LocalTensor<ElementC> commTmpBuffer[COMM_BUFFER_NUM];
        uint64_t commBufferOffset = 0;
        for (uint32_t i = 0; i < COMM_BUFFER_NUM; ++i) {
            commTmpBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementC>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::LocalTensor<ElementC> commAddBuffer[1];
        for (uint32_t i = 0; i < 1; ++i) {
            commAddBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementC>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::TEventID commEventIds[COMM_BUFFER_NUM] = {EVENT_ID0};

        MatrixCoord actualCommBlockShape{params.commBlockShape.row(), params.commBlockShape.column()};

        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAivFinishAllReduce);

        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        AscendC::PipeBarrier<PIPE_ALL>();

        if (((actualCommBlockShape.row()) <= (aivNum / 2))) {
            int sp = (aivNum / 2) / (actualCommBlockShape.row());
            int targetid = rankId / 2;
            targetid = targetid * 2 + (rankId + 1) % 2;
            aclshmemx_signal_op(
                sigAddr + ((2 * aivNum * rankSize) + rankId) * 128, signaltag, ACLSHMEM_SIGNAL_SET, targetid);
            AllReduceSplitN(
                tensorC, tensorSymmtric, sigAddr, sp, signaltag, commTmpBuffer, commAddBuffer, commEventIds);
        } else {
            int targetid = rankId / 2;
            targetid = targetid * 2 + (rankId + 1) % 2;
            aclshmemx_signal_op(
                sigAddr + ((2 * aivNum * rankSize) + rankId) * 128, signaltag, ACLSHMEM_SIGNAL_SET, targetid);
            AllReduce(tensorC, tensorSymmtric, sigAddr, signaltag, commTmpBuffer, commAddBuffer, commEventIds);
        }
    }

private:
    static constexpr uint32_t COMM_BUFFER_NUM = 1;
    static constexpr uint32_t COMM_BUFFER_BYTES = 96 * 1024;
    static constexpr uint32_t COMM_COMPUTE_LENGTH = COMM_BUFFER_BYTES / sizeof(ElementC);

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<
        AscendC::LocalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm =
        tla::Tensor<AscendC::GlobalTensor<ElementC>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyUb2GM = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerUb, TensorInnerDstGm>;
    using TileRemoteCopyShmemToLocalUb =
        Catccos::Comm::Tile::TileRemoteCopyShmemToLocalUb<ArchTag, TensorInnerUb, TensorInnerSrcGm>;
    using TileRemoteCopyLocalUbToShmem =
        Catccos::Comm::Tile::TileRemoteCopyLocalUbToShmem<ArchTag, TensorInnerDstGm, TensorInnerUb>;

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE void AllReduce(
        TensorOut& tensorDst, TensorIn& tensorSrc, __gm__ int32_t* signal, int32_t tag,
        AscendC::LocalTensor<ElementC> (&tmpBuffer)[COMM_BUFFER_NUM], AscendC::LocalTensor<ElementC> (&addBuffer)[1],
        AscendC::TEventID const (&eventIds)[COMM_BUFFER_NUM])
    {
        CopyUb2GM copyUb2Gm;
        TileRemoteCopyShmemToLocalUb copyShmemToLocalUb;
        TileRemoteCopyLocalUbToShmem copyLocalUbToShmem;
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicoreId = AscendC::GetBlockIdx() / 2;
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        int64_t rankId = aclshmem_my_pe();
        int64_t rankSize = aclshmem_n_pes();

        uint32_t tilesNum = tla::get<0>(tensorSrc.shape());
        uint32_t tileLen = tla::get<1>(tensorSrc.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(ElementC)>(tileLen);

        bool singleaiv = 1;
        uint32_t tilesPerAiv = tilesNum / aicoreNum;
        uint32_t tileRemain = tilesNum % aicoreNum;
        if (aicoreId < tileRemain) {
            tilesPerAiv++;
        }
        int buffersize = tilesPerAiv * roundTileLen;
        if ((tileRemain != 0) && (aicoreId >= tileRemain)) {
            buffersize += roundTileLen;
        }
        if (buffersize > COMM_COMPUTE_LENGTH) {
            singleaiv = 0;
            tilesPerAiv = tilesNum / aivNum;
            tileRemain = tilesNum % aivNum;
            if (aivId < tileRemain) {
                tilesPerAiv++;
            }
        }

        uint32_t mIdx = singleaiv ? aicoreId * tilesPerAiv : aivId * tilesPerAiv;
        if ((singleaiv) && (aicoreId >= tileRemain)) {
            mIdx += tileRemain;
        } else if ((!singleaiv) && (aivId >= tileRemain)) {
            mIdx += tileRemain;
        }

        uint32_t coreLoops{0};
        uint32_t tilesPerLoop = COMM_COMPUTE_LENGTH / roundTileLen;
        coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;

        auto layoutWorkspace = tla::MakeLayout(
            tla::MakeShape(static_cast<uint32_t>(1), static_cast<uint32_t>(1)),
            tla::MakeStride(static_cast<int64_t>(1), tla::Int<1>{}));
        auto tensorWorkspace = tla::MakeTensor(tmpBuffer[0], layoutWorkspace, Arch::PositionUB{});

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            if ((singleaiv) && (aivSubId == 1)) {
                continue;
            }

            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = tilesPerLoop;

            if (tilesPerAiv - tileIdx < tilesPerLoop) {
                actualTilesNum = tilesPerAiv - tileIdx;
            }

            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            auto tensorTileSymmSrc = GetTile(tensorSrc, offset, tla::MakeShape(actualTilesNum, tileLen));

            auto layoutUb = tla::MakeLayout(
                tla::MakeShape(actualTilesNum, tileLen),
                tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));
            auto tensoraddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            copyShmemToLocalUb(tensoraddUb, tensorTileSymmSrc, rankId, eventIds[0]);

            for (int i = 1; i < rankSize; i *= 2) {
                auto tensorUb = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
                int targetid = rankId / (i * 2);
                targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                int nextid = -1;
                if ((i * 2) < rankSize) {
                    nextid = rankId / (i * 4);
                    nextid = nextid * (i * 4) + (rankId + (i * 2)) % (i * 4);
                }

                if (i > 1) {
                    aclshmem_signal_wait_until(signal + ((aivId * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                } else {
                    aclshmem_signal_wait_until(
                        signal + ((2 * aivNum * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                }
                copyShmemToLocalUb(tensorUb, tensorTileSymmSrc, targetid, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                aclshmemx_signal_op(
                    signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 128, tag, ACLSHMEM_SIGNAL_SET,
                    targetid);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], (actualTilesNum * roundTileLen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                if (nextid != -1) {
                    aclshmem_signal_wait_until(
                        signal + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    copyLocalUbToShmem(tensorTileSymmSrc, tensoraddUb, rankId, eventIds[0]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    aclshmemx_signal_op(signal + (aivId * rankSize + rankId) * 128, tag, ACLSHMEM_SIGNAL_SET, nextid);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileA = GetTile(tensorDst, offset, tla::MakeShape(actualTilesNum, tileLen));
            copyUb2Gm(tensorTileA, tensoraddUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    }

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE void AllReduceSplitN(
        TensorOut& tensorDst, TensorIn& tensorSrc, __gm__ int32_t* signal, int spl, int32_t tag,
        AscendC::LocalTensor<ElementC> (&tmpBuffer)[COMM_BUFFER_NUM], AscendC::LocalTensor<ElementC> (&addBuffer)[1],
        AscendC::TEventID const (&eventIds)[COMM_BUFFER_NUM])
    {
        CopyUb2GM copyUb2Gm;
        TileRemoteCopyShmemToLocalUb copyShmemToLocalUb;
        TileRemoteCopyLocalUbToShmem copyLocalUbToShmem;
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicoreId = AscendC::GetBlockIdx() / 2;
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        int64_t rankId = aclshmem_my_pe();
        int64_t rankSize = aclshmem_n_pes();

        int split = spl;
        uint32_t tilesNum = tla::get<0>(tensorSrc.shape()) * split;
        uint32_t tileLen = (tla::get<1>(tensorSrc.shape()) + split - 1) / split;
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(ElementC)>(tileLen);
        split = (tla::get<1>(tensorSrc.shape()) + roundTileLen - 1) / roundTileLen;
        tilesNum = tla::get<0>(tensorSrc.shape()) * split;
        tileLen = (tla::get<1>(tensorSrc.shape()) + split - 1) / split;
        roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(ElementC)>(tileLen);

        uint32_t tilesPerAiv = tilesNum / aicoreNum;
        uint32_t tileRemain = tilesNum % aicoreNum;
        if (aicoreId < tileRemain) {
            tilesPerAiv++;
        }

        auto layoutWorkspace = tla::MakeLayout(
            tla::MakeShape(static_cast<uint32_t>(1), static_cast<uint32_t>(1)),
            tla::MakeStride(static_cast<int64_t>(1), tla::Int<1>{}));
        auto tensorWorkspace = tla::MakeTensor(tmpBuffer[0], layoutWorkspace, Arch::PositionUB{});

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        for (int splitloopid = 0; splitloopid < tilesNum; splitloopid++) {
            if (aivSubId == 1) {
                continue;
            }
            if ((splitloopid % aicoreNum) != aicoreId) {
                continue;
            }

            uint32_t midx = splitloopid / split;
            uint32_t sidx = splitloopid % split;

            uint32_t actualTilesLen = roundTileLen;
            if (sidx == (split - 1)) {
                actualTilesLen = tla::get<1>(tensorSrc.shape()) - (sidx * roundTileLen);
            }

            auto offset = tla::MakeCoord(midx, sidx * roundTileLen);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            auto tensorTileSymmSrc = GetTile(tensorSrc, offset, tla::MakeShape(1, actualTilesLen));

            auto layoutUb = tla::MakeLayout(
                tla::MakeShape(1, actualTilesLen),
                tla::MakeStride(
                    static_cast<int64_t>(RoundUp<BYTE_PER_BLK / sizeof(ElementC)>(tla::get<1>(tensorSrc.shape()))),
                    tla::Int<1>{}));
            auto tensoraddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            copyShmemToLocalUb(tensoraddUb, tensorTileSymmSrc, rankId, eventIds[0]);

            int finalid = rankId;
            for (int i = 1; i < rankSize; i *= 2) {
                auto tensorUb = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
                int targetid = rankId / (i * 2);
                targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                int nextid = -1;
                if ((i * 2) < rankSize) {
                    nextid = rankId / (i * 4);
                    nextid = nextid * (i * 4) + (rankId + (i * 2)) % (i * 4);
                }

                if (i > 1) {
                    aclshmem_signal_wait_until(signal + ((aivId * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                } else {
                    aclshmem_signal_wait_until(
                        signal + ((2 * aivNum * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                }
                copyShmemToLocalUb(tensorUb, tensorTileSymmSrc, targetid, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                aclshmemx_signal_op(
                    signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 128, tag, ACLSHMEM_SIGNAL_SET,
                    targetid);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], (actualTilesLen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                if (nextid != -1) {
                    aclshmem_signal_wait_until(
                        signal + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 128, ACLSHMEM_CMP_EQ, tag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    copyLocalUbToShmem(tensorTileSymmSrc, tensoraddUb, rankId, eventIds[0]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    aclshmemx_signal_op(signal + (aivId * rankSize + rankId) * 128, tag, ACLSHMEM_SIGNAL_SET, nextid);
                } else {
                    finalid = targetid;
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileA = GetTile(tensorDst, offset, tla::MakeShape(1, actualTilesLen));
            copyUb2Gm(tensorTileA, tensoraddUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    }

    static constexpr Arch::FlagID FLAG_AIV_FINISH_ALLREDUCE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_ALLREDUCE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishAllReduce{FLAG_AIV_FINISH_ALLREDUCE, RV_FLAG_AIV_FINISH_ALLREDUCE};

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 2;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};

    static constexpr Arch::FlagID FLAG_START_STORE = 3;
    Arch::CrossCoreFlag flagstart{FLAG_START_STORE};

    Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_MATMUL_ALLREDUCE_SHALLOW_FUSION_HPP
