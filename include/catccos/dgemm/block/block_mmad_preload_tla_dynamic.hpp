/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_DYNAMIC_HPP
#define CATCCOS_DGEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_DYNAMIC_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Block {
using namespace Catlass;
using namespace Catlass::Gemm;
template <
    class DispatchPolicy, class ElementA, class ElementB, class ElementC, class ElementBias, class TileCopy,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A> >
struct BlockMmadTlaDynamic {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};
} // namespace Catccos::DGemm::Block

namespace Catccos::DGemm::Block {
using namespace Catlass;
using namespace Catlass::Gemm;

template <
    bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_, class TensorA_, class TensorB_, class TensorC_, class TensorBias_,
    class TileCopy_, class TileMmad_>
struct BlockMmadTlaDynamic<
    MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>, TensorA_, TensorB_, TensorC_, TensorBias_, TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TensorA = TensorA_;
    using TensorB = TensorB_;
    using TensorC = TensorC_;
    using ElementA = typename TensorA::Element;
    using LayoutA = typename TensorA::Layout;
    using ElementB = typename TensorB::Element;
    using LayoutB = typename TensorB::Layout;
    using ElementC = typename TensorC::Element;
    using LayoutC = typename TensorC::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator = typename CopyL0CToGm::ElementSrc;

    using LayoutTagL1A = typename TileCopy_::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy_::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy_::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy_::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmadTlaDynamic(GemmCoord const& l1TileShape_, Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
        : l1TileShape(l1TileShape_)
    {
        uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
        uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;
        kPartLenMax =
            min(L0A_PINGPONG_BUF_SIZE / sizeof(ElementA) / l1TileShape.m() / L1AAlignHelper::ELE_NUM_PER_C0 *
                    L1AAlignHelper::ELE_NUM_PER_C0,
                L0B_PINGPONG_BUF_SIZE / sizeof(ElementB) / l1TileShape.n() / L1BAlignHelper::ELE_NUM_PER_C0 *
                    L1BAlignHelper::ELE_NUM_PER_C0);

        // L1 tile size
        L1A_TILE_SIZE = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        L1B_TILE_SIZE = l1TileShape.n() * l1TileShape.k() * sizeof(ElementB);
        // L0 tile size
        L0A_TILE_SIZE = l1TileShape.m() * kPartLenMax * sizeof(ElementA);
        L0B_TILE_SIZE = kPartLenMax * l1TileShape.n() * sizeof(ElementB);
        L0C_TILE_SIZE = l1TileShape.m() * l1TileShape.n() * sizeof(ElementAccumulator);

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stage.
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);

            // Assign event IDs for each stage.
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            // Mark each buffer as reusable before the first iteration.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTlaDynamic()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(
        TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, TensorA& tensorNextA, TensorB& tensorNextB,
        GemmCoord const& actualShape, GemmCoord const& actualShapeNext, bool isFirstBlock, bool hasNextBlock)
    {
        auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(l1TileShape.m(), l1TileShape.k());
        auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL1B>(l1TileShape.k(), l1TileShape.n());

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t mNextBlockActual = actualShapeNext.m();
        uint32_t kNextBlockActual = actualShapeNext.k();
        uint32_t nNextBlockActual = actualShapeNext.n();

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mBlockActual);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nBlockActual);

        auto layoutInL0C = tla::MakeLayoutL0C(mRound, nRound);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }

        uint32_t kTileCount = (kBlockActual + l1TileShape.k() - 1) / l1TileShape.k();
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? l1TileShape.k() : (kBlockActual - firstTileIdx * l1TileShape.k());
        uint32_t kTileCountNext = (kNextBlockActual + l1TileShape.k() - 1) / l1TileShape.k();
        uint32_t firstTileIdxNext = startTileIdx % kTileCountNext;

        if (isFirstBlock) {
            // load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListId], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTile(
                tensorA, tla::MakeCoord(0, firstTileIdx * l1TileShape.k()), tla::MakeShape(mBlockActual, kActual));
            copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mBlockActual, kActual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListId], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(
                tensorB, tla::MakeCoord(firstTileIdx * l1TileShape.k(), 0), tla::MakeShape(kActual, nBlockActual));
            copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActual, nBlockActual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }

        uint32_t mPartLoop = (mRound + l1TileShape.m() - 1) / l1TileShape.m();
        uint32_t nPartLoop = (nRound + l1TileShape.n() - 1) / l1TileShape.n();

        // main loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ? l1TileShape.k() :
                                                                   (kBlockActual - shuffleKIdxNext * l1TileShape.k());

                auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListIdNext], L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListIdNext], L1B_LAYOUT, Arch::PositionL1{});
                auto tensorTileA = GetTile(
                    tensorA, tla::MakeCoord(0, shuffleKIdxNext * l1TileShape.k()),
                    tla::MakeShape(mBlockActual, kActualNext));
                auto tensorTileB = GetTile(
                    tensorB, tla::MakeCoord(shuffleKIdxNext * l1TileShape.k(), 0),
                    tla::MakeShape(kActualNext, nBlockActual));

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mBlockActual, kActualNext));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActualNext, nBlockActual));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Prefetch the first K tile of the next output block into the next L1 stage.
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                kActualNext = (firstTileIdxNext < kTileCountNext - 1) ?
                                  l1TileShape.k() :
                                  (kNextBlockActual - firstTileIdxNext * l1TileShape.k());

                // Get L1 tensor for next stage
                auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListIdNext], L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListIdNext], L1B_LAYOUT, Arch::PositionL1{});
                // Get GM tile for next stage
                auto tensorTileA = GetTile(
                    tensorNextA, tla::MakeCoord(0, firstTileIdxNext * l1TileShape.k()),
                    tla::MakeShape(mNextBlockActual, kActualNext));
                auto tensorTileB = GetTile(
                    tensorNextB, tla::MakeCoord(firstTileIdxNext * l1TileShape.k(), 0),
                    tla::MakeShape(kActualNext, nNextBlockActual));

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mNextBlockActual, kActualNext));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActualNext, nNextBlockActual));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
            auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kPartLoop = (kActual + kPartLenMax - 1) / kPartLenMax;
            ;

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual =
                    (mPartIdx < mPartLoop - 1) ? l1TileShape.m() : (mRound - mPartIdx * l1TileShape.m());

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual =
                        (kPartIdx < kPartLoop - 1) ? kPartLenMax : (kActual - kPartIdx * kPartLenMax);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mPartActual, kPartActual);
                    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTile(
                        tensorL1A, tla::MakeCoord(mPartIdx * l1TileShape.m(), kPartIdx * kPartLenMax),
                        tla::MakeShape(mPartActual, kPartActual));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual =
                            (nPartIdx < nPartLoop - 1) ? l1TileShape.n() : (nRound - nPartIdx * l1TileShape.n());

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kPartActual, nPartActual);
                        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                        // Locate the current tile of matrix B on L1
                        auto tensorTileL1B = GetTile(
                            tensorL1B, tla::MakeCoord(kPartIdx * kPartLenMax, nPartIdx * l1TileShape.n()),
                            tla::MakeShape(kPartActual, nPartActual));

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        auto tensorTileL0C = GetTile(
                            tensorL0C, tla::MakeCoord(mPartIdx * l1TileShape.m(), nPartIdx * l1TileShape.n()),
                            tla::MakeShape(mPartActual, nPartActual));

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(
                            tensorTileL0C, tensorL0A, tensorL0B, mPartActual, nPartActual, kPartActual, initC,
                            unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    GemmCoord l1TileShape;
    uint32_t kPartLenMax;

    uint32_t L1A_TILE_SIZE;
    uint32_t L1B_TILE_SIZE;
    uint32_t L0A_TILE_SIZE;
    uint32_t L0B_TILE_SIZE;
    uint32_t L0C_TILE_SIZE;

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catccos::DGemm::Block

#endif // CATCCOS_DGEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_DYNAMIC_HPP
