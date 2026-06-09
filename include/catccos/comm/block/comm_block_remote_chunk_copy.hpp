/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_REMOTE_CHUNK_COPY_HPP
#define CATCCOS_COMM_BLOCK_REMOTE_CHUNK_COPY_HPP

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/detail/remote_copy_type.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catccos::Comm::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class ArchTag_,
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class SrcType_,
    class Dst0Type_,
    class Dst1Type_,
    class BlockShape_,
    class TileRemoteCopy_,
    class TileSwizzle_
>
class CommBlock <
    AtlasCommRemoteChunkCopy<ArchTag_, UB_STAGES_, IsDynamic_>,
    SrcType_,
    Dst0Type_,
    Dst1Type_,
    BlockShape_,
    TileRemoteCopy_,
    TileSwizzle_
> {
public:
    // Type aliases
    using DispatchPolicy = AtlasCommRemoteChunkCopy<ArchTag_, UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst0 = typename Dst0Type_::Element;
    using LayoutDst0 = typename Dst0Type_::Layout;
    using ElementDst1 = typename Dst1Type_::Element;
    using LayoutDst1 = typename Dst1Type_::Layout;

    using LayoutComputeInUb = Catlass::layout::RowMajor;
    using TensorCoord = Catlass::layout::VectorLayout::TensorCoord;

    using BlockShape = BlockShape_;
    using TileRemoteCopy = TileRemoteCopy_;
    using TileSwizzle = TileSwizzle_;
    using TileParams = typename TileRemoteCopy::Params;
    static constexpr detail::CopyDirect RemoteCopyDirect = TileRemoteCopy::RemoteCopyDirect;

    using CopyGmToUbC = typename TileRemoteCopy::CopyGmToUbC;
    using CopyUbToGmD0 = typename TileRemoteCopy::CopyUbToGmD0;
    using CopyUbToGmD1 = typename TileRemoteCopy::CopyUbToGmD1;

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        uint32_t chunkByteOffset;

        CATLASS_HOST_DEVICE
        ParamsBase(int64_t chunkByteOffset_): chunkByteOffset(chunkByteOffset_) {}

        CATLASS_DEVICE
        static MatrixCoord BlockShape() { return BlockShape::ToCoord(); }
        CATLASS_DEVICE
        static MatrixCoord TileShape() { return TileRemoteCopy::TileShape::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord blockShape;
        TileParams tileParams;
        uint32_t chunkByteOffset;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord blockShape_, const TileParams &tileParams_, int64_t chunkByteOffset_)
            : blockShape(blockShape_), tileParams(tileParams_), chunkByteOffset(chunkByteOffset_) {}

        CATLASS_DEVICE
        MatrixCoord BlockShape() const { return blockShape; }
        CATLASS_DEVICE
        MatrixCoord TileShape() const { return tileParams.TileShape(); }
    };

    using Params = ParamsBase<IsDynamic>;

    CATLASS_DEVICE
    CommBlock(Catlass::Arch::Resource<ArchTag> &resource, Params const &params) : params(params)
    {
        size_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubSrcList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
            ubDst0List[i] = resource.ubBuf.template GetBufferByByte<ElementDst0>(ubOffset);
            ubDst1List[i] = resource.ubBuf.template GetBufferByByte<ElementDst1>(ubOffset + params.chunkByteOffset);
            ubOffset += params.TileShape().row() * params.TileShape().column() * sizeof(ElementSrc);
        }
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]); 
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
        ubListId = 0;
    }

    CATLASS_DEVICE
    ~CommBlock()
    {
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const &layoutSrc,
        AscendC::GlobalTensor<ElementDst0> const& gmDst0, LayoutDst0 const &layoutDst0,
        AscendC::GlobalTensor<ElementDst1> const& gmDst1, LayoutDst1 const &layoutDst1,
        MatrixCoord const &actualCommBlockShape, uint32_t rankIdx
    )
    {
        if (actualCommBlockShape.row() == 0) {
            return;
        }

        auto tileShape = params.TileShape();
        TileSwizzle tileSwizzle{actualCommBlockShape, tileShape};
        uint32_t tileLoops = tileSwizzle.GetLoops();

        for (uint32_t tileIdx = 0; tileIdx < tileLoops; tileIdx++) {
            auto tileCoord = tileSwizzle.GetTileCoord(tileIdx);
            auto tileOffsetInBlock = tileCoord * tileShape; // {rowIdx, 0}
            auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);

            // gm src
            auto gmTileSrc = gmSrc[layoutSrc.GetOffset(tileOffsetInBlock)];
            auto layoutTileSrc = layoutSrc.GetTileLayout(actualTileShape);
            // gmTile dst0
            auto gmTileDst0 = gmDst0[layoutDst0.GetOffset(tileOffsetInBlock)];
            auto actualDst0TileShape = MatrixCoord{actualTileShape.row(), layoutDst0.shape(1)};
            auto layoutTileDst0 = layoutDst0.GetTileLayout(actualDst0TileShape);
            // gmTile dst1
            auto gmTileDst1 = gmDst1[layoutDst1.GetOffset(TensorCoord{tileOffsetInBlock[0]})];
            auto layoutTileDst1 = layoutDst1.GetTileLayout(TensorCoord{actualTileShape.row()});

            // ub src
            auto layoutUbSrc = LayoutComputeInUb::template MakeLayoutInUb<ElementSrc>(actualTileShape);
            // ub dst0
            auto actualTileShapeAsElemDst0 = MatrixCoord{actualTileShape.row(),
                static_cast<uint32_t>(actualTileShape.column() * sizeof(ElementSrc) / sizeof(ElementDst0))};
            auto layoutUbAsElemDst0 = LayoutComputeInUb::template MakeLayoutInUb<ElementDst0>(actualTileShapeAsElemDst0);
            auto layoutUbDst0 = layoutUbAsElemDst0.GetTileLayout(
                MatrixCoord{layoutUbAsElemDst0.shape(0), static_cast<uint32_t>(params.chunkByteOffset / sizeof(ElementDst0))});
            // ub dst1
            auto actualTileShapeAsElemDst1 = MatrixCoord{actualTileShape.row(),
                static_cast<uint32_t>(actualTileShape.column() * sizeof(ElementSrc) / sizeof(ElementDst1))};
            auto layoutUbAsElemDst1 = LayoutComputeInUb::template MakeLayoutInUb<ElementDst1>(actualTileShapeAsElemDst1);
            auto layoutUbDst1 = layoutUbAsElemDst1.GetTileLayout(MatrixCoord{layoutUbAsElemDst1.shape(0), 1});

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);

            copyGmToUbC(ubSrcList[ubListId], gmTileSrc, layoutUbSrc, layoutTileSrc);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);

            copyUbToGmD0(gmTileDst0, ubDst0List[ubListId], layoutTileDst0, layoutUbDst0);
            copyUbToGmD1(gmTileDst1, ubDst1List[ubListId], layoutTileDst1, layoutUbDst1);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    AscendC::LocalTensor<ElementSrc> ubSrcList[UB_STAGES];
    AscendC::LocalTensor<ElementDst0> ubDst0List[UB_STAGES];
    AscendC::LocalTensor<ElementDst1> ubDst1List[UB_STAGES];

    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
    
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD0 copyUbToGmD0;
    CopyUbToGmD1 copyUbToGmD1;
};

} // namespace Catccos::Comm::Block 

#endif // CATCCOS_COMM_BLOCK_REMOTE_CHUNK_COPY_HPP