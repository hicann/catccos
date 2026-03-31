/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_TO_SHARE_MEM_HPP
#define CATCCOS_COMM_BLOCK_TO_SHARE_MEM_HPP

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/detail/remote_copy_type.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catccos::Comm::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class BlockShape_,
    class TileRemoteCopy_,
    class TileSwizzle_,
    class GemmRemapper_
>
class CommBlock <
    AtlasA2CommToShareMem<UB_STAGES_, IsDynamic_>,
    SrcType_,
    DstType_,
    BlockShape_,
    TileRemoteCopy_,
    TileSwizzle_,
    GemmRemapper_
> {
public:
    // Type aliases
    using DispatchPolicy = AtlasA2CommToShareMem<UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using BlockShape = BlockShape_;
    using TileRemoteCopy = TileRemoteCopy_;
    using TileSwizzle = TileSwizzle_;
    using TileParams = typename TileRemoteCopy::Params;
    using GemmRemapper = GemmRemapper_;
    static constexpr detail::CopyDirect RemoteCopyDirect = TileRemoteCopy::RemoteCopyDirect;

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord BlockShape() { return BlockShape::ToCoord(); }
        CATLASS_DEVICE
        static MatrixCoord TileShape() { return TileRemoteCopy::TileShape::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        __gm__ ElementDst *shmemPtr{nullptr};
        LayoutDst shmemLayout;
        GemmRemapper gemmRemapper;
        MatrixCoord blockShape;
        TileParams tileParams;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(__gm__ ElementDst *shmemPtr_, LayoutDst const &shmemLayout_, GemmRemapper const &gemmRemapper_,
            MatrixCoord blockShape_, const TileParams &tileParams_) 
            : shmemPtr(shmemPtr_), shmemLayout(shmemLayout_), gemmRemapper(gemmRemapper_),
              blockShape(blockShape_), tileParams(tileParams_) {}

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
            ubSList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
            ubOffset += params.TileShape().row() * params.TileShape().column() * sizeof(ElementDst);
        }
    }

    CATLASS_DEVICE
    void AllocEventID()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
    }

    CATLASS_DEVICE
    void ReleaseEventID()
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
        MatrixCoord const &gemmBlockShape,
        MatrixCoord const &outputBlockOffset,
        MatrixCoord const &inputBlockOffset,
        MatrixCoord const &commBlockShape,
        AscendC::GlobalTensor<ElementSrc> const& gmC,
        LayoutSrc const &layoutC,
        uint32_t const &globalLoopIdx,
        uint32_t const &rankIdx)
    {
        // Remap the idx & actual shape of the gemm block
        GemmCoord remapOutputBlockCoordMNK = params.gemmRemapper.GetBlockCoord(globalLoopIdx);
        MatrixCoord actualGemmBlockShape = params.gemmRemapper.GetActualBlockShape(remapOutputBlockCoordMNK).GetCoordMN();

        // Calculate the actual output offset of the communication block
        MatrixCoord blockInnerOffset = outputBlockOffset % gemmBlockShape;

        // Get actual communication block shape
        MatrixCoord actualCommBlockShape;
        if (blockInnerOffset.row() < actualGemmBlockShape.row()) {
            actualCommBlockShape = MatrixCoord::Min(actualGemmBlockShape - blockInnerOffset, commBlockShape);
        } else {
            return;
        }
        
        auto tileShape = params.TileShape();
        TileSwizzle tileSwizzle{actualCommBlockShape, tileShape};
        uint32_t tileLoops = tileSwizzle.GetLoops();
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < tileLoops; innerLoopIdx++) {
            auto tileCoord = tileSwizzle.GetTileCoord(innerLoopIdx);
            auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            
            auto inTileOffset = inputBlockOffset + tileOffsetInBlock;
            auto outTileOffset = outputBlockOffset + tileOffsetInBlock;

            // Get the data and layout of output
            AscendC::GlobalTensor<ElementDst> gmS;
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementDst *>(params.shmemPtr));
            auto gmSubblockS = gmS[params.shmemLayout.GetOffset(outTileOffset)];
            auto layoutSubblockS = params.shmemLayout.GetTileLayout(actualTileShape);

            // Get the data and layout of output
            auto gmSubblockC = gmC[layoutC.GetOffset(inTileOffset)];
            auto layoutSubblockC = layoutC.GetTileLayout(actualTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            tileRemoteCopy(
                gmSubblockS, layoutSubblockS,
                gmSubblockC, layoutSubblockC,
                actualTileShape,
                ubSList[ubListId],
                copyEventIdList[ubListId],
                rankIdx
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    AscendC::LocalTensor<ElementDst> ubSList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
};

} // namespace Catccos::Comm::Block 

#endif // CATCCOS_COMM_BLOCK_TO_SHARE_MEM_HPP