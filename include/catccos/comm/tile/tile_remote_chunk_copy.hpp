/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_TILE_TILE_CHUNK_COPY_HPP
#define CATCCOS_COMM_TILE_TILE_CHUNK_COPY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catccos/comm/tile/copy_ub_to_gm.hpp"

namespace Catccos::Comm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class Dst0Type_,
    class Dst1UbType_,
    class Dst1Type_,
    class TileShape_,
    detail::CopyDirect CopyDirect_,
    detail::CopyTransport CopyTransport_
>
struct TileRemoteChunkCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class Dst0Type_,
    class Dst1UbType_,
    class Dst1Type_,
    class TileShape_
>
struct TileRemoteChunkCopy<
    ArchTag,
    IsDynamic_, 
    SrcType_,
    Dst0Type_,
    Dst1UbType_,
    Dst1Type_,
    TileShape_,
    detail::CopyDirect::Get,
    detail::CopyTransport::Mte
> {
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst0 = typename Dst0Type_::Element;
    using LayoutDst0 = typename Dst0Type_::Layout;
    using ElementDst1 = typename Dst1Type_::Element;
    using LayoutDst1 = typename Dst1Type_::Layout;
    
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;

    using CopyGmToUbC = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, SrcType_>;
    using CopyUbToGmD0 = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Dst0Type_>;
    using CopyUbToGmD1 = CopyUb2Gm<ArchTag, Dst1UbType_, Dst1Type_>;

    CATLASS_DEVICE
    TileRemoteChunkCopy() {}

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord TileShape() { return TileShape::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord tileShape_) : tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord TileShape() const { return tileShape; }
    };

    using Params = ParamsBase<IsDynamic>;

    Params params;
};


} // namespace Catccos::Comm::Tile

#endif  // CATCCOS_COMM_TILE_TILE_CHUNK_COPY_HPP