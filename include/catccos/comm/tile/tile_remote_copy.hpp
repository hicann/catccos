/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_TILE_REMOTE_COPY_HPP
#define CATCCOS_TILE_REMOTE_COPY_HPP

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"

// from catlass
#include "catlass/catlass.hpp"

// from shmem
#include "shmem.h"

namespace Catccos::Comm::Tile {

using Catlass::MatrixCoord;

template <
    /// Tag indicating architecture
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class TileShape_,
    detail::CopyDirect CopyDirect_,
    detail::CopyTransport CopyTransport_
>
struct TileRemoteCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class TileShape_
>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Get, detail::CopyTransport::Mte> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

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

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);
        aclshmemx_mte_get_nbi(dstTensor, srcTensor, tmpUb, copyParams, peerIdx, copyEventId);
    }
};

template <
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class TileShape_
>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Put, detail::CopyTransport::Mte> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

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

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);
        aclshmemx_mte_put_nbi(dstTensor, srcTensor, tmpUb, copyParams, peerIdx, copyEventId);
    }
};

template <
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class TileShape_
>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Get, detail::CopyTransport::Rdma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

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

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {   
        AscendC::LocalTensor<uint32_t> ubLocal32;
        ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
        ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
        AscendC::LocalTensor<uint64_t> ubLocal64;
        ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
        ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen = repeat * stride * sizeof(ElementSrc);
        auto ptr = shmem_ptr((__gm__ void *)srcTensor.GetPhyAddr(), peerIdx);
        aclshmemi_roce_read((__gm__ uint8_t*)(dstTensor.GetPhyAddr()), (__gm__ uint8_t*)ptr, peerIdx, 0, messageLen, ubLocal64, ubLocal32, 0);

        aclshmemi_roce_quiet(peerIdx, 0, ubLocal64, ubLocal32, 0);
    }
};

template <
    class ArchTag,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class TileShape_
>
struct TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, detail::CopyDirect::Put, detail::CopyTransport::Rdma> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using TileShape = TileShape_;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;
    static constexpr bool IsDynamic = IsDynamic_;

    CATLASS_DEVICE
    TileRemoteCopy() {}

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

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {   
        AscendC::LocalTensor<uint32_t> ubLocal32;
        ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
        ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
        AscendC::LocalTensor<uint64_t> ubLocal64;
        ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
        ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

        uint32_t repeat = copyShape.row();
        uint32_t stride = srcLayout.stride(0);
        uint64_t messageLen = repeat * stride * sizeof(ElementSrc);
        auto ptr = shmem_ptr((__gm__ void *)dstTensor.GetPhyAddr(), peerIdx);
        aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(srcTensor.GetPhyAddr()), peerIdx, 0, messageLen, ubLocal64, ubLocal32, 0);

        aclshmemi_roce_quiet(peerIdx, 0, ubLocal64, ubLocal32, 0);
    }
};

} // namespace Catccos::Comm::Tile

#endif // CATCCOS_TILE_REMOTE_COPY_HPP
