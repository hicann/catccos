/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_HPP
#define CATCCOS_HPP

#include <kernel_operator.h>

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/dist_coord.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catccos {

namespace detail {
 
template <class T, class Enable = void>
struct MinTraits {
    CATLASS_HOST_DEVICE static constexpr
    auto Apply(T const &lhs, T const &rhs)
    {
        return (lhs < rhs) ? lhs : rhs;
    }
};
 
template <class T, class Enable = void>
struct MaxTraits {
    CATLASS_HOST_DEVICE static constexpr
    auto Apply(T const &lhs, T const &rhs)
    {
        return (lhs > rhs) ? lhs : rhs;
    }
};
 
template <class T, class Enable = void>
struct ClipSubTraits {
    CATLASS_HOST_DEVICE static constexpr
    auto Apply(T const &lhs, T const &rhs)
    {
        return (lhs > rhs) ? (lhs - rhs) : 0;
    }
};
 
#define CATCCOS_COORD_DERIVED_TRAITS(OP)                                                                                \
    template <class T>                                                                                                 \
    struct OP##Traits<T, std::enable_if_t<std::is_base_of_v<Catlass::Coord<T::RANK, typename T::Index>, T>>> {         \
        CATLASS_HOST_DEVICE static constexpr                                                                           \
        auto Apply(T const &lhs, T const &rhs)                                                                         \
        {                                                                                                              \
            T result;                                                                                                  \
            for (int i = 0; i < T::RANK; ++i) {                                                                        \
                result[i] = OP##Traits<typename T::Index>::Apply(lhs[i], rhs[i]);                                      \
            }                                                                                                          \
            return result;                                                                                             \
        }                                                                                                              \
    };
 
CATCCOS_COORD_DERIVED_TRAITS(Min);
CATCCOS_COORD_DERIVED_TRAITS(Max);
CATCCOS_COORD_DERIVED_TRAITS(ClipSub);
 
#undef CATCCOS_COORD_DERIVED_TRAITS
 
}  // namespace detail

template <typename Index, typename LongIndex, int RANK>
CATLASS_HOST_DEVICE
LongIndex Numel(Catlass::Coord<RANK, Index, LongIndex> const &coord)
{
    LongIndex product = 1;
    for (int i = 0; i < RANK; ++i) {
        product *= static_cast<LongIndex>(coord[i]);
    }
    return product;
}

template <typename Index, typename LongIndex, int RANK>
CATLASS_HOST_DEVICE
LongIndex Dot(Catlass::Coord<RANK, Index> const &coord, Catlass::Coord<RANK, LongIndex> const &stride,
    LongIndex accumulator = {})
{
    for (int i = 0; i < RANK; ++i) {
        accumulator += static_cast<LongIndex>(coord[i]) * stride[i];
    }
    return accumulator;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
auto Min(T const &lhs, T const &rhs)
{
    return detail::MinTraits<T>::Apply(lhs, rhs);
}

template <typename Index, int RANK>
CATLASS_HOST_DEVICE constexpr
auto Min(Catlass::Coord<RANK, Index> const &lhs, Catlass::Coord<RANK, Index> const &rhs)
{
    Catlass::Coord<RANK, Index> result;
    for (int i = 0; i < RANK; ++i) {
        result[i] = Min(lhs[i], rhs[i]);
    }
    return result;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
auto Max(T const &lhs, T const &rhs)
{
    return detail::MaxTraits<T>::Apply(lhs, rhs);
}

template <typename Index, int RANK>
CATLASS_HOST_DEVICE constexpr
auto Max(Catlass::Coord<RANK, Index> const &lhs, Catlass::Coord<RANK, Index> const &rhs)
{
    Catlass::Coord<RANK, Index> result;
    for (int i = 0; i < RANK; ++i) {
        result[i] = Max(lhs[i], rhs[i]);
    }
    return result;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
auto ClipSub(T const &lhs, T const &rhs)
{
    return detail::ClipSubTraits<T>::Apply(lhs, rhs);
}
 
template <uint32_t SWIZZLE_OFFSET = 1, uint32_t SWIZZLE_DIRECTION = 0>
struct MatrixSwizzle {
    CATLASS_DEVICE static
    Catlass::MatrixCoord GetCoord(Catlass::MatrixCoord const &gridShape, uint32_t loopIdx)
    {
        constexpr uint32_t ROW_DIM = SWIZZLE_DIRECTION;
        constexpr uint32_t COL_DIM = 1 - SWIZZLE_DIRECTION;
 
        uint32_t groupSize = SWIZZLE_OFFSET * gridShape[COL_DIM];
        uint32_t groupIdx = loopIdx / groupSize;
        uint32_t groupOffset = loopIdx - groupIdx * groupSize;
 
        Catlass::MatrixCoord coord{};
        uint32_t inGroupRows = Min(SWIZZLE_OFFSET, gridShape[ROW_DIM] - groupIdx * SWIZZLE_OFFSET);
        coord[COL_DIM] = groupOffset / inGroupRows;
        uint32_t inGroupRowIdx = groupOffset - coord[COL_DIM] * inGroupRows;
        coord[ROW_DIM] = groupIdx * SWIZZLE_OFFSET + inGroupRowIdx;
        if ((groupIdx & 0b1) == 1) {
            coord[COL_DIM] = gridShape[COL_DIM] - coord[COL_DIM] - 1;
        }
 
        return coord;
    }
};

template <uint32_t SWIZZLE_DIRECTION = 0, bool IS_DETERMINISTIC = false>
struct CommSwizzle {
    CATLASS_DEVICE static
    DistMatrixCoord GetCoord(DistMatrixCoord const &gridShape, Catlass::MatrixCoord const &coreSplit, uint32_t loopIdx)
    {
        constexpr uint32_t ROW_DIM = SWIZZLE_DIRECTION;
        constexpr uint32_t COL_DIM = 1 - SWIZZLE_DIRECTION;
        uint32_t swizzleOffset = coreSplit[ROW_DIM];

        Catlass::MatrixCoord flattenGridShape{gridShape.row() * gridShape.column(), gridShape.rank()};
        uint32_t groupSize = swizzleOffset * flattenGridShape[COL_DIM];
        uint32_t groupIdx = loopIdx / groupSize;
        uint32_t groupOffset = loopIdx - groupIdx * groupSize;
        uint32_t inGroupRows = IS_DETERMINISTIC ? 
            swizzleOffset : Min(swizzleOffset, flattenGridShape[ROW_DIM] - groupIdx * swizzleOffset);

        Catlass::MatrixCoord coord{};
        coord[COL_DIM] = groupOffset / inGroupRows;
        uint32_t inGroupRowIdx = groupOffset - coord[COL_DIM] * inGroupRows;
        coord[ROW_DIM] = groupIdx * swizzleOffset + inGroupRowIdx;
        
        // Shift
        uint32_t nStride = gridShape.rank() / coreSplit.column();
        uint32_t offset = coord[1] * nStride;
        coord[1] = (offset + offset / gridShape.rank() + coord[0]) % gridShape.rank();
 
        return DistMatrixCoord{coord[0] / gridShape.column(), coord[0] % gridShape.column(), coord[1]};
    }
};

namespace layout {

// Compute the number of contiguous elements needed to store a tensor with the given size
// 这个函数应该作为每个layout的成员函数，后续应该在catlass里修改，这里是一个临时实现
template <class Layout>
CATLASS_HOST_DEVICE constexpr
int64_t Capacity(Layout const &layout)
{
    if constexpr (std::is_same_v<Layout, Catlass::layout::RowMajor>) {
        return layout.shape(0) * layout.stride(0);
    } else if constexpr (std::is_same_v<Layout, Catlass::layout::ColumnMajor>) {
        return layout.shape(1) * layout.stride(1);
    } else {
        static_assert(DEPENDENT_FALSE<Layout>, "Unsupport layout for Capacity.");
    }
}

template <int RANK_, typename Index_=int32_t>
struct AffineRankN {
    static int const RANK = RANK_;
    using Index = Index_;
    using LongIndex = int64_t;
    using TensorCoord = Catlass::Coord<RANK, Index>;
    using Stride = Catlass::Coord<RANK, LongIndex>;

private:
    Stride stride_;

public:
    CATLASS_HOST_DEVICE
    AffineRankN(Stride const &stride = Stride()) : stride_(stride) {}

    CATLASS_HOST_DEVICE
    static AffineRankN Packed(TensorCoord const &extent)
    {
        AffineRankN layout;
        layout.stride_[RANK - 1] = 1;

        for (int i = RANK - 1; i > 0; --i) {
            layout.stride_[i - 1] = layout.stride_[i] * extent[i];
        }

        return layout;
    }

    CATLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const &coord) const
    {
        return Dot(coord, stride_);
    }
};

};

namespace Padding {
 
template <class Layout> size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}
 
CATLASS_HOST_DEVICE
bool IsNeedPadding(Catlass::layout::RowMajor layout, uint32_t align)
{
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}
 
CATLASS_HOST_DEVICE
bool IsNeedPadding(Catlass::layout::ColumnMajor layout, uint32_t align)
{
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}
 
template <class Type, bool PADDING> struct PaddingHelper {
    using ArchTag = typename Catlass::Arch::AtlasA2;
    using Layout = typename Type::Layout;
    using Element = typename Type::Element;
 
    using LayoutPadding = std::conditional_t<std::is_same_v<Layout, Catlass::layout::RowMajor>, Catlass::layout::PaddingRowMajor,
                                             Catlass::layout::PaddingColumnMajor>;
    using ActualType = std::conditional_t<PADDING, Catlass::Gemm::GemmType<Element, LayoutPadding>, Type>;
    static const uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(Element);
    using GlobalPadding = std::conditional_t<
        PADDING, Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, Element, Layout, LayoutPadding, COMPUTE_LENGTH>, void>;
    using LayoutW = std::conditional_t<PADDING, LayoutPadding, Layout>;
 
    CATLASS_DEVICE
    static LayoutW GetLayoutW(uint32_t a, uint32_t b, uint32_t padA, uint32_t padB)
    {
        if constexpr (PADDING) {
            LayoutPadding layoutW = LayoutPadding(a, b, padA, padB);
            return layoutW;
        } else {
            Layout layoutW = Layout(a, b);
            return layoutW;
        }
    }
};
 
};

 __BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_timerBufferAddr;
 	  	 
CATLASS_DEVICE inline void SetTimerBuffer(__gm__ uint8_t* timerBuffer) {
    g_timerBufferAddr = timerBuffer;
}
    
CATLASS_DEVICE inline __gm__ uint8_t* GetTimerBuffer() {
    return g_timerBufferAddr;
}

} // namespace Catccos

#endif // CATCCOS_HPP
