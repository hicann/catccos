/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_TILE_PADDING_NZ_PACKED_TILE_COPY_TLA_HPP
#define CATCCOS_DGEMM_TILE_PADDING_NZ_PACKED_TILE_COPY_TLA_HPP

#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catccos::DGemm::Tile {
using namespace Catlass;
namespace helper = Catlass::Gemm::helper;
namespace detail = Catlass::detail;
template <
    /// Tag indicating architecture
    class ArchTag, class TensorA, class LayoutTagA, class TensorB, class LayoutTagB, class TensorC, class LayoutTagC,
    class TensorBias = void, class LayoutTagBias = void, bool IS_PADDING_A = false, bool IS_PADDING_B = false>
struct PaddingNZPackedTileCopyTla : Catlass::Gemm::Tile::PaddingPackedTileCopyTla<
                                        ArchTag, TensorA, LayoutTagA, TensorB, LayoutTagB, TensorC, LayoutTagC,
                                        TensorBias, LayoutTagBias, IS_PADDING_A, IS_PADDING_B> {
    using Base = Catlass::Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorA, LayoutTagA, TensorB, LayoutTagB, TensorC, LayoutTagC, TensorBias, LayoutTagBias, IS_PADDING_A,
        IS_PADDING_B>;
    using TensorL1A = typename Base::TensorL1A;
    using TensorL1B = typename Base::TensorL1B;
    using LayoutTagL1A = typename Base::LayoutTagL1A;
    using LayoutTagL1B = typename Base::LayoutTagL1B;

    using LayoutPaddingTagA = std::conditional_t<std::is_same_v<LayoutTagA, layout::RowMajor>, layout::zN, layout::nZ>;
    using LayoutPaddingTagB = std::conditional_t<std::is_same_v<LayoutTagB, layout::RowMajor>, layout::zN, layout::nZ>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A, Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutTagA, LayoutTagL1A> >;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B, Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutTagB, LayoutTagL1B> >;
};
} // namespace Catccos::DGemm::Tile

#endif // CATCCOS_DGEMM_TILE_PADDING_NZ_PACKED_TILE_COPY_TLA_HPP
