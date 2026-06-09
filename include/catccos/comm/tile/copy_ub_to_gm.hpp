/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_TILE_TILE_COPY_UB_TO_GM_HPP
#define CATCCOS_COMM_TILE_TILE_COPY_UB_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catccos::Comm::Tile {

template <
    class ArchTag,
    class GmSrcType,
    class GMDstType
>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
};

// new add vectorlayout version
template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, 
        Gemm::GemmType<Element, Catlass::layout::RowMajor>,
        Gemm::GemmType<Element, Catlass::layout::VectorLayout>> {
    using LayoutSrc = Catlass::layout::RowMajor;
    using LayoutDst = Catlass::layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0),
            sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK,
            0,
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    };
};

}  // Catccos::Comm::Tile

#endif // CATCCOS_COMM_TILE_TILE_COPY_UB_TO_GM_HPP
