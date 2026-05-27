/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L0C_TO_GM_EXPAND_HPP
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_EXPAND_HPP

#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CatccosCopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// Fixpipe with quant VDEQF16
template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CatccosCopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_CHANNEL,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
                    AscendC::LocalTensor<uint64_t> cbufWorkspace, LayoutDst const &dstLayout,
                    LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, cbufWorkspace, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L0C_TO_GM_EXPAND_HPP
