/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "catccos/epilogue/dispatch_policy.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block
{
template <class DispatchPolicy, class... Args>
class BlockEpilogue;

namespace detail
{

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

static constexpr AscendC::Reg::CastTrait kSwigluBf16ToFloat = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::CastTrait kSwigluFloatToBf16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void SwigluBf16RegBaseVf(__ubuf__ bfloat16_t *dst, __ubuf__ bfloat16_t *firstSrc,
                                            __ubuf__ bfloat16_t *secondSrc, uint16_t m, uint32_t h, uint32_t srcStride,
                                            uint32_t dstStride, uint16_t repeatTimes)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);

    AscendC::Reg::RegTensor<bfloat16_t> firstBf16;
    AscendC::Reg::RegTensor<bfloat16_t> secondBf16;
    AscendC::Reg::RegTensor<bfloat16_t> dstBf16;
    AscendC::Reg::RegTensor<float> first;
    AscendC::Reg::RegTensor<float> second;
    AscendC::Reg::RegTensor<float> denominator;
    AscendC::Reg::RegTensor<float> result;
    AscendC::Reg::MaskReg mask;

    for (uint16_t row = 0; row < m; ++row)
    {
        uint32_t remaining = h;
        for (uint16_t i = 0; i < repeatTimes; ++i)
        {
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + i * oneRepeatSize;
            uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + i * oneRepeatSize;

            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(firstBf16,
                                                                                         firstSrc + srcOffset);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(secondBf16,
                                                                                         secondSrc + srcOffset);
            AscendC::Reg::Cast<float, bfloat16_t, kSwigluBf16ToFloat>(first, firstBf16, mask);
            AscendC::Reg::Cast<float, bfloat16_t, kSwigluBf16ToFloat>(second, secondBf16, mask);

            AscendC::Reg::Muls(denominator, first, -1.0f, mask);
            AscendC::Reg::Exp(denominator, denominator, mask);
            AscendC::Reg::Adds(denominator, denominator, 1.0f, mask);
            AscendC::Reg::Div(result, first, denominator, mask);
            AscendC::Reg::Mul(result, result, second, mask);

            AscendC::Reg::Cast<bfloat16_t, float, kSwigluFloatToBf16>(dstBf16, result, mask);
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, dstBf16,
                                                                                         mask);
        }
    }
}

__aicore__ inline void SwigluRegBase(__ubuf__ bfloat16_t *dst, __ubuf__ bfloat16_t *firstSrc,
                                     __ubuf__ bfloat16_t *secondSrc, uint16_t m, uint32_t h, uint32_t srcStride,
                                     uint32_t dstStride)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>((h + oneRepeatSize - 1) / oneRepeatSize);
    asc_vf_call<SwigluBf16RegBaseVf>(dst, firstSrc, secondSrc, m, h, srcStride, dstStride, repeatTimes);
}

#endif

}  // namespace detail

template <uint32_t UB_STAGES_, uint32_t TILE_M_, class CType_, class DType_, class TileCopy_,
          class EpilogueTileSwizzle_>
class BlockEpilogue<Catccos::Epilogue::EpilogueAscend950RegBaseSwiglu<UB_STAGES_, TILE_M_>, CType_, DType_, TileCopy_,
                    EpilogueTileSwizzle_>
{
   public:
    using DispatchPolicy = Catccos::Epilogue::EpilogueAscend950RegBaseSwiglu<UB_STAGES_, TILE_M_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t TILE_M = TILE_M_;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    static_assert(UB_STAGES >= 1 && UB_STAGES <= 2, "UB stages must be in [1, 2] because event ids are limited.");
    static_assert(TILE_M >= 1, "The RegBase SwiGLU tile M must be positive.");
    static_assert(AscendC::IsSameType<ElementC, bfloat16_t>::value && AscendC::IsSameType<ElementD, bfloat16_t>::value,
                  "EpilogueAscend950RegBaseSwiglu currently supports BF16 input and output only.");

    struct Params
    {
        MatrixCoord tileShape;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(MatrixCoord const &tileShape_) : tileShape{static_cast<MatrixCoord::Index>(TILE_M), tileShape_.column()}
        {
        }
    };
    CATLASS_DEVICE
    static void ComputeFromUb(AscendC::LocalTensor<ElementD> dstTensor, AscendC::LocalTensor<ElementC> firstTensor,
                              AscendC::LocalTensor<ElementC> secondTensor, uint16_t m, uint32_t h, uint32_t srcStride,
                              uint32_t dstStride)
    {
        static_assert(
            AscendC::IsSameType<ElementC, bfloat16_t>::value && AscendC::IsSameType<ElementD, bfloat16_t>::value,
            "RegBase UB-input SwiGLU currently supports BF16 input and output only.");

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(dstTensor.GetPhyAddr());
        __ubuf__ bfloat16_t *firstSrc = reinterpret_cast<__ubuf__ bfloat16_t *>(firstTensor.GetPhyAddr());
        __ubuf__ bfloat16_t *secondSrc = reinterpret_cast<__ubuf__ bfloat16_t *>(secondTensor.GetPhyAddr());
        detail::SwigluRegBase(dst, firstSrc, secondSrc, m, h, srcStride, dstStride);
#else
        (void)dstTensor;
        (void)firstTensor;
        (void)secondTensor;
        (void)m;
        (void)h;
        (void)srcStride;
        (void)dstStride;
#endif
    }

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i)
        {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += params.tileShape.row() * params.tileShape.column() * sizeof(ElementC);
            ubOffset = (ubOffset + 31U) & ~size_t(31U);

            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += params.tileShape.row() * (params.tileShape.column() >> 1) * sizeof(ElementD);
            ubOffset = (ubOffset + 31U) & ~size_t(31U);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i)
        {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_) { params = params_; }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutBlockC,
                    AscendC::GlobalTensor<ElementD> const &gmBlockD, LayoutC const &layoutBlockD,
                    MatrixCoord const &actualBlockShape, Callback &&callback = Callback{})
    {
        callback();

        auto tileShape = params.tileShape;
        auto ubTileStride = MakeCoord(static_cast<int64_t>(tileShape.column()), 1L);
        auto ubChunkTileStride = MakeCoord(static_cast<int64_t>(tileShape.column() >> 1), 1L);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();

        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum)
        {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            MatrixCoord tileOffset = tileCoord * tileShape;

            auto actualChunkTileShape = MakeCoord(actualTileShape.row(), actualTileShape.column() >> 1);
            auto chunkTileOffset = MakeCoord(tileOffset.row(), tileOffset.column() >> 1);

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffset)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto &ubC = ubCList[ubListId];
            auto &ubD = ubDList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};
            LayoutD layoutUbD{actualChunkTileShape, ubChunkTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            uint16_t m = static_cast<uint16_t>(actualChunkTileShape.At(0));
            uint32_t h = static_cast<uint32_t>(actualChunkTileShape.At(1));
            uint32_t srcStride = static_cast<uint32_t>(tileShape.column());
            uint32_t dstStride = static_cast<uint32_t>(tileShape.column() >> 1);
            __ubuf__ bfloat16_t *firstSrc = (__ubuf__ bfloat16_t *)ubC.GetPhyAddr();
            __ubuf__ bfloat16_t *secondSrc = firstSrc + h;
            __ubuf__ bfloat16_t *dst = (__ubuf__ bfloat16_t *)ubD.GetPhyAddr();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            detail::SwigluRegBase(dst, firstSrc, secondSrc, m, h, srcStride, dstStride);
#endif

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileD = gmBlockD[layoutBlockD.GetOffset(chunkTileOffset)];
            auto layoutGmTileD = layoutBlockD.GetTileLayout(actualChunkTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
    }

   private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block
