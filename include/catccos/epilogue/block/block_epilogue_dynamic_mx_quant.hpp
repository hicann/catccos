/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "catccos/epilogue/dispatch_policy.hpp"


namespace Catccos::Epilogue::Block {

namespace mx_quant_detail {

// ---- Shared constants ----
constexpr uint16_t FP8_E8M0_NAN = 0x00ff;
constexpr int32_t  NEG_ZERO     = 0x80000000;
constexpr int32_t  FP32_BIAS    = 127;
constexpr int32_t  FP32_BIAS_NEG = -127;
constexpr int32_t  NEG_ONE      = -1;
constexpr float    FOUR         = 4.0f;
constexpr float    ONE_FOURTH   = 0.25f;
constexpr int16_t  SHR_NUM_FP32 = 23;

// ---- Type traits: maps ElementC → calc types and constants ----
// bf16 path: operate directly on uint16_t
// half path: cast to float, operate on uint32_t (matches ops-nn intCalcType)
template <typename T> struct MxQuantTraits;
template <> struct MxQuantTraits<bfloat16_t> {
    using CalcType    = bfloat16_t;
    using IntCalcType = uint16_t;
    static constexpr uint32_t VF_LEN     = 256 / sizeof(uint16_t);  // 128
    static constexpr IntCalcType EXP_MASK  = 0x7F80;
    static constexpr IntCalcType ABS_MASK  = 0x7FFF;
    static constexpr IntCalcType EXP_BIAS  = 0x7F00;
    static constexpr IntCalcType NAN_SCALE = 0x7F81;
    static constexpr IntCalcType SPECIAL_EXP = 0x0040;
    static constexpr int16_t     SHR_NUM   = 7;
    // Target emax fields (in bf16 exponent field format)
    static constexpr IntCalcType FP4_E2M1_EMAX = 0x0100;
    static constexpr IntCalcType FP4_E1M2_EMAX = 0x0000;
    static constexpr IntCalcType FP8_E4M3_EMAX = 0x0400;
    static constexpr IntCalcType FP8_E5M2_EMAX = 0x0780;
};

template <> struct MxQuantTraits<half> {
    using CalcType    = float;
    using IntCalcType = uint32_t;
    static constexpr uint32_t VF_LEN     = 256 / sizeof(uint32_t);  // 64
    static constexpr IntCalcType EXP_MASK  = 0x7F800000;
    static constexpr IntCalcType ABS_MASK  = 0x7FFFFFFF;
    static constexpr IntCalcType EXP_BIAS  = 0x7F000000;
    static constexpr IntCalcType NAN_SCALE = 0x7F810000;
    static constexpr IntCalcType SPECIAL_EXP = 0x00400000;
    static constexpr int16_t     SHR_NUM   = 23;
    static constexpr IntCalcType FP4_E2M1_EMAX = 0x01000000;
    static constexpr IntCalcType FP4_E1M2_EMAX = 0x00000000;
    static constexpr IntCalcType FP8_E4M3_EMAX = 0x04000000;
    static constexpr IntCalcType FP8_E5M2_EMAX = 0x07800000;
};

template <typename DstType, typename IntCalcType>
__aicore__ inline constexpr IntCalcType GetTargetEmaxField()
{
    if constexpr (std::is_same_v<DstType, float4_e2m1x2_t>) {
        return MxQuantTraits<bfloat16_t>::FP4_E2M1_EMAX; // placeholder, overridden by caller
    } else { return 0; }
}

template <typename DstType>
__aicore__ inline constexpr bool IsFp4Type()
{ return std::is_same_v<DstType, float4_e2m1x2_t> || std::is_same_v<DstType, float4_e1m2x2_t>; }

template <typename DstType>
__aicore__ inline constexpr bool IsFp8Type()
{ return std::is_same_v<DstType, float8_e4m3_t> || std::is_same_v<DstType, float8_e5m2_t>; }

// ---- CalcElement: inner (float path only, handles FP4 truncation + neg-zero) ----
// Ported from ops-nn dynamic_mx_quant_common.h
// NOTE: Function parameters use template types because CCE compiler cannot
// resolve AscendC::MicroAPI::RegTensor in parameter declarations.
template <AscendC::RoundMode roundMode, typename OutType, typename RegFloat, typename RegInt32, typename MaskT>
__aicore__ inline void CalcElementInner(RegFloat& in, RegInt32& maxEle, MaskT mask)
{
    // Cast template-deduced types to Reg::RegTensor for correct Reg:: API dispatch
    auto& regIn = reinterpret_cast<AscendC::MicroAPI::RegTensor<float>&>(in);
    auto& regMaxEle = reinterpret_cast<AscendC::MicroAPI::RegTensor<int32_t>&>(maxEle);

    AscendC::MicroAPI::RegTensor<float> y1;
    AscendC::MicroAPI::MaskReg negValueMask, zeroMask, negZeroMask, zeroNegMask;
    AscendC::MicroAPI::RegTensor<int32_t> negZero;
    AscendC::MicroAPI::Duplicate(negZero, NEG_ZERO);
    AscendC::MicroAPI::CompareScalar<int32_t, AscendC::CMPMODE::EQ>(
        zeroNegMask, (AscendC::MicroAPI::RegTensor<int32_t>&)regIn, NEG_ZERO, mask);

    if constexpr (std::is_same_v<OutType, float4_e2m1x2_t>) {
        AscendC::MicroAPI::RegTensor<int32_t> exp1, exp2;
        AscendC::MicroAPI::And(exp1, (AscendC::MicroAPI::RegTensor<int32_t>&)regIn, regMaxEle, mask);
        AscendC::MicroAPI::ShiftRights(exp1, exp1, SHR_NUM_FP32, mask);
        AscendC::MicroAPI::Adds(exp1, exp1, FP32_BIAS_NEG, mask);
        AscendC::MicroAPI::Maxs(exp1, exp1, 0, mask);
        AscendC::MicroAPI::Adds(exp1, exp1, NEG_ONE, mask);
        AscendC::MicroAPI::Muls(exp2, exp1, NEG_ONE, mask);
        AscendC::MicroAPI::Adds(exp2, exp2, FP32_BIAS, mask);
        AscendC::MicroAPI::ShiftLefts(exp2, exp2, SHR_NUM_FP32, mask);
        AscendC::MicroAPI::Mul(y1, regIn, (AscendC::MicroAPI::RegTensor<float>&)exp2, mask);
        AscendC::MicroAPI::Adds(exp1, exp1, FP32_BIAS, mask);
        AscendC::MicroAPI::ShiftLefts(exp1, exp1, SHR_NUM_FP32, mask);
        AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::MicroAPI::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::MicroAPI::Mul(regIn, y1, (AscendC::MicroAPI::RegTensor<float>&)exp1, mask);
    } else {
        AscendC::MicroAPI::Muls(y1, regIn, FOUR, mask);
        AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::MicroAPI::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::MicroAPI::Muls(regIn, y1, ONE_FOURTH, mask);
    }
    AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::EQ>(zeroMask, regIn, 0, mask);
    AscendC::MicroAPI::MaskAnd(negZeroMask, zeroMask, negValueMask, mask);
    AscendC::MicroAPI::MaskOr(zeroMask, negZeroMask, zeroNegMask, mask);
    AscendC::MicroAPI::Copy((AscendC::MicroAPI::RegTensor<int32_t>&)regIn, negZero, zeroMask);
}


// ---- CalcElement: outer (multiply by 1/scale, quantize, pack to uint8) ----
template <AscendC::RoundMode roundMode, typename OutType, typename CalcType, typename IntCalcType,
          typename RegCalc, typename RegICT, typename RegU8, typename MaskT>
__aicore__ inline void CalcElementQuant(
    RegCalc& in, RegICT& scaleReprocal, RegICT& maxEle, RegU8& out, MaskT mask)
{
    // Cast template-deduced types to Reg::RegTensor for correct Reg:: API dispatch
    auto& regIn = reinterpret_cast<AscendC::MicroAPI::RegTensor<CalcType>&>(in);
    auto& regScale = reinterpret_cast<AscendC::MicroAPI::RegTensor<IntCalcType>&>(scaleReprocal);
    auto& regMaxEle = reinterpret_cast<AscendC::MicroAPI::RegTensor<IntCalcType>&>(maxEle);
    auto& regOut = reinterpret_cast<AscendC::MicroAPI::RegTensor<uint8_t>&>(out);

    static constexpr AscendC::MicroAPI::CastTrait castTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

    AscendC::MicroAPI::Mul(regIn, regIn, (AscendC::MicroAPI::RegTensor<CalcType>&)regScale, mask);

    if constexpr (std::is_same_v<CalcType, float>) {
        static constexpr AscendC::MicroAPI::CastTrait castFp32ToBf16 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        CalcElementInner<roundMode, OutType>(regIn, (AscendC::MicroAPI::RegTensor<int32_t>&)regMaxEle, mask);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16Val;
        AscendC::MicroAPI::Cast<bfloat16_t, float, castFp32ToBf16>(bf16Val, regIn, mask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint16_t>&)bf16Val, (AscendC::MicroAPI::RegTensor<uint32_t>&)bf16Val);
        AscendC::MicroAPI::RegTensor<OutType> y;
        AscendC::MicroAPI::Cast<OutType, bfloat16_t, castTrait>(y, bf16Val, mask);
        AscendC::MicroAPI::RegTensor<uint16_t> yU16;
        AscendC::MicroAPI::Pack(yU16, (AscendC::MicroAPI::RegTensor<uint32_t>&)y);
        AscendC::MicroAPI::Pack(regOut, yU16);
    } else {
        AscendC::MicroAPI::RegTensor<OutType> y;
        AscendC::MicroAPI::Cast<OutType, CalcType, castTrait>(y, regIn, mask);
        AscendC::MicroAPI::RegTensor<uint16_t> yU16;
        AscendC::MicroAPI::Pack(yU16, (AscendC::MicroAPI::RegTensor<uint32_t>&)y);
        AscendC::MicroAPI::Pack(regOut, yU16);
    }
}


}  // namespace mx_quant_detail

}  // namespace Catccos::Epilogue::Block