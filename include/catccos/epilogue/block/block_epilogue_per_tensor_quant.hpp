/**

 * This program is free software, you can redistribute it and/or modify.

 * Copyright (c) 2025 Huawei Technologies Co., Ltd.

 * This file is a part of the CANN Open Software.

 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").

 * Please refer to the License for details. You may not use this file except in compliance with the License.

 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

 * See LICENSE in the root of the software repository for the full text of the License.

 */

#ifndef CATCCOS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TENSOR_QUANT_HPP
#define CATCCOS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TENSOR_QUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catccos/epilogue/dispatch_policy.hpp"

namespace Catlass::Epilogue::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class CType_, // 输入 Gemm::GemmType
    class DType_, // 输出
    class SType_, // scale的数据类型
    class TileSwizzle_
>

class BlockEpilogue <
    EpilogueAtlasA5PerTensorQuant<UB_STAGES_, IsDynamic_>,
    CType_,
    DType_,
    SType_,
    TileSwizzle_
> {

public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA5PerTensorQuant<UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using ElementCompute = ElementC;
    using ElementOut = ElementD;
    using ElementScale = SType_;
    using LayoutComputeInUb = Catlass::layout::RowMajor;
    uint32_t tileShape = 0;
    using TileSwizzle = TileSwizzle_;
    static constexpr uint32_t flagSize = 2 * 1024;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t ubAlignSize = 64;
    // Check the layout type of C and D
    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor> && std::is_same_v<LayoutD, Catlass::layout::RowMajor>,
        "Layout type of C, D must be RowMajor");
    // Epilogue params definition
    struct Params {
        __gm__ ElementC *ptrC{nullptr};
        LayoutC layoutC;
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD;
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(__gm__ ElementC *ptrC_, LayoutC const &layoutC_, __gm__ ElementD *ptrD_, LayoutD const &layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), layoutD(layoutD_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Catlass::Arch::Resource<ArchTag> &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = flagSize;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        uint32_t elementSize = UB_STAGES * (sizeof(ElementC) + sizeof(ElementD)) + sizeof(ElementScale);
        // Align tileShape to ubAlignSize to avoid address conflicts between UB stages
        tileShape = (AscendC::GetUBSizeInBytes() - flagSize) / elementSize / ubAlignSize * ubAlignSize;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubInputList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += tileShape * sizeof(ElementC);
            ubOutputList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += tileShape * sizeof(ElementD);
            eventInputVtoMTE2[i] = eventVMTE2++;
            eventInputMTE2toV[i] = eventMTE2V++;
            eventOutputMTE3toV[i] = eventMTE3V++;
            eventOutputVtoMTE3[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[i]);
        }
        ubScale = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
        ubOffset += tileShape * sizeof(ElementScale);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)

    {
        params = params_;
    }

    template<typename ElementD, typename ElementScale>
    static __simd_vf__ inline void CastHighToLow(__ubuf__ ElementD *dst, __ubuf__ ElementScale *src, const uint32_t calCount)

    {
        static constexpr AscendC::MicroAPI::CastTrait Fp8LayoutCast = { AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT };
        static constexpr AscendC::MicroAPI::CastTrait Hif8LayoutCast = { AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND };
        static constexpr AscendC::MicroAPI::CastTrait HalfLayoutCast = { AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND };
        static constexpr AscendC::MicroAPI::CastTrait FloatLayoutCast = { AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::RegTensor<ElementScale> srcReg;
        AscendC::MicroAPI::RegTensor<ElementD> dstReg;
        AscendC::MicroAPI::RegTensor<float> floatReg;
        uint32_t sregPlt = (uint32_t)calCount;
        uint32_t srcRepeatSize;
        uint32_t dstRepeatSize;
        AscendC::MicroAPI::MaskReg preg;
        uint16_t repeatTimes; // repeatTimes要根据源类型来，因为原类型是高精度
        if constexpr ((std::is_same<ElementD, float8_e4m3_t>::value && std::is_same<ElementScale, half>::value) ||
                      (std::is_same<ElementD, float8_e5m2_t>::value && std::is_same<ElementScale, half>::value)) { // half与FP8转换
            repeatTimes = AscendC::CeilDivision(calCount, AscendC::GetVecLen() / sizeof(float));
            srcRepeatSize = AscendC::GetVecLen() / sizeof(float);
            dstRepeatSize = AscendC::GetVecLen() / sizeof(float);
        } else {
            repeatTimes = AscendC::CeilDivision(calCount, AscendC::GetVecLen() / sizeof(ElementScale));
            srcRepeatSize = AscendC::GetVecLen() / sizeof(ElementScale);
            dstRepeatSize = AscendC::GetVecLen() / sizeof(ElementScale);
        }

        __VEC_SCOPE__

        {
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                // Phase 1: Load source from UB to register and cast to destination type
                if constexpr ((std::is_same<ElementD, float8_e4m3_t>::value && std::is_same<ElementScale, half>::value) ||
                              (std::is_same<ElementD, float8_e5m2_t>::value && std::is_same<ElementScale, half>::value)) {
                    // half -> float -> FP8 (two-step cast via float intermediate)
                    sregPlt = (uint32_t)calCount;
                    preg = AscendC::MicroAPI::UpdateMask<float>(sregPlt);
                    AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
                    AscendC::MicroAPI::DataCopy<ElementScale, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * srcRepeatSize);
                    AscendC::MicroAPI::Cast<float, ElementScale, FloatLayoutCast>(floatReg, srcReg, preg);
                    AscendC::MicroAPI::Cast<ElementD, float, Fp8LayoutCast>(dstReg, floatReg, preg);
                } else {
                    // Direct cast: source type -> destination type
                    sregPlt = (uint32_t)calCount;
                    preg = AscendC::MicroAPI::UpdateMask<ElementScale>(sregPlt);
                    AscendC::MicroAPI::DataCopy<ElementScale>(srcReg, src + i * srcRepeatSize);
                    if constexpr (std::is_same<ElementD, hifloat8_t>::value && (std::is_same<ElementScale, float>::value ||
                        std::is_same<ElementScale, half>::value)) {
                        AscendC::MicroAPI::Cast<ElementD, ElementScale, Hif8LayoutCast>(dstReg, srcReg, preg);
                    }
                    if constexpr ((std::is_same<ElementD, half>::value || std::is_same<ElementD, bfloat16_t>::value) &&

                                  std::is_same<ElementScale, float>::value) {
                        AscendC::MicroAPI::Cast<ElementD, ElementScale, HalfLayoutCast>(dstReg, srcReg, preg);
                    }
                    if constexpr ((std::is_same<ElementD, float8_e4m3_t>::value || std::is_same<ElementD, float8_e5m2_t>::value) &&

                                  std::is_same<ElementScale, float>::value) {
                        AscendC::MicroAPI::Cast<ElementD, ElementScale, Fp8LayoutCast>(dstReg, srcReg, preg);
                    }
                }
                // Phase 2: Store destination register back to UB
                if constexpr (std::is_same<ElementD, hifloat8_t>::value && std::is_same<ElementScale, half>::value) {
                    AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(dst +
                        i * dstRepeatSize, dstReg, preg);
                }
                if constexpr ((std::is_same<ElementD, float8_e4m3_t>::value || std::is_same<ElementD, float8_e5m2_t>::value) &&
                              std::is_same<ElementScale, half>::value) {
                    AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst +
                        i * dstRepeatSize, dstReg, preg);
                }
                if constexpr ((std::is_same<ElementD, float8_e4m3_t>::value || std::is_same<ElementD, float8_e5m2_t>::value ||

                               std::is_same<ElementD, hifloat8_t>::value) && std::is_same<ElementScale, float>::value) {
                    AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst +
                        i * dstRepeatSize, dstReg, preg);
                }
                if constexpr ((std::is_same<ElementD, half>::value || std::is_same<ElementD, bfloat16_t>::value) &&
                              std::is_same<ElementScale, float>::value) {
                    AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst +
                        i * dstRepeatSize, dstReg, preg);
                }
            }
        }
    }

    CATLASS_DEVICE

    void operator() (

        uint32_t const &actualBlockShape,

        AscendC::GlobalTensor<ElementCompute> const &gmInput,

        ElementScale const &scaleVal,

        Callback &&callback = Callback{}

    )

    {
        callback();
        if (actualBlockShape <= 0) {
            return;
        }
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));
        uint32_t tileLoops = AscendC::CeilDivision(actualBlockShape, tileShape);
        for (uint32_t loopIdx = 0; loopIdx < tileLoops; loopIdx++) {
            uint32_t offset = loopIdx * tileShape;
            ubListId = loopIdx % UB_STAGES;
            uint32_t processNum = (loopIdx == (tileLoops - 1)) ? (actualBlockShape - loopIdx * tileShape) : tileShape;
            auto &ubC = ubInputList[ubListId];
            auto &ubD = ubOutputList[ubListId];
            __ubuf__ ElementScale *scaleLocal = (__ubuf__ ElementScale *)(ubScale.GetPhyAddr());
            __ubuf__ ElementD *outputLocal = (__ubuf__ ElementD *)(ubD.GetPhyAddr());
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            AscendC::DataCopyExtParams srcCopyParams(1, processNum * sizeof(ElementCompute), 0, 0, 0);
            AscendC::DataCopyPadExtParams<ElementCompute> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(ubC, gmInput[offset], srcCopyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
            // step 1: bf16/fp16 -> fp32
            AscendC::Cast(ubScale, ubC, AscendC::RoundMode::CAST_NONE, processNum);
            AscendC::PipeBarrier<PIPE_V>();
            // step 2: fp32 data * fp32 scale
            AscendC::Muls(ubScale, ubScale, scaleVal, processNum);
            AscendC::PipeBarrier<PIPE_V>();
            // step 3: fp32 -> target type (e.g. hifloat8)
            AscendC::VF_CALL<CastHighToLow<ElementD, ElementScale>>(outputLocal, scaleLocal, processNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);
            uint32_t processSize = processNum * sizeof(ElementD);
            AscendC::DataCopyExtParams copyParams{1, processSize, 0, 0, 0};
            AscendC::DataCopyPad(gmD[offset], ubD, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
        }
    }
private:

    Params params;
    AscendC::LocalTensor<ElementC> ubInputList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubOutputList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScale;
    int32_t eventInputVtoMTE2[UB_STAGES];
    int32_t eventInputMTE2toV[UB_STAGES];
    int32_t eventOutputMTE3toV[UB_STAGES];
    int32_t eventOutputVtoMTE3[UB_STAGES];
    uint32_t ubListId{0};
};



}  // namespace Catlass::Epilogue::Block



#endif  // CATCCOS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TENSOR_QUANT_HPP
