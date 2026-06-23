/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_mx_quant_allgather/mx_quant_allgather_device.h"

using LayoutInput = Catlass::layout::RowMajor;
using LayoutOutput = Catlass::layout::RowMajor;

using ElementInput = bfloat16_t;
using ElementOutput = float8_e4m3_t;

void LaunchAscend950BF16QuantAllGatherFP8E4M3FN(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    MxQuantAllGather<ElementInput, LayoutInput, ElementOutput, LayoutOutput>
        <<<blockNum, nullptr, stream>>>(
            fftsAddr, kernelParams.ptrA, kernelParams.ptrC, kernelParams.ptrB, symmetricPtr, cocTiling, 10);
}