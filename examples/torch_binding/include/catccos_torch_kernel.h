/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_TORCH_KERNEL_H
#define CATCCOS_TORCH_KERNEL_H

#include <acl/acl.h>
#include <vector>

// Include Catccos types
#include "utils/info.h"

namespace CatccosKernel {

/**
 * @brief AllGather Matmul kernel wrapper
 * @param blockDim number of AICore blocks
 * @param stream ACL stream
 * @param fftsAddr FFTS configuration address
 * @param aPtr pointer to matrix A on device
 * @param bPtr pointer to matrix B on device
 * @param cPtr pointer to matrix C on device (output)
 * @param gmSymmetric pointer to symmetric memory
 * @param m rows of matrix A
 * @param n columns of matrix B
 * @param k columns of matrix A / rows of matrix B
 * @param rankId current rank index
 * @param rankSize number of ranks
 */
void catccos_allgather_matmul_wrapper(
    uint32_t blockDim,
    aclrtStream stream,
    uint64_t fftsAddr,
    uint8_t* aPtr,
    uint8_t* bPtr,
    uint8_t* cPtr,
    uint8_t* gmSymmetric,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    int rankId,
    int rankSize
    );

} // namespace CatccosKernel

#endif // CATCCOS_TORCH_KERNEL_H