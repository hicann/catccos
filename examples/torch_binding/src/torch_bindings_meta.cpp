/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/torch.h>
#include <torch/library.h>

#include <vector>

namespace catccos::meta {

at::Tensor allgather_matmul_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t rank_size)
{
    auto a_sizes = a.sym_sizes();
    auto b_sizes = b.sym_sizes();
    std::vector<c10::SymInt> out_shape = {
        a_sizes[0] * c10::SymInt(rank_size),
        b_sizes[1],
    };
    return at::empty_symint(out_shape, a.options());
}

}  // namespace catccos::meta

TORCH_LIBRARY_IMPL(catccos, Meta, m) {
    m.impl("allgather_matmul", &catccos::meta::allgather_matmul_meta);
}
