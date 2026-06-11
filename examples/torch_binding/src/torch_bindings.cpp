/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
#include <algorithm>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "shmem.h"
#include "catccos_torch_kernel.h"
#include "utils.h"
#include "shmem_init.h"
#include "info.h"

namespace {

struct CatccosTorchOpState {
    bool initialized = false;
    void* symmPtr = nullptr;
    size_t symmBytes = 0;
    int64_t rankId = -1;
    int64_t rankSize = 0;
};

CatccosTorchOpState& get_catccos_state()
{
    static CatccosTorchOpState state;
    return state;
}

void* get_catccos_symm_ptr(size_t bytes)
{
    auto& state = get_catccos_state();
    TORCH_CHECK(state.initialized,
                "catccos TORCH_LIBRARY not initialized, "
                "call torch.ops.catccos.init(...) first");

    if (state.symmPtr == nullptr) {
        state.symmPtr = shmem_malloc(bytes);
        TORCH_CHECK(state.symmPtr != nullptr,
                    "catccos TORCH_LIBRARY shmem_malloc failed, bytes=", bytes);
        state.symmBytes = bytes;

        auto ret = aclrtMemset(state.symmPtr, bytes, 0, bytes);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemset failed, ret=", ret);
    } else {
        TORCH_CHECK(state.symmBytes >= bytes,
                    "existing symm buffer too small, existing=",
                    state.symmBytes, ", required=", bytes);
    }

    return state.symmPtr;
}

uint64_t get_catccos_ffts()
{
    TORCH_CHECK(get_catccos_state().initialized,
                "catccos TORCH_LIBRARY not initialized, "
                "call torch.ops.catccos.init(...) first");

    uint64_t addr = shmemx_get_ffts_config();
    TORCH_CHECK(addr != 0, "shmemx_get_ffts_config returned 0");
    return addr;
}

}  // namespace

namespace catccos {

int64_t init(int64_t rank_id, int64_t rank_size,
             int64_t local_mem_size, const std::string& ip_port)
{
    auto& state = get_catccos_state();
    if (state.initialized) {
        return 0;
    }

    TORCH_CHECK(rank_id >= 0, "invalid rank_id: ", rank_id);
    TORCH_CHECK(rank_size > 0, "invalid rank_size: ", rank_size);
    TORCH_CHECK(local_mem_size > 0, "invalid local_mem_size: ", local_mem_size);
    TORCH_CHECK(!ip_port.empty(), "ip_port is empty");

    int64_t status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "aclshmemx_set_conf_store_tls failed: ", status);

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    int32_t ret = set_attr(
        static_cast<int32_t>(rank_id),
        static_cast<int32_t>(rank_size),
        static_cast<uint64_t>(local_mem_size),
        ip_port.c_str(),
        &attributes,
        &default_flag_uid);
    TORCH_CHECK(ret == ACLSHMEM_SUCCESS, "set_attr failed, ret=", ret);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "aclshmemx_init_attr failed: ", status);

    state.initialized = true;
    state.rankId = rank_id;
    state.rankSize = rank_size;
    return 0;
}

at::Tensor allgather_matmul(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t rank_size)
{
    c10_npu::OptionalNPUGuard guard(a.device());

    auto& state = get_catccos_state();
    TORCH_CHECK(state.initialized,
                "catccos TORCH_LIBRARY not initialized, "
                "call torch.ops.catccos.init(...) first");
    TORCH_CHECK(rank_size == state.rankSize,
                "rank_size mismatch: input=", rank_size,
                ", initialized=", state.rankSize);

    TORCH_CHECK(a.dtype() == at::kHalf, "Only float16 supported for a, got ", a.dtype());
    TORCH_CHECK(b.dtype() == at::kHalf, "Only float16 supported for b, got ", b.dtype());
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
                "Only NPU supported for a, got ", a.device());
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1,
                "Only NPU supported for b, got ", b.device());
    TORCH_CHECK(a.device() == b.device(), "a and b must be on same device");
    TORCH_CHECK(a.dim() == 2, "a must be 2D, got dim=", a.dim());
    TORCH_CHECK(b.dim() == 2, "b must be 2D, got dim=", b.dim());

    int64_t m = a.size(0);
    int64_t k = a.size(1);
    int64_t n = b.size(1);
    TORCH_CHECK(b.size(0) == k,
                "A/K shape mismatch, a.size(1)=", k,
                ", b.size(0)=", b.size(0));

    int32_t my_pe = shmem_my_pe();
    int32_t n_pes = shmem_n_pes();
    TORCH_CHECK(my_pe >= 0, "invalid my_pe: ", my_pe);
    TORCH_CHECK(n_pes > 0, "invalid n_pes: ", n_pes);
    TORCH_CHECK(rank_size == n_pes,
                "rank_size mismatch: input=", rank_size,
                ", shmem_n_pes=", n_pes);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = at::empty({m * n_pes, n}, a.options()).contiguous();

    uint8_t* aPtr = static_cast<uint8_t*>(a_contig.data_ptr());
    uint8_t* bPtr = static_cast<uint8_t*>(b_contig.data_ptr());
    uint8_t* cPtr = static_cast<uint8_t*>(c.data_ptr());
    uint8_t* symmPtr = static_cast<uint8_t*>(get_catccos_symm_ptr(SHMEM_BUFF_BYTES));
    uint64_t ffts = get_catccos_ffts();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "NPU stream is null");

    at_npu::native::OpCommand cmd;
    cmd.Name("catccos_allgather_matmul");
    cmd.Input(a_contig);
    cmd.Input(b_contig);
    cmd.Output(c);
    cmd.SetCustomHandler([stream, ffts, aPtr, bPtr, cPtr, symmPtr,
                          m, n, k, my_pe, n_pes]() -> int {
        CatccosKernel::catccos_allgather_matmul_wrapper(
            BLOCK_NUM,
            stream,
            ffts,
            aPtr,
            bPtr,
            cPtr,
            symmPtr,
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            my_pe,
            n_pes);
        return 0;
    });
    cmd.Run();

    return c;
}

int64_t finalize()
{
    auto& state = get_catccos_state();
    if (!state.initialized) {
        return 0;
    }

    if (state.symmPtr != nullptr) {
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        if (stream != nullptr) {
            auto sync_ret = aclrtSynchronizeStream(stream);
            TORCH_CHECK(sync_ret == ACL_SUCCESS,
                        "aclrtSynchronizeStream before shmem_free failed, ret=", sync_ret);
        }
        shmem_free(state.symmPtr);
        state.symmPtr = nullptr;
        state.symmBytes = 0;
    }

    int64_t status = aclshmem_finalize();
    state.initialized = false;
    state.rankId = -1;
    state.rankSize = 0;
    TORCH_CHECK(status == ACLSHMEM_SUCCESS,
                "aclshmem_finalize failed: ", status);
    return 0;
}

}  // namespace catccos

TORCH_LIBRARY(catccos, m) {
    m.def("init(int rank_id, int rank_size, int local_mem_size, str ip_port) -> int");
    m.def("allgather_matmul(Tensor a, Tensor b, int rank_size) -> Tensor");
    m.def("finalize() -> int");

    m.impl("init", &catccos::init);
    m.impl("allgather_matmul", torch::kPrivateUse1, &catccos::allgather_matmul);
    m.impl("finalize", &catccos::finalize);
}
