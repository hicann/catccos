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
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "torch_npu/csrc/aten/common/from_blob.h"

#include "shmem.h"
#include "torch_register.h"
#include "catccos_torch_kernel.h"
#include "utils.h"
#include "shmem_init.h"
#include "info.h"


using half = __fp16;

namespace CatccosOps {



class Manager : public torch::jit::CustomClassHolder {
public:
    // Default constructor
    Manager() : name_("Manager") {}

    std::string get_name() const
    {
        return name_;
    }

    int64_t attr_init(int64_t my_pe, int64_t n_ranks, int64_t local_mem_size, const std::string& ip_port)
    {
        int64_t status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
        if (status != ACLSHMEM_SUCCESS) {
            std::cerr << "[ERROR] aclshmemx_set_conf_store_tls failed: " << status << std::endl;
            return status;
        }

        aclshmemx_uniqueid_t default_flag_uid;
        aclshmemx_init_attr_t attributes;
        set_attr(my_pe, n_ranks, local_mem_size, ip_port.c_str(), &attributes, &default_flag_uid);

        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        if (status != ACLSHMEM_SUCCESS) {
            std::cerr << "[ERROR] aclshmemx_init_attr failed: " << status << std::endl;
        }
        return status;
    }
    int64_t finalize()
    {
        return aclshmem_finalize();
    }

    at::Tensor malloc_tensor(int64_t size)
    {
        void *symmPtr = aclshmem_malloc(size);
        at::Tensor aclshmem_tensor = at_npu::native::from_blob(symmPtr, size, torch::dtype(torch::kUInt8));

        return aclshmem_tensor;
    }

    at::Tensor malloc_like(const at::Tensor& npu_tensor)
    {
        void *npu_tensor_ptr = static_cast<void *>(const_cast<void *>(npu_tensor.storage().data()));
        int64_t size = npu_tensor.storage().nbytes();
        void *symmPtr = aclshmem_malloc(size);
        at::Tensor aclshmem_tensor = at_npu::native::from_blob(symmPtr, npu_tensor.sizes(), npu_tensor.dtype());
        aclrtMemcpy(symmPtr, size, npu_tensor_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        return aclshmem_tensor;
    }

    void free_tensor(const at::Tensor& aclshmem_tensor)
    {
        void *aclshmem_ptr = static_cast<void *>(const_cast<void *>(aclshmem_tensor.storage().data()));
        aclshmem_free(aclshmem_ptr);
        return;
    }

private:
    std::string name_;
};



class AllGatherMatmul : public torch::jit::CustomClassHolder {
public:
    AllGatherMatmul() : name_("CatccosAllGatherMatmul"), count_(0), fftsAddr_(shmemx_get_ffts_config()), symmPtr_(nullptr)
    {
        // Allocate symmetric memory for inter-rank communication
        symmPtr_ = static_cast<uint8_t*>(shmem_malloc(SHMEM_BUFF_BYTES));
        aclrtMemset(symmPtr_, SHMEM_BUFF_BYTES, 0, SHMEM_BUFF_BYTES);
    }

    ~AllGatherMatmul()
    {
        if (symmPtr_ != nullptr) {
            shmem_free(symmPtr_);
            symmPtr_ = nullptr;
        }
    }

    std::string get_name() const
    {
        return name_;
    }

    void compute(const at::Tensor& c_tensor,
                 const at::Tensor& a_tensor,
                 const at::Tensor& b_tensor)
    {
        // 1. Parameter validation
        TORCH_CHECK(a_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ",
                    "Current dtype: ", a_tensor.dtype().name());

        TORCH_CHECK(b_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ",
                    "Current dtype: ", b_tensor.dtype().name());

        TORCH_CHECK(c_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ",
                    "Current dtype: ", c_tensor.dtype().name());

        TORCH_CHECK(a_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! Current device: ", a_tensor.device().type());

        TORCH_CHECK(b_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! Current device: ", b_tensor.device().type());

        TORCH_CHECK(c_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! Current device: ", c_tensor.device().type());

        // 2. Shape validation
        TORCH_CHECK(a_tensor.dim() == 2, "A tensor must be 2D! Current dim: ", a_tensor.dim());
        TORCH_CHECK(b_tensor.dim() == 2, "B tensor must be 2D! Current dim: ", b_tensor.dim());
        TORCH_CHECK(c_tensor.dim() == 2, "C tensor must be 2D! Current dim: ", c_tensor.dim());

        uint32_t m = a_tensor.size(0);
        uint32_t k = a_tensor.size(1);
        uint32_t n = b_tensor.size(1);

        TORCH_CHECK(b_tensor.size(0) == k, "A/K mismatch! A.size(1)=", k, ", B.size(0)=", b_tensor.size(0));

        int32_t n_pes = shmem_n_pes();
        int64_t expected_c_rows = n_pes * m;
        TORCH_CHECK(c_tensor.size(0) == expected_c_rows && c_tensor.size(1) == n,
                    "Compute Error: C tensor shape mismatch! ",
                    "Expected shape: (", expected_c_rows, ",", n, "), Current shape: (", c_tensor.size(0), ",", c_tensor.size(1), ")");

        // 3. Make tensors contiguous
        at::Tensor a_contig = a_tensor.contiguous();
        at::Tensor b_contig = b_tensor.contiguous();
        at::Tensor c_contig = c_tensor.contiguous();

        // 4. Get device pointers
        uint8_t* aPtr = static_cast<uint8_t*>(const_cast<void*>(a_contig.storage().data()));
        uint8_t* bPtr = static_cast<uint8_t*>(const_cast<void*>(b_contig.storage().data()));
        uint8_t* cPtr = static_cast<uint8_t*>(const_cast<void*>(c_contig.storage().data()));

        // 5. Get NPU stream
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        count_++;

        // 6. Call Catccos AllGatherMatmul kernel with fixed template parameters (float16, RowMajor)
        int32_t my_pe = shmem_my_pe();
        CatccosKernel::catccos_allgather_matmul_wrapper(
            BLOCK_NUM,
            stream,
            fftsAddr_,
            aPtr,
            bPtr,
            cPtr,
            symmPtr_,
            m,
            n,
            k,
            my_pe,
            n_pes
        );
    }

private:
    std::string name_;
    int32_t count_;
    uint64_t fftsAddr_;
    uint8_t* symmPtr_;

    uint32_t BLOCK_NUM = 20;
};

} // namespace CatccosOps



// Register the class
REGISTER_CATCCOS_OPS_CLASS(Manager, attr_init, finalize, malloc_tensor, free_tensor, malloc_like, get_name);
REGISTER_CATCCOS_OPS_CLASS(AllGatherMatmul, compute, get_name);
