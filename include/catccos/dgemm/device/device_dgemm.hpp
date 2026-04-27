/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP
#define CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP

#include <acl/acl.h>
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "catlass/detail/kernel_adapter.hpp"

namespace Catccos::DGemm::Device {

template <class DGemmKernel>
class DeviceDGemm {
public:
    using Kernel = DGemmKernel;
    /// Argument structure: User API
    using Arguments = typename DGemmKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename DGemmKernel::Params;

private:
    /// kernel API parameters object
    Params params_;

public:
    DeviceDGemm() {}
    ~DeviceDGemm() {}

    /// Access the Params structure
    Params const &params() const
    {
        return params_;
    }

    /// Gets the workspace size
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        return DGemmKernel::GetWorkspaceSize(args);
    }

    /// Initializes DGemm state from arguments
    Catlass::Status Initialize(Arguments const &args, uint8_t *workspace = nullptr)
    {
        params_ = DGemmKernel::ToUnderlyingArguments(args, workspace);
        return Catlass::Status::kSuccess;
    }

    /// Primary run() entry point API
    inline Catlass::Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        if (fftsAddr == 0) {
            Catlass::KernelAdapter<DGemmKernel><<<blockDim, nullptr, stream>>>(params_);
        } else {
            Catlass::KernelAdapter<DGemmKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr);
        }
        return Catlass::Status::kSuccess;
    }

    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        return Run(stream, blockDim, fftsAddr);
    }
};

} // namespace Catccos::DGemm::Device

#endif // CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP
