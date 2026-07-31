/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_ARCH_SHMEM_COMM_HPP
#define CATCCOS_ARCH_SHMEM_COMM_HPP

// from catlass
#include "catlass/catlass.hpp"

#include "shmem.h"

namespace Catccos::Arch {

class ShmemComm {
public:
    struct Params {
        GM_ADDR ptrSymmetric;

        CATLASS_HOST_DEVICE
        Params() : ptrSymmetric(nullptr)
        {
        }

        CATLASS_HOST_DEVICE
        explicit Params(GM_ADDR ptrSymmetric_) : ptrSymmetric(ptrSymmetric_)
        {
        }
    };

    CATLASS_DEVICE
    ShmemComm()
    {
    }

    CATLASS_DEVICE
    ShmemComm(Params const &params_) : params(params_)
    {
    }

    CATLASS_DEVICE
    void Init(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void CrossRankSync()
    {
        aclshmemx_barrier_all_vec();
    }

    CATLASS_DEVICE
    auto GetPeerMem() const
    {
        return params.ptrSymmetric;
    }

    CATLASS_DEVICE
    auto GetPeerMem(int32_t rankId) const
    {
        return shmem_ptr(params.ptrSymmetric, rankId);
    }

    CATLASS_DEVICE
    auto GetRankIdx() const
    {
        return shmem_my_pe();
    }

    CATLASS_DEVICE
    auto GetRankSize() const
    {
        return shmem_n_pes();
    }

private:
    Params params;
};

} // namespace Catccos::Arch

#endif
