/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_KERNEL_H 
#define ALLGATHER_KERNEL_H

#include "catlass/arch/resource.hpp"
#include "catccos/catccos.hpp"
#include "shmem.h"

template <
    class ArchTag_,
    class Element_
>
class AllGather {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;

    using DType = Catlass::Gemm::GemmType<Element, Catlass::layout::RowMajor>;
    using CopyGmToUb = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, DType>;
    using CopyUbToGm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, DType>;

    /// Parameters structure
    struct Params {
        uint32_t elementNum;

        __gm__ Element *srcPtr;
        __gm__ Element *dstPtr;
        GM_ADDR symmetricPtr;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            uint32_t elementNum_,
            GM_ADDR srcPtr_,
            GM_ADDR dstPtr_,
            GM_ADDR symmetricPtr_
        ) : elementNum(elementNum_),
            srcPtr(reinterpret_cast<__gm__ Element *>(srcPtr_)),
            dstPtr(reinterpret_cast<__gm__ Element *>(dstPtr_)),
            symmetricPtr(symmetricPtr_)
        {
        }
    };
    
    CATLASS_DEVICE
    void operator()(Params &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        uint32_t rankIdx = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        if (g_coreType == AIC) {
            return;
        }
        tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        auto elementNum = params.elementNum;
        auto layout = Catlass::layout::RowMajor{1, elementNum};

        AscendC::GlobalTensor<Element> src;
        src.SetGlobalBuffer(params.srcPtr);
        AscendC::GlobalTensor<Element> peerMem;
        
        for (int32_t dstRank = aicoreIdx; dstRank < rankSize && subcoreIdx == 0; dstRank += aicoreNum) {
            if (dstRank == rankIdx) {
                continue;
            }
            auto peerMemPtr = reinterpret_cast<__gm__ Element *>(shmem_ptr(params.symmetricPtr, dstRank));
            peerMem.SetGlobalBuffer(peerMemPtr);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copyGmToUb(tmpBuffer, src[0], layout, layout);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            copyUbToGm(peerMem[rankIdx * elementNum], tmpBuffer, layout, layout);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        aclshmemx_barrier_all_vec();
    }

private:
    AscendC::LocalTensor<Element> tmpBuffer;
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
};

#endif