/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_KERNEL_WITH_FLAG_H
#define ALLGATHER_KERNEL_WITH_FLAG_H
 
#include "catlass/arch/resource.hpp"
#include "catccos/catccos.hpp"
#include "shmem.h"

constexpr static int32_t CACHE_LINE = 512;

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


    CATLASS_DEVICE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
        do {
            AscendC::LocalTensor<int32_t> ub;
            ub.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
            ub.address_.bufferAddr = 0;
            AscendC::GlobalTensor<int32_t> sig;
            sig.SetGlobalBuffer(sig_addr);
            AscendC::DataCopy(ub, sig, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            if (ub(0) != cmp_val) {
                return;
            }
        } while (true);
        return;
    }

    CATLASS_DEVICE
    void operator()(Params &params, Catlass::Arch::Resource<ArchTag> resource)
    {
        if (g_coreType == AIC) {
            return;
        }
        
        uint32_t rankIdx = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        auto elementNum = params.elementNum;
        auto layout = Catlass::layout::RowMajor{1, elementNum};
        auto layoutUb = Catlass::layout::RowMajor{1, elementNum, RoundUp<Catlass::BYTE_PER_C0 / sizeof(Element)>(elementNum)};
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[elementNum];

        AscendC::GlobalTensor<Element> src;
        src.SetGlobalBuffer(params.srcPtr);

        for (int32_t dstRank = aicoreIdx; dstRank < rankSize  && subcoreIdx == 0; dstRank += aicoreNum) {
            if (dstRank == rankIdx) {
                continue;
            }
            AscendC::GlobalTensor<Element> peerMem;
            auto peerMemPtr = reinterpret_cast<__gm__ Element *>(shmem_ptr(params.symmetricPtr, dstRank));
            peerMem.SetGlobalBuffer(peerMemPtr);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            
            copyGmToUb(tmpBuffer, src[0], layoutUb, layout);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, elementNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGm(peerMem[rankIdx * elementNum], tmpBuffer, layout, layoutUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        AscendC::GlobalTensor<Element> peerMem;
        peerMem.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(params.symmetricPtr));

        for (int32_t dstRank = aicoreIdx; dstRank < rankSize  && subcoreIdx == 0; dstRank += aicoreNum) {
            if (dstRank != rankIdx) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for(int32_t checkIdx = 0; checkIdx < elementNum; checkIdx += intPer512) {
                    __gm__ int32_t* sync_check = reinterpret_cast<__gm__ int32_t*>(params.symmetricPtr) + dstRank * elementNum + checkIdx;
                    gm_signal_wait_until_ne(sync_check, 0);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(tmpBuffer, peerMem[dstRank * elementNum], layoutUb, layout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, elementNum);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                copyUbToGm(peerMem[dstRank * elementNum], tmpBuffer, layout, layoutUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            } 
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::SyncAll<true>();
    }

private:
    AscendC::LocalTensor<Element> tmpBuffer;
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
};

#endif