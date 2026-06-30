/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_ARCH_HCCL_COMM_H
#define CATCCOS_ARCH_HCCL_COMM_H

// from catlass
#include "catlass/catlass.hpp"

namespace Catccos::Arch {

inline constexpr uint64_t MB_SIZE = 1024 * 1024UL;

template <typename T>
CATLASS_DEVICE
void StoreElement(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template <typename T>
CATLASS_DEVICE
T LoadElement(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

CATLASS_DEVICE
void Dcci(__gm__ uint8_t * addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

CATLASS_DEVICE
int32_t WaitUntilEqualForBarrier(__gm__ int32_t *sigAddr, int32_t cmpVal)
{
    do {
        Dcci((__gm__ uint8_t *)sigAddr);

        if (*sigAddr == cmpVal) {
            return *sigAddr;
        }

        // in case when peer pe enters next barrier
        if (*sigAddr == cmpVal + 1) {
            return *sigAddr;
        }
    } while (true);

    // never reach
    return -1;
}

template <int32_t MAX_RANK_NUM_>
class HcclComm {
public:
    static constexpr int32_t MAX_RANK_NUM = MAX_RANK_NUM_;

    struct Params {
        GM_ADDR mc2InitTiling;
        GM_ADDR mc2CcTiling;
        uint64_t segmentSize;

        CATLASS_HOST_DEVICE
        Params() : mc2InitTiling(nullptr), mc2CcTiling(nullptr), segmentSize(0)
        {
        }

        CATLASS_HOST_DEVICE
        explicit Params(GM_ADDR mc2InitTiling_, GM_ADDR mc2CcTiling_ = nullptr, uint64_t segmentSize_ = 0)
            : mc2InitTiling(mc2InitTiling_), mc2CcTiling(mc2CcTiling_), segmentSize(segmentSize_)
        {
        }
    };

    CATLASS_DEVICE
    HcclComm()
    {
    }

    CATLASS_DEVICE
    HcclComm(Params const &params)
    {
        Init(params);
    }

    CATLASS_DEVICE
    void Init(Params const &params)
    {
        auto hcclContext = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        hccl_.Init(hcclContext, reinterpret_cast<__gm__ void *>(params.mc2InitTiling));
        if (params.mc2CcTiling != nullptr) {
            hccl_.SetCcTiling(reinterpret_cast<__gm__ void *>(params.mc2CcTiling));
        }

        rankId_ = hccl_.GetRankId();
        rankSize_ = hccl_.GetRankDim();
        segmentSize_ = params.segmentSize;

        for (int i = 0; i < rankSize_; i++) {
            ptrArr_[i] = hccl_.GetWindowsInAddr(i);
        }
    }

    CATLASS_DEVICE
    void CrossRankSync() {
        uint64_t flagOffset = (segmentSize_ - MB_SIZE) / sizeof(int32_t);
        auto syncCounter = reinterpret_cast<__gm__ int32_t *>(ptrArr_[rankId_]) + flagOffset;
        auto syncBase = reinterpret_cast<__gm__ int32_t *>(ptrArr_[rankId_]) + flagOffset + 2048;
        int32_t count = LoadElement(syncBase) + 1;
        int32_t vectorId = AscendC::GetBlockIdx();
        int32_t vectorNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for(int i = vectorId; i < rankSize_; i += vectorNum) {
            auto syncRemote = reinterpret_cast<__gm__ int32_t *>(ptrArr_[i]) + flagOffset + rankId_ * 16;
            StoreElement(syncRemote, count);
            Dcci(reinterpret_cast<__gm__ uint8_t *>(syncRemote));
        }
        for(int i = vectorId; i < rankSize_; i += vectorNum) {
            auto syncCheck = syncCounter + i * 16;
            WaitUntilEqualForBarrier(syncCheck, count);
        }

        AscendC::SyncAll<true>();
        StoreElement(syncBase, count);
    }

    CATLASS_DEVICE
    auto GetPeerMem() const
    {
        return ptrArr_[rankId_];
    }

    CATLASS_DEVICE
    auto GetPeerMem(int32_t rankId) const
    {
        return ptrArr_[rankId];
    }

    CATLASS_DEVICE
    auto GetRankIdx() const
    {
        return rankId_;
    }

    CATLASS_DEVICE
    auto GetRankSize() const
    {
        return rankSize_;
    }

private:
    AscendC::Hccl<AscendC::HCCL_SERVER_TYPE_AICPU> hccl_{};
    int32_t rankId_{-1};
    int32_t rankSize_{0};
    uint64_t segmentSize_{0};
    GM_ADDR ptrArr_[MAX_RANK_NUM]{nullptr};
};

} // namespace Catccos::Arch

#endif
