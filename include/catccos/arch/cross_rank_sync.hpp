/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_ARCH_CROSS_RANK_SYNC_HPP
#define CATCCOS_ARCH_CROSS_RANK_SYNC_HPP

#include "catccos/catccos.hpp"

// from catlass
#include "catlass/catlass.hpp"

// from shmem
#include "shmem.h"

namespace Catccos::Arch {

constexpr int32_t FLAG_ZERO_IDX = 0;
constexpr int32_t FLAG_ONE_IDX = 1;
constexpr int32_t FLAG_TWO_IDX = 2;
constexpr int32_t FLAG_NUM = 3;

constexpr int64_t IPC_BUFF_MAX_SIZE = 1024UL * 1024UL * 1000;
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);

CATLASS_DEVICE
void SetRankFlag(__ubuf__ int32_t *ctrlFlagsUB, __gm__ int32_t *buff, int64_t flag)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);

    AscendC::GlobalTensor<int64_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buff), FLAG_UNIT_INT_NUM);

    AscendC::LocalTensor<int64_t> ubTensor;
    AscendC::TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubAddr.bufferAddr = reinterpret_cast<int64_t>(ctrlFlagsUB);
    ubTensor.SetAddr(ubAddr);
    ubTensor.SetValue(0, flag);

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::DataCopy(gmTensor, ubTensor, FLAG_UNIT_INT_NUM);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}

CATLASS_DEVICE
void SetRemoteRankFlag(__gm__ int32_t *buff, int64_t flag, uint32_t peerIdx)
{
    AscendC::LocalTensor<uint32_t> ubLocal32;
    ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ubLocal64;
    ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
    ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

    AscendC::GlobalTensor<int64_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buff), FLAG_UNIT_INT_NUM);

    auto ptr = shmem_ptr((__gm__ void *)gmTensor.GetPhyAddr(), peerIdx);
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(gmTensor.GetPhyAddr()), peerIdx, 0, FLAG_UNIT_INT_NUM, ubLocal64, ubLocal32, 0);
}

CATLASS_DEVICE
void CheckRankFlag(__ubuf__ int32_t *ctrlFlagsUB, __gm__ int32_t *buff, int64_t flag)
{
    AscendC::LocalTensor<int64_t> ubTensor;
    AscendC::TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<int64_t>(ctrlFlagsUB);
    ubTensor.SetAddr(ubAddr);
    AscendC::GlobalTensor<int64_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buff));
    bool isSync = false;
    do {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(ubTensor, gmTensor, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        int64_t v = ubTensor.GetValue(0);
        if (v == flag) {
            isSync = true;
        }
    } while (!isSync);
}

CATLASS_DEVICE
void CheckRemoteRankFlag(__ubuf__ int32_t *ctrlFlagsUB, __gm__ int32_t *buff, int64_t flag, uint32_t peerIdx)
{
    AscendC::LocalTensor<uint32_t> ubLocal32;
    ubLocal32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ubLocal32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ubLocal64;
    ubLocal64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR + UB_ALIGN_SIZE);
    ubLocal64.address_.dataLen = UB_ALIGN_SIZE;

    AscendC::LocalTensor<int64_t> ubTensor;
    AscendC::TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<int64_t>(ctrlFlagsUB);
    ubTensor.SetAddr(ubAddr);

    AscendC::GlobalTensor<int64_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buff));

    bool isSync = false;
    do {
        AscendC::PipeBarrier<PIPE_ALL>();
        auto ptr = shmem_ptr((__gm__ void *)gmTensor.GetPhyAddr(), peerIdx);
        aclshmemi_roce_read((__gm__ uint8_t*)(gmTensor.GetPhyAddr()), (__gm__ uint8_t*)ptr, peerIdx, 0, FLAG_UNIT_INT_NUM, ubLocal64, ubLocal32, 0);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(ubTensor, gmTensor, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        int64_t v = ubTensor.GetValue(0);
        if (v == flag) {
            isSync = true;
        }
    } while (!isSync);
}

CATLASS_DEVICE
void CrossRankSync(int32_t flagIdx, int64_t flagData, int32_t rank, int32_t rankSize,
    __ubuf__ int32_t *ctrlFlagsUB, __gm__ int32_t *buff)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t aivIdx = AscendC::GetSubBlockIdx();
    int32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    // write check
    if (aivIdx == 0 && coreIdx == rank) {
        GM_ADDR syncAddr = (GM_ADDR)buff + IPC_BUFF_MAX_SIZE + FLAG_NUM * SYNC_UNIT_SIZE * coreIdx + flagIdx * SYNC_UNIT_SIZE;
        SetRankFlag(ctrlFlagsUB, (__gm__ int32_t *)syncAddr, flagData);
        for (uint32_t rankIdx = 0; rankIdx < rankSize; rankIdx++) {
            if (rankIdx == rank) continue;
            SetRemoteRankFlag((__gm__ int32_t *)syncAddr, flagData, rankIdx);
        }
    } else if (aivIdx == 0 && coreIdx < rankSize) {
        GM_ADDR syncAddr = (GM_ADDR)buff + IPC_BUFF_MAX_SIZE + FLAG_NUM * SYNC_UNIT_SIZE * coreIdx + flagIdx * SYNC_UNIT_SIZE;
        CheckRankFlag(ctrlFlagsUB, (__gm__ int32_t *)syncAddr, flagData);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

CATLASS_DEVICE
void ResetFlags(int32_t numFlags, int32_t rank, int32_t rankSize, __ubuf__ int32_t *ctrlFlagsUB, __gm__ int32_t *buff)
{
    for (int32_t flagIdx = 0; flagIdx < numFlags; flagIdx++) {
        CrossRankSync(flagIdx, 0, rank, rankSize, ctrlFlagsUB, buff);
    }
}

} // namespace Catccos::Arch

#endif // CATCCOS_ARCH_CROSS_RANK_SYNC_HPP