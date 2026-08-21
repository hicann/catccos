/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "acl/acl.h"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_gmm.hpp"

namespace
{

using Scheduler = Catlass::Gemm::Block::BlockCommSchedulerAllToAllVGmm;

constexpr uint32_t TOKEN_COUNT = 32;
constexpr uint32_t RESULT_COUNT = 16;

CATLASS_DEVICE void FlushResult(GM_ADDR resultData)
{
    AscendC::GlobalTensor<uint8_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(resultData));
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        result);
    __asm__ __volatile__("");
}

CATLASS_GLOBAL void TestSetCommCore(GM_ADDR tokenData, GM_ADDR resultData)
{
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenData), TOKEN_COUNT);

    AscendC::GlobalTensor<int32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(resultData), RESULT_COUNT);

    Scheduler scheduler;
    scheduler.rank = 2;
    scheduler.EP = 4;
    scheduler.expertPerRank = 2;
    scheduler.tokenPerExpert = tokenPerExpert;
    scheduler.commContext = {7, 11, 13};

    scheduler.SetCommCore(1, 3);
    result.SetValue(0, static_cast<int32_t>(scheduler.coreIdx));
    result.SetValue(1, static_cast<int32_t>(scheduler.coreNum));
    result.SetValue(2, scheduler.GetLocalExpertIdx());
    result.SetValue(3, scheduler.GetPrevGroupSum());
    result.SetValue(4, scheduler.GetSrcPrevSum());
    result.SetValue(5, static_cast<int32_t>(scheduler.GetCommLoops()));
    result.SetValue(6, static_cast<int32_t>(scheduler.GetBlockOffset(0).row()));
    result.SetValue(7, static_cast<int32_t>(scheduler.GetBlockOffset(1).row()));

    scheduler.commContext = {5, 9, 17};
    scheduler.SetCommCore(4, 6);
    result.SetValue(8, static_cast<int32_t>(scheduler.coreIdx));
    result.SetValue(9, scheduler.GetSrcPrevSum());
    result.SetValue(10, static_cast<int32_t>(scheduler.GetCommLoops()));
    result.SetValue(11, static_cast<int32_t>(scheduler.GetBlockOffset(0).row()));

    scheduler.commContext = {3, 8, 21};
    scheduler.SetCommCore(0, 3);
    result.SetValue(12, static_cast<int32_t>(scheduler.coreIdx));
    result.SetValue(13, scheduler.GetSrcPrevSum());
    result.SetValue(14, static_cast<int32_t>(scheduler.GetCommLoops()));
    result.SetValue(15, static_cast<int32_t>(scheduler.GetBlockOffset(1).row()));

    FlushResult(resultData);
}

bool CheckAcl(aclError status, const char *operation)
{
    if (status == ACL_ERROR_NONE)
    {
        return true;
    }
    std::cerr << "[FAILED] " << operation << ", aclError=" << status << std::endl;
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    std::array<int32_t, TOKEN_COUNT> tokenData{};
    std::array<int32_t, RESULT_COUNT> actual{};
    const std::array<int32_t, RESULT_COUNT> expected = {
        1, 3, 0, 0, 42, 1, 1, 4, 4, 0, 0, 4, 0, 10, 2, 3,
    };
    for (uint32_t i = 0; i < TOKEN_COUNT; ++i)
    {
        tokenData[i] = static_cast<int32_t>(i + 1);
    }

    aclrtStream stream = nullptr;
    uint8_t *tokenDevice = nullptr;
    uint8_t *resultDevice = nullptr;
    bool aclInitialized = false;
    bool deviceSet = false;
    bool passed = false;

    do
    {
        if (!CheckAcl(aclInit(nullptr), "aclInit"))
        {
            break;
        }
        aclInitialized = true;
        if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice"))
        {
            break;
        }
        deviceSet = true;
        if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
            !CheckAcl(
                aclrtMalloc(reinterpret_cast<void **>(&tokenDevice), sizeof(tokenData), ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(tokens)") ||
            !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&resultDevice), sizeof(actual), ACL_MEM_MALLOC_HUGE_FIRST),
                      "aclrtMalloc(result)") ||
            !CheckAcl(aclrtMemcpy(tokenDevice, sizeof(tokenData), tokenData.data(), sizeof(tokenData),
                                  ACL_MEMCPY_HOST_TO_DEVICE),
                      "aclrtMemcpy(tokens)"))
        {
            break;
        }

        TestSetCommCore<<<1, nullptr, stream>>>(tokenDevice, resultDevice);
        if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
            !CheckAcl(
                aclrtMemcpy(actual.data(), sizeof(actual), resultDevice, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(result)"))
        {
            break;
        }

        passed = actual == expected;
        if (!passed)
        {
            for (uint32_t i = 0; i < RESULT_COUNT; ++i)
            {
                if (actual[i] != expected[i])
                {
                    std::cerr << "[FAILED] result[" << i << "]: expected " << expected[i] << ", got " << actual[i]
                              << std::endl;
                }
            }
        }
        else
        {
            std::cout << "[PASSED] BlockCommSchedulerAllToAllVGmm::SetCommCore" << std::endl;
        }
    } while (false);

    if (resultDevice != nullptr)
    {
        aclrtFree(resultDevice);
    }
    if (tokenDevice != nullptr)
    {
        aclrtFree(tokenDevice);
    }
    if (stream != nullptr)
    {
        aclrtDestroyStream(stream);
    }
    if (deviceSet)
    {
        aclrtResetDevice(deviceId);
    }
    if (aclInitialized)
    {
        aclFinalize();
    }
    return passed ? 0 : 1;
}
