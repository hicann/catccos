/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mte_cost_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

MTECostModel::MTECostModel(CostModelHardwareConfig const &config) : config_(config) {}

MTECacheConfig const &MTECostModel::GetCacheConfig(CacheStatus status) const
{
    return status == CacheStatus::HIT ? config_.cacheHit : config_.cacheMiss;
}

double MTECostModel::Nd2NzContinuous(uint32_t coreNum, uint32_t instructionNum, uint32_t nValue, uint32_t dValue,
                                     CacheStatus cacheStatus) const
{
    // TODO(A5_MEASURE): calibrate ND2NZ dValue=512.
    if (dValue != 256 && dValue != 512)
    {
        throw std::invalid_argument("only dValue=256/512 is supported");
    }
    if (coreNum == 0 || instructionNum == 0 || nValue == 0 || config_.inputElementBits == 0 ||
        config_.readOtsdBound == 0 || config_.nd2nzCmdOtsd == 0)
    {
        throw std::invalid_argument("ND2NZ arguments must be positive");
    }

    // The fitted request model uses dValue=256 and 16-bit input as its
    // baseline. Preserve the actual ND2NZ byte volume for other precisions
    // and dValue values before applying the OSTD model.
    constexpr uint64_t BASELINE_D_VALUE = 256;
    constexpr uint64_t BASELINE_ELEMENT_BITS = 16;
    uint64_t requestN = (static_cast<uint64_t>(nValue) * dValue * config_.inputElementBits +
                         BASELINE_D_VALUE * BASELINE_ELEMENT_BITS - 1) /
                        (BASELINE_D_VALUE * BASELINE_ELEMENT_BITS);
    if (requestN > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("normalized ND2NZ request count is too large");
    }

    auto const &cache = GetCacheConfig(cacheStatus);
    uint32_t cmdOtsd = config_.nd2nzCmdOtsd;
    constexpr uint32_t CMD_ROUNDS = 2;
    auto estimateWithinBound = [&](uint32_t boundedRequestN)
    {
        double averageTimeNs;
        if (boundedRequestN <= 32)
        {
            uint32_t requestCount = boundedRequestN * (cmdOtsd + 1);
            double scheduleTime = cache.requestIntervalNs * requestCount;
            averageTimeNs = (scheduleTime + cache.readRttNs * cmdOtsd) / (cmdOtsd * CMD_ROUNDS);
        }
        else
        {
            uint32_t requestCount = boundedRequestN * cmdOtsd * CMD_ROUNDS;
            double scheduleTime = cache.requestIntervalNs * requestCount;
            averageTimeNs = (scheduleTime + cache.readRttNs) / (cmdOtsd * CMD_ROUNDS);
        }

        double timeUs = averageTimeNs * instructionNum / 1000.0;
        if (coreNum == config_.coreNum && cacheStatus == CacheStatus::HIT)
        {
            timeUs /= config_.fullCoreHitEfficiency;
        }
        return timeUs;
    };

    if (requestN > config_.readOtsdBound)
    {
        double baseTime = estimateWithinBound(config_.readOtsdBound);
        double scaledTime = baseTime * requestN / config_.readOtsdBound;
        return std::round(scaledTime * 1000.0) / 1000.0;
    }
    return estimateWithinBound(static_cast<uint32_t>(requestN));
}
