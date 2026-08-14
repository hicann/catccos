/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cost_model.h"

namespace
{

constexpr double A2_INT8_CUBE_FLOPS_PER_US = 27840672.0;
constexpr uint32_t A5_DEFAULT_CORE_NUM = 28;
constexpr double A5_WRITE_RTT_NS = 87.880;
constexpr double A5_REMOTE_READ_SCHEDULE_NS = 13.685075;
constexpr double A5_REMOTE_READ_RTT_NS = 662.657;
constexpr uint32_t A5_REMOTE_READ_REQUEST_BYTES = 512;
constexpr double A5_HCCS_BANDWIDTH_GIB_PER_S = 35.1495;
constexpr double A5_CUBE_FLOPS_PER_US = 13500000.0;
constexpr double A5_SYNC_TIME_US = 1.297780;
constexpr double A5_CROSS_CORE_BARRIER_TIME_US = 0.183095;
constexpr double A5_LAUNCH_TIME_US = 6.47;  // TODO(A5_MEASURE): launch time
constexpr uint32_t A5_READ_OTSD_BOUND = 128;
constexpr uint32_t A5_ND2NZ_CMD_OTSD = 2;                  // TODO(A5_MEASURE): ND2NZ OTSD
constexpr double A5_CACHE_MISS_READ_RTT_NS = 235.0;        // TODO(A5_MEASURE): miss RTT
constexpr double A5_CACHE_MISS_REQUEST_INTERVAL_NS = 4.5;  // TODO(A5_MEASURE): miss interval
constexpr double A5_CACHE_HIT_READ_RTT_NS = 110.0;         // TODO(A5_MEASURE): hit RTT
constexpr double A5_CACHE_HIT_REQUEST_INTERVAL_NS = 1.9;   // TODO(A5_MEASURE): hit interval
constexpr double A5_FULL_CORE_HIT_EFFICIENCY = 0.8;        // TODO(A5_MEASURE): hit efficiency

uint32_t GetElementBits(CocDataType dataType)
{
    switch (dataType)
    {
        case FP16:
        case BF16:
            return 16;
        case INT8:
        case FP8E4M3FN:
            return 8;
        default:
            return 0;
    }
}

}  // namespace

CostModelStatus GetA2CostModelHardwareConfig(CocDataType dataType, CostModelHardwareConfig &hardware)
{
    uint32_t elementBits = GetElementBits(dataType);
    if (elementBits == 0)
    {
        return CostModelStatus::INVALID_ARGUMENT;
    }

    hardware = CostModelHardwareConfig{};
    if (dataType == CocDataType::INT8)
    {
        hardware.cubeFlopsPerUs = A2_INT8_CUBE_FLOPS_PER_US;
    }
    hardware.inputElementBits = elementBits;
    hardware.communicationElementBits = elementBits;
    return CostModelStatus::SUCCESS;
}

CostModelStatus GetA3CostModelHardwareConfig(CocDataType dataType, CostModelHardwareConfig &hardware)
{
    (void)dataType;
    (void)hardware;
    return CostModelStatus::UNSUPPORTED;
}

CostModelStatus GetA5CostModelHardwareConfig(CocDataType dataType, CostModelHardwareConfig &hardware)
{
    uint32_t elementBits = GetElementBits(dataType);
    if (elementBits == 0)
    {
        return CostModelStatus::INVALID_ARGUMENT;
    }

    hardware = CostModelHardwareConfig{};
    hardware.coreNum = A5_DEFAULT_CORE_NUM;
    hardware.writeRttNs = A5_WRITE_RTT_NS;
    hardware.remoteReadScheduleNs = A5_REMOTE_READ_SCHEDULE_NS;
    hardware.remoteReadRttNs = A5_REMOTE_READ_RTT_NS;
    hardware.remoteReadRequestBytes = A5_REMOTE_READ_REQUEST_BYTES;
    hardware.hccsBandwidth = A5_HCCS_BANDWIDTH_GIB_PER_S;
    hardware.cubeFlopsPerUs = A5_CUBE_FLOPS_PER_US;
    hardware.inputElementBits = elementBits;
    hardware.communicationElementBits = elementBits;
    hardware.syncTimeUs = A5_SYNC_TIME_US;
    hardware.crossCoreBarrierTimeUs = A5_CROSS_CORE_BARRIER_TIME_US;
    hardware.launchTimeUs = A5_LAUNCH_TIME_US;
    hardware.readOtsdBound = A5_READ_OTSD_BOUND;
    hardware.nd2nzCmdOtsd = A5_ND2NZ_CMD_OTSD;
    hardware.cacheMiss = {A5_CACHE_MISS_READ_RTT_NS, A5_CACHE_MISS_REQUEST_INTERVAL_NS};
    hardware.cacheHit = {A5_CACHE_HIT_READ_RTT_NS, A5_CACHE_HIT_REQUEST_INTERVAL_NS};
    hardware.fullCoreHitEfficiency = A5_FULL_CORE_HIT_EFFICIENCY;
    return CostModelStatus::SUCCESS;
}

CostModelStatus GetCostModelHardwareConfig(CostModelConfig const &config, CostModelHardwareConfig &hardware)
{
    auto status = GetCostModelHardwareConfig(config.hardwareType, config.dataType, hardware);
    if (status != CostModelStatus::SUCCESS)
    {
        return status;
    }
    if (config.inputElementBits > 0)
    {
        hardware.inputElementBits = config.inputElementBits;
    }
    if (config.communicationElementBits > 0)
    {
        hardware.communicationElementBits = config.communicationElementBits;
    }
    if (config.cubeFlopsPerUsOverride > 0.0)
    {
        hardware.cubeFlopsPerUs = config.cubeFlopsPerUsOverride;
    }
    if (config.aicCoreNum > 0)
    {
        // The fused kernels launch one logical AIV group per AIC.
        hardware.coreNum = config.aicCoreNum;
    }
    return CostModelStatus::SUCCESS;
}

CostModelStatus GetCostModelHardwareConfig(CostModelHardwareType hardwareType, CocDataType dataType,
                                           CostModelHardwareConfig &hardware)
{
    switch (hardwareType)
    {
        case CostModelHardwareType::A2:
            return GetA2CostModelHardwareConfig(dataType, hardware);
        case CostModelHardwareType::A3:
            return GetA3CostModelHardwareConfig(dataType, hardware);
        case CostModelHardwareType::A5:
            return GetA5CostModelHardwareConfig(dataType, hardware);
        default:
            return CostModelStatus::UNSUPPORTED;
    }
}
