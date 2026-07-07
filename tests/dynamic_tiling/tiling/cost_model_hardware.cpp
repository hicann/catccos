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

namespace {

uint32_t GetElementSize(CocDataType dataType)
{
    switch (dataType) {
        case FP16:
        case BF16:
            return 2;
        case INT8:
            return 1;
        default:
            return 0;
    }
}

} // namespace

CostModelStatus GetA2CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware)
{
    uint32_t elementSize = GetElementSize(dataType);
    if (elementSize == 0) {
        return CostModelStatus::INVALID_ARGUMENT;
    }

    hardware = CostModelHardwareConfig{};
    hardware.elementSize = elementSize;
    return CostModelStatus::SUCCESS;
}

CostModelStatus GetA3CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware)
{
    (void)dataType;
    (void)hardware;
    return CostModelStatus::UNSUPPORTED;
}

CostModelStatus GetA5CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware)
{
    (void)dataType;
    (void)hardware;
    return CostModelStatus::UNSUPPORTED;
}

CostModelStatus GetCostModelHardwareConfig(
    CostModelHardwareType hardwareType, CocDataType dataType,
    CostModelHardwareConfig &hardware)
{
    switch (hardwareType) {
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
