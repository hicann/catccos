/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MTE_COST_MODEL_H
#define MTE_COST_MODEL_H

#include <cstdint>

#include "cost_model.h"

enum class CacheStatus {
    HIT,
    MISS,
};

class MTECostModel {
public:
    explicit MTECostModel(CostModelHardwareConfig const &config);

    double Nd2NzContinuous(
        uint32_t coreNum, uint32_t instructionNum, uint32_t nValue,
        uint32_t dValue = 256,
        CacheStatus cacheStatus = CacheStatus::MISS) const;

private:
    MTECacheConfig const &GetCacheConfig(CacheStatus status) const;

    CostModelHardwareConfig const &config_;
};

#endif // MTE_COST_MODEL_H
