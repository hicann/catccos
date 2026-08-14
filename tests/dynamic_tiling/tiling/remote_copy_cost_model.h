/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REMOTE_COPY_COST_MODEL_H
#define REMOTE_COPY_COST_MODEL_H

#include <cstdint>
#include <vector>

#include "cost_model.h"

struct RemoteCopyWindow
{
    uint64_t rows = 0;
    uint64_t columns = 0;
    uint64_t blockRows = 0;
    uint64_t blockColumns = 0;
    uint64_t tileRows = 0;
    uint64_t tileColumns = 0;
    uint32_t activeCoreNum = 0;
    // Scheduler rank-axis count, including a local-rank task that may skip
    // remote copy after core assignment.
    uint32_t taskRankCount = 0;
    // Keep UB ping-pong state across consecutive communication blocks.
    bool continuousBlockStream = false;
    // 0 uses CostModelHardwareConfig::communicationElementBits.
    uint32_t elementBits = 0;
    // 0 derives the outstanding request count from tile bytes.
    uint32_t singleCoreOstd = 0;
};

double EstimateRemoteCopyWindowTime(RemoteCopyWindow const &window, CostModelHardwareConfig const &hardware);

double ApplyCommBlockM64Penalty(double cost, uint32_t commBlockM);

double SimulateDoubleBufferPipeline(std::vector<double> const &producerTimes, std::vector<double> const &consumerTimes,
                                    uint32_t stageCount);

#endif  // REMOTE_COPY_COST_MODEL_H
