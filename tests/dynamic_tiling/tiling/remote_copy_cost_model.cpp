/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "remote_copy_cost_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

constexpr double BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0;
constexpr double BITS_PER_BYTE = 8.0;
constexpr double NS_PER_SECOND = 1000.0 * 1000.0 * 1000.0;
constexpr double US_PER_SECOND = 1000.0 * 1000.0;
// In the transition region, two ping-pong stages exceed the effective OSTD
// limit while one stage does not. The empirical weight is the exposed part of
// that additional scheduling pressure.
constexpr double REMOTE_READ_TRANSITION_EXTRA_SCHEDULE_WEIGHT = 0.75;
constexpr uint32_t PENALTY_COMM_BLOCK_M_64 = 64;
constexpr double COMM_BLOCK_M_64_PENALTY_RATIO = 1.50;

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    return dividend / divisor + static_cast<uint64_t>(dividend % divisor != 0);
}

}  // namespace

double EstimateRemoteCopyWindowTime(RemoteCopyWindow const &window, CostModelHardwareConfig const &hardware)
{
    uint32_t elementBits = window.elementBits > 0 ? window.elementBits : hardware.communicationElementBits;
    if (window.rows == 0 || window.columns == 0 || window.taskRankCount == 0)
    {
        return 0.0;
    }
    if (window.blockRows == 0 || window.blockColumns == 0 || window.tileRows == 0 || window.tileColumns == 0 ||
        window.activeCoreNum == 0 || elementBits == 0 || hardware.remoteReadRequestBytes == 0 ||
        hardware.hccsBandwidth <= 0.0)
    {
        return std::numeric_limits<double>::infinity();
    }

    uint64_t blockRows = CeilDiv(window.rows, window.blockRows);
    uint64_t blockColumns = CeilDiv(window.columns, window.blockColumns);
    uint64_t blocksPerRank = blockRows * blockColumns;
    uint32_t peerCoreNum = window.activeCoreNum / window.taskRankCount;
    uint64_t blockRounds;
    if (peerCoreNum > 0)
    {
        blockRounds = CeilDiv(blocksPerRank, peerCoreNum);
    }
    else
    {
        blockRounds = CeilDiv(blocksPerRank * window.taskRankCount, window.activeCoreNum);
        peerCoreNum = 1;
    }

    uint64_t rowTilesPerBlock = CeilDiv(window.blockRows, window.tileRows);
    uint64_t columnTilesPerBlock = CeilDiv(window.blockColumns, window.tileColumns);
    uint64_t pingPongRounds = CeilDiv(rowTilesPerBlock, UB_STAGES) * columnTilesPerBlock;
    double scheduleRounds = static_cast<double>(pingPongRounds);
    double scheduleSequenceCount = static_cast<double>(blockRounds);
    if (window.continuousBlockStream)
    {
        scheduleRounds *= blockRounds;
        scheduleSequenceCount = 1.0;
    }

    double bytes = static_cast<double>(window.rows) * window.columns * elementBits / BITS_PER_BYTE;
    double bandwidthBytesPerUs = hardware.hccsBandwidth * BYTES_PER_GB / US_PER_SECOND;
    double bandwidthTimeUs = bytes / bandwidthBytesPerUs;

    uint64_t requestBits = static_cast<uint64_t>(hardware.remoteReadRequestBytes) * BITS_PER_BYTE;
    uint64_t tileBits = window.tileRows * window.tileColumns * elementBits;
    double singleCoreOstd =
        window.singleCoreOstd > 0 ? window.singleCoreOstd : static_cast<double>(CeilDiv(tileBits, requestBits));
    double peerOstd = singleCoreOstd * peerCoreNum;
    double effectiveOstd = (hardware.hccsBandwidth * BYTES_PER_GB / NS_PER_SECOND * hardware.remoteReadRttNs /
                            hardware.remoteReadRequestBytes);
    effectiveOstd = std::max(1.0, effectiveOstd);

    double writeTimeNs = 4.0 + 2.0 * singleCoreOstd + hardware.writeRttNs;
    double scheduleNs = hardware.remoteReadScheduleNs;
    double rttNs = hardware.remoteReadRttNs;
    double blockTimeNs;
    if (2.0 * peerOstd <= effectiveOstd)
    {
        blockTimeNs =
            ((rttNs + singleCoreOstd * scheduleNs + writeTimeNs) * scheduleRounds + singleCoreOstd * scheduleNs);
    }
    else if (peerOstd < effectiveOstd)
    {
        double extraNs = (2.0 * peerOstd - effectiveOstd) * scheduleNs;
        blockTimeNs = ((rttNs + singleCoreOstd * scheduleNs + writeTimeNs +
                        REMOTE_READ_TRANSITION_EXTRA_SCHEDULE_WEIGHT * extraNs) *
                           scheduleRounds +
                       rttNs - (effectiveOstd / peerCoreNum - singleCoreOstd) * scheduleNs);
    }
    else
    {
        double requestCount = 2.0 * peerOstd * scheduleRounds;
        double waveCount = std::ceil(requestCount / effectiveOstd);
        double tailOstd = requestCount - (waveCount - 1.0) * effectiveOstd;
        double tailSingleCoreOstd = std::ceil(tailOstd / peerCoreNum);
        double tailWriteNs = 4.0 + 2.0 * tailSingleCoreOstd + hardware.writeRttNs;
        blockTimeNs = (waveCount * rttNs + tailSingleCoreOstd * scheduleNs + tailWriteNs);
    }

    double scheduleTimeUs = blockTimeNs * scheduleSequenceCount / 1000.0;
    return std::max(scheduleTimeUs, bandwidthTimeUs);
}

double ApplyCommBlockM64Penalty(double cost, uint32_t commBlockM)
{
    return commBlockM == PENALTY_COMM_BLOCK_M_64 ? cost * COMM_BLOCK_M_64_PENALTY_RATIO : cost;
}

double SimulateDoubleBufferPipeline(std::vector<double> const &producerTimes, std::vector<double> const &consumerTimes,
                                    uint32_t stageCount)
{
    if (stageCount == 0 || producerTimes.size() != consumerTimes.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    std::vector<double> consumerFinish(consumerTimes.size(), 0.0);
    double producerDone = 0.0;
    double consumerDone = 0.0;
    for (size_t i = 0; i < producerTimes.size(); ++i)
    {
        double producerStart = producerDone;
        if (i >= stageCount)
        {
            producerStart = std::max(producerStart, consumerFinish[i - stageCount]);
        }
        double producerFinish = producerStart + producerTimes[i];
        producerDone = producerFinish;

        double consumerStart = std::max(consumerDone, producerFinish);
        consumerFinish[i] = consumerStart + consumerTimes[i];
        consumerDone = consumerFinish[i];
    }
    return consumerDone;
}
