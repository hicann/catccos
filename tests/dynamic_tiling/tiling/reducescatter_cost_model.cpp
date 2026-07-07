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
#include "mte_cost_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

struct AivWorkload {
    double tileNum;
    double dataSizeKb;
};

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}

bool IsValidConfig(
    COCMatMulInfo const &info, uint32_t rankSize,
    CostModelConfig const &config,
    CostModelHardwareConfig const &hardware)
{
    if (info.m <= 0 || info.k <= 0 || info.n <= 0 ||
        rankSize == 0 || rankSize > std::numeric_limits<uint32_t>::max() / 2) {
        return false;
    }
    if (info.m > std::numeric_limits<uint32_t>::max() ||
        info.k > std::numeric_limits<uint32_t>::max() ||
        info.n > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (config.commIntervalList.empty() ||
        config.m0List.empty() || config.aivCoreList.empty()) {
        return false;
    }
    return hardware.coreNum > 0 &&
        hardware.elementSize > 0 &&
        hardware.hccsBandwidth > 0.0 &&
        hardware.cubeFlopsPerUs > 0.0 &&
        hardware.readOtsdBound > 0 &&
        hardware.nd2nzCmdOtsd > 0 &&
        hardware.fullCoreHitEfficiency > 0.0;
}

AivWorkload GetAivWorkload(
    uint64_t tileCount, uint32_t n, uint32_t m0, uint32_t n0,
    uint32_t blockM, uint32_t rankSize,
    CostModelHardwareConfig const &hardware)
{
    uint32_t tileWidth = std::min(n, n0);
    double elementCount = static_cast<double>(tileCount) * m0 * tileWidth;
    double dataSizeKb = (
        elementCount * hardware.elementSize
    ) / 1024.0 / rankSize;
    double tileNum = (
        elementCount * hardware.elementSize
    ) / (static_cast<double>(n0) * blockM) / rankSize;
    return {tileNum, dataSizeKb};
}

double EstimateAivTime(
    AivWorkload const &workload, uint32_t t, uint32_t aivCoreNum,
    CostModelHardwareConfig const &hardware)
{
    uint32_t burstlenOtsd = t / 2;
    double writeTime = 4.0 + 2.0 * burstlenOtsd + hardware.writeRttNs;
    double overlapTime = (
        burstlenOtsd * hardware.remoteReadScheduleNs
    ) - writeTime;
    double pingPongTime = (
        hardware.remoteReadRttNs
        + 2.0 * burstlenOtsd * hardware.remoteReadScheduleNs
        + writeTime
    );

    double roundedCount = std::round(
        workload.tileNum / aivCoreNum * 100000.0
    ) / 100000.0;
    uint64_t pingPongCount = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(roundedCount)));
    double timeUs = (
        25.0
        + pingPongTime * pingPongCount
        - overlapTime * (pingPongCount - 1)
    ) / 1000.0;

    double bandwidth = workload.dataSizeKb / timeUs;
    if (bandwidth > hardware.hccsBandwidth) {
        timeUs = workload.dataSizeKb / hardware.hccsBandwidth;
    }
    return timeUs;
}

} // namespace

CostModelResult SelectReduceScatterTiling(
    COCMatMulInfo const &info, uint32_t rankSize,
    CostModelConfig const &config)
{
    CostModelResult best;
    CostModelHardwareConfig hardware;
    auto hardwareStatus = GetCostModelHardwareConfig(
        config.hardwareType, config.dataType, hardware);
    if (hardwareStatus != CostModelStatus::SUCCESS) {
        best.status = hardwareStatus;
        return best;
    }
    if (!IsValidConfig(info, rankSize, config, hardware)) {
        best.status = CostModelStatus::INVALID_ARGUMENT;
        return best;
    }

    uint32_t m = static_cast<uint32_t>(info.m);
    uint32_t k = static_cast<uint32_t>(info.k);
    uint32_t n = static_cast<uint32_t>(info.n);
    constexpr uint32_t K0 = 256;
    MTECostModel mteModel{hardware};

    for (uint32_t m0 : config.m0List) {
        if (m0 != 128 && m0 != 256) {
            continue;
        }
        uint32_t n0 = m0 == 128 ? 256 : 128;
        uint64_t mLoops = CeilDiv(m, m0);
        uint64_t kLoops = CeilDiv(k, K0);
        uint64_t nLoops = CeilDiv(n, n0);
        uint64_t totalTiles = mLoops * nLoops;
        uint64_t pMax = CeilDiv(totalTiles, hardware.coreNum);

        double leftHitTime = mteModel.Nd2NzContinuous(
            hardware.coreNum, kLoops, 128, 256, CacheStatus::HIT);
        double rightHitTime = mteModel.Nd2NzContinuous(
            hardware.coreNum, kLoops, 256, 256, CacheStatus::HIT);
        double nd2nzHitTimePerCore = leftHitTime + rightHitTime;
        double cubeTimePerCore = (
            2.0 * m0 * k * n0
        ) / hardware.cubeFlopsPerUs;

        for (uint32_t p : config.commIntervalList) {
            if (p == 0 ||
                (static_cast<uint64_t>(hardware.coreNum) * p) % rankSize != 0) {
                continue;
            }

            uint64_t activeRounds = std::min<uint64_t>(p, pMax);
            double aicCubeTime = cubeTimePerCore * activeRounds;
            double aicNd2nzHitTime = nd2nzHitTimePerCore * activeRounds;
            double aicTime = std::max(aicCubeTime, aicNd2nzHitTime);

            uint64_t tilesPerComm = static_cast<uint64_t>(hardware.coreNum) * p;
            uint64_t commCount = CeilDiv(totalTiles, tilesPerComm);
            uint64_t fullRoundTiles = std::min(totalTiles, tilesPerComm);
            uint64_t remainderTiles = totalTiles % tilesPerComm;
            uint64_t remainderLoops = CeilDiv(remainderTiles, hardware.coreNum);

            double lastAicTime = aicTime;
            if (remainderLoops > 0) {
                lastAicTime = std::max(
                    cubeTimePerCore * remainderLoops,
                    nd2nzHitTimePerCore * remainderLoops);
            }

            // ReduceScatter uses rankSize as the model tile value.
            // The runtime tiling stores twice this value in commTileM.
            uint32_t t = rankSize;
            uint32_t blockM = 2 * t;
            auto fullWorkload = GetAivWorkload(
                fullRoundTiles, n, m0, n0, blockM, rankSize, hardware);

            for (uint32_t aivCoreNum : config.aivCoreList) {
                if (aivCoreNum != 16 && aivCoreNum != 20) {
                    continue;
                }

                CostModelTiling candidate{
                    m0, K0, n0, blockM, p, 1,
                    aivCoreNum == 16 ? 16U : 20U, blockM};
                if (!config.IsCandidateValid(candidate)) {
                    continue;
                }

                double aivTime = EstimateAivTime(
                    fullWorkload, t, aivCoreNum, hardware);
                double tailTime = aivTime;
                if (remainderTiles > 0) {
                    auto tailWorkload = GetAivWorkload(
                        remainderTiles, n, m0, n0,
                        blockM, rankSize, hardware);
                    tailTime = EstimateAivTime(
                        tailWorkload, t, aivCoreNum, hardware);
                }

                double pipelineTime;
                if (aivTime >= aicTime) {
                    pipelineTime = (
                        aicTime
                        + aivTime * (commCount - 1)
                        + tailTime
                    );
                } else {
                    pipelineTime = (
                        aicTime * (commCount - 1)
                        + std::max(lastAicTime, aivTime)
                        + tailTime
                    );
                }
                double totalTime = (
                    pipelineTime
                    + hardware.syncTimeUs * commCount
                    + hardware.launchTimeUs
                );

                if (!std::isfinite(totalTime) || totalTime >= best.cost) {
                    continue;
                }

                best.cost = totalTime;
                best.status = CostModelStatus::SUCCESS;
                best.tiling = candidate;
            }
        }
    }

    if (!best.IsSuccess()) {
        best.status = CostModelStatus::NO_VALID_CANDIDATE;
    }
    return best;
}
