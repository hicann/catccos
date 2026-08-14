/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cost_model.h"
#include "mte_cost_model.h"
#include "remote_copy_cost_model.h"

namespace
{

// Each round synchronizes before reduce-scatter, between phases, and after all-gather.
constexpr uint32_t SYNC_COUNT_PER_COMM = 3;

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor) { return (dividend + divisor - 1) / divisor; }

bool IsValidConfig(COCMatMulInfo const &info, uint32_t rankSize, CostModelConfig const &config,
                   CostModelHardwareConfig const &hardware)
{
    if (info.m <= 0 || info.k <= 0 || info.n <= 0 || rankSize == 0 ||
        rankSize > std::numeric_limits<uint32_t>::max() / 2)
    {
        return false;
    }
    if (info.m > std::numeric_limits<uint32_t>::max() || info.k > std::numeric_limits<uint32_t>::max() ||
        info.n > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    if (config.commIntervalList.empty() || config.m0List.empty() || config.aivCoreList.empty())
    {
        return false;
    }
    uint64_t requestBits = static_cast<uint64_t>(hardware.remoteReadRequestBytes) * 8;
    return hardware.coreNum > 0 && hardware.inputElementBits > 0 && hardware.communicationElementBits > 0 &&
           requestBits >= hardware.communicationElementBits && requestBits % hardware.communicationElementBits == 0 &&
           hardware.hccsBandwidth > 0.0 && hardware.cubeFlopsPerUs > 0.0 && hardware.readOtsdBound > 0 &&
           hardware.nd2nzCmdOtsd > 0 && hardware.fullCoreHitEfficiency > 0.0;
}

double EstimateAllReducePhaseTime(uint64_t blockCount, uint32_t n, uint32_t m0, uint32_t n0, uint32_t commTileM,
                                  uint32_t commBlockM, uint32_t aivCoreNum, uint32_t rankSize,
                                  CostModelHardwareConfig const &hardware)
{
    uint32_t tileWidth = std::min(n, n0);
    uint64_t commBlockCount = CeilDiv(blockCount * m0, commBlockM);
    uint64_t blockCountInRank = CeilDiv(commBlockCount, rankSize);
    RemoteCopyWindow window{
        blockCountInRank * commBlockM,          tileWidth, commBlockM, n0, std::max<uint32_t>(1, commTileM / 2), n0,
        std::min(hardware.coreNum, aivCoreNum), rankSize,  true};
    return EstimateRemoteCopyWindowTime(window, hardware);
}

}  // namespace

CostModelResult SelectAllReduceTiling(COCMatMulInfo const &info, uint32_t rankSize, CostModelConfig const &config)
{
    CostModelResult best;
    CostModelHardwareConfig hardware;
    auto hardwareStatus = GetCostModelHardwareConfig(config, hardware);
    if (hardwareStatus != CostModelStatus::SUCCESS)
    {
        best.status = hardwareStatus;
        return best;
    }
    if (!IsValidConfig(info, rankSize, config, hardware))
    {
        best.status = CostModelStatus::INVALID_ARGUMENT;
        return best;
    }

    uint32_t m = static_cast<uint32_t>(info.m);
    uint32_t k = static_cast<uint32_t>(info.k);
    uint32_t n = static_cast<uint32_t>(info.n);
    constexpr uint32_t K0 = 256;
    MTECostModel mteModel{hardware};

    for (uint32_t m0 : config.m0List)
    {
        if (m0 != 128 && m0 != 256)
        {
            continue;
        }
        uint32_t n0 = m0 == 128 ? 256 : 128;
        uint64_t mLoops = CeilDiv(m, m0);
        uint64_t kLoops = CeilDiv(k, K0);
        uint64_t nLoops = CeilDiv(n, n0);
        uint64_t totalTiles = mLoops * nLoops;

        double leftHitTime = mteModel.Nd2NzContinuous(hardware.coreNum, kLoops, 128, 256, CacheStatus::HIT);
        double rightHitTime = mteModel.Nd2NzContinuous(hardware.coreNum, kLoops, 256, 256, CacheStatus::HIT);
        double nd2nzHitTimePerCore = leftHitTime + rightHitTime;
        double cubeTimePerCore = (2.0 * m0 * k * n0) / hardware.cubeFlopsPerUs;

        for (uint32_t p : config.commIntervalList)
        {
            if (p == 0 || (static_cast<uint64_t>(hardware.coreNum) * p) % rankSize != 0)
            {
                continue;
            }

            // ReduceScatter uses rankSize as the model tile value.
            // The runtime tiling stores twice this value in commTileM.
            uint32_t t = rankSize;
            uint32_t blockM = 2 * t;
            uint64_t blocksPerComm = static_cast<uint64_t>(hardware.coreNum) * p;
            uint64_t commCount = CeilDiv(totalTiles, blocksPerComm);

            for (uint32_t aivCoreNum : config.aivCoreList)
            {
                if (aivCoreNum != 16 && aivCoreNum != 20)
                {
                    continue;
                }

                CostModelTiling candidate{m0, K0, n0, blockM, p, 1, aivCoreNum == 16 ? 16U : 20U, blockM};
                if (!config.IsCandidateValid(candidate))
                {
                    continue;
                }

                std::vector<double> aicTimes;
                std::vector<double> aivTimes;
                aicTimes.reserve(commCount);
                aivTimes.reserve(commCount);
                for (uint64_t i = 0; i < commCount; ++i)
                {
                    uint64_t actualBlocks = std::min(blocksPerComm, totalTiles - i * blocksPerComm);
                    uint64_t aicRounds = CeilDiv(actualBlocks, hardware.coreNum);
                    aicTimes.push_back(std::max(cubeTimePerCore * aicRounds, nd2nzHitTimePerCore * aicRounds));

                    // Both phases schedule the full rank axis. ReduceScatter
                    // skips local copy only after cores have been assigned.
                    double reduceScatterTime = EstimateAllReducePhaseTime(actualBlocks, n, m0, n0, blockM, blockM,
                                                                          aivCoreNum, rankSize, hardware);
                    double allGatherTime = EstimateAllReducePhaseTime(actualBlocks, n, m0, n0, blockM, blockM,
                                                                      aivCoreNum, rankSize, hardware);
                    aivTimes.push_back(reduceScatterTime + allGatherTime + hardware.syncTimeUs * SYNC_COUNT_PER_COMM);
                }

                double pipelineTime = SimulateDoubleBufferPipeline(aicTimes, aivTimes, WORKSPACE_STAGES);
                double totalTime = (pipelineTime + hardware.launchTimeUs);
                totalTime = ApplyCommBlockM64Penalty(totalTime, candidate.commBlockM);

                if (!std::isfinite(totalTime) || totalTime >= best.cost)
                {
                    continue;
                }

                best.cost = totalTime;
                best.status = CostModelStatus::SUCCESS;
                best.tiling = candidate;
            }
        }
    }

    if (!best.IsSuccess())
    {
        best.status = CostModelStatus::NO_VALID_CANDIDATE;
    }
    return best;
}
