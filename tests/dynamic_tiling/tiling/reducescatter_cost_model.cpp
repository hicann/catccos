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

// Each communication round has one barrier before and one after remote copy.
constexpr uint32_t SYNC_COUNT_PER_COMM = 2;

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor) { return (dividend + divisor - 1) / divisor; }

bool IsValidConfig(COCMatMulInfo const &info, uint32_t rankSize, CostModelConfig const &config,
                   CostModelHardwareConfig const &hardware)
{
    if (info.m <= 0 || info.k <= 0 || info.n <= 0 || rankSize == 0 ||
        rankSize > std::numeric_limits<uint32_t>::max() / 2)
    {
        return false;
    }
    if (info.m % rankSize != 0)
    {
        return false;
    }
    if (info.m > std::numeric_limits<uint32_t>::max() || info.k > std::numeric_limits<uint32_t>::max() ||
        info.n > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    if (config.commIntervalList.empty() || config.m0List.empty())
    {
        return false;
    }
    if (config.hardwareType == CostModelHardwareType::A5)
    {
        if (config.commTileList.empty() || config.splitCandidates.empty())
        {
            return false;
        }
    }
    else if (config.aivCoreList.empty())
    {
        return false;
    }
    uint64_t requestBits = static_cast<uint64_t>(hardware.remoteReadRequestBytes) * 8;
    return hardware.coreNum > 0 && config.k0 > 0 && hardware.inputElementBits > 0 &&
           hardware.communicationElementBits > 0 && requestBits >= hardware.communicationElementBits &&
           requestBits % hardware.communicationElementBits == 0 && hardware.hccsBandwidth > 0.0 &&
           hardware.cubeFlopsPerUs > 0.0 && hardware.readOtsdBound > 0 && hardware.nd2nzCmdOtsd > 0 &&
           hardware.fullCoreHitEfficiency > 0.0;
}

double EstimateReduceScatterAivWindowTime(uint64_t blockCountInRank, uint32_t n, uint32_t m0, uint32_t n0,
                                          uint32_t commTileM, uint32_t commBlockM, uint32_t aivCoreNum,
                                          uint32_t rankSize, CostModelHardwareConfig const &hardware)
{
    uint32_t tileWidth = std::min(n, n0);
    RemoteCopyWindow window{blockCountInRank * m0,
                            tileWidth,
                            commBlockM,
                            n0,
                            std::max<uint32_t>(1, commTileM / 2),
                            n0,
                            std::min(hardware.coreNum, aivCoreNum),
                            rankSize,
                            true};
    return EstimateRemoteCopyWindowTime(window, hardware);
}

}  // namespace

CostModelResult SelectReduceScatterTiling(COCMatMulInfo const &info, uint32_t rankSize, CostModelConfig const &config)
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
    uint32_t k0 = config.k0;
    MTECostModel mteModel{hardware};

    for (uint32_t m0 : config.m0List)
    {
        if (m0 != 128 && m0 != 256)
        {
            continue;
        }
        uint32_t n0 = m0 == 128 ? 256 : 128;
        uint64_t mInRank = m / rankSize;
        uint64_t mLoopsInRank = CeilDiv(mInRank, m0);
        uint64_t kLoops = CeilDiv(k, k0);
        uint64_t nLoops = CeilDiv(n, n0);
        uint64_t totalRankBlocks = mLoopsInRank * nLoops * rankSize;

        uint32_t leftMteN = config.useMxMteShape ? m0 : 128;
        uint32_t rightMteN = config.useMxMteShape ? n0 : 256;
        double leftHitTime = mteModel.Nd2NzContinuous(hardware.coreNum, kLoops, leftMteN, k0, CacheStatus::HIT);
        double rightHitTime = mteModel.Nd2NzContinuous(hardware.coreNum, kLoops, rightMteN, k0, CacheStatus::HIT);
        double nd2nzHitTimePerCore = leftHitTime + rightHitTime;
        double cubeTimePerCore = (2.0 * m0 * k * n0) / hardware.cubeFlopsPerUs;

        for (uint32_t p : config.commIntervalList)
        {
            if (p == 0 || (static_cast<uint64_t>(hardware.coreNum) * p) % rankSize != 0)
            {
                continue;
            }

            uint64_t blocksPerComm = static_cast<uint64_t>(hardware.coreNum) * p;
            uint64_t commCount = CeilDiv(totalRankBlocks, blocksPerComm);

            std::vector<CostModelTiling> candidates;
            if (config.hardwareType == CostModelHardwareType::A5)
            {
                for (uint32_t commTileM : config.commTileList)
                {
                    if (commTileM == 0 || commTileM % WORKSPACE_STAGES != 0)
                    {
                        continue;
                    }
                    for (auto const &split : config.splitCandidates)
                    {
                        uint64_t activeAivCoreNum = static_cast<uint64_t>(split.commNpuSplit) * split.commDataSplit;
                        if (split.commNpuSplit == 0 || split.commDataSplit == 0 ||
                            activeAivCoreNum > std::numeric_limits<uint32_t>::max())
                        {
                            continue;
                        }
                        candidates.push_back(CostModelTiling{m0, k0, n0, commTileM, p, split.commNpuSplit,
                                                             split.commDataSplit, commTileM});
                    }
                }
            }
            else
            {
                // Keep the existing A2/A3 candidate space unchanged.
                uint32_t blockM = 2 * rankSize;
                for (uint32_t aivCoreNum : config.aivCoreList)
                {
                    if (aivCoreNum != 16 && aivCoreNum != 20)
                    {
                        continue;
                    }
                    candidates.push_back(
                        CostModelTiling{m0, k0, n0, blockM, p, 1, aivCoreNum == 16 ? 16U : 20U, blockM});
                }
            }

            for (auto const &candidate : candidates)
            {
                if (!config.IsCandidateValid(candidate))
                {
                    continue;
                }
                uint32_t activeAivCoreNum = candidate.commNpuSplit * candidate.commDataSplit;

                std::vector<double> aicTimes;
                std::vector<double> aivTimes;
                aicTimes.reserve(commCount);
                aivTimes.reserve(commCount);
                for (uint64_t i = 0; i < commCount; ++i)
                {
                    uint64_t actualBlocks = std::min(blocksPerComm, totalRankBlocks - i * blocksPerComm);
                    uint64_t aicRounds = CeilDiv(actualBlocks, hardware.coreNum);
                    aicTimes.push_back(std::max(cubeTimePerCore * aicRounds, nd2nzHitTimePerCore * aicRounds));

                    uint64_t blockCountInRank = CeilDiv(actualBlocks, rankSize);
                    double aivTime =
                        EstimateReduceScatterAivWindowTime(blockCountInRank, n, m0, n0, candidate.commTileM,
                                                           candidate.commBlockM, activeAivCoreNum, rankSize, hardware);
                    aivTimes.push_back(aivTime + hardware.syncTimeUs * SYNC_COUNT_PER_COMM);
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
