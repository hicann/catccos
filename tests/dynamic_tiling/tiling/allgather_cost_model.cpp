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

constexpr uint32_t SHMEM_BARRIER_COUNT = 2;
constexpr uint32_t CROSS_CORE_BARRIER_PER_LOOP = 2;
constexpr double AG_TIE_BREAK_RATIO = 0.02;
// TODO(COSTMODEL_CALIBRATE): refine with more fixed-tiling interval samples.
constexpr double ALLGATHER_EXPOSED_COMM_LOOP_OVERHEAD_US = 1.1;
constexpr uint32_t PENALTY_COMM_BLOCK_M_32 = 32;
constexpr double COMM_BLOCK_M_32_PENALTY_RATIO = 1.25;
constexpr double BYTES_PER_MIB = 1024.0 * 1024.0;
constexpr uint32_t BITS_PER_BYTE = 8;
constexpr double A_STAGE_FAST_LIMIT_MIB = 20.0;
constexpr double A_STAGE_SLOW_LIMIT_MIB = 24.0;
constexpr double A_STAGE_FAST_US_PER_MIB = 0.65;
constexpr double A_STAGE_SLOW_US_PER_MIB = 2.48;
constexpr double A_STAGE_SLOW_US_OFFSET = -10.5;
constexpr uint32_t A_STAGE_BASELINE_AIC_CORE_NUM = 20;

struct TilingCandidate
{
    uint32_t m0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
};

uint64_t CeilDiv(uint64_t dividend, uint64_t divisor) { return (dividend + divisor - 1) / divisor; }

uint64_t RoundUp(uint64_t value, uint64_t align) { return CeilDiv(value, align) * align; }

uint64_t ElementsToBytes(uint64_t elementCount, uint32_t elementBits)
{
    return CeilDiv(elementCount * elementBits, BITS_PER_BYTE);
}

uint32_t EstimateSingleCoreOstd(uint32_t tileRows, uint32_t tileColumns, uint32_t elementBits, uint32_t requestBytes)
{
    uint64_t tileBits = static_cast<uint64_t>(tileRows) * tileColumns * elementBits;
    return static_cast<uint32_t>(CeilDiv(tileBits, static_cast<uint64_t>(requestBytes) * BITS_PER_BYTE));
}

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
    if (config.commIntervalList.empty() || config.commTileList.empty() || config.m0List.empty() ||
        config.splitCandidates.empty())
    {
        return false;
    }
    uint64_t requestBits = static_cast<uint64_t>(hardware.remoteReadRequestBytes) * BITS_PER_BYTE;
    bool scaleConfigValid = (config.mxScaleGroupSize == 0 && config.mxScaleElementBits == 0) ||
                            (config.mxScaleGroupSize > 0 && config.mxScaleElementBits > 0);
    return hardware.coreNum > 0 && config.k0 > 0 && hardware.inputElementBits > 0 &&
           hardware.communicationElementBits > 0 && requestBits >= hardware.communicationElementBits &&
           requestBits % hardware.communicationElementBits == 0 && hardware.hccsBandwidth > 0.0 &&
           hardware.cubeFlopsPerUs > 0.0 && hardware.crossCoreBarrierTimeUs >= 0.0 && hardware.readOtsdBound > 0 &&
           hardware.nd2nzCmdOtsd > 0 && hardware.fullCoreHitEfficiency > 0.0 && scaleConfigValid;
}

bool IsWorkspaceValid(uint32_t m, uint32_t k, uint32_t rankSize, uint32_t m0, uint32_t commInterval,
                      CostModelConfig const &config, CostModelHardwareConfig const &hardware)
{
    uint64_t alignK =
        (static_cast<uint64_t>(hardware.remoteReadRequestBytes) * BITS_PER_BYTE / hardware.communicationElementBits);
    uint64_t alignedK = RoundUp(k, alignK);
    uint64_t commSizeM = static_cast<uint64_t>(commInterval) * m0;
    uint64_t aStageSize =
        config.includeAStage ? ElementsToBytes(static_cast<uint64_t>(m) * alignedK, hardware.inputElementBits) : 0;
    uint64_t gatheredSize = ElementsToBytes(static_cast<uint64_t>(WORKSPACE_STAGES) * rankSize * commSizeM * alignedK,
                                            hardware.communicationElementBits);
    uint64_t gatheredScaleSize = 0;
    if (config.mxScaleGroupSize > 0)
    {
        uint64_t scaleK = CeilDiv(k, config.mxScaleGroupSize);
        gatheredScaleSize = ElementsToBytes(static_cast<uint64_t>(WORKSPACE_STAGES) * rankSize * commSizeM * scaleK,
                                            config.mxScaleElementBits);
    }
    return aStageSize + gatheredSize + gatheredScaleSize <= static_cast<uint64_t>(SHMEM_BUFF_BYTES);
}

double EstimateAStageTime(uint32_t m, uint32_t k, uint32_t aicCoreNum, CostModelHardwareConfig const &hardware)
{
    // A staging is an E2E-only term and does not participate in tiling ranking.
    // The measured piecewise model uses 20 AIC cores as the baseline. Scale
    // the MiB boundaries by runtimeAicCoreNum / 20, and scale the resulting
    // time by 20 / runtimeAicCoreNum.
    double aMiB = (static_cast<double>(m) * k * hardware.inputElementBits / BITS_PER_BYTE / BYTES_PER_MIB);
    double coreScale = static_cast<double>(aicCoreNum) / A_STAGE_BASELINE_AIC_CORE_NUM;
    double fastLimitMiB = A_STAGE_FAST_LIMIT_MIB * coreScale;
    double slowLimitMiB = A_STAGE_SLOW_LIMIT_MIB * coreScale;
    double stageTimeUs;
    if (aMiB <= fastLimitMiB)
    {
        stageTimeUs = A_STAGE_FAST_US_PER_MIB * aMiB;
    }
    else if (aMiB >= slowLimitMiB)
    {
        stageTimeUs = A_STAGE_SLOW_US_PER_MIB * aMiB + A_STAGE_SLOW_US_OFFSET;
    }
    else
    {
        double fastTimeUs = A_STAGE_FAST_US_PER_MIB * fastLimitMiB;
        double slowTimeUs = (A_STAGE_SLOW_US_PER_MIB * slowLimitMiB + A_STAGE_SLOW_US_OFFSET);
        double ratio = ((aMiB - fastLimitMiB) / (slowLimitMiB - fastLimitMiB));
        stageTimeUs = fastTimeUs + ratio * (slowTimeUs - fastTimeUs);
    }
    return stageTimeUs * A_STAGE_BASELINE_AIC_CORE_NUM / aicCoreNum;
}

double EstimateAicTileTime(uint32_t m0, uint32_t n0, uint32_t k, uint32_t k0, MTECostModel const &mteModel,
                           CostModelHardwareConfig const &hardware)
{
    uint64_t kLoops = CeilDiv(k, k0);
    double cubeTime = (2.0 * m0 * static_cast<double>(k) * n0) / hardware.cubeFlopsPerUs;
    double leftHitTime =
        mteModel.Nd2NzContinuous(hardware.coreNum, static_cast<uint32_t>(kLoops), m0, k0, CacheStatus::HIT);
    double rightHitTime =
        mteModel.Nd2NzContinuous(hardware.coreNum, static_cast<uint32_t>(kLoops), n0, k0, CacheStatus::HIT);
    return std::max(cubeTime, leftHitTime + rightHitTime);
}

double EstimateAicWindowTime(uint32_t actualCommSizeM, uint32_t n, uint32_t m0, uint32_t n0, uint32_t rankSize,
                             double aicTileTime, uint32_t aicCoreNum)
{
    uint64_t remoteTiles = CeilDiv(actualCommSizeM, m0) * CeilDiv(n, n0) * static_cast<uint64_t>(rankSize);
    return CeilDiv(remoteTiles, aicCoreNum) * aicTileTime;
}

double EstimateAivWindowTime(uint32_t actualCommSizeM, uint32_t k, uint32_t commTileM, uint32_t commBlockM, uint32_t k0,
                             uint32_t commNpuSplit, uint32_t commDataSplit, uint32_t rankSize,
                             CostModelConfig const &config, CostModelHardwareConfig const &hardware)
{
    uint32_t tileRows = std::max<uint32_t>(1, commTileM / 2);
    uint32_t activeAivCore = std::min(hardware.coreNum, commNpuSplit * commDataSplit);
    RemoteCopyWindow window{actualCommSizeM, k, commBlockM, k, tileRows, k0, activeAivCore, rankSize};
    window.singleCoreOstd =
        EstimateSingleCoreOstd(tileRows, k0, hardware.communicationElementBits, hardware.remoteReadRequestBytes);
    double dataTime = EstimateRemoteCopyWindowTime(window, hardware);
    if (config.mxScaleGroupSize == 0)
    {
        return dataTime;
    }

    uint32_t scaleK = static_cast<uint32_t>(CeilDiv(k, config.mxScaleGroupSize));
    uint32_t scaleK0 = std::max<uint32_t>(1, static_cast<uint32_t>(CeilDiv(k0, config.mxScaleGroupSize)));
    RemoteCopyWindow scaleWindow{actualCommSizeM, scaleK,        commBlockM, scaleK, tileRows,
                                 scaleK0,         activeAivCore, rankSize,   false,  config.mxScaleElementBits};
    scaleWindow.singleCoreOstd =
        EstimateSingleCoreOstd(tileRows, scaleK0, config.mxScaleElementBits, hardware.remoteReadRequestBytes);
    double scaleTime = EstimateRemoteCopyWindowTime(scaleWindow, hardware);
    return std::max(dataTime, scaleTime);
}

double ApplyCommBlockPenalty(double cost, uint32_t commBlockM, CostModelHardwareType hardwareType, CocDataType dataType)
{
    cost = ApplyCommBlockM64Penalty(cost, commBlockM);
    if (hardwareType != CostModelHardwareType::A5 && dataType != CocDataType::INT8 &&
        commBlockM == PENALTY_COMM_BLOCK_M_32)
    {
        return cost * COMM_BLOCK_M_32_PENALTY_RATIO;
    }
    return cost;
}

int CommTilePriority(uint32_t commTileM, CostModelHardwareType hardwareType, CocDataType dataType)
{
    if (commTileM == 8 || commTileM == 16 ||
        ((hardwareType == CostModelHardwareType::A5 || dataType == CocDataType::INT8) && commTileM == 32))
    {
        return 0;
    }
    return 1;
}

int SplitNpuPriority(uint32_t commNpuSplit, uint32_t rankSize)
{
    if (rankSize == 2)
    {
        if (commNpuSplit == 4)
        {
            return 0;
        }
        if (commNpuSplit == 2)
        {
            return 1;
        }
        return 2;
    }
    if (rankSize == 4)
    {
        if (commNpuSplit == 4)
        {
            return 0;
        }
        if (commNpuSplit == 2)
        {
            return 1;
        }
        return 2;
    }
    if (rankSize == 8)
    {
        if (commNpuSplit == 1)
        {
            return 0;
        }
        if (commNpuSplit == 2)
        {
            return 1;
        }
        return 2;
    }
    return 0;
}

int SplitProductPriority(TilingCandidate const &candidate)
{
    return candidate.commNpuSplit * candidate.commDataSplit == 16 ? 0 : 1;
}

bool BetterByTieBreak(TilingCandidate const &lhs, TilingCandidate const &rhs, uint32_t rankSize,
                      CostModelHardwareType hardwareType, CocDataType dataType)
{
    int lhsCommTilePriority = CommTilePriority(lhs.commTileM, hardwareType, dataType);
    int rhsCommTilePriority = CommTilePriority(rhs.commTileM, hardwareType, dataType);
    if (lhsCommTilePriority != rhsCommTilePriority)
    {
        return lhsCommTilePriority < rhsCommTilePriority;
    }
    int lhsNpuPriority = SplitNpuPriority(lhs.commNpuSplit, rankSize);
    int rhsNpuPriority = SplitNpuPriority(rhs.commNpuSplit, rankSize);
    if (lhsNpuPriority != rhsNpuPriority)
    {
        return lhsNpuPriority < rhsNpuPriority;
    }
    int lhsProductPriority = SplitProductPriority(lhs);
    int rhsProductPriority = SplitProductPriority(rhs);
    if (lhsProductPriority != rhsProductPriority)
    {
        return lhsProductPriority < rhsProductPriority;
    }
    if (lhs.commDataSplit != rhs.commDataSplit)
    {
        return lhs.commDataSplit > rhs.commDataSplit;
    }
    return false;
}

bool IsBetter(double cost, TilingCandidate const &candidate, bool hasBest, double bestCost,
              TilingCandidate const &bestCandidate, uint32_t rankSize, CostModelHardwareType hardwareType,
              CocDataType dataType)
{
    if (!std::isfinite(cost))
    {
        return false;
    }
    if (!hasBest)
    {
        return true;
    }
    double threshold = bestCost * (1.0 - AG_TIE_BREAK_RATIO);
    if (cost < threshold)
    {
        return true;
    }
    if (std::abs(cost - bestCost) / std::min(cost, bestCost) <= AG_TIE_BREAK_RATIO)
    {
        if (BetterByTieBreak(candidate, bestCandidate, rankSize, hardwareType, dataType))
        {
            return true;
        }
        if (BetterByTieBreak(bestCandidate, candidate, rankSize, hardwareType, dataType))
        {
            return false;
        }
        return cost < bestCost;
    }
    return false;
}

}  // namespace

CostModelResult SelectAllGatherTiling(COCMatMulInfo const &info, uint32_t rankSize, CostModelConfig const &config)
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
    uint32_t aicCoreNum = config.aicCoreNum > 0 ? config.aicCoreNum : hardware.coreNum;
    double aStageTime = config.includeAStage ? EstimateAStageTime(m, k, aicCoreNum, hardware) : 0.0;
    double bestRankingCost = std::numeric_limits<double>::max();
    uint32_t k0 = config.k0;
    MTECostModel mteModel{hardware};
    TilingCandidate bestCandidate;

    for (uint32_t m0 : config.m0List)
    {
        if (m0 != 128 && m0 != 256)
        {
            continue;
        }
        uint32_t n0 = m0 == 128 ? 256 : 128;
        double aicTileTime = EstimateAicTileTime(m0, n0, k, k0, mteModel, hardware);

        for (uint32_t commInterval : config.commIntervalList)
        {
            if (commInterval == 0 || !IsWorkspaceValid(m, k, rankSize, m0, commInterval, config, hardware))
            {
                continue;
            }
            uint64_t commSizeM = static_cast<uint64_t>(commInterval) * m0;
            uint64_t commLoops = CeilDiv(m, commSizeM);

            for (uint32_t commTileM : config.commTileList)
            {
                if (commTileM == 0 || commTileM % 2 != 0)
                {
                    continue;
                }
                uint32_t commBlockM = commTileM;
                for (auto const &split : config.splitCandidates)
                {
                    uint32_t splitProduct = split.commNpuSplit * split.commDataSplit;
                    if (split.commNpuSplit == 0 || split.commDataSplit == 0 ||
                        (splitProduct != 16 && splitProduct != 20))
                    {
                        continue;
                    }
                    std::vector<double> aivTimes;
                    std::vector<double> aicTimes;
                    aivTimes.reserve(commLoops);
                    aicTimes.reserve(commLoops);
                    for (uint64_t i = 0; i < commLoops; ++i)
                    {
                        uint32_t offsetM = static_cast<uint32_t>(i * commSizeM);
                        uint32_t actualCommSizeM = std::min<uint32_t>(static_cast<uint32_t>(commSizeM), m - offsetM);
                        double aivTime =
                            EstimateAivWindowTime(actualCommSizeM, k, commTileM, commBlockM, k0, split.commNpuSplit,
                                                  split.commDataSplit, rankSize, config, hardware);
                        if (config.allGatherBarrierPerLoop)
                        {
                            aivTime += hardware.syncTimeUs * SHMEM_BARRIER_COUNT;
                        }
                        aivTimes.push_back(aivTime);
                        aicTimes.push_back(
                            EstimateAicWindowTime(actualCommSizeM, n, m0, n0, rankSize, aicTileTime, aicCoreNum));
                    }

                    uint32_t fixedShmemBarrierRounds = config.allGatherBarrierPerLoop ? 0 : 1;
                    double exposedCommLoopOverhead = ALLGATHER_EXPOSED_COMM_LOOP_OVERHEAD_US * commLoops;
                    double rankingCost = (SimulateDoubleBufferPipeline(aivTimes, aicTimes, WORKSPACE_STAGES) +
                                          hardware.syncTimeUs * SHMEM_BARRIER_COUNT * fixedShmemBarrierRounds +
                                          hardware.crossCoreBarrierTimeUs * CROSS_CORE_BARRIER_PER_LOOP * commLoops +
                                          exposedCommLoopOverhead + hardware.launchTimeUs);
                    rankingCost = ApplyCommBlockPenalty(rankingCost, commBlockM, config.hardwareType, config.dataType);

                    TilingCandidate candidate{
                        m0, n0, commTileM, commInterval, split.commNpuSplit, split.commDataSplit, commBlockM};
                    if (!IsBetter(rankingCost, candidate, best.IsSuccess(), bestRankingCost, bestCandidate, rankSize,
                                  config.hardwareType, config.dataType))
                    {
                        continue;
                    }

                    best.cost = rankingCost + aStageTime;
                    best.status = CostModelStatus::SUCCESS;
                    best.tiling.m0 = m0;
                    best.tiling.k0 = k0;
                    best.tiling.n0 = n0;
                    best.tiling.commTileM = commTileM;
                    best.tiling.commInterval = commInterval;
                    best.tiling.commNpuSplit = split.commNpuSplit;
                    best.tiling.commDataSplit = split.commDataSplit;
                    best.tiling.commBlockM = commBlockM;
                    bestRankingCost = rankingCost;
                    bestCandidate = candidate;
                }
            }
        }
    }

    if (!best.IsSuccess())
    {
        best.status = CostModelStatus::NO_VALID_CANDIDATE;
    }
    return best;
}
