/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "info.h"

enum class CostModelStatus {
    SUCCESS,
    UNSUPPORTED,
    INVALID_ARGUMENT,
    NO_VALID_CANDIDATE,
};

enum class CostModelHardwareType {
    A2,
    A3,
    A5,
};

struct MTECacheConfig {
    double readRttNs;
    double requestIntervalNs;
};

struct CostModelHardwareConfig {
    uint32_t coreNum = 20;
    double writeRttNs = 123.0;
    double remoteReadScheduleNs = 20.0;
    double remoteReadRttNs = 500.0;
    double hccsBandwidth = 19.5;
    double cubeFlopsPerUs = 13920336.0;
    uint32_t elementSize = 0;
    double syncTimeUs = 6.7;
    double launchTimeUs = 6.47;

    uint32_t readOtsdBound = 64;
    uint32_t nd2nzCmdOtsd = 2;
    MTECacheConfig cacheMiss{235.0, 4.5};
    MTECacheConfig cacheHit{110.0, 1.9};
    double fullCoreHitEfficiency = 0.8;
};

struct CostModelTiling {
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
};

struct CostModelConfig {
    std::vector<uint32_t> commIntervalList{1, 2, 4, 6, 8, 10, 12, 14};
    std::vector<uint32_t> commTileList{2, 4, 8, 16, 32, 64};
    std::vector<uint32_t> m0List{128, 256};
    std::vector<uint32_t> aivCoreList{16, 20};
    CostModelHardwareType hardwareType = CostModelHardwareType::A2;
    CocDataType dataType = CocDataType::FP16;
    std::function<bool(CostModelTiling const &)> tilingValidator;

    bool IsCandidateValid(CostModelTiling const &tiling) const
    {
        return !tilingValidator || tilingValidator(tiling);
    }
};

struct CostModelResult {
    CostModelTiling tiling{};
    double cost = std::numeric_limits<double>::max();
    CostModelStatus status = CostModelStatus::UNSUPPORTED;

    bool IsSuccess() const
    {
        return status == CostModelStatus::SUCCESS;
    }
};

CostModelStatus GetA2CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware);

CostModelStatus GetA3CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware);

CostModelStatus GetA5CostModelHardwareConfig(
    CocDataType dataType, CostModelHardwareConfig &hardware);

CostModelStatus GetCostModelHardwareConfig(
    CostModelHardwareType hardwareType, CocDataType dataType,
    CostModelHardwareConfig &hardware);

CostModelResult SelectReduceScatterTiling(
    COCMatMulInfo const &info, uint32_t rankSize,
    CostModelConfig const &config = CostModelConfig{});

CostModelResult SelectAllGatherTiling(
    COCMatMulInfo const &info, uint32_t rankSize,
    CostModelConfig const &config = CostModelConfig{});

CostModelResult SelectAllReduceTiling(
    COCMatMulInfo const &info, uint32_t rankSize,
    CostModelConfig const &config = CostModelConfig{});

CostModelResult SelectCostModelTiling(
    COCMatMulInfo const &info, CocCommType type, uint32_t rankSize,
    CostModelConfig const &config = CostModelConfig{});

bool ApplyCostModel(
    COCMatMulInfo const &info, CocCommType type, uint32_t rankSize,
    CocTilingParams &tiling, CostModelConfig const &config = CostModelConfig{});

#endif // COST_MODEL_H
