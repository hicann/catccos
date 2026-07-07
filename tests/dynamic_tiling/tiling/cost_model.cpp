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

CostModelResult SelectCostModelTiling(
    COCMatMulInfo const &info, CocCommType type, uint32_t rankSize,
    CostModelConfig const &config)
{
    switch (type) {
        case MATMUL_REDUCE_SCATTER:
        case ASCEND950_MATMUL_REDUCE_SCATTER:
            return SelectReduceScatterTiling(info, rankSize, config);
        case MATMUL_ALLREDUCE:
            return SelectAllReduceTiling(info, rankSize, config);
        case ALLGATHER_MATMUL:
            return SelectAllGatherTiling(info, rankSize, config);
        default:
            return {};
    }
}

bool ApplyCostModel(
    COCMatMulInfo const &info, CocCommType type, uint32_t rankSize,
    CocTilingParams &tiling, CostModelConfig const &config)
{
    auto result = SelectCostModelTiling(info, type, rankSize, config);
    if (!result.IsSuccess() || !config.IsCandidateValid(result.tiling)) {
        return false;
    }

    tiling.m0 = result.tiling.m0;
    tiling.k0 = result.tiling.k0;
    tiling.n0 = result.tiling.n0;
    tiling.commTileM = result.tiling.commTileM;
    tiling.commInterval = result.tiling.commInterval;
    tiling.commNpuSplit = result.tiling.commNpuSplit;
    tiling.commDataSplit = result.tiling.commDataSplit;
    tiling.commBlockM = result.tiling.commBlockM;
    tiling.rankSize = rankSize;
    return true;
}
