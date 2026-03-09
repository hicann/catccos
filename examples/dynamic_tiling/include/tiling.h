/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_H
#define TILING_H

#include "info.h"
#include "utils.h"
#include "launch_map.h"
#include "operator_registry.h"
#include <sstream>
#include <vector>

std::vector<uint32_t> vCommInterval = {1, 2, 4, 6, 8, 10, 12, 14};
std::vector<uint32_t> vCommTileM    = {2, 4, 8, 16, 32, 64, 128};
std::vector<uint32_t> vM0           = {128, 256};
std::vector<std::pair<uint32_t, uint32_t>> vCommSplitNpuDataPair = {{1, 16}, {2, 8}, {4, 4}, {1, 20}, {2, 10}, {4, 5}};
std::vector<std::vector<uint32_t>> allParams = {vCommInterval, vCommTileM, vM0};

void GetParamFromSearchSpace(std::vector<uint32_t>& curParams,
                             std::vector<std::vector<uint32_t>> &results,
                             int pos) {
    if (pos == allParams.size()) {
        for (int i = 0; i < vCommSplitNpuDataPair.size(); i++) {
            std::vector<uint32_t> tmpParams(curParams.begin(), curParams.end());
            tmpParams.push_back(vCommSplitNpuDataPair[i].first);
            tmpParams.push_back(vCommSplitNpuDataPair[i].second);
            results.push_back(tmpParams);
        }
    } else {
        for (int i = 0; i < allParams[pos].size(); i++) {
            curParams[pos] = allParams[pos][i];
            GetParamFromSearchSpace(curParams, results, pos + 1);
        }
    }
}

void GetTilings(std::vector<CocTilingParams> &tilings, CocTilingParams &t,
    const std::string &opName, int rankSize) {
    auto op = OperatorRegistry::Instance().CreateOperator(opName);
    if (!op) {
        std::cout << "Operator " << opName << " not found!" << std::endl;
        return;
    }
    std::vector<uint32_t> curParams(allParams.size(), 0);
    std::vector<std::vector<uint32_t>> allTilings;
    GetParamFromSearchSpace(curParams, allTilings, 0);
    for (const auto &tiling : allTilings) {
        uint32_t idx = 0;
        t.commInterval = tiling[idx++];
        t.commTileM    = tiling[idx++] * 2;
        t.commBlockM   = t.commTileM;
        t.m0           = tiling[idx++];
        t.k0           = 256;
        t.n0           = (t.m0 == 128) ? 256 : 128;
        t.commNpuSplit = tiling[idx++];
        t.commDataSplit = tiling[idx++];

        if (!op->CheckCocTilingParams(rankSize, t)) continue;
        
        tilings.push_back(t);
    }
}

bool CreateTilingFile(const std::string filename)
{
    std::ofstream outFile(filename, std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Open file failed." << std::endl;
        return false;
    }
    outFile << "Op,M,K,N,Transpose A,Transpose B,M0,commInterval,commTileM,commBlockM,commNpuSplit,commDataSplit,Time(us)\n";
    outFile.close();
    return true;
}

bool WriteTilingInfos(std::string opName, std::vector<CocTilingParams> &cocTilings, const std::string filename,
                      int transA = 0, int transB = 1) {
    std::ofstream outputFile(filename, std::ios::out | std::ios::app);
    if (!outputFile) {
        ERROR_LOG("Open file failed. path = %s, error = %s", filename.c_str(), strerror(errno));
        return false;
    }

    for (CocTilingParams cocTiling : cocTilings) {
        outputFile << opName
                   << "," << cocTiling.m
                   << "," << cocTiling.k
                   << "," << cocTiling.n
                   << "," << transA
                   << "," << transB
                   << "," << cocTiling.m0
                   << "," << cocTiling.commInterval
                   << "," << cocTiling.commTileM
                   << "," << cocTiling.commBlockM
                   << "," << cocTiling.commNpuSplit
                   << "," << cocTiling.commDataSplit
                   << "," << "\n";
    }
    outputFile.close();
    return true;
}

#endif // TILING_H