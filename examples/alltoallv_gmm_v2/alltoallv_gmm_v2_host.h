/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTOALLV_GMM_MATMUL_V2_HOST_H
#define ALLTOALLV_GMM_MATMUL_V2_HOST_H

#include "operator_registry.h"

class AllToAllVGMMV2Operator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t maxOutputSize = cocTiling.m * cocTiling.rankSize;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(__fp16);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * expertPerRank * sizeof(half);
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16) * cocTiling.rankSize;
        size_t tokenPerExpertSize = EP * EP * expertPerRank * sizeof(int32_t);
        
        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/a_gm_" + std::to_string(rankId) + ".bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixA(cocTiling.m * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/b_gm_" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(cocTiling.k * cocTiling.n * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        
        uint8_t *tokenPerExpertHost;
        uint8_t *tokenPerExpertDevice;
        ACL_CHECK(aclrtMalloc((void **)(&tokenPerExpertDevice), tokenPerExpertSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&tokenPerExpertHost), tokenPerExpertSize));
            ReadFile(dataFile + "/global_tokens_per_expert_matrix.bin", tokenPerExpertHost, tokenPerExpertSize);
            ACL_CHECK(aclrtMemcpy(tokenPerExpertDevice, tokenPerExpertSize, tokenPerExpertHost, tokenPerExpertSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixTokenPerEP(EP * EP * expertPerRank, cocTiling.m / (EP * expertPerRank));
            ACL_CHECK(aclrtMemcpy(tokenPerExpertDevice, tokenPerExpertSize, matrixTokenPerEP.data(), tokenPerExpertSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        params.SetKernelParams(aDevice, bDevice, cDevice, tokenPerExpertDevice);

        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(tokenPerExpertHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16) * cocTiling.rankSize;

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t maxOutputSize = cocTiling.m * cocTiling.rankSize;
        size_t workspaceSize = static_cast<size_t>(cocTiling.m * cocTiling.rankSize) * cocTiling.k * sizeof(half) + EP * EP * expertPerRank * sizeof(int32_t);
        return workspaceSize;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ALLTOALLV_GMM_V2;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoop = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = IPC_BUFF_MAX_SIZE / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoop >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("AllToAllVGMMV2", AllToAllVGMMV2Operator);

#endif