/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_GMM_DEQUANT_SWIGLU_HOST_H
#define DISPATCH_GMM_DEQUANT_SWIGLU_HOST_H

#include "operator_registry.h"

class DispatchGmmDequantSwigluOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t maxOutputSize = cocTiling.m * cocTiling.topK * cocTiling.rankSize;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(half);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * expertPerRank * sizeof(int8_t);
        size_t scaleSize = static_cast<size_t>(cocTiling.n) * expertPerRank * sizeof(uint64_t);
        size_t cSize = static_cast<size_t>(maxOutputSize) * cocTiling.n / 2 * sizeof(half);
        size_t expertIdxSize = cocTiling.m * cocTiling.topK * sizeof(int32_t);

        uint8_t *aDevice, *aHost;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/a_gm_" + std::to_string(rankId) + ".bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixA(cocTiling.m * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice, *bHost;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/b_gm_" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(cocTiling.k * cocTiling.n * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *scaleDevice, *scaleHost;
        ACL_CHECK(aclrtMalloc((void **)(&scaleDevice), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleSize));
            ReadFile(dataFile + "/scale_gm_" + std::to_string(rankId) + ".bin", scaleHost, scaleSize);
            ACL_CHECK(aclrtMemcpy(scaleDevice, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<uint64_t> matrixScale(cocTiling.n * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(scaleDevice, scaleSize, matrixScale.data(), scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *expertIdxDevice, *expertIdxHost;
        ACL_CHECK(aclrtMalloc((void **)(&expertIdxDevice), expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void **)(&expertIdxHost), expertIdxSize));
        ReadFile(dataFile + "/expert_idx_" + std::to_string(rankId) + ".bin", expertIdxHost, expertIdxSize);
        ACL_CHECK(aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxHost, expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        params.SetKernelParams(aDevice, bDevice, cDevice, scaleDevice, expertIdxDevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(expertIdxHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.topK * (cocTiling.n / 2) * sizeof(half) * cocTiling.rankSize;
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
        uint32_t maxOutputSize = cocTiling.m * cocTiling.rankSize * cocTiling.topK;
        size_t workspaceSize = EP * EP * expertPerRank * sizeof(int32_t) // ptrCumsumMM
            + maxOutputSize * sizeof(float) // ptrPerTokenScale
            + static_cast<size_t>(maxOutputSize) * cocTiling.n * sizeof(half) // ptrC
            + static_cast<size_t>(maxOutputSize) * cocTiling.k * sizeof(int8_t); // ptrA
        return workspaceSize;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::DISPATCH_GMM_DEQUANT_SWIGLU;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams &cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoop = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = IPC_BUFF_MAX_SIZE / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoop >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("DispatchGmmDequantSwiglu", DispatchGmmDequantSwigluOperator);

#endif // DISPATCH_GMM_DEQUANT_SWIGLU_HOST_H