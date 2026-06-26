/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV_HOST_H
#define ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV_HOST_H

#include "operator_registry.h"
#include "catlass/detail/alignment.hpp"

class Ascend950Fp4MxGroupedMatmulAllToAllVOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        uint32_t localExpertNum = cocTiling.expertNum / cocTiling.epSize;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k / 2;
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n / 2 * localExpertNum;
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16) * cocTiling.rankSize;
        size_t localTokensPerExpertSize = static_cast<size_t>(cocTiling.expertNum) * sizeof(int32_t);
        size_t globalTokensPerLocalExpertSize = static_cast<size_t>(cocTiling.rankSize) * localExpertNum * sizeof(int32_t);
        size_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(cocTiling.k);
        size_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
        size_t lenMxScaleA = cocTiling.m * mxScaleAlignedK;
        size_t lenMxScaleB = mxScaleAlignedK * cocTiling.n * localExpertNum;
        size_t sizeMxScaleA = lenMxScaleA * sizeof(int8_t);
        size_t sizeMxScaleB = lenMxScaleB * sizeof(int8_t);

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
            std::vector<half> matrixB(cocTiling.k * cocTiling.n * localExpertNum, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *aMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aMxScaleDevice), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aMxScaleHost), sizeMxScaleA));
            ReadFile(dataFile + "/a_scale_" + std::to_string(rankId) + ".bin", aMxScaleHost, sizeMxScaleA);
            ACL_CHECK(aclrtMemcpy(aMxScaleDevice, sizeMxScaleA, aMxScaleHost, sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixAMxScale(lenMxScaleA, 1);
            ACL_CHECK(aclrtMemcpy(aMxScaleDevice, sizeMxScaleA, matrixAMxScale.data(), sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bMxScaleDevice), sizeMxScaleB, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bMxScaleHost), sizeMxScaleB));
            ReadFile(dataFile + "/b_scale_" + std::to_string(rankId) + ".bin", bMxScaleHost, sizeMxScaleB);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeMxScaleB, bMxScaleHost, sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixBMxScale(lenMxScaleB, 1);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeMxScaleB, matrixBMxScale.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        std::string fileName;
        uint32_t lSize = cocTiling.expertNum * sizeof(uint32_t);
        uint32_t gSize = cocTiling.expertNum / cocTiling.epSize * cocTiling.rankSize * sizeof(uint32_t);
        uint8_t *localTokensPerExpertDevice;
        ACL_CHECK(aclrtMalloc((void **)(&localTokensPerExpertDevice), localTokensPerExpertSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *localTokensPerExpertHost;
        uint8_t *globalTokensPerLocalExpertDevice;
        ACL_CHECK(aclrtMalloc((void **)(&globalTokensPerLocalExpertDevice), globalTokensPerLocalExpertSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *globalTokensPerLocalExpertHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&localTokensPerExpertHost), localTokensPerExpertSize));
            fileName = dataFile + "/local_tokens_per_expert_" + std::to_string(rankId) + ".bin";
            ReadFile(fileName, localTokensPerExpertHost, localTokensPerExpertSize);
            ACL_CHECK(aclrtMemcpy(localTokensPerExpertDevice, localTokensPerExpertSize, localTokensPerExpertHost, localTokensPerExpertSize, ACL_MEMCPY_HOST_TO_DEVICE));
                
            ACL_CHECK(aclrtMallocHost((void **)(&globalTokensPerLocalExpertHost), globalTokensPerLocalExpertSize));
            fileName = dataFile + "/global_tokens_per_local_expert_" + std::to_string(rankId) + ".bin";
            ReadFile(fileName, globalTokensPerLocalExpertHost, globalTokensPerLocalExpertSize);
            ACL_CHECK(aclrtMemcpy(globalTokensPerLocalExpertDevice, globalTokensPerLocalExpertSize, globalTokensPerLocalExpertHost, globalTokensPerLocalExpertSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<uint32_t> matrixL(lSize / sizeof(uint32_t), cocTiling.m / cocTiling.expertNum);
            ACL_CHECK(aclrtMemcpy(localTokensPerExpertDevice, lSize, matrixL.data(), lSize, ACL_MEMCPY_HOST_TO_DEVICE));
                
            std::vector<uint32_t> matrixG(gSize / sizeof(uint32_t), cocTiling.m / cocTiling.expertNum * cocTiling.rankSize / cocTiling.epSize);
            ACL_CHECK(aclrtMemcpy(globalTokensPerLocalExpertDevice, gSize, matrixG.data(), gSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        params.SetKernelParams(aDevice, bDevice, cDevice, aMxScaleDevice, bMxScaleDevice,
            localTokensPerExpertDevice, globalTokensPerLocalExpertDevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(aMxScaleHost));
            ACL_CHECK(aclrtFreeHost(bMxScaleHost));
            ACL_CHECK(aclrtFreeHost(localTokensPerExpertHost));
            ACL_CHECK(aclrtFreeHost(globalTokensPerLocalExpertHost));
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
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams &cocTiling) override {
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;

        if (product % rankSize != 0) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("Ascend950Fp4MxGroupedMatmulAllToAllV", Ascend950Fp4MxGroupedMatmulAllToAllVOperator);

#endif // ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV_HOST_H
