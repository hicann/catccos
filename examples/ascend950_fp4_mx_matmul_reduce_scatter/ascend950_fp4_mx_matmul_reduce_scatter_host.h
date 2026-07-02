/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER_HOST_H
#define ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER_HOST_H

#include "operator_registry.h"
#include "catlass/detail/alignment.hpp"

class Ascend950Fp4MxMatmulReduceScatterOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k / 2;
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n / 2;
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16);
        size_t cSizeScatter = cSize / cocTiling.rankSize;
        size_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(cocTiling.k);
        size_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
        size_t lenMxScaleA = cocTiling.m * mxScaleAlignedK;
        size_t lenMxScaleB = mxScaleAlignedK * cocTiling.n;
        size_t sizeMxScaleA = lenMxScaleA * sizeof(int8_t);
        size_t sizeMxScaleB = lenMxScaleB * sizeof(int8_t);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
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
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(cocTiling.k * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *aMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aMxScaleDevice), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aMxScaleHost), sizeMxScaleA));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_a_scale.bin", aMxScaleHost, sizeMxScaleA);
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
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_b_scale.bin", bMxScaleHost, sizeMxScaleB);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeMxScaleB, bMxScaleHost, sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixBMxScale(lenMxScaleB, 1);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeMxScaleB, matrixBMxScale.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSizeScatter, 0, cSizeScatter));

        params.SetKernelParams(aDevice, bDevice, cDevice, aMxScaleDevice, bMxScaleDevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(aMxScaleHost));
            ACL_CHECK(aclrtFreeHost(bMxScaleHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16);
        size_t cSizeScatter = cSize / cocTiling.rankSize;

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSizeScatter));
        ACL_CHECK(aclrtMemcpy(cHost, cSizeScatter, cDevice, cSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, cSizeScatter, rankId * cSizeScatter);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;

        if (product % rankSize != 0) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("Ascend950Fp4MxMatmulReduceScatter", Ascend950Fp4MxMatmulReduceScatterOperator);

#endif // ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER_HOST_H
