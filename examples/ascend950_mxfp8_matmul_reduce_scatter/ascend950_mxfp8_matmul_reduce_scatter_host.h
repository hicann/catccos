/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_HOST_H
#define ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_HOST_H

#include "operator_registry.h"

using host_float8_e5m2_t = int8_t;
using host_float8_e8m0_t = int8_t;
using host_bfloat16_t = int16_t;

template <typename T1, typename T2>
auto Ceil(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

class Ascend950MxFp8MatmulReduceScatterOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t lenA = static_cast<size_t>(cocTiling.m) * cocTiling.k;
        size_t lenB = static_cast<size_t>(cocTiling.k) * cocTiling.n;
        size_t lenC = static_cast<size_t>(cocTiling.m) * cocTiling.n / cocTiling.rankSize;
        size_t lenAScale = static_cast<size_t>(cocTiling.m) * Ceil(static_cast<size_t>(cocTiling.k), 64) * 2;
        size_t lenBScale = static_cast<size_t>(cocTiling.n) * Ceil(static_cast<size_t>(cocTiling.k), 64) * 2;

        size_t aSize = lenA * sizeof(host_float8_e5m2_t);
        size_t bSize = lenB * sizeof(host_float8_e5m2_t);
        size_t cSize = lenC * sizeof(host_bfloat16_t);
        size_t aScaleSize = lenAScale * sizeof(host_float8_e8m0_t);
        size_t bScaleSize = lenBScale * sizeof(host_float8_e8m0_t);

        // Allocate and fill matrix A
        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *aHost;
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(aHost));
        } else {
            std::vector<host_float8_e5m2_t> matrixA(lenA, 60);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and fill matrix B
        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *bHost;
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(bHost));
        } else {
            std::vector<host_float8_e5m2_t> matrixB(lenB, 60);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate output C
        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));

        // Allocate and fill A scale
        uint8_t *aScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aScaleDevice), aScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *aScaleHost;
            ACL_CHECK(aclrtMallocHost((void **)(&aScaleHost), aScaleSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_a_scale.bin", aScaleHost, aScaleSize);
            ACL_CHECK(aclrtMemcpy(aScaleDevice, aScaleSize, aScaleHost, aScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(aScaleHost));
        } else {
            std::vector<host_float8_e8m0_t> scaleA(lenAScale, 121);
            ACL_CHECK(aclrtMemcpy(aScaleDevice, aScaleSize, scaleA.data(), aScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and fill B scale
        uint8_t *bScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bScaleDevice), bScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *bScaleHost;
            ACL_CHECK(aclrtMallocHost((void **)(&bScaleHost), bScaleSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_b_scale.bin", bScaleHost, bScaleSize);
            ACL_CHECK(aclrtMemcpy(bScaleDevice, bScaleSize, bScaleHost, bScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(bScaleHost));
        } else {
            std::vector<host_float8_e8m0_t> scaleB(lenBScale, 121);
            ACL_CHECK(aclrtMemcpy(bScaleDevice, bScaleSize, scaleB.data(), bScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        params.SetKernelParams(aDevice, bDevice, cDevice, aScaleDevice, bScaleDevice);
        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t lenC = static_cast<size_t>(cocTiling.m) * cocTiling.n / cocTiling.rankSize;
        size_t cSize = lenC * sizeof(host_bfloat16_t);

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, cSize, rankId * cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER;
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

// 注册这个算子
REGISTER_OPERATOR("Ascend950MxFp8MatmulReduceScatter", Ascend950MxFp8MatmulReduceScatterOperator);

#endif // ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER_HOST_H
