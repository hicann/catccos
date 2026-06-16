/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_FP4_MX_ALLGATHER_MATMUL_HOST_H
#define ASCEND950_FP4_MX_ALLGATHER_MATMUL_HOST_H

#include "operator_registry.h"
#include "catlass/detail/alignment.hpp"

class Ascend950Fp4MxAllGatherMatmulOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        int m = cocTiling.m;
        int n = cocTiling.n;
        int k = cocTiling.k;
        size_t lenA = static_cast<size_t>(m) * k / 2;
        size_t lenB = static_cast<size_t>(k) * n / 2;
        size_t lenC = static_cast<size_t>(m) * n * cocTiling.rankSize;
        size_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(cocTiling.k);
        size_t lenAScale = cocTiling.m * mxScaleK;
        size_t lenBScale = mxScaleK * cocTiling.n;

        size_t sizeA = lenA * sizeof(int8_t);
        size_t sizeB = lenB * sizeof(int8_t);
        size_t sizeC = lenC * sizeof(__fp16);
        size_t sizeAScale = lenAScale * sizeof(int8_t);
        size_t sizeBScale = lenBScale * sizeof(int8_t);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), sizeA));
            ReadFile(dataFile + "/input_a_" + std::to_string(rankId) + ".bin", aHost, sizeA);
            ACL_CHECK(aclrtMemcpy(aDevice, sizeA, aHost, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixA(lenA, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, sizeA, matrixA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), sizeB));
            ReadFile(dataFile + "/input_b_" + std::to_string(rankId) + ".bin", bHost, sizeB);
            ACL_CHECK(aclrtMemcpy(bDevice, sizeB, bHost, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixB(lenB, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, sizeB, matrixB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *aMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aMxScaleDevice), sizeAScale, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aMxScaleHost), sizeAScale));
            ReadFile(dataFile + "/input_a_scale_" + std::to_string(rankId) + ".bin", aMxScaleHost, sizeAScale);
            ACL_CHECK(aclrtMemcpy(aMxScaleDevice, sizeAScale, aMxScaleHost, sizeAScale, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixAMxScale(lenAScale, 1);
            ACL_CHECK(aclrtMemcpy(aMxScaleDevice, sizeAScale, matrixAMxScale.data(), sizeAScale, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bMxScaleDevice), sizeBScale, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bMxScaleHost), sizeBScale));
            ReadFile(dataFile + "/input_b_scale_" + std::to_string(rankId) + ".bin", bMxScaleHost, sizeBScale);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeBScale, bMxScaleHost, sizeBScale, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixBMxScale(lenBScale, 1);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, sizeBScale, matrixBMxScale.data(), sizeBScale, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

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
        size_t sizeC = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(__fp16);

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), sizeC));
        ACL_CHECK(aclrtMemcpy(cHost, sizeC, cDevice, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", cHost, sizeC);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ASCEND950_FP4_MX_ALLGATHER_MATMUL;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoops = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = 2 * SHMEM_BUFF_BYTES / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoops >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("Ascend950Fp4MxAllGatherMatmul", Ascend950Fp4MxAllGatherMatmulOperator);

#endif // ASCEND950_FP4_MX_ALLGATHER_MATMUL_HOST_H