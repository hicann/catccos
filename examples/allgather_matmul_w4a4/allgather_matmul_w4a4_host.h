/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_W4A4_HOST_H
#define ALLGATHER_MATMUL_W4A4_HOST_H

#include "operator_registry.h"

class AllGatherMatmulW4A4Operator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override
    {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(int8_t) / 2;
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(int8_t) / 2;
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(half);
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(bfloat16_t);
        size_t scaleSize = static_cast<size_t>(cocTiling.n) * sizeof(uint64_t);
        size_t perTokenScaleSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * sizeof(float);

        uint8_t *aDevice = static_cast<uint8_t *>(shmem_malloc(aSize));
        if (dataFile != "") {
            uint8_t *aHost;
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/a_gm_rank" + std::to_string(rankId) + ".bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(aHost));
        } else {
            std::vector<int8_t> matrixA(cocTiling.m * cocTiling.k / 2, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *bHost;
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/b_gm_rank" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(bHost));
        } else {
            std::vector<int8_t> matrixB(cocTiling.k * cocTiling.n / 2, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *dDevice;
        ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *scaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&scaleDevice), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *scaleHost;
            ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleSize));
            ReadFile(dataFile + "/scale_gm.bin", scaleHost, scaleSize);
            ACL_CHECK(aclrtMemcpy(scaleDevice, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(scaleHost));
        } else {
            std::vector<uint64_t> matrixScale(cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(scaleDevice, scaleSize, matrixScale.data(), scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *perTokenScaleDevice;
        ACL_CHECK(aclrtMalloc((void **)(&perTokenScaleDevice), perTokenScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t *perTokenScaleHost;
            ACL_CHECK(aclrtMallocHost((void **)(&perTokenScaleHost), perTokenScaleSize));
            ReadFile(dataFile + "/per_token_scale_gm.bin", perTokenScaleHost, perTokenScaleSize);
            ACL_CHECK(aclrtMemcpy(
                perTokenScaleDevice, perTokenScaleSize, perTokenScaleHost, perTokenScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(perTokenScaleHost));
        } else {
            std::vector<float> matrixPerTokenScale(cocTiling.m * cocTiling.rankSize, 1.0f);
            ACL_CHECK(aclrtMemcpy(
                perTokenScaleDevice, perTokenScaleSize, matrixPerTokenScale.data(), perTokenScaleSize,
                ACL_MEMCPY_HOST_TO_DEVICE));
        }

        params.SetKernelParams(aDevice, bDevice, dDevice, cDevice, scaleDevice, perTokenScaleDevice);
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override
    {
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(bfloat16_t);

        uint8_t *dDevice = params.ptrC;
        uint8_t *dHost;
        ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
        ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_rank" + std::to_string(rankId) + ".bin", dHost, dSize);
        ACL_CHECK(aclrtFreeHost(dHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams & /*cocTiling*/) override
    {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams & /*cocTiling*/) override
    {
        return CocCommType::UNKNOWN;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams &cocTiling) override
    {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoop = CeilDev(cocTiling.k, 1024);
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * 1024 * kLoop >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("AllGatherMatmulW4A4", AllGatherMatmulW4A4Operator);

#endif // ALLGATHER_MATMUL_W4A4_HOST_H
