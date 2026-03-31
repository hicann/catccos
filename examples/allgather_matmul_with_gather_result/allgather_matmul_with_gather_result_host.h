/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_WITH_GATHER_RESULT_HOST_H
#define ALLGATHER_MATMUL_WITH_GATHER_RESULT_HOST_H

#include "operator_registry.h"

class AllGatherMatmulWithGatherResultOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(__fp16);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(__fp16);
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(__fp16);
        size_t gatherASize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.k * sizeof(__fp16);

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

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *gatherADevice;
        ACL_CHECK(aclrtMalloc((void **)(&gatherADevice), gatherASize, ACL_MEM_MALLOC_HUGE_FIRST));

        params.SetKernelParams(aDevice, bDevice, cDevice, gatherADevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(__fp16);
        size_t gatherASize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.k * sizeof(__fp16);

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, cSize);

        uint8_t *gatherADevice = params.customPtrs[0];
        uint8_t *gatherAHost;
        ACL_CHECK(aclrtMallocHost((void **)(&gatherAHost), gatherASize));
        ACL_CHECK(aclrtMemcpy(gatherAHost, gatherASize, gatherADevice, gatherASize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_gather_a.bin", gatherAHost, gatherASize);

        ACL_CHECK(aclrtFreeHost(cHost));
        ACL_CHECK(aclrtFreeHost(gatherAHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ALLGATHER_MATMUL_WITH_GATHER_RESULT;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoops = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoops >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("AllGatherMatmulWithGatherResult", AllGatherMatmulWithGatherResultOperator);

#endif // ALLGATHER_MATMUL_WITH_GATHER_RESULT_HOST_H