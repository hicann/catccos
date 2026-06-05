/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_FFN_COMBINE_H_HOST_H
#define DISPATCH_FFN_COMBINE_H_HOST_H

#include "operator_registry.h"

class DispatchFFNCombineOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t k2 = cocTiling.n / 2;
        uint32_t n2 = cocTiling.k;

        uint32_t maxOutputSize = cocTiling.m * cocTiling.topK * cocTiling.rankSize;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(half);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * expertPerRank * sizeof(half);
        size_t b2Size = static_cast<size_t>(k2) * n2 * expertPerRank * sizeof(half);
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(half);

        size_t expertIdxSize = cocTiling.m * cocTiling.topK * sizeof(int32_t);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/in_routing_matrix_a_" + std::to_string(rankId) + ".bin", aHost, aSize);
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
            ReadFile(dataFile + "/in_gmm_matrix_b_" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(cocTiling.k * cocTiling.n * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *b2Device;
        ACL_CHECK(aclrtMalloc((void **)(&b2Device), b2Size, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *b2Host;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&b2Host), b2Size));
            ReadFile(dataFile + "/in_gmm_matrix_b2_" + std::to_string(rankId) + ".bin", b2Host, b2Size);
            ACL_CHECK(aclrtMemcpy(b2Device, b2Size, b2Host, b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB2(k2 * n2 * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(b2Device, b2Size, matrixB2.data(), b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *expertIdxDevice;
        uint8_t *expertIdxHost;
        ACL_CHECK(aclrtMalloc((void **)(&expertIdxDevice), expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void **)(&expertIdxHost), expertIdxSize));
        ReadFile(dataFile + "/in_routing_expert_idx_" + std::to_string(rankId) + ".bin", expertIdxHost, expertIdxSize);
        ACL_CHECK(aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxHost, expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t *probsDevice;
        uint8_t *probsHost;
        size_t probsSize = cocTiling.m * cocTiling.topK * sizeof(float32_t);
        ACL_CHECK(aclrtMalloc((void **)(&probsDevice), probsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void **)(&probsHost), probsSize));
        ReadFile(dataFile + "/in_gather_gate_weight_" + std::to_string(rankId) + ".bin", probsHost, probsSize);
        ACL_CHECK(aclrtMemcpy(probsDevice, probsSize, probsHost, probsSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        // uint8_t *outputDevice;
        // size_t outputSize = cocTiling.m * cocTiling.k * sizeof(half);
        // ACL_CHECK(aclrtMalloc((void **)(&outputDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // ACL_CHECK(aclrtMemset(outputDevice, outputSize, 0, outputSize));

        params.SetKernelParams(aDevice, bDevice, cDevice, b2Device, expertIdxDevice, probsDevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(b2Host));
            ACL_CHECK(aclrtFreeHost(expertIdxHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(half);
        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));

        // size_t outputSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(half);
        // uint8_t *outputDevice = params.customPtrs[3];
        // uint8_t *outputHost;
        // ACL_CHECK(aclrtMallocHost((void **)(&outputHost), outputSize));
        // ACL_CHECK(aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
        // WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", outputHost, outputSize);

        // ACL_CHECK(aclrtFreeHost(outputHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t maxOutputSize = cocTiling.m * cocTiling.rankSize * cocTiling.topK;
        size_t workspaceSize = RoundUp<size_t>(cocTiling.m * sizeof(int32_t), 512) +                      // expandedRowIdx
                               RoundUp<size_t>(maxOutputSize * cocTiling.k * sizeof(half), 512) +         // AllToAllV_GMM workspace
                               EP * EP * expertPerRank * sizeof(int32_t) +                                // metaInfo
                               RoundUp<size_t>(maxOutputSize * cocTiling.n * sizeof(half), 512) +         // allToAllVGmmOut
                               RoundUp<size_t>(maxOutputSize * cocTiling.n * sizeof(half), 512) +         // swigluOutput
                               RoundUp<size_t>(maxOutputSize * cocTiling.k * sizeof(half), 512) +         // gmmAllToAllWorkspace
                               EP * EP * expertPerRank * sizeof(int32_t);                                 // metaInfo
        return workspaceSize;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::ALLTOALLV_GMM_V2;
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

REGISTER_OPERATOR("DispatchFFNCombine", DispatchFFNCombineOperator);

#endif // DISPATCH_FFN_COMBINE_H_HOST_H