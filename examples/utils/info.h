/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INFO_H
#define INFO_H

#include "utils.h"
#include <map>
#include <acl/acl.h>
#include <limits.h>

static uint64_t SHMEM_MALLOC_MAX_SIZE = 1024UL * 1024UL * 1024;
constexpr uint32_t M0 = 128;
constexpr uint32_t N0 = 256;
constexpr uint32_t K0 = 256;
constexpr uint32_t TILE_SHAPE_64 = 64;
constexpr uint32_t TILE_SHAPE_128 = 128;
constexpr uint32_t TILE_SHAPE_256 = 256;
constexpr uint32_t TILE_SHAPE_512 = 512;
constexpr uint32_t WORKSPACE_STAGES = 2;
constexpr uint32_t UB_STAGES = 2;
constexpr uint32_t BLOCK_NUM = 20;
constexpr uint32_t WARM_UP_TIMES = 10;
constexpr uint32_t PERF_TEST_CYCLE_TIMES = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t SHMEM_BUFF_BYTES = 1004UL * 1024 * 1024;
constexpr uint32_t IPC_BUFF_MAX_SIZE = 1000UL * 1024 * 1024;
constexpr uint32_t SYNC_UNIT_SIZE = 4 * sizeof(int64_t);
constexpr int32_t FLAG_BUFF_BYTES = 5 * 512 * 1024;
constexpr int32_t INPUT_DTYPE = 2;

using half = __fp16;

enum CocCommType {
    MATMUL_ALLREDUCE = 0,
    ALLGATHER_MATMUL,
    MATMUL_REDUCE_SCATTER,
    ALLGATHER_MATMUL_WITH_GATHER_RESULT,
    GROUPED_MATMUL_ALLTOALLV,
    ALLTOALLV_GROUPED_MATMUL,
    ALLGATHER_MATMUL_RDMA,
    ALLTOALLV_GMM_V2,
    ALLGATHER_MATMUL_DEQUANT_BIAS,
    ALLGATHER_MATMUL_DEQUANT,
    ALLGATHER_MATMUL_DEQUANT_PADDING,
    TYPE_NUM,
    UNKNOWN
};

enum CocDataType {
    FP16 = 1,
    INT8 = 2,
    BF16 = 27
};

struct CocTilingParams {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t transA = 0;
    uint32_t transB = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
    uint32_t rankSize = 0;
    uint32_t epSize = 0;
    uint32_t expertNum = 0;
};

struct COCMatMulInfo{
    int64_t m;
    int64_t k;
    int64_t n;
};

struct KernelParams {
    uint8_t *ptrA;
    uint8_t *ptrB;
    uint8_t *ptrC;
    uint8_t **customPtrs;
    uint32_t customCount;

    KernelParams() : ptrA(nullptr), ptrB(nullptr), ptrC(nullptr), 
                         customPtrs(nullptr), customCount(0) {}

    template<typename... Args>
    void SetKernelParams(uint8_t *a, uint8_t *b, uint8_t *c, Args... args) {
        ptrA = a;
        ptrB = b;
        ptrC = c;

        const int argsCount = sizeof...(args);
        customCount = argsCount;
    
        if (customPtrs != nullptr) {
            ACL_CHECK(aclrtFreeHost(customPtrs));
        }
    
        if (argsCount > 0) {
            ACL_CHECK(aclrtMallocHost((void **)(&customPtrs), argsCount * sizeof(uint8_t *)));
            uint8_t *pointers[] = {args...};
    
            for (size_t i = 0; i < argsCount; ++i) {
                customPtrs[i] = pointers[i];
            }
        }
    }
    
    ~KernelParams() {
        if (customPtrs != nullptr) {
            ACL_CHECK(aclrtFreeHost(customPtrs));
        }
    }
};

inline void FreeDeviceSpace(KernelParams &params)
{
    ACL_CHECK(aclrtFree(params.ptrA));
    ACL_CHECK(aclrtFree(params.ptrB));
    ACL_CHECK(aclrtFree(params.ptrC));
    params.ptrA = nullptr;
    params.ptrB = nullptr;
    params.ptrC = nullptr;
    for (uint32_t i = 0; i < params.customCount; i++) {
        ACL_CHECK(aclrtFree(params.customPtrs[i]));
    }
}

// kernel缩写
const std::map<std::string, CocCommType> CommTypeMap = {
    {"mmar", CocCommType::MATMUL_ALLREDUCE},
    {"agmm", CocCommType::ALLGATHER_MATMUL},
    {"mmrs", CocCommType::MATMUL_REDUCE_SCATTER},
    {"agmmwg", CocCommType::ALLGATHER_MATMUL_WITH_GATHER_RESULT},
    {"gmmata", CocCommType::GROUPED_MATMUL_ALLTOALLV},
    {"atagmm", CocCommType::ALLTOALLV_GROUPED_MATMUL},
    {"agmmrdma", CocCommType::ALLGATHER_MATMUL_RDMA},
    {"atavgmmv2", CocCommType::ALLTOALLV_GMM_V2},
    {"agmmdq", CocCommType::ALLGATHER_MATMUL_DEQUANT},
    {"agmmdqbs", CocCommType::ALLGATHER_MATMUL_DEQUANT_BIAS},
    // 新增算子继续添加...
};

inline CocCommType GetCommType(const std::string& kernelName) {
    auto it = CommTypeMap.find(kernelName);
    if (it != CommTypeMap.end()) {
        return it->second;
    }
    return CocCommType::UNKNOWN;
}

const std::map<CocCommType, std::string> CommTypeOpNameMap = {
    { MATMUL_ALLREDUCE, "MatmulAllReduce" },
    { ALLGATHER_MATMUL, "AllGatherMatmul" },
    { MATMUL_REDUCE_SCATTER, "MatmulReduceScatter" },
    { ALLGATHER_MATMUL_WITH_GATHER_RESULT, "AllGatherMatmulWithGatherResult" },
    { GROUPED_MATMUL_ALLTOALLV, "GroupedMatmulAllToAllV" },
    { ALLTOALLV_GROUPED_MATMUL, "AllToAllVGroupedMatmul" },
    { ALLGATHER_MATMUL_RDMA, "AllGatherMatmulRdma" },
    { ALLTOALLV_GMM_V2, "AllToAllVGMMV2" },
    { ALLGATHER_MATMUL_DEQUANT, "AllGatherMatmulDequant" },
    { ALLGATHER_MATMUL_DEQUANT_BIAS, "AllGatherMatmulDequantBias" },
};

inline int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

#endif // INFO_H