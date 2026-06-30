/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "aclnn_all_gather_matmul_hccl.h"
#include "allgather_matmul_host.h"
#include "info.h"

#ifdef __cplusplus
extern "C" {
#endif
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
#ifdef __cplusplus
}
#endif

#define HCCL_CHECK(ret)                                                                                                \
    do {                                                                                                               \
        if ((ret) != (HCCL_SUCCESS)) {                                                                                     \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, (ret));                      \
            return (ret);                                                                                              \
        }                                                                                                              \
    } while (0)

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size m n k data_path [device_id_list]\n";
    static constexpr size_t MAX_RANK_SIZE = 16;

    int rankSize;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum class ArgsIndex : int {
            RANK_SIZE_INDEX = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };
        auto argIndex = [](ArgsIndex index) {
            return static_cast<int>(index);
        };

        if (argc < argIndex(ArgsIndex::DATA_PATH_INDEX) + 1 || argc > argIndex(ArgsIndex::INDEX_MAX)) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[argIndex(ArgsIndex::RANK_SIZE_INDEX)]);
        if (rankSize <= 0 || rankSize > static_cast<int>(MAX_RANK_SIZE)) {
            printf("rankSize is illegal\n");
            return -1;
        }
        m = std::atoi(argv[argIndex(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[argIndex(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[argIndex(ArgsIndex::K_INDEX)]);
        dataPath = argv[argIndex(ArgsIndex::DATA_PATH_INDEX)];
        if (argc > argIndex(ArgsIndex::DEVICE_LIST_INDEX)) {
            char *idListStr = argv[argIndex(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        if (deviceIdList.size() < static_cast<size_t>(rankSize)) {
            printf("device_id_list has fewer entries than rankSize\n");
            return -1;
        }
        return 0;
    }

    std::string GetDataPath() const
    {
        return dataPath;
    }
};

struct ThreadContext {
    Options options;
    HcclComm comm;
    int32_t device;
    int32_t rankId;
};

std::mutex mtx;

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    aclrtStream stream;
    ACL_CHECK(aclrtSetDevice(ctx->device));
    ACL_CHECK(aclrtCreateStream(&stream));

    char hcomName[128] = {0};
    HCCL_CHECK(HcclGetCommName(ctx->comm, hcomName));

    std::cout << "HcomName: " << hcomName << std::endl;

    auto op = OperatorRegistry::Instance().CreateOperator("AllGatherMatmulHccl");
    if (!op) {
        std::cout << "Operator AllGatherMatmulHccl not found!" << std::endl;
        return -1;
    }
    KernelParams kernelParams;
    CocTilingParams cocTiling;
    cocTiling.m = ctx->options.m;
    cocTiling.n = ctx->options.n;
    cocTiling.k = ctx->options.k;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 20;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = ctx->options.rankSize;

    op->AllocateDeviceSpace(kernelParams, cocTiling, ctx->rankId, ctx->options.GetDataPath());

    uint8_t *aDevice = kernelParams.ptrA;
    uint8_t *bDevice = kernelParams.ptrB;
    uint8_t *cDevice = kernelParams.ptrC;

    std::vector<int64_t> aDims{ctx->options.m, ctx->options.k};
    std::vector<int64_t> aStride{ctx->options.k, 1};
    std::vector<int64_t> bDims{ctx->options.k, ctx->options.n};
    std::vector<int64_t> bStride{ctx->options.n, 1};
    std::vector<int64_t> cDims{ctx->options.m * ctx->options.rankSize, ctx->options.n};
    std::vector<int64_t> cStride{ctx->options.n, 1};

    auto a = aclCreateTensor(aDims.data(), aDims.size(), ACL_FLOAT16, aStride.data(), 0, ACL_FORMAT_ND,
        aDims.data(), aDims.size(), aDevice);
    auto b = aclCreateTensor(bDims.data(), bDims.size(), ACL_FLOAT16, bStride.data(), 0, ACL_FORMAT_ND,
        bDims.data(), bDims.size(), bDevice);
    auto c = aclCreateTensor(cDims.data(), cDims.size(), ACL_FLOAT16, cStride.data(), 0, ACL_FORMAT_ND,
        cDims.data(), cDims.size(), cDevice);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (auto ret = aclnnAllGatherMatmulHcclGetWorkspaceSize(a, b, hcomName, ctx->options.rankSize, c,
        &workspaceSize, &executor); ret != ACL_SUCCESS) {
        std::cerr << aclGetRecentErrMsg() << std::endl;
        return ret;
    }

    uint8_t *workspaceDevice = nullptr;
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    if (auto ret = aclnnAllGatherMatmulHccl(workspaceDevice, workspaceSize, executor, stream); ret != ACL_SUCCESS) {
        std::cerr << "Call aclnnAllGatherMatmulHccl error: " << aclGetRecentErrMsg() << std::endl;
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (workspaceDevice != nullptr) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    op->WriteResultFile(kernelParams, cocTiling, ctx->rankId, ctx->options.GetDataPath());

    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[TEST] begin to exit...... rankId: " << ctx->rankId << std::endl;
    }

    FreeDeviceSpace(kernelParams);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(ctx->device));
    return 0;
}

int main(int argc, char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    // 设备资源初始化
    int ndev = options.rankSize;
    ACL_CHECK(aclInit(nullptr));
    int32_t devices[Options::MAX_RANK_SIZE];
    for (int i = 0; i < ndev; ++i) {
        devices[i] = options.deviceIdList[i];
    }
    for (int32_t i = 0; i < ndev; i++) {
        ACL_CHECK(aclrtSetDevice(devices[i]));
    }

    HcclComm comms[Options::MAX_RANK_SIZE];
    // 初始化通信域
    HCCL_CHECK(HcclCommInitAll(ndev, devices, comms));
    // 启动线程执行集合通信操作
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    struct ThreadContext args[Options::MAX_RANK_SIZE];
    for (uint32_t i = 0; i < ndev; i++) {
        args[i].options = options;
        args[i].device = devices[i];
        args[i].comm = comms[i];
        args[i].rankId = i;
        threads[i].reset(new (std::nothrow) std::thread(&Sample, (void *)&args[i]));
        if (threads[i] == nullptr) {
            std::cerr << "Failed to create thread for rankId: " << i << std::endl;
            for (uint32_t j = 0; j < i; j++) {
                threads[j]->join();
            }
            return 1;
        }
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }
    // 释放通信域等相关资源
    for (uint32_t i = 0; i < ndev; i++) {
        HCCL_CHECK(HcclCommDestroy(comms[i]));
    }
    std::cout << "end end end" << std::endl;
    // 设备资源去初始化
    ACL_CHECK(aclFinalize());
    return 0;
}
