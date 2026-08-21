/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ascend950_dispatch_ffn_combine_device.h"
#include "ascend950_dispatch_ffn_combine_host.h"

using namespace AscendC;
using namespace Catccos;

using ElementA = bfloat16_t;
using ElementB = float8_e4m3_t;
using ElementC = bfloat16_t;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::ColumnMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::ColumnMajor;

static uint32_t gNpuNum = 8;

std::vector<int32_t> ParseDeviceIdList(char *deviceIdList)
{
    std::vector<int32_t> result;
    if (deviceIdList == nullptr)
    {
        return result;
    }

    for (char *idToken = std::strtok(deviceIdList, ","); idToken; idToken = std::strtok(nullptr, ","))
    {
        result.push_back(std::atoi(idToken));
    }
    return result;
}

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    int rankSize = atoi(argv[1]);
    uint32_t rankId = atoi(argv[2]);
    std::string ipPort = argv[3];
    auto deviceIdList = argc > 12 ? ParseDeviceIdList(argv[12]) : std::vector<int32_t>{};
    int32_t deviceId = deviceIdList.empty() ? atoi(argv[4]) + rankId % gNpuNum : deviceIdList[rankId];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " device_id:" << deviceId
              << " input_ip: " << ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aicCoreNum = platform == nullptr ? 0 : static_cast<uint32_t>(platform->GetCoreNumAic());
    uint32_t aivCoreNum = platform == nullptr ? 0 : static_cast<uint32_t>(platform->GetCoreNumAiv());
    uint32_t hardwareBlockNum = std::min(aicCoreNum, aivCoreNum / 2);
    uint32_t aclshmemBlockNum = ACLSHMEM_MAX_AIV_PER_NPU / 2;
    uint32_t blockNum = std::min(hardwareBlockNum, aclshmemBlockNum);
    if (blockNum == 0)
    {
        std::cerr << "[ERROR] Invalid platform core counts: AIC=" << aicCoreNum << " AIV=" << aivCoreNum << std::endl;
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    uint32_t launchAivNum = 2 * blockNum;
    std::cout << "[CONFIG] platform_aic=" << aicCoreNum << " platform_aiv=" << aivCoreNum
              << " hardware_mixed_1_2_block_num=" << hardwareBlockNum << " launch_block_num=" << blockNum
              << " launch_aiv_num=" << launchAivNum << " aclshmem_aiv_limit=" << ACLSHMEM_MAX_AIV_PER_NPU << std::endl;
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    std::cout << "[TEST] after shmem init rankId: " << rankId << " status: " << status << std::endl;
    if (status != ACLSHMEM_SUCCESS)
    {
        std::cerr << "[ERROR] aclshmemx_init_attr failed rankId: " << rankId << " status: " << status << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return status;
    }

    uint64_t fftsAddr = shmemx_get_ffts_config();

    uint32_t m = atoi(argv[5]);
    uint32_t k = atoi(argv[6]);
    uint32_t n = atoi(argv[7]);
    uint32_t EP = rankSize;
    uint32_t expertPerRank = atoi(argv[8]);
    uint32_t dataType = atoi(argv[9]);
    uint32_t weightNz = atoi(argv[10]);
    uint32_t transB = atoi(argv[11]);

    uint32_t m0 = 10;
    uint32_t k0 = 10;
    uint32_t n0 = 10;
    uint32_t swizzleDirect = 1;
    uint32_t swizzleOffset = 2;
    uint32_t ubMoveNum = 8192;
    uint32_t pValue = 1;
    uint32_t commNpuSplit = rankSize;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;
    uint32_t topK = argc > 13 ? atoi(argv[13]) : 4;
    uint32_t gmm1MTile = argc > 14 ? atoi(argv[14]) : M0;
    if (gmm1MTile != 128 && gmm1MTile != 256)
    {
        std::cerr << "[ERROR] GMM1 M tile must be 128 or 256, got " << gmm1MTile << std::endl;
        shmem_finalize();
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    int64_t tilingKey = 0;
    uint32_t maxOutputSize = m * topK * rankSize;

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    COCMatMulInfo info{int64_t(m), int64_t(k), int64_t(n)};
    cocTiling.m0 = gmm1MTile;
    cocTiling.n0 = N0;
    cocTiling.k0 = K0;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = rankSize;
    cocTiling.commDataSplit = 1;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    cocTiling.epSize = EP;
    cocTiling.expertNum = EP * expertPerRank;
    cocTiling.topK = topK;

    optiling::MoeInitRoutingV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t expertCapacity = 0;
    int64_t activeNum = cocTiling.m * cocTiling.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t inuptXDtypeSize = sizeof(ElementA);
    int64_t quantMode = -1;
    int64_t scaleDim0 = 0;
    uint32_t aivNum = launchAivNum;
    int64_t ubSize = 196352;

    moeInitRoutingQuantV2TilingBase.DoTiling(m, k, topK, expertCapacity, cocTiling.expertNum, activeNum, dropPadMode,
                                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                             inuptXDtypeSize, quantMode, scaleDim0, aivNum, ubSize);

    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    uint64_t initRoutingTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    if (initRoutingTilingKey == optiling::TILING_KEY_HIGH_PERFORMANCE)
    {
        initRoutingTilingKey = optiling::TILING_KEY_DROPLESS_SORT_ONE_CORE;
    }

    MoeInitRoutingQuantV2Tiling moeTiling{moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData,
                                          initRoutingTilingKey};

    printf("tiling key: %lu\n", initRoutingTilingKey);

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950DispatchFFNCombine");
    if (!op)
    {
        std::cout << "Operator Ascend950DispatchFFNCombine not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, "./output");

    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    size_t workSpaceSize = op->GetWorkspaceSize(cocTiling);
    size_t expandedRowIdxSize = (cocTiling.m + 255) / 256 * 256 * cocTiling.topK * sizeof(int32_t);
    auto alignBytes = [](size_t bytes) { return (bytes + 511) / 512 * 512; };
    size_t expandedM = static_cast<size_t>(cocTiling.m) * cocTiling.topK;
    size_t symmetricElementBytes = std::max(sizeof(ElementA), sizeof(ElementC));
    size_t symmetricABytes = alignBytes(expandedM * cocTiling.k * symmetricElementBytes);
    uint8_t *tokensPerExpertDevice = symmetricPtr + symmetricABytes;
    size_t tokensPerExpertBytes =
        alignBytes(static_cast<size_t>(cocTiling.epSize) * cocTiling.epSize * expertPerRank * sizeof(int32_t));

    uint8_t *workspaceDevice{nullptr};
    if (workSpaceSize > 0)
    {
        ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *b2Ptr = kernelParams.customPtrs[0];
    uint8_t *expertIdxPtr = kernelParams.customPtrs[1];
    uint8_t *probsPtr = kernelParams.customPtrs[2];
    uint8_t *bScalePtr = kernelParams.customPtrs[3];
    uint8_t *b2ScalePtr = kernelParams.customPtrs[4];

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(tokensPerExpertDevice, tokensPerExpertBytes, 0, tokensPerExpertBytes));
    aclshmem_barrier_all();
    aclshmemx_barrier_all_on_stream(stream);

    std::cout << "[TEST] before DispatchFFNCombine kernel rankId: " << rankId << std::endl;
    DispatchFFNCombine<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
        <<<blockNum, nullptr, stream>>>(fftsAddr, aPtr, bPtr, bScalePtr, b2Ptr, b2ScalePtr, cPtr, expertIdxPtr,
                                        probsPtr, workspaceDevice, symmetricPtr, cocTiling, moeTiling);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "[TEST] after DispatchFFNCombine kernel rankId: " << rankId << std::endl;

    op->WriteResultFile(kernelParams, cocTiling, rankId, "./output");

    uint8_t *tokensPerExpertHost;
    int32_t size = cocTiling.epSize * cocTiling.epSize * expertPerRank;
    ACL_CHECK(aclrtMallocHost((void **)(&tokensPerExpertHost), size * sizeof(int32_t)));
    ACL_CHECK(aclrtMemcpy(tokensPerExpertHost, size * sizeof(int32_t), tokensPerExpertDevice, size * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    uint8_t *expandedRowIdxDevice = workspaceDevice;
    uint8_t *expandedRowIdxHost;
    ACL_CHECK(aclrtMallocHost((void **)(&expandedRowIdxHost), expandedRowIdxSize));
    ACL_CHECK(aclrtMemcpy(expandedRowIdxHost, expandedRowIdxSize, expandedRowIdxDevice, expandedRowIdxSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtFreeHost(tokensPerExpertHost));
    ACL_CHECK(aclrtFreeHost(expandedRowIdxHost));

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);
    if (workSpaceSize > 0)
    {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
