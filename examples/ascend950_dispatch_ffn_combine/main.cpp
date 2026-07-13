/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <vector>

#include "ascend950_dispatch_ffn_combine_device.h"
#include "ascend950_dispatch_ffn_combine_host.h"

using namespace AscendC;
using namespace Catccos;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;
 
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

static uint32_t gNpuNum = 8;

std::vector<int32_t> ParseDeviceIdList(char *deviceIdList)
{
    std::vector<int32_t> result;
    if (deviceIdList == nullptr) {
        return result;
    }

    for (char *idToken = std::strtok(deviceIdList, ","); idToken; idToken = std::strtok(nullptr, ",")) {
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

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId
              << " device_id:" << deviceId << " input_ip: " << ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
 	aclshmemx_uniqueid_t default_flag_uid;
 	set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
 	status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    std::cout << "[TEST] after shmem init rankId: " << rankId << " status: " << status << std::endl;

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
    int64_t tilingKey = 0;
    uint32_t maxOutputSize = m * topK * rankSize;

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    COCMatMulInfo info{ int64_t(m), int64_t(k), int64_t(n) };
    cocTiling.m0 = M0;
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
    uint32_t aivNum = 2 * BLOCK_NUM;
    int64_t ubSize = 196352;

    moeInitRoutingQuantV2TilingBase.DoTiling(m, k, topK, expertCapacity, cocTiling.expertNum, activeNum, dropPadMode,
                                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                             inuptXDtypeSize, quantMode, scaleDim0, aivNum, ubSize);

    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    uint64_t initRoutingTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    if (initRoutingTilingKey == optiling::TILING_KEY_HIGH_PERFORMANCE) {
        initRoutingTilingKey = optiling::TILING_KEY_DROPLESS_SORT_ONE_CORE;
    }

    MoeInitRoutingQuantV2Tiling moeTiling{
        moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData,
        initRoutingTilingKey
    }; 

    printf("tiling key: %lu\n", initRoutingTilingKey);

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950DispatchFFNCombine");
    if (!op) {
        std::cout << "Operator Ascend950DispatchFFNCombine not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, "./output");

    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    size_t workSpaceSize = op->GetWorkspaceSize(cocTiling);
    size_t expandedRowIdxSize = (cocTiling.m + 255) / 256 * 256 * cocTiling.topK * sizeof(int32_t);

    uint8_t *workspaceDevice{nullptr};
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *b2Ptr = kernelParams.customPtrs[0];
    uint8_t *expertIdxPtr = kernelParams.customPtrs[1];
    uint8_t *probsPtr = kernelParams.customPtrs[2];

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "[TEST] before DispatchFFNCombine kernel rankId: " << rankId << std::endl;

    for (int i = 0; i < 1; i++) {
        DispatchFFNCombine<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(
                fftsAddr,
                aPtr,
                bPtr,
                b2Ptr,
                cPtr,
                expertIdxPtr,
                probsPtr,
                workspaceDevice,
                symmetricPtr,
                cocTiling,
                moeTiling
            );
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "[TEST] after DispatchFFNCombine kernel rankId: " << rankId << std::endl;

    op->WriteResultFile(kernelParams, cocTiling, rankId, "./output");

    uint8_t *tokensPerExpertDevice = symmetricPtr + cocTiling.m * cocTiling.topK * cocTiling.k * sizeof(ElementA);
    uint8_t *tokensPerExpertHost;
    int32_t size = cocTiling.epSize * cocTiling.epSize * expertPerRank;
    ACL_CHECK(aclrtMallocHost((void **)(&tokensPerExpertHost), size * sizeof(int32_t)));
    ACL_CHECK(aclrtMemcpy(tokensPerExpertHost, size * sizeof(int32_t), tokensPerExpertDevice, size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    uint8_t *expandedRowIdxDevice = workspaceDevice;
    uint8_t *expandedRowIdxHost;
    ACL_CHECK(aclrtMallocHost((void **)(&expandedRowIdxHost), expandedRowIdxSize));
    ACL_CHECK(aclrtMemcpy(expandedRowIdxHost, expandedRowIdxSize, expandedRowIdxDevice, expandedRowIdxSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtFreeHost(tokensPerExpertHost));

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
