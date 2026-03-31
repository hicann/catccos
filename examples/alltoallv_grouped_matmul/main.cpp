/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "alltoallv_grouped_matmul_device.h"
#include "alltoallv_grouped_matmul_host.h"
 
using namespace AscendC;
using namespace Catccos;
 
using ElementA = half;
using ElementB = half;
using ElementC = half;
 
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;
 
struct Options {
    static constexpr auto helper = "Usage: matmul_allreduce m n k ep_size expert_num device_list\n";
 
    int rankSize;
    int rankId;
    std::string ipPort{};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t epSize{0};
    uint32_t expertNum{0};
    std::vector<int> deviceIdList{};
 
    int Parse(int argc, char **argv)
    {
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            EP_SIZE_INDEX,
            EXPERT_NUM_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };
 
        if (argc > static_cast<int>(ArgsIndex::INDEX_MAX) || argc <= static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            printf(helper);
            return -1;
        }
 
        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        epSize = std::atoi(argv[static_cast<int>(ArgsIndex::EP_SIZE_INDEX)]);
        expertNum = std::atoi(argv[static_cast<int>(ArgsIndex::EXPERT_NUM_INDEX)]);
 
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char *idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (int i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }
};
 
int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    uint32_t epSize = options.epSize;
    uint32_t expertNum = options.expertNum;
    int32_t deviceId = options.deviceIdList[rankId];

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
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 16;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    cocTiling.epSize = epSize;
    cocTiling.expertNum = expertNum;
 
    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";
 
    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto op = OperatorRegistry::Instance().CreateOperator("AllToAllVGroupedMatmul");
    if (!op) {
        std::cout << "Operator AllToAllVGroupedMatmul not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, "./output");
    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *localExpertPtr = kernelParams.customPtrs[0];
    uint8_t *globalExpertPtr = kernelParams.customPtrs[1];
 
    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; i++) {
        uint64_t fftsAddr = shmemx_get_ffts_config();
        AllToAllVGroupedMatmul<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
 
    op->WriteResultFile(kernelParams, cocTiling, rankId, "./output");
    std::printf("Rank %d test finished\n", rankId);
 
    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);
 
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
 
    return 0;
}