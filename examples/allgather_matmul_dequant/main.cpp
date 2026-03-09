/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "allgather_matmul_dequant_host.h"
#include "allgather_matmul_dequant_device.h"
#include "allgather_matmul_dequant_padding_device.h"

using namespace AscendC;
using namespace Catccos;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementD = half;
using ElementScale = uint64_t;
 
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul_dequant rank_size rank_id ip_port m n k [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath() const
    {
        return dataPath;
    }
};

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    COCMatMulInfo info{ int64_t(m), int64_t(k), int64_t(n) };
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 16;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto op = OperatorRegistry::Instance().CreateOperator("AllGatherMatmulDequant");
    if (!op) {
        std::cout << "Operator AllGatherMatmulDequant not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    size_t workSpaceSize = op->GetWorkspaceSize(cocTiling);
    uint8_t *workspaceDevice{nullptr};
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *dPtr = kernelParams.ptrC;
    uint8_t *scalePtr = kernelParams.customPtrs[0];

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM_DQ kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        uint64_t fftsAddr = shmemx_get_ffts_config();
        if (workSpaceSize > 0) {
            AllGatherMatmulDequantPadding<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementScale, LayoutScale>
                <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, dPtr, scalePtr, workspaceDevice, gmSymmetric, cocTiling);
        } else {
            AllGatherMatmulDequant<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementScale, LayoutScale>
                <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, dPtr, scalePtr, workspaceDevice, gmSymmetric, cocTiling);
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM_DQ kernel " << std::endl;

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    FreeDeviceSpace(kernelParams);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}