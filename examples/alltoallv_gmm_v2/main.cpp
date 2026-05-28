/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "alltoallv_gmm_v2_device.h"
#include "alltoallv_gmm_v2_host.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using ElementA = half;
using ElementB = half;
using ElementC = half;

using Config = AllToAllVGMMV2Config_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
using DeviceOp = Config::Device;

struct Options {
    static constexpr auto helper = "Usage: alltoallv_gmm_v2 rank_size rank_id ip m n k expert_per_rank device_list\n";
    int rankSize; uint32_t rankId; std::string ipPort{};
    uint32_t m{0}; uint32_t n{0}; uint32_t k{0};
    uint32_t expertPerRank{0};
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv) {
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1, RANK_ID_INDEX, IP_PORT_INDEX,
            M_INDEX, N_INDEX, K_INDEX, EXPERT_PER_RANK_INDEX,
            DEVICE_LIST_INDEX, INDEX_MAX
        };
        if (argc > static_cast<int>(ArgsIndex::INDEX_MAX) || argc <= static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            printf(helper); return -1;
        }
        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        expertPerRank = std::atoi(argv[static_cast<int>(ArgsIndex::EXPERT_PER_RANK_INDEX)]);
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char *s = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char *t = std::strtok(s, ","); t; t = std::strtok(nullptr, ",")) deviceIdList.push_back(std::atoi(t));
        } else { for (int i = 0; i < rankSize; ++i) deviceIdList.push_back(i); }
        return 0;
    }
};

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) return 1;

    int rankSize = options.rankSize;
    uint32_t rankId = options.rankId;
    uint32_t m = options.m, n = options.n, k = options.k;
    uint32_t EP = rankSize;
    uint32_t expertPerRank = options.expertPerRank;
    int32_t deviceId = options.deviceIdList[rankId];

    std::cout << "[TEST] rank_size:" << rankSize << " rank_id:" << rankId << " ip:" << options.ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, options.ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    CocTilingParams cocTiling;
    cocTiling.m = m; cocTiling.n = n; cocTiling.k = k;
    cocTiling.m0 = M0; cocTiling.n0 = N0; cocTiling.k0 = K0;
    cocTiling.commTileM = 64; cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = rankSize; cocTiling.commDataSplit = 1;
    cocTiling.commBlockM = 64; cocTiling.rankSize = rankSize;
    cocTiling.epSize = EP; cocTiling.expertNum = EP * expertPerRank;

    auto op = OperatorRegistry::Instance().CreateOperator("AllToAllVGMMV2");
    if (!op) { std::cout << "Operator not found!" << std::endl; return -1; }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, "./output");
    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    size_t workSpaceSize = op->GetWorkspaceSize(cocTiling);
    uint8_t *workspaceDevice{nullptr};
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Construct DeviceDGemm Arguments
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};

    DeviceOp::Arguments args{
        problemShape,
        rankId, static_cast<uint32_t>(rankSize),
        EP, EP * expertPerRank,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
        kernelParams.customPtrs[0],
        workspaceDevice,
        symmetricPtr,
        commBlockShape, commTileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    uint64_t fftsAddr = shmemx_get_ffts_config();
    for (int i = 0; i < 1; i++) { deviceOp.Run(stream, blockNum, fftsAddr); }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    op->WriteResultFile(kernelParams, cocTiling, rankId, "./output");
    std::printf("Rank %d test finished\n", rankId);

    shmem_free(symmPtr);
    FreeDeviceSpace(kernelParams);
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
