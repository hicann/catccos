/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "allgather_matmul_with_gather_result_device.h"
#include "allgather_matmul_with_gather_result_host.h"

using namespace AscendC;
using namespace Catccos;

using ElementA = half;
using ElementB = half;
using ElementC = half;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using Config = AllGatherMatmulWithGatherResultConfig_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
using DeviceOp = Config::Device;

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size rank_id ip_port m n k data_path [device_id_list]\n";

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
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1, RANK_ID_INDEX, IP_PORT_INDEX,
            M_INDEX, N_INDEX, K_INDEX, DATA_PATH_INDEX,
            DEVICE_LIST_INDEX, INDEX_MAX
        };
        if (argc > static_cast<int>(ArgsIndex::INDEX_MAX)) { printf(HELPER); return -1; }
        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char *s = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char *t = std::strtok(s, ","); t; t = std::strtok(nullptr, ",")) deviceIdList.push_back(std::atoi(t));
        } else { for (size_t i = 0; i < rankSize; ++i) deviceIdList.push_back(i); }
        return 0;
    }
    std::string GetDataPath() const { return dataPath; }
};

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) { std::cerr << "Invalid arguments\n"; return 1; }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    uint32_t m = options.m, n = options.n, k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    CocTilingParams cocTiling;
    cocTiling.m = m; cocTiling.n = n; cocTiling.k = k;
    cocTiling.m0 = 128; cocTiling.n0 = 256; cocTiling.k0 = 256;
    cocTiling.commTileM = 64; cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1; cocTiling.commDataSplit = 16;
    cocTiling.commBlockM = 64; cocTiling.rankSize = rankSize;

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

    auto op = OperatorRegistry::Instance().CreateOperator("AllGatherMatmulWithGatherResult");
    if (!op) { std::cout << "Operator not found!" << std::endl; return -1; }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    // Construct DeviceDGemm Arguments
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, RoundUp(k, cocTiling.k0)};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.k0};

    uint32_t commCoreNum = cocTiling.commDataSplit * cocTiling.commNpuSplit;
    uint32_t copyCoreNum = blockNum - commCoreNum;
    uint32_t copyBlockM = CeilDiv((cocTiling.commInterval * cocTiling.m0), CeilDiv(copyCoreNum, (uint32_t)rankSize));
    uint32_t copyBlockN = RoundUp<32 / sizeof(ElementA)>(k);
    Catlass::MatrixCoord copyGatherABlockShape{copyBlockM, copyBlockN};

    constexpr uint32_t UB_STAGES_VAL = 2;
    uint32_t maxCopyLength = Catlass::Arch::AtlasA2::UB_SIZE / UB_STAGES_VAL / sizeof(ElementA);
    uint32_t copyTileN = Min<uint32_t>(maxCopyLength, copyBlockN);
    uint32_t copyTileM = maxCopyLength / copyTileN;
    Catlass::MatrixCoord copyGatherATileShape{copyTileM, copyTileN};

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC,
        kernelParams.customPtrs[0], gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape,
        copyGatherABlockShape, copyGatherATileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    uint64_t fftsAddr = shmemx_get_ffts_config();
    for (int i = 0; i < 1; i++) { deviceOp.Run(stream, blockNum, fftsAddr); }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    if (rankId == 0) std::printf("test finished\n");

    shmem_free(symmPtr);
    FreeDeviceSpace(kernelParams);
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
