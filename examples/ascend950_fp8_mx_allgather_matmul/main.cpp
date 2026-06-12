/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_fp8_mx_allgather_matmul_host.h"
#include "ascend950_fp8_mx_allgather_matmul_device.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutMxScaleA = Catlass::layout::RowMajor;
using LayoutMxScaleB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = float8_e4m3_t;
using ElementB = float8_e4m3_t;
using ElementMxScaleA = float8_e8m0_t;
using ElementMxScaleB = float8_e8m0_t;
using ElementC = half;

using Config = Ascend950Fp8MxAllGatherMatmulConfig_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB, ElementC, LayoutC>;
using DeviceOp = Config::Device;

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size rank_id ip_port m n k [device_id_list]\n";

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

        if (argc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char *idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];   
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
    cocTiling.k0 = 512;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 2;
    cocTiling.commDataSplit = 16;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    if (cocTiling.commNpuSplit > cocTiling.rankSize) {
        std::cout << "[ERROR] CommNpuSplit must <= npu num!" << std::endl;
        return -1;
    }

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950Fp8MxAllGatherMatmul");
    if (!op) {
        std::cout << "Operator Ascend950Fp8MxAllGatherMatmul not found!" << std::endl;
        return -1;
    }
    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void *symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    // Construct DeviceDGemm Arguments
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};

    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *aMxScalePtr = kernelParams.customPtrs[0];
    uint8_t *bMxScalePtr = kernelParams.customPtrs[1];

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr,
        symmetricPtr,
        commCoreSplit, commBlockShape, commTileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling FP8_MX_AG_MM kernel " << std::endl;
    uint64_t fftsAddr = shmemx_get_ffts_config();
    for (int i = 0; i < 1; i++) {
        deviceOp.Run(stream, blockNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling FP8_MX_AG_MM kernel " << std::endl;

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}