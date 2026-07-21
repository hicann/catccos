/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <tiling/platform/platform_ascendc.h>

#include "allgather_matmul_w4a4_host.h"
#include "allgather_matmul_w4a4_device.h"
#include "catlass/arch/arch.hpp"

using namespace AscendC;
using namespace Catccos;

using ElementA = int4b_t;
using ElementB = int4b_t;
using ElementC = half;
using ElementD = bfloat16_t;
using ElementScale = uint64_t;
using ElementPerTokenScale = float;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::zN;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;
using LayoutPerTokenScale = Catlass::layout::VectorLayout;

using Config = AllGatherMatmulW4A4Config_M0_128<
    ElementA, LayoutA, ElementB, LayoutB,
    ElementC, LayoutC, ElementD, LayoutD,
    ElementScale, LayoutScale, ElementPerTokenScale, LayoutPerTokenScale>;

struct Options {
    static constexpr auto HELPER =
        "Usage: allgather_matmul_w4a4 rank_size rank_id ip_port m n k data_path [device_id_list]\n";

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
            for (size_t i = 0; i < static_cast<size_t>(rankSize); ++i) {
                deviceIdList.push_back(static_cast<int>(i));
            }
        }
        return 0;
    }

    std::string GetDataPath() const { return dataPath; }
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
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[static_cast<size_t>(rankId)];

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 16;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    uint32_t blockNum =
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, options.ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto op = OperatorRegistry::Instance().CreateOperator("AllGatherMatmulW4A4");
    if (!op) {
        std::cerr << "Operator AllGatherMatmulW4A4 not found!" << std::endl;
        return -1;
    }

    if (!op->CheckCocTilingParams(rankSize, cocTiling)) {
        std::cerr << "Invalid tiling params for symmetric buffer" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t *gmSymmetric = static_cast<uint8_t *>(symmPtr);

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / WORKSPACE_STAGES};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / WORKSPACE_STAGES, cocTiling.k0};

    ACL_CHECK(aclrtSynchronizeStream(stream));

    uint64_t fftsAddr = shmemx_get_ffts_config();

    using DeviceOp = Config::Device;
    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        kernelParams.ptrA,
        kernelParams.ptrB,
        kernelParams.customPtrs[0],
        kernelParams.ptrC,
        kernelParams.customPtrs[1],
        kernelParams.customPtrs[2],
        gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    deviceOp.Run(stream, blockNum, fftsAddr);

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());

    shmem_free(symmPtr);
    shmem_free(kernelParams.ptrA);
    kernelParams.ptrA = nullptr;
    FreeDeviceSpace(kernelParams);
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
