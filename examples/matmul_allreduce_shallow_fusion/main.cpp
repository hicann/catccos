/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#include <sstream>

#include "matmul_allreduce_shallow_fusion_device.h"
#include "matmul_allreduce_shallow_fusion_host.h"

struct Options {
    uint32_t rankSize, rankId;
    CocTilingParams tiling;
    std::string ipPort, dataPath;
    std::vector<int> devices;
    int Parse(int argc, char** argv)
    {
        if (argc < 8 || argc > 10)
            return -1;
        rankSize = std::stoul(argv[1]);
        rankId = std::stoul(argv[2]);
        ipPort = argv[3];
        tiling.m = std::stoul(argv[4]);
        tiling.n = std::stoul(argv[5]);
        tiling.k = std::stoul(argv[6]);
        dataPath = argv[7];
        if (argc > 8) {
            std::stringstream ids(argv[8]);
            std::string id;
            while (std::getline(ids, id, ','))
                devices.push_back(std::stoi(id));
        } else
            for (uint32_t i = 0; i < rankSize; ++i)
                devices.push_back(i);
        if (argc > 9)
            tiling.transB = std::stoul(argv[9]);
        tiling.rankSize = tiling.epSize = rankSize;
        return rankId < rankSize && devices.size() == rankSize && !dataPath.empty() ? 0 : -1;
    }
};

int main(int argc, char** argv)
{
    Options options;
    try {
        if (options.Parse(argc, argv) != 0)
            throw std::runtime_error("Invalid arguments");
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nUsage: " << argv[0]
                  << " rank_size rank_id ip_port m n k data_path [device_id_list "
                     "[transB]]\n";
        return 1;
    }
    auto op = OperatorRegistry::Instance().CreateOperator("MatmulAllReduceShallowFusion");
    auto& t = options.tiling;
    if (!op || !op->CheckCocTilingParams(options.rankSize, t))
        return 1;
    aclrtStream stream = nullptr;
    int device = options.devices[options.rankId];
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes{};
    aclshmemx_uniqueid_t uid{};
    int status = set_attr(
        options.rankId, options.rankSize, op->GetSymmetricSize(t) + 20UL * 1024 * 1024, options.ipPort.c_str(),
        &attributes, &uid);
    if (!status)
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status)
        return status;
    int result = 0;
    try {
        KernelParams params;
        op->AllocateDeviceSpace(params, t, options.rankId, options.dataPath);
        uint8_t* workspace = nullptr;
        size_t workspaceBytes = op->GetWorkspaceSize(t);
        if (workspaceBytes)
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&workspace), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        auto* symmetric = static_cast<uint8_t*>(aclshmem_calloc(1, op->GetSymmetricSize(t)));
        if (!symmetric)
            throw std::runtime_error("aclshmem_calloc symmetric workspace failed");
        auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        uint64_t ffts = util_get_ffts_config();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        Catccos::Examples::MatmulAllReduceShallowFusionExample::Launch(
            stream, cores, ffts, params, workspace, symmetric, t, t.transA, t.transB);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        op->WriteResultFile(params, t, options.rankId, options.dataPath);
        aclshmem_free(symmetric);
        if (workspace)
            ACL_CHECK(aclrtFree(workspace));
        FreeDeviceSpace(params);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        result = 1;
    }
    aclshmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device));
    ACL_CHECK(aclFinalize());
    return result;
}
