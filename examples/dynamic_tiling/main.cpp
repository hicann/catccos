/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op.h"

struct Options {
    std::string kernelName{};
    CocDataType dataType;
    int rankSize;
    int rankId;
    std::string ipPort{};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::vector<int> deviceIdList{};
    uint32_t test_start_line{0};
    uint32_t test_collect_rows{0};
    uint32_t warmUpTimes{0};
    uint32_t testCycleTimes{0};
    std::string parentPath{};
    std::string csv_file{};
    std::string dataFile{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            KERNEL_NAME_INDEX = 1,
            DATA_TYPE_INDEX,
            RANK_SIZE_INDEX,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            START_LINE_INDEX,
            COLLECT_ROWS_INDEX,
            PARENT_PATH_INDEX,
            CSV_FILE_INDEX,
            WARM_UP_TIMES_INDEX,
            TEST_CYCLE_TIMES_INDEX,
            DEVICE_LIST_INDEX,
            DATA_FILE_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            return -1;
        }

        kernelName = argv[KERNEL_NAME_INDEX];
        dataType = static_cast<CocDataType>(std::atoi(argv[DATA_TYPE_INDEX]));
        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        test_start_line = std::atoi(argv[START_LINE_INDEX]);
        test_collect_rows = std::atoi(argv[COLLECT_ROWS_INDEX]);
        parentPath = argv[PARENT_PATH_INDEX];
        csv_file = argv[CSV_FILE_INDEX];
        warmUpTimes = std::atoi(argv[WARM_UP_TIMES_INDEX]);
        testCycleTimes = std::atoi(argv[TEST_CYCLE_TIMES_INDEX]);
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
        if (argc > DATA_FILE_INDEX) {
            dataFile = argv[DATA_FILE_INDEX];
        }
        return 0;
    }
};

std::vector<std::vector<uint32_t>> InitTestShapes(const Options &options)
{
    uint32_t startLine = options.test_start_line;
    uint32_t collectRows = options.test_collect_rows;
    std::string shapeFileName = options.csv_file;
    std::vector<std::string> headers = {};
    std::vector<std::vector<uint32_t>> shapes = {};
    std::ifstream file(shapeFileName);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << shapeFileName << std::endl;
        return shapes;
    }

    std::string line;

    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        while (getline(ss, header, ',')) {
            headers.push_back(header);
        }
    } else {
        std::cerr << "The file is empty or the header line fails to be read." << std::endl;
        return shapes;
    }

    int rowIndex = 0;
    int added = 0;

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (rowIndex < startLine) {
            ++rowIndex;
            continue;
        }
        if (added >= collectRows) {
            break;
        }

        std::stringstream ss(line);
        std::vector<uint32_t> shape;
        std::string cell;
        while (getline(ss, cell, ',')) {
            shape.push_back(std::stoi(cell));
        }

        if (shape.size() != headers.size()) {
            std::cerr << "The number of data columns in row " << rowIndex << " does not match the number of header columns: " << line << std::endl;
        } else {
            shapes.push_back(shape);
            ++added;
        }
        ++rowIndex;
    }
    file.close();

    return shapes;
}

std::string GetCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    options.Parse(argc, argv);
    CocCommType commType = GetCommType(options.kernelName);
    CocDataType dataType = options.dataType;
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    int epSize = rankSize;
    int expertNum = 8;
    uint32_t warmUpTimes = options.warmUpTimes;
    uint32_t testCycleTimes = options.testCycleTimes;
    std::string ipPort = options.ipPort;
    int32_t deviceId = options.deviceIdList[rankId];
    std::string dataFile = options.dataFile;
    const std::vector<std::vector<uint32_t>> shapes = InitTestShapes(options);

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id: " << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
#ifdef RDMA_TRANSPORT
    if (commType == ALLGATHER_MATMUL_RDMA) {
        attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    }
#endif

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    std::string currentTime = GetCurrentTime();
    std::string opName = CommTypeOpNameMap.at(commType);
    std::string currentDir = options.parentPath;
    std::string tilingFileName = currentDir + "/output/tiling/tilingData_" + currentTime + ".csv";
    if (rankId == 0) {
        CreateTilingFile(tilingFileName);
    }

    for (size_t i = 0; i < shapes.size(); i++) {
        uint32_t m = shapes[i][0];
        uint32_t k = shapes[i][1];
        uint32_t n = shapes[i][2];
        uint32_t transA = shapes[i][3];
        uint32_t transB = shapes[i][4];

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

        auto op = OperatorRegistry::Instance().CreateOperator(opName);
        if (op) {
            if (warmUpTimes == 0 && !op->CheckCocTilingParams(rankSize, cocTiling)) {
                std::printf("M: %d, K: %d, N: %d coc params check failed!\n", cocTiling.m, cocTiling.k, cocTiling.n);
                continue;
            }
        } else {
            std::cout << "Operator AllGatherMatmul not found!" << std::endl;
            return -1;
        }

        CocCommType actualKernelType = op->GetActualKernelType(cocTiling);

        KernelParams kernelParams;

        op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, dataFile);

        size_t workSpaceSize = op->GetWorkspaceSize(cocTiling);
        uint8_t *workspaceDevice{nullptr};
        if (workSpaceSize > 0) {
            ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
        uint8_t *gmSymmetric = (uint8_t *)symmPtr;

        uint32_t searchparams = (std::getenv("SEARCH_PARAMS") == nullptr) ? 1U : std::stoul(std::getenv("SEARCH_PARAMS"));

        std::vector<CocTilingParams> cocTilings;
        if (warmUpTimes == 0) {
            cocTilings.push_back(cocTiling);
        } else {
            if (searchparams == 1) {
                // 搜索 tiling
                GetTilings(cocTilings, cocTiling, opName, rankSize);
            } else {
                bool ok = ApplyLookupTable(info, actualKernelType, rankSize, cocTiling);
                if (!ok) {
                    std::cerr << "[LUT] no table for (" << opName << "," << rankSize << "), using defaults\n";
                }
                cocTilings.push_back(cocTiling);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        auto kernelFunc = KernelDispatcher::GetKernelFunc(actualKernelType, dataType);

        for (size_t i = 0; i < warmUpTimes; i++) {
            kernelFunc(stream, fftsAddr, kernelParams, workspaceDevice, gmSymmetric, cocTilings[0], transA, transB);
        }

        for (CocTilingParams tiling : cocTilings) {
            for (size_t i = 0; i < testCycleTimes; i++) {
                kernelFunc(stream, fftsAddr, kernelParams, workspaceDevice, gmSymmetric, tiling, transA, transB);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        if (dataFile != "") {
            op->WriteResultFile(kernelParams, cocTiling, rankId, dataFile);
        }

        if (rankId == 0) {
            WriteTilingInfos(opName, cocTilings, tilingFileName, transA, transB);
            std::printf("M: %d, K: %d, N: %d aclrtSynchronizeStream success!\n", cocTiling.m, cocTiling.k, cocTiling.n);
        }

        FreeDeviceSpace(kernelParams);

        if (workSpaceSize > 0) {
            ACL_CHECK(aclrtFree(workspaceDevice));
        }
        shmem_free(symmPtr);
    }
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = aclshmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}