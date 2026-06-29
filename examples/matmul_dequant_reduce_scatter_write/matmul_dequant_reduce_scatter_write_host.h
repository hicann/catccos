#ifndef MATMUL_DEQUANT_REDUCE_SCATTER_WRITE_HOST_H
#define MATMUL_DEQUANT_REDUCE_SCATTER_WRITE_HOST_H

#include "operator_registry.h"

class MatmulDequantReduceScatterWriteOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
            uint32_t rankId, std::string dataFile) override {
        // Memory sizes
        size_t x1Size = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(int8_t);
        size_t x2Size = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(int8_t);
        size_t scaleX1Size = static_cast<size_t>(cocTiling.m) * sizeof(float);
        size_t scaleX2Size = static_cast<size_t>(cocTiling.n) * sizeof(float);
        size_t biasSize = static_cast<size_t>(cocTiling.n) * sizeof(int32_t);
        size_t dOutSize = static_cast<size_t>(cocTiling.m) * cocTiling.n / cocTiling.rankSize * sizeof(half);

        // Allocate and copy x1
        uint8_t *x1Device, *x1Host;
        ACL_CHECK(aclrtMalloc((void **)(&x1Device), x1Size, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&x1Host), x1Size));
            std::string x1_filename = "/x1_gm_rank" + std::to_string(rankId) + ".bin";
            ReadFile(dataFile + x1_filename, x1Host, x1Size);
            ACL_CHECK(aclrtMemcpy(x1Device, x1Size, x1Host, x1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixX1(cocTiling.m * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(x1Device, x1Size, matrixX1.data(), x1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and copy x2
        uint8_t *x2Device, *x2Host;
        ACL_CHECK(aclrtMalloc((void **)(&x2Device), x2Size, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&x2Host), x2Size));
            std::string x2_filename = "/x2_gm_rank" + std::to_string(rankId) + ".bin";
            ReadFile(dataFile + x2_filename, x2Host, x2Size);
            ACL_CHECK(aclrtMemcpy(x2Device, x2Size, x2Host, x2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixX2(cocTiling.k * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(x2Device, x2Size, matrixX2.data(), x2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and copy scale_x1
        uint8_t *scaleX1Device, *scaleX1Host;
        ACL_CHECK(aclrtMalloc((void **)(&scaleX1Device), scaleX1Size, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&scaleX1Host), scaleX1Size));
            ReadFile(dataFile + "/scale_x1_gm.bin", scaleX1Host, scaleX1Size);
            ACL_CHECK(aclrtMemcpy(scaleX1Device, scaleX1Size, scaleX1Host, scaleX1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<float> matrixScaleX1(cocTiling.m, 1);
            ACL_CHECK(aclrtMemcpy(scaleX1Device, scaleX1Size, matrixScaleX1.data(), scaleX1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and copy scale_x2
        uint8_t *scaleX2Device, *scaleX2Host;
        ACL_CHECK(aclrtMalloc((void **)(&scaleX2Device), scaleX2Size, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&scaleX2Host), scaleX2Size));
            ReadFile(dataFile + "/scale_x2_gm.bin", scaleX2Host, scaleX2Size);
            ACL_CHECK(aclrtMemcpy(scaleX2Device, scaleX2Size, scaleX2Host, scaleX2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<float> matrixScaleX2(cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(scaleX2Device, scaleX2Size, matrixScaleX2.data(), scaleX2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and copy bias
        uint8_t *biasDevice = nullptr;
        uint8_t *biasHost = nullptr;
        if (rankId == 0 && dataFile != "") {
            ACL_CHECK(aclrtMalloc((void **)(&biasDevice), biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
            ACL_CHECK(aclrtMallocHost((void **)(&biasHost), biasSize));
            ReadFile(dataFile + "/bias_gm.bin", biasHost, biasSize);
            ACL_CHECK(aclrtMemcpy(biasDevice, biasSize, biasHost, biasSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate intermediate and final output buffers
        uint8_t *dOutDevice, *dOutHost;
        ACL_CHECK(aclrtMalloc((void **)(&dOutDevice), dOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(dOutDevice, dOutSize, 0, dOutSize));
        ACL_CHECK(aclrtMallocHost((void **)(&dOutHost), dOutSize));

        params.SetKernelParams(x1Device, x2Device, dOutDevice, scaleX1Device, scaleX2Device, biasDevice);

        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(x1Host));
            ACL_CHECK(aclrtFreeHost(x2Host));
            ACL_CHECK(aclrtFreeHost(scaleX1Host));
            ACL_CHECK(aclrtFreeHost(scaleX2Host));
        }

        ACL_CHECK(aclrtFreeHost(dOutHost));
        if (rankId == 0 && dataFile != "") {
            ACL_CHECK(aclrtFreeHost(biasHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t dOutSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(uint16_t) / cocTiling.rankSize;

        uint8_t *cDevice = params.ptrC;
        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), dOutSize));
        ACL_CHECK(aclrtMemcpy(cHost, dOutSize, cDevice, dOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, dOutSize, rankId * dOutSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        uint32_t blockPerCommInRank = blockNum * cocTiling.commInterval / cocTiling.rankSize;
        return static_cast<size_t>(WORKSPACE_STAGES) * blockPerCommInRank *
            cocTiling.m0 * cocTiling.n0 * cocTiling.rankSize * sizeof(int32_t);
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::MATMUL_DEQUANT_REDUCE_SCATTER_WRITE;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams &cocTiling) override {
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;

        if (product % rankSize != 0) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("MatmulDequantReduceScatterWrite", MatmulDequantReduceScatterWriteOperator);

#endif // MATMUL_DEQUANT_REDUCE_SCATTER_WRITE_HOST_H
