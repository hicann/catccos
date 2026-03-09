#ifndef MATMUL_ALLREDUCE_HOST_H
#define MATMUL_ALLREDUCE_HOST_H

#include "operator_registry.h"

class MatmulAllReduceOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(__fp16);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(__fp16);
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixA(cocTiling.m * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(cocTiling.k * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *dDevice;
        ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *dHost;
        ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
        memset(dHost, 0, dSize);
        ACL_CHECK(aclrtMemcpy(dDevice, dSize, dHost, dSize, ACL_MEMCPY_HOST_TO_DEVICE));
        
        params.SetKernelParams(aDevice, bDevice, dDevice);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
        }

        ACL_CHECK(aclrtFreeHost(dHost));

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(__fp16);

        uint8_t *dDevice = params.ptrC;
        uint8_t *dHost;
        ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
        ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", dHost, dSize);

        ACL_CHECK(aclrtFreeHost(dHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        return 0;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::MATMUL_ALLREDUCE;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoops = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * BLOCK_NUM >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("MatmulAllReduce", MatmulAllReduceOperator);

#endif // MATMUL_ALLREDUCE_HOST_H