#ifndef ALLGATHER_MATMUL_DEQUANT_HOST_H
#define ALLGATHER_MATMUL_DEQUANT_HOST_H

#include "operator_registry.h"
#include "padding.h"

class AllGatherMatmulDequantOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(int8_t);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(int8_t);
        size_t scaleSize = static_cast<size_t>(cocTiling.n) * sizeof(uint64_t);
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(half);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(dataFile + "/a_gm_rank" + std::to_string(rankId) + ".bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixA(cocTiling.m * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(dataFile + "/b_gm_rank" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixB(cocTiling.k * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *dDevice;
        ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *deviceScale;
        ACL_CHECK(aclrtMalloc((void **)(&deviceScale), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *scaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleSize));
            ReadFile(dataFile + "/scale_gm.bin", scaleHost, scaleSize);
            ACL_CHECK(aclrtMemcpy(deviceScale, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<uint64_t> matrixScale(cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(deviceScale, scaleSize, matrixScale.data(), scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        params.SetKernelParams(aDevice, bDevice, dDevice, deviceScale);
        
        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(scaleHost));
        }

        return;
    }

    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override {
        size_t dSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(half);

        uint8_t *dDevice = params.ptrC;
        uint8_t *dHost;
        ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
        ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
        std::string output_filename = "/output_rank" + std::to_string(rankId) + ".bin";
        WriteFile(dataFile + output_filename, dHost, dSize);

        ACL_CHECK(aclrtFreeHost(dHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override {
        constexpr uint32_t alignByByte = 512;
        constexpr uint32_t alignByElement = alignByByte / sizeof(int8_t);

        bool isNeedPaddingB = IsNeedPadding(cocTiling.k, cocTiling.n, cocTiling.transB, alignByElement);
        size_t sizeWB = 0;
        if (isNeedPaddingB) {
            if (cocTiling.m0 == 128) {
                sizeWB = GetWorkspaceLen(cocTiling.k, cocTiling.n, TILE_SHAPE_512, TILE_SHAPE_256) * sizeof(int8_t);
            } else {
                sizeWB = GetWorkspaceLen(cocTiling.k, cocTiling.n, TILE_SHAPE_512, TILE_SHAPE_128) * sizeof(int8_t);
            }
        }

        return sizeWB;
    }

    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        constexpr uint32_t alignByByte = 512;
        constexpr uint32_t alignByElement = alignByByte / sizeof(int8_t);

        bool isNeedPaddingB = IsNeedPadding(cocTiling.k, cocTiling.n, cocTiling.transB, alignByElement);
        if (isNeedPaddingB) {
            return CocCommType::ALLGATHER_MATMUL_DEQUANT_PADDING;
        }
        return CocCommType::ALLGATHER_MATMUL_DEQUANT;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams &cocTiling) override {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoop = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoop >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("AllGatherMatmulDequant", AllGatherMatmulDequantOperator);

#endif // ALLGATHER_MATMUL_DEQUANT_HOST_H