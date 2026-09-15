/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#include <cstddef>
#include <stdexcept>

#include "catlass/gemm_coord.hpp"
#include "matmul_reduce_scatter_shallow_fusion_tiling.h"
#include "operator_registry.h"

namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample {
using Catlass::GemmCoord;
constexpr size_t DATA_REGION_BYTES = 1024UL * 1024 * 1024;
struct RuntimeTiling {
    GemmCoord shape;
    GemmCoord tile{128, 256, 256};
    bool paddingA = false, paddingB = false;
    size_t aBytes, bBytes, cBytes, waBytes = 0, wbBytes = 0, wcBytes = 0, flagBytes = 0;
    explicit RuntimeTiling(const CocTilingParams& t) : shape(t.m, t.n, t.k)
    {
        aBytes = size_t(t.m) * t.k * sizeof(fp16_t);
        bBytes = size_t(t.k) * t.n * sizeof(fp16_t);
        cBytes = size_t(t.m) * t.n * sizeof(fp16_t);
        auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        if (t.transB)
            DoTilingB16Layout01(shape, tile, cores);
        else
            DoTilingB16Layout00(shape, tile, cores);
        paddingA = paddingB = true;
        waBytes = size_t(RoundUp(t.m, tile.m())) * RoundUp(t.k, tile.k()) * sizeof(fp16_t);
        wbBytes = size_t(RoundUp(t.k, tile.k())) * RoundUp(t.n, tile.n()) * sizeof(fp16_t);
        size_t chunks = CeilDiv(t.n, tile.n()); // Kernel N-chunk count for nBlocksPerChunk == 0.
        flagBytes = std::max(size_t(1024 * 1024), (size_t(t.rankSize) * t.rankSize * chunks + t.rankSize) * 512) *
                    sizeof(int32_t);
    }
    size_t WorkspaceSize() const { return waBytes + wbBytes; }
};
} // namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample

class MatmulReduceScatterShallowFusionOperator : public CatccosOperator {
public:
    using RuntimeTiling = Catccos::Examples::MatmulReduceScatterShallowFusionExample::RuntimeTiling;
    bool CheckCocTilingParams(uint32_t ranks, const CocTilingParams& t) override
    {
        if (!t.m || !t.n || !t.k || ranks < 2 || ranks > 16 || t.transA || t.transB > 1)
            return false;
        if (t.m % ranks)
            return false;
        // The full-sync ReduceScatterTile path keeps a complete N row in a 64 KiB buffer.
        if (t.n > 64 * 1024 / sizeof(fp16_t)) {
            std::cerr << "ReduceScatter requires N <= 32768 (64 KiB communication buffer)\n";
            return false;
        }
        return size_t(t.m) * t.n * sizeof(fp16_t) <
               Catccos::Examples::MatmulReduceScatterShallowFusionExample::DATA_REGION_BYTES;
    }
    size_t GetWorkspaceSize(const CocTilingParams& t) override { return RuntimeTiling(t).WorkspaceSize(); }
    size_t GetSymmetricSize(const CocTilingParams& t) override
    {
        // SHMEM VMM heap reservations require a multiple of the 2 MiB page size.
        return RoundUp(
            Catccos::Examples::MatmulReduceScatterShallowFusionExample::DATA_REGION_BYTES + RuntimeTiling(t).flagBytes,
            size_t(2 * 1024 * 1024));
    }
    CocCommType GetActualKernelType(const CocTilingParams&) override { return MATMUL_REDUCE_SCATTER_SHALLOW_FUSION; }
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& t, uint32_t rank, std::string dataFile) override
    {
        RuntimeTiling p(t);
        if (rank == 0) {
            std::cout << "Tiling M1=" << p.tile.m() << " N1=" << p.tile.n() << " K1=" << p.tile.k()
                      << " paddingA=" << p.paddingA << " paddingB=" << p.paddingB << std::endl;
        }
        std::vector<fp16_t> a(p.aBytes / sizeof(fp16_t)), b(p.bBytes / sizeof(fp16_t));
        if (!ReadFile(dataFile + "/rank_" + std::to_string(rank) + "_a.bin", a.data(), p.aBytes) ||
            !ReadFile(dataFile + "/rank_" + std::to_string(rank) + "_b.bin", b.data(), p.bBytes))
            throw std::runtime_error("Unable to load rank input");
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrA), p.aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrB), p.bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrC), p.cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(params.ptrA, p.aBytes, a.data(), p.aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(params.ptrB, p.bBytes, b.data(), p.bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& t, uint32_t rank, std::string dataFile) override
    {
        RuntimeTiling p(t);
        size_t bytes = p.cBytes / t.rankSize;
        std::vector<uint8_t> out(bytes);
        ACL_CHECK(aclrtMemcpy(out.data(), bytes, params.ptrC, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", out.data(), bytes, size_t(rank) * bytes);
    }
};
REGISTER_OPERATOR("MatmulReduceScatterShallowFusion", MatmulReduceScatterShallowFusionOperator);
