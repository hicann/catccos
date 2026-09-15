/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include <cstddef>
#include <stdexcept>

#include "catlass/gemm_coord.hpp"
#include "matmul_allreduce_shallow_fusion_tiling.h"
#include "operator_registry.h"

namespace Catccos::Examples::MatmulAllReduceShallowFusionExample {
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
        GetPaddingTag(shape, tile, false, t.transB != 0, cores, paddingA, paddingB);
        waBytes = paddingA ? size_t(RoundUp(t.m, 16U)) * RoundUp(t.k, 16U) * sizeof(fp16_t) : 0;
        wbBytes = paddingB ? size_t(RoundUp(t.k, 16U)) * RoundUp(t.n, 16U) * sizeof(fp16_t) : 0;
        flagBytes = size_t(3) * 2 * cores * t.rankSize * 512 * 100 * sizeof(int32_t);
    }
    size_t WorkspaceSize() const { return waBytes + wbBytes; }
};
} // namespace Catccos::Examples::MatmulAllReduceShallowFusionExample

class MatmulAllReduceShallowFusionOperator : public CatccosOperator {
public:
    using RuntimeTiling = Catccos::Examples::MatmulAllReduceShallowFusionExample::RuntimeTiling;
    bool CheckCocTilingParams(uint32_t ranks, const CocTilingParams& t) override
    {
        if (!t.m || !t.n || !t.k || ranks < 2 || ranks > 16 || t.transA || t.transB > 1)
            return false;
        if (ranks & (ranks - 1))
            return false;
        auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        // AllReduceSplitN divides a row among floor(cores / M) cores when M <= cores.
        auto split = std::max(1U, cores / t.m);
        size_t maxN = size_t(96 * 1024 / sizeof(fp16_t)) * split;
        if (t.n > maxN) {
            std::cerr << "AllReduce requires N <= " << maxN << " for M=" << t.m << " and " << cores << " AIC cores\n";
            return false;
        }
        return size_t(t.m) <=
               (Catccos::Examples::MatmulAllReduceShallowFusionExample::DATA_REGION_BYTES - 1) / sizeof(fp16_t) / t.n;
    }
    size_t GetWorkspaceSize(const CocTilingParams& t) override { return RuntimeTiling(t).WorkspaceSize(); }
    size_t GetSymmetricSize(const CocTilingParams& t) override
    {
        // SHMEM VMM heap reservations require a multiple of the 2 MiB page size.
        return RoundUp(
            Catccos::Examples::MatmulAllReduceShallowFusionExample::DATA_REGION_BYTES + RuntimeTiling(t).flagBytes,
            size_t(2 * 1024 * 1024));
    }
    CocCommType GetActualKernelType(const CocTilingParams&) override { return MATMUL_ALLREDUCE_SHALLOW_FUSION; }
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
        ACL_CHECK(aclrtMemset(params.ptrC, p.cBytes, 0, p.cBytes));
    }
    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& t, uint32_t rank, std::string dataFile) override
    {
        RuntimeTiling p(t);
        size_t bytes = p.cBytes;
        std::vector<uint8_t> out(bytes);
        ACL_CHECK(aclrtMemcpy(out.data(), bytes, params.ptrC, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/rank_" + std::to_string(rank) + "_output.bin", out.data(), bytes);
    }
};
REGISTER_OPERATOR("MatmulAllReduceShallowFusion", MatmulAllReduceShallowFusionOperator);
