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
#include "operator_registry.h"

namespace Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample {
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
        aBytes *= t.rankSize;
        bBytes *= t.expertNum / t.rankSize;
        wcBytes = cBytes * t.rankSize;
    }
    size_t WorkspaceSize() const { return waBytes + wbBytes; }
};
} // namespace Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample

class GroupedMatmulAlltoAllvSmallMOperator : public CatccosOperator {
public:
    using RuntimeTiling = Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample::RuntimeTiling;
    bool CheckCocTilingParams(uint32_t ranks, const CocTilingParams& t) override
    {
        if (!t.m || !t.n || !t.k || ranks < 2 || ranks > 16 || t.transA || t.transB > 1)
            return false;
        if (t.epSize != ranks || !t.expertNum || t.expertNum % ranks)
            return false;
        // Every destination rank has its own M-by-N slice in the symmetric data region.
        // Divide before comparing to avoid overflow for invalid large dimensions.
        return size_t(t.m) <= (Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample::DATA_REGION_BYTES - 1) /
                                  sizeof(fp16_t) / ranks / t.n;
    }
    size_t GetWorkspaceSize(const CocTilingParams& t) override { return RuntimeTiling(t).WorkspaceSize(); }
    size_t GetSymmetricSize(const CocTilingParams& t) override
    {
        return Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample::DATA_REGION_BYTES + RuntimeTiling(t).flagBytes;
    }
    CocCommType GetActualKernelType(const CocTilingParams&) override { return GROUPED_MATMUL_ALLTOALLV_SMALL_M; }
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& t, uint32_t rank, std::string dataFile) override
    {
        RuntimeTiling p(t);
        if (rank == 0) {
            std::cout << "Tiling M1=" << p.tile.m() << " N1=" << p.tile.n() << " K1=" << p.tile.k()
                      << " paddingA=" << p.paddingA << " paddingB=" << p.paddingB << std::endl;
        }
        std::vector<fp16_t> a(p.aBytes / sizeof(fp16_t)), b(p.bBytes / sizeof(fp16_t));
        if (!ReadFile(dataFile + "/a_gm_" + std::to_string(rank) + ".bin", a.data(), p.aBytes) ||
            !ReadFile(dataFile + "/b_gm_" + std::to_string(rank) + ".bin", b.data(), p.bBytes))
            throw std::runtime_error("Unable to load rank input");
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrA), p.aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrB), p.bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&params.ptrC), p.cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(params.ptrA, p.aBytes, a.data(), p.aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(params.ptrB, p.bBytes, b.data(), p.bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<uint32_t> values(size_t(t.rankSize) * t.expertNum);
        if (!ReadFile(
                dataFile + "/global_tokens_per_expert_" + std::to_string(rank) + ".bin", values.data(),
                values.size() * sizeof(uint32_t)))
            throw std::runtime_error("Unable to load token table");
        uint8_t *tokens = nullptr, *wc = nullptr;
        ACL_CHECK(aclrtMalloc(
            reinterpret_cast<void**>(&tokens), values.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&wc), p.wcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(
            tokens, values.size() * sizeof(uint32_t), values.data(), values.size() * sizeof(uint32_t),
            ACL_MEMCPY_HOST_TO_DEVICE));
        params.SetKernelParams(params.ptrA, params.ptrB, params.ptrC, tokens, wc);
    }
    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& t, uint32_t rank, std::string dataFile) override
    {
        RuntimeTiling p(t);
        size_t bytes = p.cBytes;
        // The public generator pads each result to rankSize * M rows.
        std::vector<uint8_t> out(bytes * t.rankSize, 0);
        ACL_CHECK(aclrtMemcpy(out.data(), out.size(), params.ptrC, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rank) + ".bin", out.data(), out.size());
    }
};
REGISTER_OPERATOR("GroupedMatmulAlltoAllvSmallM", GroupedMatmulAlltoAllvSmallMOperator);
