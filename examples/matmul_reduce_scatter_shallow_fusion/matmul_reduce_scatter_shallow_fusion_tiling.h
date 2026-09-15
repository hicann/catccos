/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#include <cstddef>

#include "catlass/gemm_coord.hpp"
#include "operator_registry.h"
namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample {
using Catlass::GemmCoord;
enum class PaddingTag : uint8_t { PADDING_NONE = 0, PADDING_ND = 1, PADDING_BLOCK_ND = 2, PADDING_NZ = 3 };
inline void BalanceWorkload(uint32_t m, uint32_t n, uint32_t& m1, uint32_t& n1, uint32_t threshold, uint32_t coreNum)
{
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
    while (m1 > threshold && (CeilDiv(m, m1 - 16) * CeilDiv(n, n1) <= maxBlocks)) {
        m1 -= 16;
    }
    if (m < m1) {
        m1 = RoundUp(m, uint32_t(16));
    }
    if (n < n1) {
        n1 = RoundUp(n, uint32_t(16));
    }
}
template <class DType>
bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1)
{
    uint64_t l1Size{512 * 1024};
    uint64_t l0CSize{128 * 1024};
    bool judgeL1 = (m1 * k1 * 2 * sizeof(DType) + k1 * n1 * 2 * sizeof(DType) <= l1Size);
    bool judgeL0C = (m1 * n1 * 4 <= l0CSize) ? true : false;
    return judgeL1 && judgeL0C;
}

template <class DType>
uint32_t GetMaxK1(uint32_t m1, uint32_t n1)
{
    std::vector<uint32_t> k1List = {1024, 512, 256, 128};
    uint32_t k1 = 512 / sizeof(DType);
    for (const auto& k1t : k1List) {
        if (JudgeSpace<DType>(m1, n1, k1t)) {
            k1 = k1t;
            break;
        }
    }
    return k1;
}

inline void DoTilingB16Layout00(const GemmCoord& problemShape, GemmCoord& L1Shape, uint32_t coreNum)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();
    uint32_t k = problemShape.k();
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // Start with n1 = 256; consider n1 = 512 only if it reduces scheduling rounds.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
        BalanceWorkload(m, n, m1, n1, 32, coreNum);
        uint32_t blocks = CeilDiv(m, uint32_t(64)) * CeilDiv(n, uint32_t(512));
        if (blocks <= maxBlocks - coreNum && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUp(n, uint32_t(16));
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 32, coreNum);
    }
    if (k >= 65536 || n >= 65536) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1);
    L1Shape.m() = m1;
    L1Shape.n() = n1;
    L1Shape.k() = k1;
}

inline void DoTilingB16Layout01(const GemmCoord& problemShape, GemmCoord& L1Shape, uint32_t coreNum)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();
    uint32_t k = problemShape.k();
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    // This heuristic prioritizes balanced M/N work for RowMajor A and ColumnMajor B.
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
    } else {
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
    }
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
    if (m < n) {
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(m1, n1t + 16, k1)) {
            n1t += 16;
            uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1t);
            if (blocks <= maxBlocks - coreNum) {
                n1 = n1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
    } else {
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
    }
    if (k >= 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    k1 = GetMaxK1<fp16_t>(m1, n1);
    L1Shape.m() = m1;
    L1Shape.n() = n1;
    L1Shape.k() = k1;
}

} // namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample
