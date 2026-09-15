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
namespace Catccos::Examples::MatmulAllReduceShallowFusionExample {
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
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
        platform_ascendc::CoreMemType::L0_C, l0CSize);
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

inline double Getbandwidth(uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    double a6 = 0.000000000000020146121020;
    double a5 = -0.000000000012456944162142;
    double a4 = -0.000000006738536427145036;
    double a3 = 0.000007301215580838747961;
    double a2 = -0.002146456956750821074703;
    double a1 = 0.312849910814454512664184;
    double a0 = 0.1;
    double unalignBand = a6 * pow(static_cast<double>(dValue), 6) + a5 * pow(static_cast<double>(dValue), 5) +
                         a4 * pow(static_cast<double>(dValue), 4) + a3 * pow(static_cast<double>(dValue), 3) +
                         a2 * pow(static_cast<double>(dValue), 2) + a1 * static_cast<double>(dValue) + a0;

    if (dValue == srcDValue && dValue <= 128) {
        if (dValue % 16 == 0) {
            unalignBand = 60;
        }
    }
    if (srcDValue >= 65536) {
        unalignBand = 1;
    }

    if (srcDValue % 256 == 0) {
        unalignBand = static_cast<double>(100) / 30 * unalignBand;
    } else if (srcDValue % 128 == 0) {
        unalignBand = static_cast<double>(80) / 30 * unalignBand;
    } else if (srcDValue % 64 == 0) {
        unalignBand = static_cast<double>(50) / 30 * unalignBand;
    } else if (srcDValue % 16 == 0) {
        unalignBand = static_cast<double>(40) / 30 * unalignBand;
    }

    unalignBand = std::min(unalignBand, 80.0);

    if (dValue % 256 == 0) {
        if (nValue < 16) {
            double b2 = -0.003332381309698882569659;
            double b1 = 0.113578920178116271610946;
            double b0 = 0.016102868630357251855667;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * nValue + b0);
        }
    } else if (dValue % 32 == 0) {
        if (nValue < 32) {
            double b2 = -0.000298086120946179481978;
            double b1 = 0.045309519479127147167929;
            double b0 = 0.035130178145161221336945;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * nValue + b0);
        }
    } else {
        if (nValue < 64) {
            double b3 = 0.000001809180573350345869;
            double b2 = -0.000469676727179688081274;
            double b1 = 0.038963259596073690493867;
            double b0 = 0.003942641759904389614499;
            unalignBand = unalignBand * (b3 * pow(nValue, 3) + b2 * pow(nValue, 2) + b1 * nValue + b0);
        }
    }
    return unalignBand;
}

inline void GetPaddingTag(
    const GemmCoord& problemShape, GemmCoord& L1Shape, bool lA, bool lB, uint32_t coreNum, bool& isneedpaddingA,
    bool& isneedpaddingB)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();
    uint32_t k = problemShape.k();
    uint32_t m1 = L1Shape.m();
    uint32_t n1 = L1Shape.n();
    uint32_t k1 = L1Shape.k();

    uint64_t outterAxisA = m;
    uint64_t innerAxisA = k;
    uint32_t nValueA = std::min(m, m1);
    uint32_t dValueA = std::min(k, k1);
    if (lA == 1) {
        outterAxisA = k;
        innerAxisA = m;
        nValueA = std::min(k, k1);
        dValueA = std::min(m, m1);
    }

    uint64_t outterAxisB = k;
    uint64_t innerAxisB = n;
    uint32_t nValueB = std::min(k, k1);
    uint32_t dValueB = std::min(n, n1);
    if (lB == 1) {
        outterAxisB = n;
        innerAxisB = k;
        nValueB = std::min(n, n1);
        dValueB = std::min(k, k1);
    }

    double aBandwidthAiv = 30; // Assumed per-AIV bandwidth in GB/s.
    size_t matrixASize = static_cast<size_t>(m) * k * 2;
    if (matrixASize > 192 * 1024 * 1024) { // L2-capacity threshold assumed by this cost model.
        aBandwidthAiv = 10;
    }
    double aBandwidthBeforePaddingAic = Getbandwidth(nValueA, dValueA, innerAxisA);

    uint32_t tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1);
    uint32_t blockDimAic = tasksAic > coreNum ? coreNum : tasksAic;
    if (CeilDiv(m, m1) < blockDimAic / 2 && k <= k1 && CeilDiv(m, m1) <= 2) {
        aBandwidthBeforePaddingAic = aBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(m, m1)) * 1.5;
    }
    double aBandwidthAfterPaddingAic = 80;
    if (nValueA < 16) {
        aBandwidthAfterPaddingAic *= (static_cast<double>(nValueA) / 16);
    }

    double bBandwidthAiv = 30; // Assumed per-AIV bandwidth in GB/s.
    size_t matrixBSize = static_cast<size_t>(k) * n * 2;
    if (matrixBSize > 192 * 1024 * 1024) { // L2-capacity threshold assumed by this cost model.
        bBandwidthAiv = 10;
    }
    double bBandwidthBeforePaddingAic = Getbandwidth(nValueB, dValueB, innerAxisB);
    if (CeilDiv(n, n1) < blockDimAic / 2 && k <= k1 && CeilDiv(n, n1) <= 2) {
        bBandwidthBeforePaddingAic = bBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(n, n1)) * 1.5;
    }
    double bBandwidthAfterPaddingAic = 80;
    if (nValueB < 16) {
        bBandwidthAfterPaddingAic *= (static_cast<double>(nValueB) / 16);
    }

    uint32_t actualM = std::min(m, m1);
    uint32_t actualN = std::min(n, n1);
    uint32_t roundMax = CeilDiv(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
    size_t aMaxDataSizeAic = static_cast<size_t>(roundMax) * actualM * k * 2; // Byte
    size_t bMaxDataSizeAic = static_cast<size_t>(roundMax) * actualN * k * 2; // Byte

    // Estimate per-AIV padding traffic for the cost model.
    size_t aMaxDataSizeAiv{0};
    uint32_t tasksAivA{0};
    {
        uint32_t taskRows = 16;
        uint32_t taskCols = 48 * 1024 / 2 / taskRows;
        if (innerAxisA < taskCols) {
            taskCols = innerAxisA;
        }
        if (outterAxisA < taskRows) {
            taskRows = outterAxisA;
        }
        taskCols = RoundUp(innerAxisA / CeilDiv(innerAxisA, taskCols), 16);
        tasksAivA = CeilDiv(outterAxisA, taskRows) * CeilDiv(innerAxisA, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivA, coreNum * 2);
        aMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 2;
    }

    size_t bMaxDataSizeAiv{0};
    uint32_t tasksAivB{0};
    {
        uint32_t taskRows = 16;
        uint32_t taskCols = 48 * 1024 / 2 / taskRows;
        if (innerAxisB < taskCols) {
            taskCols = innerAxisB;
        }
        if (outterAxisB < taskRows) {
            taskRows = outterAxisB;
        }
        taskCols = RoundUp(innerAxisB / CeilDiv(innerAxisB, taskCols), 16);
        tasksAivB = CeilDiv(outterAxisB, taskRows) * CeilDiv(innerAxisB, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivB, coreNum * 2);
        bMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 2;
    }

    double headCost = 1 + 7 * static_cast<double>(blockDimAic) / coreNum; // us
    double t00 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000 +
                 static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000;
    double t01 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000 +
                 static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000 +
                 static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost;
    double t10 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000 +
                 static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000 +
                 static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000 + headCost;
    double t11 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000 +
                 static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000 +
                 static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000 +
                 static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost + 2;

    double minCost = std::numeric_limits<double>::max();
    PaddingTag paddingTagA = PaddingTag::PADDING_NONE;
    PaddingTag paddingTagB = PaddingTag::PADDING_NONE;
    if (minCost > t00) {
        minCost = t00;
    }
    if (minCost > t01) {
        minCost = t01;
        paddingTagA = PaddingTag::PADDING_NONE;
        paddingTagB = PaddingTag::PADDING_NZ;
    }
    if (minCost > t10) {
        minCost = t10;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NONE;
    }
    if (minCost > t11) {
        minCost = t11;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    if ((innerAxisA < 8 || (innerAxisA < 32 && (innerAxisA % 16 != 0))) && outterAxisA > 512) {
        paddingTagA = PaddingTag::PADDING_NZ;
    }
    if ((innerAxisB < 8 || (innerAxisB < 32 && (innerAxisB % 16 != 0))) && outterAxisB > 512) {
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    // Apply the large-stride padding heuristic to inner axes above 8192
    // that are multiples of 8192, once the outer axis reaches 2048.
    if (outterAxisA >= 2048 && innerAxisA > 8192 && innerAxisA % 8192 == 0) {
        paddingTagA = PaddingTag::PADDING_NZ;
    }
    if (outterAxisB >= 2048 && innerAxisB > 8192 && innerAxisB % 8192 == 0) {
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    PaddingTag paddingTagC = PaddingTag::PADDING_NONE;
    if (static_cast<size_t>(m) * n > 2048 * 2048 && n > 256 && (n % 128 != 0)) {
        size_t totalDataSize = static_cast<size_t>(m) * k * CeilDiv(n, n1) * 2 +
                               static_cast<size_t>(k) * n * CeilDiv(m, m1) * 2 + static_cast<size_t>(m) * n * 2;
        if (totalDataSize < 192 * 1024 * 1024) { // L2-capacity threshold assumed by this cost model.
            paddingTagC = PaddingTag::PADDING_ND;
        }
    }

    if (paddingTagA == PaddingTag::PADDING_NONE) {
        isneedpaddingA = false;
    } else {
        isneedpaddingA = true;
    }

    if (paddingTagB == PaddingTag::PADDING_NONE) {
        isneedpaddingB = false;
    } else {
        isneedpaddingB = true;
    }

    if (n * k > 70 * 1024 * 1024) {
        isneedpaddingA = true;
        isneedpaddingB = false;
    }

    if ((m * k > 450000) && (n * k > 64000000)) {
        isneedpaddingA = true;
        isneedpaddingB = false;
    }

    if (m <= 4) {
        isneedpaddingA = false;
        isneedpaddingB = true;
    }
}

} // namespace Catccos::Examples::MatmulAllReduceShallowFusionExample
