/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#include <type_traits>

#include "allgather_matmul_shallow_fusion_host.h"
#include "catccos/dgemm/kernel/allgather_matmul_shallow_fusion.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::Examples::AllGatherMatmulShallowFusionExample {
using namespace Catlass;
using namespace tla;
template <class Layout>
auto GetPaddingLayout(Layout layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = tla::MakeShape(
            tla::MakeShape(blockRows, ::CeilDiv(layout.shape(0), blockRows)),
            tla::MakeShape(blockCols, ::CeilDiv(layout.shape(1), blockCols)));
        auto stride = tla::MakeStride(
            tla::MakeStride(
                static_cast<int64_t>(blockCols),
                static_cast<int64_t>(blockRows) * ::RoundUp(layout.shape(1), blockCols)),
            tla::MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols));
        return tla::MakeLayout(shape, stride);
    } else {
        auto shape = tla::MakeShape(
            tla::MakeShape(blockRows, ::CeilDiv(layout.shape(0), blockRows)),
            tla::MakeShape(blockCols, ::CeilDiv(layout.shape(1), blockCols)));
        auto stride = tla::MakeStride(
            tla::MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            tla::MakeStride(
                static_cast<int64_t>(blockRows),
                ::RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)));
        return tla::MakeLayout(shape, stride);
    }
}

template <class Layout>
auto GetPaddingALayout(Layout layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = tla::MakeShape(layout.shape(0), layout.shape(1));
        auto stride = tla::MakeStride(static_cast<int64_t>(::RoundUp((uint32_t)layout.stride(0), blockCols)), Int<1>{});
        return tla::MakeLayout(shape, stride);
    } else {
        auto shape = tla::MakeShape(layout.shape(0), layout.shape(1));
        auto stride = tla::MakeStride(Int<1>{}, static_cast<int64_t>(::RoundUp((uint32_t)layout.stride(1), blockRows)));
        return tla::MakeLayout(shape, stride);
    }
}

template <class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return ::RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           ::RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

template <class Layout>
size_t GetSymmWorkspaceLen(Layout layout, size_t blockCols)
{
    if (std::is_same_v<Layout, layout::RowMajor>) {
        return static_cast<size_t>(layout.shape(0)) * ::RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
    }
    return ::RoundUp(static_cast<size_t>(layout.shape(0)), blockCols) * static_cast<size_t>(layout.shape(1));
}

template <
    class TensorA, class TensorB, class TensorC, class TensorWA, class TensorWB, class LayoutTagA, class LayoutTagB,
    class LayoutTagC, bool PaddingBEnabled, int SwizzleDirection>
struct Config {
    using ArchTag = Arch::AtlasA2;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, SwizzleDirection>;
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    using L0TileShape = std::conditional_t<
        std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        Shape<_256, _128, _64>, Shape<_128, _256, _64>>;
    using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, false, PaddingBEnabled>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
    using PaddingB = std::conditional_t<
        PaddingBEnabled,
        Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, 96 * 1024 / sizeof(typename TensorB::Element)>,
        void>;
    using Kernel =
        Catccos::DGemm::Kernel::AllGatherMatmulShallowFusion<BlockMmad, void, BlockScheduler, PaddingB, TensorA>;
    using DeviceOp = Gemm::Device::DeviceGemm<Kernel>;
};

template <int Direction, class TagB>
inline void LaunchConfigured(
    void* stream, uint32_t blockNum, uint64_t fftsAddr, KernelParams& params, uint8_t* workspace, uint8_t* symmetric,
    const CocTilingParams& t)
{
    RuntimeTiling plan(t);
    using Tag = Catlass::layout::RowMajor;
    using Element = half;
    Catlass::GemmCoord shape{t.m, t.n, t.k};
    Catlass::GemmCoord tile{plan.tile.m(), plan.tile.n(), plan.tile.k()};
    Tag tagA{t.m, t.k}, tagC{t.m * t.rankSize, t.n};
    TagB tagB{t.k, t.n};
    auto la = tla::MakeLayoutFromTag(tagA);
    auto lb = tla::MakeLayoutFromTag(tagB);
    auto lc = tla::MakeLayoutFromTag(tagC);
    using TA =
        Tensor<AscendC::GlobalTensor<Element>, decltype(la), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TB =
        Tensor<AscendC::GlobalTensor<Element>, decltype(lb), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TC =
        Tensor<AscendC::GlobalTensor<Element>, decltype(lc), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    auto lwa = GetPaddingALayout(tagA, 128, 256);
    auto lwb = GetPaddingLayout(tagB, 256, 256);
    using TWA =
        Tensor<AscendC::GlobalTensor<Element>, decltype(lwa), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TWB =
        Tensor<AscendC::GlobalTensor<Element>, decltype(lwb), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using DeviceOp = typename Config<TA, TB, TC, TWA, TWB, Tag, TagB, Tag, true, Direction>::DeviceOp;
    typename DeviceOp::Arguments args{shape, params.ptrA, la,  params.ptrB, lb,  params.ptrC,
                                      lc,    workspace,   lwb, symmetric,   lwa, symmetric + DATA_REGION_BYTES};
    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    deviceOp(static_cast<aclrtStream>(stream), blockNum, fftsAddr);
}
inline void Launch(
    void* stream, uint32_t blockNum, uint64_t fftsAddr, KernelParams& params, uint8_t* workspace, uint8_t* symmetric,
    CocTilingParams& t, uint32_t transA, uint32_t transB)
{
    if (transA || transB > 1 || transB != t.transB)
        throw std::runtime_error("Expected RowMajor A and matching transB in {0, 1}");
    if (transB) {
        if (t.m > t.n)
            LaunchConfigured<0, Catlass::layout::ColumnMajor>(
                stream, blockNum, fftsAddr, params, workspace, symmetric, t);
        else
            LaunchConfigured<1, Catlass::layout::ColumnMajor>(
                stream, blockNum, fftsAddr, params, workspace, symmetric, t);
    } else {
        if (t.m > t.n)
            LaunchConfigured<0, Catlass::layout::RowMajor>(stream, blockNum, fftsAddr, params, workspace, symmetric, t);
        else
            LaunchConfigured<1, Catlass::layout::RowMajor>(stream, blockNum, fftsAddr, params, workspace, symmetric, t);
    }
}
} // namespace Catccos::Examples::AllGatherMatmulShallowFusionExample
