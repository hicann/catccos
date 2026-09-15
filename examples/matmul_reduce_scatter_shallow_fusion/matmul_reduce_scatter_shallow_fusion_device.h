/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#include <type_traits>

#include "catccos/dgemm/block/block_mmad_preload_tla_dynamic.hpp"
#include "catccos/dgemm/kernel/matmul_reduce_scatter_shallow_fusion.hpp"
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
#include "matmul_reduce_scatter_shallow_fusion_host.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample {
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
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return ::RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           ::RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

template <
    class TensorA, class TensorB, class TensorC, class TensorWA, class TensorWB, class LayoutTagA, class LayoutTagB,
    class LayoutTagC, bool PaddingAEnabled, bool PaddingBEnabled, int SwizzleDirection>
struct Config {
    using ArchTag = Arch::AtlasA2;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, SwizzleDirection>;
    using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, PaddingAEnabled,
        PaddingBEnabled>;
    using BlockMmad =
        Catccos::DGemm::Block::BlockMmadTlaDynamic<MmadDispatchPolicy, TensorWA, TensorWB, TensorC, void, TileCopy>;
    using PaddingA = std::conditional_t<
        PaddingAEnabled,
        Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, 96 * 1024 / sizeof(typename TensorA::Element)>,
        void>;
    using PaddingB = std::conditional_t<
        PaddingBEnabled,
        Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, 96 * 1024 / sizeof(typename TensorB::Element)>,
        void>;
    using Kernel =
        Catccos::DGemm::Kernel::MatmulReduceScatterShallowFusion<BlockMmad, void, BlockScheduler, PaddingA, PaddingB>;
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
    Tag tagA{t.m, t.k}, tagC{t.m, t.n};
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
    auto launch = [&](auto pa, auto pb) {
        constexpr bool padA = decltype(pa)::value, padB = decltype(pb)::value;
        auto lwa = [&] {
            if constexpr (padA)
                return GetPaddingLayout(tagA, tile.m(), tile.k());
            else
                return tla::MakeLayout(la.shape(), la.stride());
        }();
        auto lwb = [&] {
            if constexpr (padB)
                return GetPaddingLayout(tagB, tile.k(), tile.n());
            else
                return tla::MakeLayout(lb.shape(), lb.stride());
        }();
        using TWA =
            Tensor<AscendC::GlobalTensor<Element>, decltype(lwa), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TWB =
            Tensor<AscendC::GlobalTensor<Element>, decltype(lwb), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using DeviceOp = typename Config<TA, TB, TC, TWA, TWB, Tag, TagB, Tag, padA, padB, Direction>::DeviceOp;
        uint8_t* wa = padA ? workspace : params.ptrA;
        uint8_t* wb = padB ? workspace + plan.waBytes : params.ptrB;
        Catlass::MatrixCoord commShape{t.m / t.rankSize, t.n};
        typename DeviceOp::Arguments args{
            shape,
            tile,
            commShape,
            params.ptrA,
            la,
            params.ptrB,
            lb,
            params.ptrC,
            lc,
            wa,
            lwa,
            wb,
            lwb,
            symmetric,
            symmetric + DATA_REGION_BYTES,
            t.rankSize,
            1,
            0,
            0};
        DeviceOp deviceOp;
        deviceOp.Initialize(args);
        deviceOp(static_cast<aclrtStream>(stream), blockNum, fftsAddr);
    };
    launch(std::true_type{}, std::true_type{});
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
} // namespace Catccos::Examples::MatmulReduceScatterShallowFusionExample
