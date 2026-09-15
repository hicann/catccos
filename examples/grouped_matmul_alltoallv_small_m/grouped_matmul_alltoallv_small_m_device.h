/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). See LICENSE in the root of the software repository for details.
 */
#pragma once
#include <type_traits>

#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_small_m.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/status.hpp"
#include "grouped_matmul_alltoallv_small_m_host.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample {
using namespace Catlass;
using namespace tla;

template <
    class TensorA, class TensorB, class TensorWC, class TensorC, class LayoutTagA, class LayoutTagB, class LayoutTagWC,
    int SwizzleDirection>
struct Config {
    using ArchTag = Arch::AtlasA2;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, SwizzleDirection>;
    using L1TileShape = Shape<_128, _256, _256>;
    using L0TileShape = Shape<_128, _256, _64>;
    using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorA, LayoutTagA, TensorB, LayoutTagB, TensorWC, LayoutTagWC, void, void, false, false>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, TensorA, TensorB, TensorWC, void, TileCopy>;
    using Kernel = Catccos::DGemm::Kernel::GroupedMatmulAlltoAllvSmallM<BlockMmad, void, BlockScheduler, TensorC>;
    using DeviceOp = Gemm::Device::DeviceGemm<Kernel>;
};

template <int Direction, class TagB>
inline void LaunchConfigured(
    void* stream, uint32_t blockNum, uint64_t fftsAddr, KernelParams& params, uint8_t* workspace, uint8_t* symmetric,
    const CocTilingParams& t)
{
    RuntimeTiling plan(t);
    workspace = params.customPtrs[1];
    using Tag = Catlass::layout::RowMajor;
    using Element = half;
    Catlass::GemmCoord shape{t.m, t.n, t.k};
    Catlass::GemmCoord tile{plan.tile.m(), plan.tile.n(), plan.tile.k()};
    Tag tagA{t.m * t.rankSize, t.k}, tagC{t.m, t.n};
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
    auto lwc = tla::MakeLayoutFromTag(Tag{t.m * t.rankSize, t.n});
    using TWC =
        Tensor<AscendC::GlobalTensor<Element>, decltype(lwc), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using DeviceOp = typename Config<TA, TB, TWC, TC, Tag, TagB, Tag, Direction>::DeviceOp;
    typename DeviceOp::Arguments args{
        shape,
        t.expertNum / t.rankSize,
        params.customPtrs[0],
        params.ptrA,
        la,
        params.ptrB,
        lb,
        workspace,
        lwc,
        symmetric,
        lc,
        t.rankSize,
        params.ptrC};
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
} // namespace Catccos::Examples::GroupedMatmulAlltoAllvSmallMExample
