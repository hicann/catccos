/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_BLOCK_SWIZZLE_ALLTOALL_MATMUL_K_SPLIT_HPP
#define CATCCOS_DGEMM_BLOCK_SWIZZLE_ALLTOALL_MATMUL_K_SPLIT_HPP

#include "catccos/catccos.hpp"
#include "catccos/dist_coord.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::DGemm::Block
{

using Catlass::MatrixCoord;

/// Swizzle for ops-transformer-style AllToAll + MatMul K-split.
///
/// Symmetric memory stores one stage as rank-contiguous blocks:
///
///     [srcRank][commM][localK_aligned]
///
/// This scheduler only decides M/N block order and workspace stage size. The
/// non-trivial logical-K to physical-rank mapping is handled inside the
/// catccos K-split MMAD block.
template <uint32_t SWIZZLE_OFFSET = 1, uint32_t SWIZZLE_DIRECTION = 0>
struct GemmBlockSwizzleAllToAllMatmulKSplit
{
    DistGemmCoord problemShape;
    DistGemmCoord loops;
    DistGemmCoord tileShape;
    uint32_t commRows{0};
    int64_t alignedLocalK{1};

    CATLASS_DEVICE
    GemmBlockSwizzleAllToAllMatmulKSplit() = default;

    CATLASS_DEVICE
    GemmBlockSwizzleAllToAllMatmulKSplit(DistGemmCoord const &problemShape_, MatrixCoord const &tileShapeMN_,
                                         uint32_t commRows_, int64_t alignedLocalK_)
        : problemShape(problemShape_), commRows(commRows_), alignedLocalK(alignedLocalK_)
    {
        tileShape = Catlass::MakeCoord<uint32_t>(tileShapeMN_[0], tileShapeMN_[1], problemShape_[2], 1);
        loops = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const { return loops[0] * loops[1] * loops[2] * loops[3]; }

    CATLASS_DEVICE
    DistGemmCoord GetBlockCoord(uint32_t loopIdx) const
    {
        uint32_t rows = loops[0] * loops[3];
        uint32_t cols = loops[1];
        uint32_t rowIdx{}, colIdx{};
        if constexpr (SWIZZLE_DIRECTION == 0)
        {
            uint32_t groupSize = SWIZZLE_OFFSET * cols;
            uint32_t groupIdx = loopIdx / groupSize;
            uint32_t groupOffset = loopIdx - groupIdx * groupSize;

            uint32_t inGroupRows = Min(SWIZZLE_OFFSET, rows - groupIdx * SWIZZLE_OFFSET);
            colIdx = groupOffset / inGroupRows;
            uint32_t inGroupRowIdx = groupOffset - colIdx * inGroupRows;
            rowIdx = groupIdx * SWIZZLE_OFFSET + inGroupRowIdx;
            if ((groupIdx & 0b1) == 1)
            {
                colIdx = cols - colIdx - 1;
            }
        }
        else if constexpr (SWIZZLE_DIRECTION == 1)
        {
            uint32_t groupSize = SWIZZLE_OFFSET * rows;
            uint32_t groupIdx = loopIdx / groupSize;
            uint32_t groupOffset = loopIdx - groupIdx * groupSize;

            uint32_t inGroupCols = Min(SWIZZLE_OFFSET, cols - groupIdx * SWIZZLE_OFFSET);
            rowIdx = groupOffset / inGroupCols;
            uint32_t inGroupColIdx = groupOffset - rowIdx * inGroupCols;
            colIdx = groupIdx * SWIZZLE_OFFSET + inGroupColIdx;
            if ((groupIdx & 0b1) == 1)
            {
                rowIdx = rows - rowIdx - 1;
            }
        }
        return {rowIdx % loops[0], colIdx, 0, rowIdx / loops[0]};
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockOffset(uint32_t loopIdx) const { return GetBlockCoord(loopIdx) * tileShape; }

    CATLASS_DEVICE
    DistGemmCoord GetActualBlockShapeByOffset(DistGemmCoord blockOffset) const
    {
        return Min(tileShape, problemShape - blockOffset);
    }

    CATLASS_DEVICE
    DistGemmCoord GetActualBlockShape(DistGemmCoord blockCoord) const
    {
        auto blockOffset = blockCoord * tileShape;
        return GetActualBlockShapeByOffset(blockOffset);
    }

    CATLASS_DEVICE
    int64_t GetSymmetricStageStride(uint32_t rankSize) const
    {
        return static_cast<int64_t>(rankSize) * commRows * alignedLocalK;
    }
};

}  // namespace Catccos::DGemm::Block

#endif  // CATCCOS_DGEMM_BLOCK_SWIZZLE_ALLTOALL_MATMUL_K_SPLIT_HPP
