/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DIST_COORD_HPP
#define CATCCOS_DIST_COORD_HPP

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/coord.hpp"

namespace Catccos {

struct DistMatrixCoord : public Catlass::Coord<3, uint32_t> {
    using Index = uint32_t;
    using Base = Catlass::Coord<3, Index>;
    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;
    static constexpr int RANK_INDEX = 2;

    CATLASS_HOST_DEVICE
    DistMatrixCoord() = default;

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Base const &coord) : Base(coord)
    {
    }

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Index row, Index column, Index rank) : Base(Catlass::MakeCoord<Index>(row, column, rank))
    {
    }

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Catlass::Coord<2, Index> matrixCoord, Index rank)
        : Base(Catlass::MakeCoord<Index>(matrixCoord[0], matrixCoord[1], rank))
    {
    }

    CATLASS_HOST_DEVICE
    Index const &row() const
    {
        return this->At(ROW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &row()
    {
        return this->At(ROW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &column() const
    {
        return this->At(COLUMN_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &column()
    {
        return this->At(COLUMN_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &rank() const
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &rank()
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    auto GetCoordInRank() const
    {
        return this->GetCoordByAxis<ROW_INDEX, COLUMN_INDEX>();
    }
};

struct DistGemmCoord : public Catlass::Coord<4, uint32_t> {
    using Index = uint32_t;
    using Base = Catlass::Coord<4, Index>;
    static constexpr int M_INDEX = 0;
    static constexpr int N_INDEX = 1;
    static constexpr int K_INDEX = 2;
    static constexpr int RANK_INDEX = 3;

    CATLASS_HOST_DEVICE
    DistGemmCoord() = default;

    CATLASS_HOST_DEVICE
    DistGemmCoord(Base const &coord) : Base(coord)
    {
    }

    CATLASS_HOST_DEVICE
    DistGemmCoord(Index m, Index n, Index k, Index rank) : Base(Catlass::MakeCoord<Index>(m, n, k, rank))
    {
    }

    CATLASS_HOST_DEVICE
    Index const &m() const
    {
        return this->At(M_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &n() const
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &k() const
    {
        return this->At(K_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &rank() const
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &m()
    {
        return this->At(M_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &n()
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &k()
    {
        return this->At(K_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &rank()
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMNK() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX, K_INDEX>();
    }
};

}  // namespace Catccos

#endif  // CATCCOS_DIST_COORD_HPP