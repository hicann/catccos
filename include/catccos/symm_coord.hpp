/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_SYMM_COORD_HPP
#define CATCCOS_SYMM_COORD_HPP
 
#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/coord.hpp"
#include "catlass/matrix_coord.hpp"
 
namespace Catccos {
 
using Catlass::MatrixCoord;
 
class SymmMatrixCoord : public Catlass::Coord<4, uint32_t> {
public:
    using Index = uint32_t;
    using Base = Catlass::Coord<4, Index>;
    static constexpr int ROW_INDEX = 0;
    static constexpr int COLUMN_INDEX = 1;
    static constexpr int LOCAL_RANK_INDEX = 2;
    static constexpr int REMOTE_RANK_INDEX = 3;
 
    CATLASS_HOST_DEVICE
    SymmMatrixCoord() = default;
 
    CATLASS_HOST_DEVICE
    SymmMatrixCoord(Base const &coord) : Base(coord)
    {
    }
 
    CATLASS_HOST_DEVICE
    SymmMatrixCoord(Index row, Index column, Index local, Index remote) :
        Base(Catlass::MakeCoord(row, column, local, remote))
    {
    }
 
    CATLASS_HOST_DEVICE
    SymmMatrixCoord(MatrixCoord const &coord, Index local, Index remote) :
        Base(Catlass::MakeCoord<Index>(coord.row(), coord.column(), local, remote))
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
    Index const &local() const
    {
        return this->At(LOCAL_RANK_INDEX);
    }
 
    CATLASS_HOST_DEVICE
    Index &local()
    {
        return this->At(LOCAL_RANK_INDEX);
    }
 
    CATLASS_HOST_DEVICE
    Index const &remote() const
    {
        return this->At(REMOTE_RANK_INDEX);
    }
 
    CATLASS_HOST_DEVICE
    Index &remote()
    {
        return this->At(REMOTE_RANK_INDEX);
    }
 
    CATLASS_HOST_DEVICE
    auto GetMatrixCoord() const
    {
        return this->GetCoordByAxis<ROW_INDEX, COLUMN_INDEX>();
    }
 
    CATLASS_HOST_DEVICE
    auto GetLocalCoord() const
    {
        return this->GetCoordByAxis<ROW_INDEX, COLUMN_INDEX, LOCAL_RANK_INDEX>();
    }
 
    CATLASS_HOST_DEVICE
    auto GetRemoteCoord() const
    {
        return this->GetCoordByAxis<ROW_INDEX, COLUMN_INDEX, REMOTE_RANK_INDEX>();
    }
};
 
}  // namespace Catccos
 
#endif  // CATCCOS_SYMM_COORD_HPP