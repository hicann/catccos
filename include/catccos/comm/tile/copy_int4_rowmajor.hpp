/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_COMM_TILE_COPY_INT4_ROWMAJOR_HPP
#define CATCCOS_COMM_TILE_COPY_INT4_ROWMAJOR_HPP

#include "catccos/catccos.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/numeric_size.hpp"

namespace Catccos::Comm::Tile {

using Catlass::layout::RowMajor;
using Catlass::MatrixCoord;

CATLASS_DEVICE
__gm__ void *Int4GmVoidAddr(AscendC::GlobalTensor<AscendC::int4b_t> const &tensor)
{
    return const_cast<__gm__ void *>(reinterpret_cast<__gm__ void const *>(tensor.GetPhyAddr()));
}

CATLASS_DEVICE
uint32_t Int4PackedRowBytes(uint32_t logicalCols)
{
    return Catlass::BitsToBytes<uint32_t>(
        logicalCols * Catlass::SizeOfBits<AscendC::int4b_t>::value);
}

// int4 GM is allocated tight (m*k/2 bytes) while layout stride stays logical (k).
// Physical row pitch in bytes matches copy_gm_to_l1 int4 srcDValue = CeilDiv(stride, 2).
CATLASS_DEVICE
uint32_t Int4TightRowByteStride(uint32_t logicalRowStride)
{
    return Int4PackedRowBytes(logicalRowStride);
}

CATLASS_DEVICE
void CopyInt4GmToUbRowMajor(
    AscendC::LocalTensor<AscendC::int4b_t> const &dstTensor,
    AscendC::GlobalTensor<AscendC::int4b_t> const &srcTensor,
    RowMajor const &layoutDst,
    RowMajor const &layoutSrc)
{
    uint32_t packedCols = Int4PackedRowBytes(layoutSrc.shape(1));
    uint32_t srcRowByteStride = Int4TightRowByteStride(static_cast<uint32_t>(layoutSrc.stride(0)));
    uint32_t dstRowByteStride = Int4TightRowByteStride(static_cast<uint32_t>(layoutDst.stride(0)));

    AscendC::DataCopyExtParams dataCopyParams(
        layoutSrc.shape(0),
        packedCols,
        srcRowByteStride - packedCols,
        (dstRowByteStride - packedCols) / Catlass::BYTE_PER_BLK,
        0
    );
    AscendC::DataCopyPadExtParams<AscendC::int4b_t> padParams(false, 0, 0, 0);
    AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

CATLASS_DEVICE
void CopyInt4UbToGmRowMajor(
    AscendC::GlobalTensor<AscendC::int4b_t> const &dstTensor,
    AscendC::LocalTensor<AscendC::int4b_t> const &srcTensor,
    RowMajor const &layoutDst,
    RowMajor const &layoutSrc)
{
    uint32_t packedCols = Int4PackedRowBytes(layoutDst.shape(1));
    uint32_t dstRowByteStride = Int4TightRowByteStride(static_cast<uint32_t>(layoutDst.stride(0)));
    uint32_t srcRowByteStride = Int4TightRowByteStride(static_cast<uint32_t>(layoutSrc.stride(0)));

    AscendC::DataCopyExtParams dataCopyParams(
        layoutDst.shape(0),
        packedCols,
        (srcRowByteStride - packedCols) / Catlass::BYTE_PER_C0,
        dstRowByteStride - packedCols,
        0
    );
    AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

} // namespace Catccos::Comm::Tile

#endif // CATCCOS_COMM_TILE_COPY_INT4_ROWMAJOR_HPP
