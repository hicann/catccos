/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PADDING_H
#define PADDING_H

#include "utils.h"


template <class T>
constexpr T Roundup(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

inline bool IsNeedPadding(uint32_t rows, uint32_t cols, uint32_t trans, uint32_t alignByElement)
{
    if (trans) {
        if (rows < 65536) {
            return rows % alignByElement != 0;
        } else {
            return true;
        }
    }

    if (cols < 65536) {
        return cols % alignByElement != 0;
    } else {
        return true;
    }
}

inline size_t GetWorkspaceLen(uint32_t shape0, uint32_t shape1, size_t blockRows, size_t blockCols)
{
    return Roundup(static_cast<size_t>(shape0), blockRows) *
           Roundup(static_cast<size_t>(shape1), blockCols);
}

#endif // PADDING_H