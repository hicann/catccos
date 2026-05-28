/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_OPERATOR_H
#define CATCCOS_OPERATOR_H

#pragma once
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

#include "shmem_init.h"

// utils
#include "utils.h"
#include "info.h"

class CatccosOperator {
public:
    virtual ~CatccosOperator() = default;
    virtual bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) = 0;
    virtual void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile = "") = 0;
    virtual void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile = "") = 0;
    virtual size_t GetWorkspaceSize(const CocTilingParams &cocTiling) = 0;
    virtual CocCommType GetActualKernelType(const CocTilingParams &cocTiling) = 0;
};

// 算子创建函数类型
using OperatorCreator = std::function<std::unique_ptr<CatccosOperator>()>;

#endif // CATCCOS_OPERATOR_H
