/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPERATOR_REGISTRY_H
#define OPERATOR_REGISTRY_H

#pragma once
#include "catccos_operator.h"
#include <mutex>

class OperatorRegistry {
public:
    static OperatorRegistry& Instance() {
        static OperatorRegistry instance;
        return instance;
    }

    bool RegisterOperator(const std::string& op_type, OperatorCreator creator) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = registry_.emplace(op_type, std::move(creator));
        return inserted;
    }

    std::unique_ptr<CatccosOperator> CreateOperator(const std::string& op_type) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry_.find(op_type);
        if (it != registry_.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    std::unordered_map<std::string, OperatorCreator> registry_;
    std::mutex mutex_;
};

// 注册宏
#define REGISTER_OPERATOR(op_type, op_class) \
    class op_class; \
    namespace { \
        struct op_class##_registrar { \
            op_class##_registrar() { \
                OperatorRegistry::Instance().RegisterOperator( \
                    op_type, \
                    []() -> std::unique_ptr<CatccosOperator> { \
                        return std::make_unique<op_class>(); \
                    } \
                ); \
            } \
        }; \
        static op_class##_registrar op_class##_instance; \
    }

#endif // OPERATOR_REGISTRY_H