/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_TORCH_REGISTER_H
#define CATCCOS_TORCH_REGISTER_H

#define REGISTER_CATCCOS_OPS_CLASS(CLASS_NAME, ...) \
    static auto registry_##CLASS_NAME = \
    torch::jit::class_<CatccosOps::CLASS_NAME>("CatccosOps", #CLASS_NAME) \
    .def(torch::jit::init<>()) \
    REGISTER_CATCCOS_OPS_FUNCS_HELPER(CLASS_NAME, ##__VA_ARGS__)

#define REGISTER_CATCCOS_OPS_FUNCS_HELPER(CLASS, ...) \
    REGISTER_CATCCOS_OPS_FUNCS_CHOOSER(__VA_ARGS__, REGISTER_CATCCOS_OPS_FUNCS_6, REGISTER_CATCCOS_OPS_FUNCS_5, REGISTER_CATCCOS_OPS_FUNCS_4, REGISTER_CATCCOS_OPS_FUNCS_3, REGISTER_CATCCOS_OPS_FUNCS_2, REGISTER_CATCCOS_OPS_FUNCS_1)(CLASS, ##__VA_ARGS__)

#define REGISTER_CATCCOS_OPS_FUNCS_CHOOSER(_1, _2, _3, _4, _5, _6, FUNC, ...) FUNC

#define REGISTER_CATCCOS_OPS_FUNCS_1(CLASS, FUNC1) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1)

#define REGISTER_CATCCOS_OPS_FUNCS_2(CLASS, FUNC1, FUNC2) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1) \
    .def(#FUNC2, &CatccosOps::CLASS::FUNC2)

#define REGISTER_CATCCOS_OPS_FUNCS_3(CLASS, FUNC1, FUNC2, FUNC3) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1) \
    .def(#FUNC2, &CatccosOps::CLASS::FUNC2) \
    .def(#FUNC3, &CatccosOps::CLASS::FUNC3)

#define REGISTER_CATCCOS_OPS_FUNCS_4(CLASS, FUNC1, FUNC2, FUNC3, FUNC4) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1) \
    .def(#FUNC2, &CatccosOps::CLASS::FUNC2) \
    .def(#FUNC3, &CatccosOps::CLASS::FUNC3) \
    .def(#FUNC4, &CatccosOps::CLASS::FUNC4)

#define REGISTER_CATCCOS_OPS_FUNCS_5(CLASS, FUNC1, FUNC2, FUNC3, FUNC4, FUNC5) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1) \
    .def(#FUNC2, &CatccosOps::CLASS::FUNC2) \
    .def(#FUNC3, &CatccosOps::CLASS::FUNC3) \
    .def(#FUNC4, &CatccosOps::CLASS::FUNC4) \
    .def(#FUNC5, &CatccosOps::CLASS::FUNC5)

#define REGISTER_CATCCOS_OPS_FUNCS_6(CLASS, FUNC1, FUNC2, FUNC3, FUNC4, FUNC5, FUNC6) \
    .def(#FUNC1, &CatccosOps::CLASS::FUNC1) \
    .def(#FUNC2, &CatccosOps::CLASS::FUNC2) \
    .def(#FUNC3, &CatccosOps::CLASS::FUNC3) \
    .def(#FUNC4, &CatccosOps::CLASS::FUNC4) \
    .def(#FUNC5, &CatccosOps::CLASS::FUNC5) \
    .def(#FUNC6, &CatccosOps::CLASS::FUNC6)

#endif // CATCCOS_TORCH_REGISTER_H