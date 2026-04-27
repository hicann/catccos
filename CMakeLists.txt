# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.10)
project(CATCCOS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT DEFINED ASCEND_HOME_PATH AND NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED SHMEM_HOME_PATH AND NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED SHMEM_HOME_PATH)
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

set(CCEC ${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng)

add_subdirectory(examples)
add_subdirectory(tests)