# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(allgather_matmul_hccl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

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

set(NPU_MODULE "Ascend910_93")
get_filename_component(CATCCOS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
set(MSOPGEN_CMAKE_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/msopgen)
set(MSOPGEN_CMAKE_BINARY_DIR ${MSOPGEN_CMAKE_SOURCE_DIR}/build_out)
set(MSOPGEN_CONFIG_FILE ${MSOPGEN_CMAKE_SOURCE_DIR}/all_gather_matmul_hccl.json)
set(CUSTOM_OP_API_INCLUDE_DIRS
    ${MSOPGEN_CMAKE_BINARY_DIR}/op_api/include
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include
)
set(CUSTOM_OP_API_LIB_DIRS
    ${MSOPGEN_CMAKE_BINARY_DIR}/op_api/lib
    ${MSOPGEN_CMAKE_BINARY_DIR}/op_host
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib
)
if(DEFINED ENV{ASCEND_OPP_PATH})
    list(APPEND CUSTOM_OP_API_INCLUDE_DIRS $ENV{ASCEND_OPP_PATH}/vendors/customize/op_api/include)
    list(APPEND CUSTOM_OP_API_LIB_DIRS $ENV{ASCEND_OPP_PATH}/vendors/customize/op_api/lib)
endif()

find_package(Threads REQUIRED)

find_program(MSOPGEN NAMES msopgen HINTS
    ${ASCEND_HOME_PATH}
    ${ASCEND_HOME_PATH}/bin
    ${ASCEND_HOME_PATH}/tools/msopgen/bin
)
if(NOT MSOPGEN)
    message(FATAL_ERROR "Cannot find msopgen, please check ASCEND_HOME_PATH.")
endif()

add_custom_command(
    OUTPUT ${MSOPGEN_CMAKE_SOURCE_DIR}/CMakeLists.txt
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MSOPGEN_CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul_hccl.json
            ${MSOPGEN_CONFIG_FILE}
    COMMAND ${MSOPGEN} gen -i ${MSOPGEN_CONFIG_FILE} -f aclnn -c ai_core-${NPU_MODULE} -lan cpp
            -out ${MSOPGEN_CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/op_host
            ${MSOPGEN_CMAKE_SOURCE_DIR}/op_host
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
            ${MSOPGEN_CMAKE_SOURCE_DIR}/op_kernel
    COMMENT "Generating AllGatherMatmulHccl op project..."
)

add_custom_target(msopgen_project
    DEPENDS ${MSOPGEN_CMAKE_SOURCE_DIR}/CMakeLists.txt
)

add_custom_target(msopgen_build
    COMMAND ${CMAKE_COMMAND} -S ${MSOPGEN_CMAKE_SOURCE_DIR} -B ${MSOPGEN_CMAKE_BINARY_DIR} --preset=default
            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
            -DASCEND_COMPUTE_UNIT=ascend910_93
            -DCATCCOS_ROOT=${CATCCOS_ROOT}
    COMMAND ${CMAKE_COMMAND} --build ${MSOPGEN_CMAKE_BINARY_DIR} --target binary package -j
    DEPENDS msopgen_project
    COMMENT "Building AllGatherMatmulHccl op project..."
)

add_executable(allgather_matmul_hccl main.cpp)
add_dependencies(allgather_matmul_hccl msopgen_build)

target_include_directories(allgather_matmul_hccl
    PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/aclnn
        ${ASCEND_HOME_PATH}/include/experiment/runtime
        ${ASCEND_HOME_PATH}/include/experiment/msprof
        ${CUSTOM_OP_API_INCLUDE_DIRS}
        ${MSOPGEN_CMAKE_BINARY_DIR}/autogen
        ${CATCCOS_ROOT}/3rdparty/shmem/include
        ${SHMEM_HOME_PATH}/shmem/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CATCCOS_ROOT}/examples/utils
        ${CATCCOS_ROOT}/utils
)

target_link_directories(allgather_matmul_hccl
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${CUSTOM_OP_API_LIB_DIRS}
)

target_link_libraries(allgather_matmul_hccl
    PRIVATE ascendcl nnopbase opapi hccl cust_opapi Threads::Threads
)
