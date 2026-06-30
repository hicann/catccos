# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

if(NOT DEFINED CATCCOS_ROOT)
    set(CATCCOS_ROOT $ENV{CATCCOS_ROOT})
endif()
if(NOT CATCCOS_ROOT)
    set(CATCCOS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()
if(NOT CATCCOS_ROOT OR NOT EXISTS ${CATCCOS_ROOT}/include/catccos)
    message(FATAL_ERROR "Cannot find CATCCOS_ROOT. Please pass -DCATCCOS_ROOT or set CATCCOS_ROOT.")
endif()
get_filename_component(CATCCOS_ROOT ${CATCCOS_ROOT} ABSOLUTE)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
endif()
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
endif()

npu_op_kernel_options(ascendc_kernels ALL OPTIONS
    -DCATLASS_ARCH=2201
    -I${CATCCOS_ROOT}/3rdparty/catlass/include
    -I${CATCCOS_ROOT}/include
    -I${CATCCOS_ROOT}/src/device
    -I${CATCCOS_ROOT}/src/device/gm2gm
    -I${CATCCOS_ROOT}/src/device/ub2gm
    -I${CATCCOS_ROOT}/3rdparty/shmem/include
    -I${CATCCOS_ROOT}/3rdparty/shmem/include/device/gm2gm/engine
    -I${CATCCOS_ROOT}/3rdparty/shmem/include/device/ub2gm/engine
    -I${CATCCOS_ROOT}/3rdparty/shmem/src
    -I${CATCCOS_ROOT}/3rdparty/shmem/src/device
    -I${CATCCOS_ROOT}/3rdparty/shmem/src/device/gm2gm/engine
    -I${CATCCOS_ROOT}/3rdparty/shmem/src/device/ub2gm/mte
    -I${CMAKE_CURRENT_SOURCE_DIR}/../..
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    -I$ENV{ASCEND_HOME_PATH}/include/experiment/runtime
    -I$ENV{ASCEND_HOME_PATH}/include/experiment/msprof
    -I$ENV{SHMEM_HOME_PATH}/shmem/include
    -I$ENV{SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    -I$ENV{SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
)

if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
