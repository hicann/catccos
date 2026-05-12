# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Torch extension support
option(CATCCOS_TORCH_EXTENSION "Build PyTorch extension" OFF)

# List of operators that support PyTorch binding
set(CATCCOS_TORCH_SUPPORTED
    allgather_matmul
)

add_custom_target(catccos_examples)

function(catccos_example_add_executable TARGET)
    set(OPTIONS)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS SOURCES INCLUDES COMPILE_DEFINITIONS)

    cmake_parse_arguments(
        ARG_PARSE
        "${OPTIONS}"
        "${ONE_VALUE_ARGS}"
        "${MULTI_VALUE_ARGS}"
        ${ARGN}
    )
    set(KERNEL_SOURCES)
    foreach(SOURCE ${ARG_PARSE_SOURCES})
        if(NOT IS_ABSOLUTE ${SOURCE})
            get_filename_component(SOURCE ${SOURCE} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        list(APPEND KERNEL_SOURCES ${SOURCE})
    endforeach()

    set(INCLUDE_DIRS "")
    foreach(INCLUDES ${ARG_PARSE_INCLUDES})
        if(NOT IS_ABSOLUTE ${DIR})
            get_filename_component(INCLUDES ${INCLUDES} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        list(APPEND INCLUDE_DIRS -I${INCLUDES})
    endforeach()

    set(COMPILE_DEFS "")
    foreach(DEF ${ARG_PARSE_COMPILE_DEFINITIONS})
        list(APPEND COMPILE_DEFS -D${DEF})
    endforeach()

    if(CATLASS_BISHENG_ARCH STREQUAL "a5")
        set(CCE_AICORE_ARCH_BASE dav-c310)
    else()
        set(CCE_AICORE_ARCH_BASE dav-c220)
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
        COMMAND ${CCEC} -O2 -std=c++17 -xcce --cce-aicore-arch=${CCE_AICORE_ARCH_BASE} ${CATLASS_ARCH_FLAGS}
                        -mllvm -cce-aicore-stack-size=0x8000
                        -mllvm -cce-aicore-function-stack-size=0x8000
                        -mllvm -cce-aicore-record-overflow=true
                        -mllvm -cce-aicore-addr-transform
                        -mllvm -cce-aicore-dcci-insert-for-scalar=false
                        -DL2_CACHE_HINT
                        -MD -MF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
                        ${COMPILE_DEFS}
                        -I${CMAKE_SOURCE_DIR}/include
                        -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
                        -I${CMAKE_SOURCE_DIR}/examples
                        -I${CMAKE_SOURCE_DIR}/examples/utils
                        -I${ASCEND_HOME_PATH}/include
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
                        -I${ASCEND_HOME_PATH}/aarch64-linux/include
                        -I${ASCEND_HOME_PATH}/include/experiment/runtime
                        -I${ASCEND_HOME_PATH}/include/experiment/msprof
                        -I${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime
                        -I${SHMEM_HOME_PATH}/shmem/include
                        -I${SHMEM_HOME_PATH}/shmem/src
                        -I${SHMEM_HOME_PATH}/shmem/src/device
                        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
                        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
                        ${INCLUDE_DIRS}
                        -L${ASCEND_HOME_PATH}/lib64
                        -L${SHMEM_HOME_PATH}/memfabric_hybrid/lib
                        -L${SHMEM_HOME_PATH}/shmem/lib
                        -lruntime -lstdc++ -lm -lascendcl
                        -lshmem -lgraph_base -ltiling_api -lplatform -lregister
                        ${KERNEL_SOURCES} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
        DEPENDS ${KERNEL_SOURCES}
        DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.d
        VERBATIM
        COMMENT "Generating executable kernel ${TARGET}"
    )

    add_custom_target(
        ${TARGET}
        DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
    )
    add_dependencies(catccos_examples ${TARGET})
endfunction()

# Function to compile wrapper to shared library for PyTorch extension
function(catccos_example_add_library TARGET)
    set(OPTIONS)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS SOURCES INCLUDES COMPILE_DEFINITIONS)

    cmake_parse_arguments(
        ARG_PARSE
        "${OPTIONS}"
        "${ONE_VALUE_ARGS}"
        "${MULTI_VALUE_ARGS}"
        ${ARGN}
    )

    if(NOT CATCCOS_TORCH_EXTENSION)
        return()
    endif()

    if(CATLASS_BISHENG_ARCH STREQUAL "a5")
        set(CCE_AICORE_ARCH_BASE dav-c310)
    else()
        set(CCE_AICORE_ARCH_BASE dav-c220)
    endif()

    set(KERNEL_SOURCES)
    foreach(SOURCE ${ARG_PARSE_SOURCES})
        if(NOT IS_ABSOLUTE ${SOURCE})
            get_filename_component(SOURCE ${SOURCE} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        list(APPEND KERNEL_SOURCES ${SOURCE})
    endforeach()

    set(INCLUDE_DIRS "")
    foreach(INCLUDES ${ARG_PARSE_INCLUDES})
        if(NOT IS_ABSOLUTE ${DIR})
            get_filename_component(INCLUDES ${INCLUDES} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        list(APPEND INCLUDE_DIRS -I${INCLUDES})
    endforeach()

    set(COMPILE_DEFS "")
    foreach(DEF ${ARG_PARSE_COMPILE_DEFINITIONS})
        list(APPEND COMPILE_DEFS -D${DEF})
    endforeach()

    # Create custom target for kernel library
    add_custom_target(${TARGET}_kernel_build)

    # Create shared library using CCEC
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/lib/lib${TARGET}_kernel.so
        COMMAND ${CCEC} -O2 -shared -fPIC -std=c++17 -xcce --cce-aicore-arch=${CCE_AICORE_ARCH_BASE} ${CATLASS_ARCH_FLAGS}
                        -mllvm -cce-aicore-stack-size=0x8000
                        -mllvm -cce-aicore-function-stack-size=0x8000
                        -mllvm -cce-aicore-record-overflow=true
                        -mllvm -cce-aicore-addr-transform
                        -mllvm -cce-aicore-dcci-insert-for-scalar=false
                        -DL2_CACHE_HINT
                        ${COMPILE_DEFS}
                        -MD -MF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_kernel.d
                        -I${CMAKE_SOURCE_DIR}/include
                        -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
                        -I${CMAKE_SOURCE_DIR}/examples
                        -I${CMAKE_SOURCE_DIR}/examples/utils
                        -I${ASCEND_HOME_PATH}/include
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
                        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
                        -I${ASCEND_HOME_PATH}/aarch64-linux/include
                        -I${ASCEND_HOME_PATH}/include/experiment/runtime
                        -I${ASCEND_HOME_PATH}/include/experiment/msprof
                        -I${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime
                        -I${SHMEM_HOME_PATH}/shmem/include
                        -I${SHMEM_HOME_PATH}/shmem/src
                        -I${SHMEM_HOME_PATH}/shmem/src/device
                        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
                        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
                        ${INCLUDE_DIRS}
                        -L${ASCEND_HOME_PATH}/lib64
                        -L${SHMEM_HOME_PATH}/memfabric_hybrid/lib
                        -L${SHMEM_HOME_PATH}/shmem/lib
                        -lruntime -lstdc++ -lm -lascendcl
                        -lshmem -lgraph_base -ltiling_api -lplatform -lregister
                        ${KERNEL_SOURCES} -o ${CMAKE_BINARY_DIR}/lib/lib${TARGET}_kernel.so
        DEPENDS ${KERNEL_SOURCES}
        DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_kernel.d
        VERBATIM
        COMMENT "Generating kernel library ${TARGET}_kernel.so"
    )

    add_custom_target(${TARGET}_kernel ALL
        DEPENDS ${CMAKE_BINARY_DIR}/lib/lib${TARGET}_kernel.so
    )
    add_dependencies(${TARGET}_kernel_build ${TARGET}_kernel)
endfunction()

# Torch binding subdirectory
if(CATCCOS_TORCH_EXTENSION)
    add_subdirectory(torch_binding)
endif()

foreach(EXAMPLE
    matmul_allreduce
    allgather_matmul
    matmul_reduce_scatter
    ascend950_matmul_reduce_scatter
    allgather_matmul_with_gather_result
    grouped_matmul_alltoallv
    grouped_matmul_alltoallv_tla
    ascend950_grouped_matmul_alltoallv
    alltoallv_grouped_matmul
    allgather_matmul_dequant
    allgather_matmul_dequant_bias
    alltoallv_gmm_v2
    dispatch_gmm
    dispatch_gmm_swiglu
    gmm_alltoallv_v2
)
    add_subdirectory(${EXAMPLE})
endforeach()

if(RDMA_TRANSPORT)
    add_subdirectory(allgather_matmul_rdma)
endif()