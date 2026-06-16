# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Default A2 (dav-c220). Set ON or pass build.sh -soc_type Ascend950 to additionally build a5agmm/a5mmrs.
option(CATCCOS_ENABLE_A5_BUILD "dynamic_tiling: additionally build a5agmm/a5mmrs kernels and hosts" OFF)

message(STATUS "dynamic_tiling: building A2 kernels (dav-c220)")
if(CATCCOS_ENABLE_A5_BUILD)
    message(STATUS "dynamic_tiling: additionally building Ascend950 kernels (dav-c310)")
endif()

option(RDMA_TRANSPORT "Enable Rdma transport" OFF)

# Profiling option (default OFF)
option(ENABLE_TIMER "Enable AscendTimer profiling instrumentation" OFF)
option(ENABLE_A5_PERF "Enable Ascend950 AIC/AIV split perf (no block-local-init)" OFF)

set(PROFILING_FLAGS_A2 "")
set(PROFILING_FLAGS_ASCEND950 "")
if(ENABLE_TIMER)
    # AscendTimer uses block-local class members; requires --cce-aicore-block-local-init on dav-c220.
    set(PROFILING_FLAGS_A2 -DENABLE_TIMER --cce-aicore-block-local-init)
    # dav-c310 (Ascend950) rejects block-local-init; a5agmm/a5mmrs kernel .so build without device timer for now.
    set(PROFILING_FLAGS_ASCEND950 "")
    message(STATUS "ENABLE_TIMER: ON for A2 (dav-c220) kernels only; Ascend950 kernels omit timer until c310 supports block-local-init")
endif()
if(ENABLE_A5_PERF)
    set(PROFILING_FLAGS_ASCEND950 -DENABLE_A5_PERF)
    message(STATUS "ENABLE_A5_PERF: ON for Ascend950 kernels (AIC=MatMul, AIV=Comm)")
endif()

# Compile kernel
set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    ${PROFILING_FLAGS_A2}
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)

set(CCEC_COMPILER_OPTIONS_ASCEND950
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c310
    ${PROFILING_FLAGS_ASCEND950}
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)

set(LIB_OPTIONS
    --shared -fPIC
)
set(INCLUDE_DIR
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
    -I${CMAKE_SOURCE_DIR}/tools
    -I${CMAKE_SOURCE_DIR}/utils
    -I${CMAKE_SOURCE_DIR}/examples/tools
    -I${CMAKE_SOURCE_DIR}/examples
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/aarch64-linux/include
    -I${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${SHMEM_HOME_PATH}/shmem/include
    -I${SHMEM_HOME_PATH}/shmem/src
    -I${SHMEM_HOME_PATH}/shmem/src/device
    -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(LINK_LIB
    -L${ASCEND_HOME_PATH}/lib64
    -L${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    -L${SHMEM_HOME_PATH}/shmem/lib
    -L${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -Wl,-rpath,${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -lruntime -lstdc++ -lascendcl
    -lmf_smem -lmf_hybm_core -lshmem
    -lgraph_base -ltiling_api -lunified_dlog -lplatform -lregister
    -Wl,--no-as-needed -ldl -Wl,--as-needed
)
set(KERNEL_LINK_LIB
    -L${ASCEND_HOME_PATH}/lib64
    -Wl,-rpath,${ASCEND_HOME_PATH}/lib64
    -lruntime -lstdc++ -lascendcl
    -lgraph_base -ltiling_api -lunified_dlog -lplatform -lregister
    -Wl,--no-as-needed -ldl -Wl,--as-needed
)
file(GLOB_RECURSE CATLASS_DEPEND_FILES ${CMAKE_SOURCE_DIR}/include/**.hpp)

add_custom_target(lib_impl)
function(add_impl_share_lib NAME)
    set(options ASCEND950)
    set(oneValueArgs "")
    set(multiValueArgs COMPILE_DEFINITIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EXTRA_DEFS "")
    foreach(DEF ${ARG_COMPILE_DEFINITIONS})
        list(APPEND EXTRA_DEFS "-D${DEF}")
    endforeach()

    if(ARG_ASCEND950)
        set(CCEC_OPTS ${CCEC_COMPILER_OPTIONS_ASCEND950})
    else()
        set(CCEC_OPTS ${CCEC_COMPILER_OPTIONS})
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        COMMAND ${CCEC} ${CCEC_OPTS} ${EXTRA_DEFS} ${LIB_OPTIONS} ${INCLUDE_DIR}
            ${NAME}.cpp ${KERNEL_LINK_LIB} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling shared library: lib${NAME}.so with defs: ${ARG_COMPILE_DEFINITIONS}"
    )
    add_custom_target(lib${NAME}.so ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so)
    add_dependencies(lib_impl lib${NAME}.so)
endfunction()

add_subdirectory(impl)

include_directories(
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    ${ASCEND_HOME_PATH}/include ${CMAKE_SOURCE_DIR}/include
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
    ${CMAKE_SOURCE_DIR}/examples
)
link_directories(${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${ASCEND_HOME_PATH}/lib64
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib ${SHMEM_HOME_PATH}/shmem/lib)

set(TILING_SOURCES
    tiling/coc_tiling_lut.cpp
    tiling/reducescatter_tiling.cpp
    tiling/allreduce_tiling.cpp
    tiling/allgather_tiling.cpp
)
add_library(tiling_lib SHARED ${TILING_SOURCES})
target_include_directories(tiling_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils
)
if(CATCCOS_ENABLE_A5_BUILD)
    target_compile_definitions(tiling_lib PUBLIC CATCCOS_ENABLE_A5_BUILD)
endif()
target_compile_options(tiling_lib PRIVATE -O3)

add_executable(dynamic_tiling main.cpp)
if(CATCCOS_ENABLE_A5_BUILD)
    target_compile_definitions(dynamic_tiling PRIVATE CATCCOS_ENABLE_A5_BUILD)
endif()
if(ENABLE_TIMER)
    target_compile_definitions(dynamic_tiling PRIVATE ENABLE_TIMER)
endif()
if(ENABLE_A5_PERF)
    target_compile_definitions(dynamic_tiling PRIVATE ENABLE_A5_PERF)
endif()
if(RDMA_TRANSPORT)
    target_compile_definitions(dynamic_tiling PRIVATE RDMA_TRANSPORT)
endif()
target_link_libraries(dynamic_tiling tiling_lib ${SHARE_LIB_LINK} runtime ascendcl shmem graph_base tiling_api unified_dlog platform register -Wl,--no-as-needed dl -Wl,--as-needed)
target_compile_options(dynamic_tiling PRIVATE -O3)
set_target_properties(dynamic_tiling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_dependencies(dynamic_tiling lib_impl tiling_lib)
