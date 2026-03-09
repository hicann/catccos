# Compile kernel
set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
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
    -I${CMAKE_SOURCE_DIR}/examples/utils
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
)
file(GLOB_RECURSE CATLASS_DEPEND_FILES ${CMAKE_SOURCE_DIR}/include/**.hpp)

add_custom_target(lib_impl)
function(add_impl_share_lib NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        COMMAND ${CCEC} ${CCEC_COMPILER_OPTIONS} ${LIB_OPTIONS} ${INCLUDE_DIR}
            ${NAME}.cpp -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling shared library: lib${NAME}.so"
    )
    add_custom_target(lib${NAME}.so ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so)
    add_dependencies(lib_impl lib${NAME}.so)
endfunction()

add_subdirectory(impl)

include_directories(
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/include ${CMAKE_SOURCE_DIR}/include
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/examples/utils
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
target_include_directories(tiling_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tiling)
target_compile_options(tiling_lib PRIVATE -O3)

add_executable(dynamic_tiling main.cpp)
target_link_libraries(dynamic_tiling tiling_lib runtime ascendcl shmem ${SHARE_LIB_LINK})
target_compile_options(dynamic_tiling PRIVATE -O3)
set_target_properties(dynamic_tiling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_dependencies(dynamic_tiling lib_impl tiling_lib)
