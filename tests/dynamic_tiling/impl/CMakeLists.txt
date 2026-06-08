# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

foreach(SHARE_LIB_NAME
    kernel_bf16
    kernel_fp16
    kernel_int8
)
    add_impl_share_lib(${SHARE_LIB_NAME} COMPILE_DEFINITIONS "CATLASS_ARCH=2201")
    LIST(APPEND SHARE_LIB_LINK ${SHARE_LIB_NAME})
endforeach()

if(CATCCOS_ENABLE_A5_BUILD)
    foreach(SHARE_LIB_NAME
        kernel_fp16_ascend950
        kernel_bf16_ascend950
    )
        add_impl_share_lib(${SHARE_LIB_NAME} ASCEND950 COMPILE_DEFINITIONS "CATLASS_ARCH=3510")
        LIST(APPEND SHARE_LIB_LINK ${SHARE_LIB_NAME})
    endforeach()
endif()

set(SHARE_LIB_LINK ${SHARE_LIB_LINK} PARENT_SCOPE)
