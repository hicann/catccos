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
    add_impl_share_lib(${SHARE_LIB_NAME} ${SHARE_LIB_NAME}.cpp)
    LIST(APPEND SHARE_LIB_LINK ${SHARE_LIB_NAME})
endforeach()

set(SHARE_LIB_LINK ${SHARE_LIB_LINK} PARENT_SCOPE)