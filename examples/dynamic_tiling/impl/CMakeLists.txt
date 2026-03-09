foreach(SHARE_LIB_NAME
    kernel_bf16
    kernel_fp16
    kernel_int8
)
    add_impl_share_lib(${SHARE_LIB_NAME} ${SHARE_LIB_NAME}.cpp)
    LIST(APPEND SHARE_LIB_LINK ${SHARE_LIB_NAME})
endforeach()

set(SHARE_LIB_LINK ${SHARE_LIB_LINK} PARENT_SCOPE)