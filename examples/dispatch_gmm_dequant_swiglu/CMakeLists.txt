catccos_example_add_executable(
    dispatch_gmm_dequant_swiglu
    SOURCES main.cpp
    INCLUDES ../aux_ops ../aux_ops/moe_init_routing_quant_v2
    COMPILE_DEFINITIONS CATLASS_ARCH=2201
)