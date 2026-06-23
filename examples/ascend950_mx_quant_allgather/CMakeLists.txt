
 	 # Copyright (c) 2026 Huawei Technologies Co., Ltd.
 	 # This file is a part of the CANN Open Software.
 	 # Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 	 # Please refer to the License for details. You may not use this file except in compliance with the License.
 	 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 	 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 	 # See LICENSE in the root of the software repository for the full text of the License.
 	 
 	 # Default: fp8_e4m3
 	 catccos_example_add_executable(
 	     ascend950_mx_quant_allgather
 	     SOURCES main.cpp
 	     COMPILE_DEFINITIONS CATLASS_ARCH=3510 MX_QUANT_DTYPE=0
 	 )

 	 # fp8_e5m2
 	 catccos_example_add_executable(
 	     ascend950_mx_quant_allgather_e5m2
 	     SOURCES main.cpp
 	     COMPILE_DEFINITIONS CATLASS_ARCH=3510 MX_QUANT_DTYPE=1
 	 )

 	 # fp4_e2m1
 	 catccos_example_add_executable(
 	     ascend950_mx_quant_allgather_fp4_e2m1
 	     SOURCES main.cpp
 	     COMPILE_DEFINITIONS CATLASS_ARCH=3510 MX_QUANT_DTYPE=2
 	 )

 	 # fp4_e1m2
 	 catccos_example_add_executable(
 	     ascend950_mx_quant_allgather_fp4_e1m2
 	     SOURCES main.cpp
 	     COMPILE_DEFINITIONS CATLASS_ARCH=3510 MX_QUANT_DTYPE=3
 	 )
