/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

#include "../op_kernel/all_gather_matmul_hccl_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto tiling = context->GetTilingData<AllGatherMatmulHcclTiling>();
    context->SetBlockDim(platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAic());

    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(0);
    uint32_t opType = 8U;
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);

    auto &params = tiling->params;
    params.m = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    params.n = context->GetInputShape(1)->GetOriginShape().GetDim(1);
    params.k = context->GetInputShape(0)->GetOriginShape().GetDim(1);

    params.m0 = 128;
    params.n0 = 256;
    params.k0 = 256;
    params.commTileM = 64;
    params.commInterval = 3;
    params.commNpuSplit = 1;
    params.commDataSplit = 20;
    params.commBlockM = 64;
    params.rankSize = *attrs->GetInt(1);
    params.segmentSize = HCCL_WINDOW_SIZE;

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = HCCL_WINDOW_SIZE;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto &aShape = context->GetInputShape(0);
    const auto &bShape = context->GetInputShape(1);
    auto cShape = context->GetOutputShape(0);

    auto m = aShape->GetDim(0);
    auto n = bShape->GetDim(1);
    auto rankSize = *context->GetAttrs()->GetInt(1);
    *cShape = {m * rankSize, n};

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AllGatherMatmulHccl : public OpDef {
public:
    explicit AllGatherMatmulHccl(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Attr("group").String();
        this->Attr("rankSize").Int(16);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");

        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AllGatherMatmulHccl);
}
