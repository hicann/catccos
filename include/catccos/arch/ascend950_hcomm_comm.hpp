/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_ARCH_ASCEND950_HCOMM_COMM_HPP
#define CATCCOS_ARCH_ASCEND950_HCOMM_COMM_HPP

#include <cstddef>
#include <cstdint>

#include "catlass/catlass.hpp"

namespace Catccos::Arch {

/// Ascend950 communication backend implemented by the CCU HCCL client.
///
/// Ascend950 does not support the legacy GetWindowsInAddr/GetRankDim device
/// interfaces.  The backend therefore gathers every communication tile into a
/// rank-local workspace and exposes that workspace to the compute side.
template <class TilingData_>
class Ascend950HcommComm {
public:
    using TilingData = TilingData_;
    static constexpr bool IS_COLLECTIVE_ALL_GATHER = true;

    struct Params {
        TilingData *tilingData;
        uint64_t ccTilingOffset;
        GM_ADDR workspace;
        uint32_t rankSize;

        CATLASS_HOST_DEVICE
        Params() : tilingData(nullptr), ccTilingOffset(0), workspace(nullptr), rankSize(0)
        {
        }

        CATLASS_HOST_DEVICE
        Params(TilingData *tilingData_, uint64_t ccTilingOffset_, GM_ADDR workspace_, uint32_t rankSize_)
            : tilingData(tilingData_),
              ccTilingOffset(ccTilingOffset_),
              workspace(workspace_),
              rankSize(rankSize_)
        {
        }
    };

    CATLASS_DEVICE
    Ascend950HcommComm()
    {
    }

    CATLASS_DEVICE
    bool Init(Params const &params)
    {
        workspace_ = params.workspace;
        rankSize_ = params.rankSize;

        if (g_coreType == AscendC::AIV) {
            GM_ADDR context = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
            hccl_.InitV2(context, params.tilingData);
            int32_t ret = hccl_.SetCcTilingV2(params.ccTilingOffset);
            if (ret != HCCL_SUCCESS_CODE) {
                if (IsDiagnosticCore()) {
                    AscendC::PRINTF("[AGMM_DIAG] SetCcTilingV2 failed, ret=%d, offset=%llu\n", ret,
                                    static_cast<unsigned long long>(params.ccTilingOffset));
                }
                return false;
            }
            if (IsDiagnosticCore()) {
                AscendC::PRINTF("[AGMM_DIAG] SetCcTilingV2 success, offset=%llu\n",
                                static_cast<unsigned long long>(params.ccTilingOffset));
            }
        }
        return true;
    }

    CATLASS_DEVICE
    bool AllGather(GM_ADDR sendBuffer, GM_ADDR recvBuffer, uint64_t sendCount, uint64_t strideCount)
    {
        uint32_t sequence = sequence_++;
        if (IsDiagnosticCore()) {
            AscendC::PRINTF("[AGMM_DIAG] AllGather begin, seq=%u, sendCount=%llu, strideCount=%llu\n", sequence,
                            static_cast<unsigned long long>(sendCount),
                            static_cast<unsigned long long>(strideCount));
        }

        auto handle = hccl_.AllGather<true>(
            sendBuffer, recvBuffer, sendCount, AscendC::HcclDataType::HCCL_DATA_TYPE_FP16, strideCount);
        if (handle < 0) {
            if (IsDiagnosticCore()) {
                AscendC::PRINTF("[AGMM_DIAG] AllGather prepare failed, seq=%u, handle=%d\n", sequence,
                                static_cast<int32_t>(handle));
            }
            return false;
        }

        if (IsDiagnosticCore()) {
            AscendC::PRINTF("[AGMM_DIAG] Wait begin, seq=%u, handle=%d\n", sequence,
                            static_cast<int32_t>(handle));
        }
        int32_t ret = hccl_.Wait(handle);
        if (ret != HCCL_SUCCESS_CODE) {
            if (IsDiagnosticCore()) {
                AscendC::PRINTF("[AGMM_DIAG] Wait failed, seq=%u, handle=%d, ret=%d\n", sequence,
                                static_cast<int32_t>(handle), ret);
            }
            return false;
        }
        if (IsDiagnosticCore()) {
            AscendC::PRINTF("[AGMM_DIAG] Wait end, seq=%u, handle=%d\n", sequence,
                            static_cast<int32_t>(handle));
        }
        return true;
    }

    CATLASS_DEVICE
    void Finalize()
    {
        if (IsDiagnosticCore()) {
            AscendC::PRINTF("[AGMM_DIAG] Finalize begin\n");
        }
        AscendC::SyncAll<true>();
        hccl_.Finalize();
        if (IsDiagnosticCore()) {
            AscendC::PRINTF("[AGMM_DIAG] Finalize end\n");
        }
    }

    CATLASS_DEVICE
    void CrossRankSync()
    {
        // AllGather + Wait is the cross-rank synchronization boundary.
    }

    CATLASS_DEVICE
    auto GetPeerMem() const
    {
        return workspace_;
    }

    CATLASS_DEVICE
    auto GetPeerMem(int32_t) const
    {
        return workspace_;
    }

    CATLASS_DEVICE
    uint32_t GetRankIdx() const
    {
        // The collective path does not need the local rank id. AllGather writes
        // results in rank order into the local receive buffer.
        return 0;
    }

    CATLASS_DEVICE
    uint32_t GetRankSize() const
    {
        return rankSize_;
    }

private:
    static constexpr int32_t HCCL_SUCCESS_CODE = 0;

    CATLASS_DEVICE
    bool IsDiagnosticCore() const
    {
        return g_coreType == AscendC::AIV && AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0;
    }

    AscendC::Hccl<AscendC::HCCL_SERVER_TYPE_CCU> hccl_{};
    GM_ADDR workspace_{nullptr};
    uint32_t rankSize_{0};
    uint32_t sequence_{0};
};

} // namespace Catccos::Arch

#endif // CATCCOS_ARCH_ASCEND950_HCOMM_COMM_HPP
