/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_KERNEL_ADAPTER_HPP
#define CATCCOS_KERNEL_ADAPTER_HPP

#include "catccos/catccos.hpp"

// 只有开启 Timer 时才需要引入具体实现头文件
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Device {

/// Profiling-aware kernel adapter: sets timer buffer conditionally
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrTimer = nullptr)
{
#ifdef ENABLE_TIMER
    Catccos::SetTimerBuffer(ptrTimer);
#endif
    Operator op;
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr, GM_ADDR ptrTimer = nullptr)
{
#ifdef ENABLE_TIMER
    Catccos::SetTimerBuffer(ptrTimer);
#endif
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
    op(params);
}

} // namespace Catccos::DGemm::Device

#endif // CATCCOS_KERNEL_ADAPTER_HPP