#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
from enum import IntEnum
import numpy as np
import torch

class DataType(IntEnum):
    FLOAT = 0
    FLOAT16 = 1
    INT8 = 2
    BF16 = 27

    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))

    @property
    def torch_type(self):
        return {
            DataType.FLOAT: torch.float,
            DataType.FLOAT16: torch.float16,
            DataType.INT8: torch.int8,
            DataType.BF16: torch.bfloat16,
        }[self]

def tensor_to_file(tensor: torch.Tensor, file_name: str) -> None:
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)

def tensor_from_file(file_name: str, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.from_numpy(np.fromfile(file_name, dtype=np.float16)).view(torch.bfloat16)
    else:
        numpy_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        return torch.from_numpy(np.fromfile(file_name, numpy_dtype))



BF16_NAN = np.uint16(0xFFFF)
BF16_POS_INF = np.uint16(0x7F80)
BF16_NEG_INF = np.uint16(0xFF80)
BF16_ROUND_FACTOR = np.uint32(0x7FFF)


def fp32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    result = np.empty(values.shape, dtype=np.uint16)

    is_zero = values == np.float32(0.0)
    is_nan = np.isnan(values)
    is_inf = np.isinf(values)
    is_finite = ~(is_zero | is_nan | is_inf)

    result[is_zero] = np.uint16(0x0000)
    result[is_nan] = BF16_NAN
    result[is_inf] = np.where(values[is_inf] > 0, BF16_POS_INF, BF16_NEG_INF).astype(np.uint16)

    if np.any(is_finite):
        rounded = bits[is_finite].copy()
        rounded += BF16_ROUND_FACTOR + ((rounded >> np.uint32(16)) & np.uint32(1))
        result[is_finite] = (rounded >> np.uint32(16)).astype(np.uint16)

    return result


def bf16_bits_to_fp32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint16)
    fp32 = (values.astype(np.uint32) << np.uint32(16)).view(np.float32)
    fp32[(values >> np.uint16(1)) == 0] = np.float32(0.0)
    return fp32


def get_rtol(dtype: torch.dtype, compute_times: int) -> float:
    if dtype == torch.float16:
        return 2 ** (-8) if compute_times < 2048 else 2 ** (-7)
    elif dtype == torch.bfloat16:
        return 2 ** (-7) if compute_times < 2048 else 2 ** (-6)
    elif dtype == torch.float32:
        return 2 ** (-11) if compute_times < 2048 else 2 ** (-10)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")

