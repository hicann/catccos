#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import argparse
import torch
import math
from typing import Tuple
import numpy as np


def tensor_to_file(tensor, file_name):
    if tensor.dtype == torch.bfloat16:
        # bf16: view as uint16 and write 2-byte per element (same as utils.py)
        tensor.view(torch.uint16).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)


class MXFP8MatrixQuantizer:
    """
    MXFP8 matrix quantizer
    Supports:
    1. Custom quantization axis (0: row-wise, 1: column-wise)
    2. Custom block size
    3. FP8_E8M0FNU scale factor
    4. Output quantized matrix and scale matrix
    """

    DATA_FORMATS = {
        'E4M3': {
            'exp_bits': 4,
            'mantissa_bits': 3,
            'bias': 7,
            'emax': 8,
            'max_value': 448.0,
            'min_value': -448.0
        },
        'E5M2': {
            'exp_bits': 5,
            'mantissa_bits': 2,
            'bias': 15,
            'emax': 15,
            'max_value': 57344.0,
            'min_value': -57344.0
        }
    }

    SCALE_FORMAT = {
        'name': 'FP8_E8M0FNU',
        'exp_bits': 8,
        'mantissa_bits': 0,
        'bias': 128,
        'max_exp': 127,
        'min_exp': -128,
        'max_value': 2**127,
        'min_value': 2**-128,
        'signed': False,
        'allow_zero': True
    }

    def __init__(self, data_format: str = 'E5M2', axis: int = 1, block_size: int = 32, epsilon: float = 1e-12):
        if data_format not in self.DATA_FORMATS:
            raise ValueError(f"Unsupported data format: {data_format}, supported: {list(self.DATA_FORMATS.keys())}")
        if axis not in [0, 1]:
            raise ValueError("axis must be 0 (row) or 1 (column)")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.data_format = data_format
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        self.config = self.DATA_FORMATS[data_format]
        self._build_fp8_lookup_table()

    def _build_fp8_lookup_table(self):
        if self.data_format == 'E4M3':
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()

    def _build_e4m3_lookup_table(self):
        values = []
        for i in range(256):
            if i < 128:
                sign = 1
                val = i
            else:
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07
                if exp == 0:
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))
                value = sign * value

            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']
            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _build_e5m2_lookup_table(self):
        values = []
        for i in range(256):
            if i < 128:
                sign = 1
                val = i
            else:
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03
                if exp == 0:
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))
                value = sign * value

            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']
            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> Tuple[float, int]:
        max_abs = torch.max(torch.abs(block_data)).item()
        if max_abs < self.epsilon:
            return 1.0, 0

        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128

        exp = int(math.floor(log2_scale)) - self.config['emax']
        exp = max(self.SCALE_FORMAT['min_exp'], min(exp, self.SCALE_FORMAT['max_exp']))
        scale = 2.0 ** exp
        return scale, exp

    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        original_shape = data.shape
        data_flat = data.flatten()
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)
        quantized_flat = torch.zeros_like(data_clamped)

        for i in range(len(data_clamped)):
            val = data_clamped[i].item()
            distances = torch.abs(self.fp8_lut - val)
            min_idx = torch.argmin(distances).item()
            quantized_flat[i] = self.fp8_lut[min_idx]

        return quantized_flat.view(original_shape)

    def _process_block(self, block_data: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)
        if abs(scale) > self.epsilon:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data
        quantized_scaled = self._quantize_to_fp8(scaled_data)
        return quantized_scaled, scale, exp

    def quantize_matrix(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if matrix.dim() != 2:
            raise ValueError(f"Input must be 2D matrix, got {matrix.dim()}D")
        M, N = matrix.shape
        if self.axis == 0:
            return self._quantize_by_rows(matrix, M, N)
        else:
            return self._quantize_by_cols(matrix, M, N)

    def _quantize_by_rows(self, matrix: torch.Tensor, M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = (M + self.block_size - 1) // self.block_size
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)
            block_data = matrix[start_row:end_row, :]
            for col in range(N):
                col_data = block_data[:, col]
                quantized_col, scale, exp = self._process_block(col_data)
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def _quantize_by_cols(self, matrix: torch.Tensor, M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = (N + self.block_size - 1) // self.block_size
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            block_data = matrix[:, start_col:end_col]
            for row in range(M):
                row_data = block_data[row, :]
                quantized_row, scale, exp = self._process_block(row_data)
                quantized_matrix[row, start_col:end_col] = quantized_row
                scale_matrix[row, block_idx] = scale

        return quantized_matrix, scale_matrix

    def dequantize_matrix(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
        M, N = quantized_matrix.shape
        if self.axis == 0:
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_rows(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        num_blocks = scale_matrix.shape[0]
        dequantized_matrix = torch.zeros_like(quantized_matrix)
        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)
            for col in range(N):
                scale = scale_matrix[block_idx, col].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[start_row:end_row, col] = quantized_matrix[start_row:end_row, col] * scale
                else:
                    dequantized_matrix[start_row:end_row, col] = quantized_matrix[start_row:end_row, col]
        return dequantized_matrix

    def _dequantize_by_cols(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        num_blocks = scale_matrix.shape[1]
        dequantized_matrix = torch.zeros_like(quantized_matrix)
        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            for row in range(M):
                scale = scale_matrix[row, block_idx].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[row, start_col:end_col] = quantized_matrix[row, start_col:end_col] * scale
                else:
                    dequantized_matrix[row, start_col:end_col] = quantized_matrix[row, start_col:end_col]
        return dequantized_matrix


def gen_data_fp8_e5m2(row, col, axis):
    """Generate MXFP8 E5M2 quantized data with scales and dequantized fp32 result."""
    matrix = torch.randn((row, col), dtype=torch.float32)

    quantizer = MXFP8MatrixQuantizer(
        data_format='E5M2',
        axis=axis,
        block_size=32
    )

    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_data(rank, m, k, n, trans_a, trans_b, output_path):
    """Generate fp8 data and scale for a single rank, return (c_fp32, c_bf16).

    c_fp32: high-precision golden (dequantized fp32 @ dequantized fp32, fp32 matmul)
    c_bf16:  low-precision golden (fp8+scale dequant via repeat_interleave, matmul in bf16)
    """
    os.makedirs(output_path, exist_ok=True)

    # A: [M, K], axis=1 (column-wise quantization)
    a_fp8, a_scale, a_fp32 = gen_data_fp8_e5m2(m, k, 1)
    # B: [K, N], axis=0 (row-wise quantization)
    b_fp8, b_scale, b_fp32 = gen_data_fp8_e5m2(k, n, 0)

    # Save dequantized fp32 and fp8/scale for golden computation before any transpose
    a_fp32_golden = a_fp32
    b_fp32_golden = b_fp32
    a_fp8_golden = a_fp8
    b_fp8_golden = b_fp8
    a_scale_golden = a_scale  # [M, scaleK], before kernel-layout reshape
    b_scale_golden = b_scale  # [scaleK, N], before kernel-layout reshape

    # Reshape scales to match host.h / kernel layout
    # A scale: [M, scaleK] -> [M, scaleK//2, 2]
    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
    # B scale: [scaleK, N] -> [scaleK//2, 2, N]
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

    # Apply transpose for A
    if trans_a == 1:
        a_fp8 = a_fp8.t()
        a_scale = a_scale.permute(1, 0, 2)

    # Apply transpose for B (lcoc_a5 logic: trans_b==1 uses permute(2,0,1), else permute(0,2,1))
    if trans_b == 1:
        b_fp8 = b_fp8.t()
        b_scale = b_scale.permute(2, 0, 1)
    else:
        b_scale = b_scale.permute(0, 2, 1)

    # Write fp8 data files (as int8 bytes)
    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile(os.path.join(output_path, f'rank_{rank}_a.bin'))
    b_np.tofile(os.path.join(output_path, f'rank_{rank}_b.bin'))

    # Write scale files (as int8 bytes)
    a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_scale_np = torch.tensor(b_scale.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_scale_np.tofile(os.path.join(output_path, f'rank_{rank}_a_scale.bin'))
    b_scale_np.tofile(os.path.join(output_path, f'rank_{rank}_b_scale.bin'))

    # High-precision golden: dequantized fp32 matmul in fp32
    # Golden computation uses original matrix orientation (before file transpose),
    # matching lcoc_a5 logic: a_fp32 @ b_fp32 regardless of trans flags.
    c_fp32 = a_fp32_golden @ b_fp32_golden

    # Low-precision golden: fp8+scale via repeat_interleave, matmul in bf16
    # Use pre-transpose copies for golden computation
    # block_size=32 in quantizer => scaleK = k//32 groups per row/col
    block_size = 32
    a_dequant = (a_fp8_golden.to(torch.float32) *
                 torch.repeat_interleave(a_scale_golden.reshape(m, k // block_size), block_size, dim=1).to(torch.float32))
    b_dequant = (b_fp8_golden.to(torch.float32) *
                 torch.repeat_interleave(b_scale_golden.reshape(k // block_size, n), block_size, dim=0).to(torch.float32))
    c_bf16 = torch.matmul(a_dequant, b_dequant).to(torch.bfloat16)
    return c_fp32, c_bf16


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MXFP8 MatmulReduceScatter test data")
    parser.add_argument('out_dtype', type=int, nargs='?', default=27, help='Output dtype (27=BF16, for compatibility)')
    parser.add_argument('rank_size', type=int, help='Number of ranks')
    parser.add_argument('m', type=int, help='Matrix M dimension')
    parser.add_argument('n', type=int, help='Matrix N dimension')
    parser.add_argument('k', type=int, help='Matrix K dimension')
    parser.add_argument('trans_a', type=int, nargs='?', default=0, help='Transpose A (for compatibility)')
    parser.add_argument('trans_b', type=int, nargs='?', default=0, help='Transpose B (for compatibility)')
    parser.add_argument('data_dir', type=str, help='Directory to save data files')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Generate data for each rank, collecting high-precision and low-precision results
    matmul_result_fp32_list = []
    matmul_result_bf16_list = []
    for rank in range(args.rank_size):
        c_fp32, c_bf16 = gen_data(rank, args.m, args.k, args.n, args.trans_a, args.trans_b, data_dir)
        matmul_result_fp32_list.append(c_fp32)
        matmul_result_bf16_list.append(c_bf16)

    # Reduce-scatter (sum all rank results, then chunk by rows)
    def reduce_scatter_sum(results_list, dtype=None):
        result = results_list[0]
        for i in range(1, len(results_list)):
            result = result + results_list[i]
        rank_result_list = torch.chunk(result, args.rank_size, dim=0)
        return torch.cat(list(rank_result_list), dim=0)

    # High-precision golden (fp32)
    golden_fp32 = reduce_scatter_sum(matmul_result_fp32_list)
    golden_fp32.numpy().tofile(os.path.join(data_dir, 'golden.bin'))

    # Low-precision golden (bf16, same format as kernel output)
    golden_bf16 = reduce_scatter_sum(matmul_result_bf16_list, dtype=torch.bfloat16)
    tensor_to_file(golden_bf16, os.path.join(data_dir, 'golden_low.bin'))

    print(f"[gen_data] Generated data for {args.rank_size} ranks, shape M={args.m} K={args.k} N={args.n}")
    print(f"[gen_data] Data directory: {data_dir}")
