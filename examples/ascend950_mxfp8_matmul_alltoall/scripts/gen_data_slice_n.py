#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
import os
import argparse
import torch
import math
from typing import Tuple
import numpy as np

def tensor_to_file(tensor, file_name):
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)


class MXFP8MatrixQuantizer:
    """
    MXFP8 (E4M3) matrix quantizer with FP8_E8M0FNU scale factors.
    Supports:
    1. Custom quantization axis (0: row-wise, 1: column-wise)
    2. Block size 32 (MX scale group)
    3. FP8_E8M0FNU scale factor
    4. Output quantized matrix and scale matrix
    """

    DATA_FORMATS = {
        'E4M3': {
            'exp_bits': 4,
            'mantissa_bits': 3,
            'bias': 7,
            'max_value': 448.0,
            'min_value': -448.0
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

    def __init__(self, data_format: str = 'E4M3', axis: int = 1, block_size: int = 32, epsilon: float = 1e-12):
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

    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> Tuple[float, int]:
        max_abs = torch.max(torch.abs(block_data)).item()
        if max_abs < self.epsilon:
            return 1.0, 0

        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128

        exp = int(math.floor(log2_scale))
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

    def _dequantize_by_rows(self, quantized_matrix, scale_matrix, M, N):
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

    def _dequantize_by_cols(self, quantized_matrix, scale_matrix, M, N):
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


def gen_data_fp8_e4m3(row, col, axis):
    """Generate MXFP8 E4M3 quantized data with scales and dequantized fp32 result."""
    matrix = torch.randn((row, col), dtype=torch.float32) * 10

    quantizer = MXFP8MatrixQuantizer(data_format='E4M3', axis=axis, block_size=32)
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_matmul_alltoall_n_data():
    """
    MXFP8 MatMul + AllToAll-along-N data generation (matches the production
    MatmulAlltoAll "split along N" semantic):
    - Each rank i has A_i of shape (M, K) (different rows per rank) and a shared
      B of shape (K, N) (replicated).
    - MatMul: C_i = dequant(A_i) @ dequant(B), producing C_i of shape (M, N),
      full N.
    - AllToAll along N: chunkN = N // rank_size. Rank j receives from each rank i
      the j-th N-slice C_i[:, j*chunkN:(j+1)*chunkN] (shape (M, chunkN)).
    - After AllToAll, rank j holds D_j of shape (rank_size*M, chunkN), where the
      row-block [i*M:(i+1)*M] came from rank i.
    - Golden for rank j: assembled D_j (fp32).
    """
    parser = argparse.ArgumentParser(description="Generate MXFP8 MatmulAllToAll-N test data")
    parser.add_argument('out_dtype', type=int, nargs='?', default=1, help='Output dtype (1=FP16)')
    parser.add_argument('rank_size', type=int, help='Number of ranks')
    parser.add_argument('m', type=int, help='Per-rank M dimension (rows of A_i)')
    parser.add_argument('n', type=int, help='Matrix N dimension (must be divisible by rank_size)')
    parser.add_argument('k', type=int, help='Matrix K dimension')
    parser.add_argument('trans_a', type=int, nargs='?', default=0, help='Transpose A (compatibility)')
    parser.add_argument('trans_b', type=int, nargs='?', default=0, help='Transpose B (compatibility)')
    parser.add_argument('data_dir', type=str, help='Directory to save data files')
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    rank_size = args.rank_size
    if N % rank_size != 0:
        raise ValueError(f"N ({N}) must be divisible by rank_size ({rank_size})")
    chunkN = N // rank_size

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Generate per-rank A matrices (E4M3 + scale), collect dequantized fp32 for golden
    a_dequant_list = []
    for i in range(rank_size):
        a_fp8, a_scale, a_fp32 = gen_data_fp8_e4m3(M, K, 1)  # axis=1 (column-wise)
        a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
        if args.trans_a == 1:
            a_fp8 = a_fp8.t()
            a_scale = a_scale.permute(1, 0, 2)
        a_gm_path = os.path.join(data_dir, f"rank_{i}_a.bin")
        a_scale_gm_path = os.path.join(data_dir, f"rank_{i}_a_scale.bin")
        a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8)
        a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8)
        tensor_to_file(a_np, a_gm_path)
        tensor_to_file(a_scale_np, a_scale_gm_path)
        a_dequant_list.append(a_fp32)

    # Generate shared B (E4M3 + scale), same content on every rank
    b_fp8, b_scale, b_fp32 = gen_data_fp8_e4m3(K, N, 0)  # axis=0 (row-wise)
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])
    if args.trans_b == 1:
        b_fp8 = b_fp8.t()
        b_scale = b_scale.permute(2, 0, 1)
    else:
        b_scale = b_scale.permute(0, 2, 1)
    for i in range(rank_size):
        b_gm_path = os.path.join(data_dir, f"rank_{i}_b.bin")
        b_scale_gm_path = os.path.join(data_dir, f"rank_{i}_b_scale.bin")
        b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8)
        b_scale_np = torch.tensor(b_scale.flatten().untyped_storage(), dtype=torch.int8)
        tensor_to_file(b_np, b_gm_path)
        tensor_to_file(b_scale_np, b_scale_gm_path)

    # Per-rank MatMul results (fp32): C_i = A_i @ B, shape (M, N)
    c_list = [torch.matmul(a_dequant_list[i], b_fp32) for i in range(rank_size)]

    # Golden for each rank after AllToAll along N:
    # rank j assembles [C_0[:, j*chunkN], C_1[:, j*chunkN], ..., C_{R-1}[:, j*chunkN]]
    # -> shape (rank_size*M, chunkN)
    for j in range(rank_size):
        chunks = [c_list[i][:, j * chunkN:(j + 1) * chunkN] for i in range(rank_size)]
        golden_j = torch.cat(chunks, dim=0)  # (rank_size*M, chunkN) fp32
        tensor_to_file(golden_j, os.path.join(data_dir, f"golden_{j}.bin"))

    print(
        f"Generated MXFP8 MatMul+AllToAll(N) data: M={M}, N={N}, K={K}, "
        f"rank_size={rank_size}, chunkN={chunkN}"
    )
    print(f"Data directory: {data_dir}")


if __name__ == "__main__":
    gen_matmul_alltoall_n_data()
