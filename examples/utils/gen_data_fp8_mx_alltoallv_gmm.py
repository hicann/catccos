import os
import argparse
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List
import numpy as np

from utils import DataType, tensor_to_file

class MXFP8MatrixQuantizer:
    """
    二维矩阵 MXFP8 量化器
    支持:
    1. 自定义量化轴 (0: 行方向, 1: 列方向)
    2. 自定义分块大小
    3. FP8_E8M0FNU 缩放因子
    4. 输出量化矩阵和缩放矩阵
    """

    # MXFP8 数据格式定义
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

    # FP8_E8M0FNU 缩放格式定义
    SCALE_FORMAT = {
        'name': 'FP8_E8M0FNU',
        'exp_bits': 8,
        'mantissa_bits': 0,
        'bias': 128,  # 偏置为128，0表示指数-128
        'max_exp': 127,
        'min_exp': -128,
        'max_value': 2**127,   # 约 1.7e38
        'min_value': 2**-128,  # 约 2.9e-39
        'signed': False,       # 无符号，仅正数
        'allow_zero': True     # 允许零值（指数-128表示零）
    }

    def __init__(self,
                 data_format: str = 'E4M3',
                 axis: int = 1,
                 block_size: int = 32,
                 epsilon: float = 1e-12):
        """
        初始化 MXFP8 矩阵量化器

        参数:
            data_format: 数据格式 'E4M3' 或 'E5M2'
            axis: 量化轴 (0: 行方向, 1: 列方向)
            block_size: 分块大小 (在量化轴上的元素数)
            epsilon: 防止除零的小值
        """
        if data_format not in self.DATA_FORMATS:
            raise ValueError(f"不支持的数据格式: {data_format}，支持: {list(self.DATA_FORMATS.keys())}")

        if axis not in [0, 1]:
            raise ValueError("axis 必须是 0 (行) 或 1 (列)")

        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")

        self.data_format = data_format
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        self.config = self.DATA_FORMATS[data_format]

        # 预构建 FP8 值查找表
        self._build_fp8_lookup_table()

    def _build_fp8_lookup_table(self):
        """构建 FP8 值查找表"""
        if self.data_format == 'E4M3':
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()

    def _build_e4m3_lookup_table(self):
        """构建 E4M3 值查找表"""
        values = []

        # E4M3: 总共 256 个可能值 (0-255)
        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:  # NaN，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _build_e5m2_lookup_table(self):
        """构建 E5M2 值查找表"""
        values = []

        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:  # NaN/Inf，替换为最大值
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> Tuple[float, int]:
        """
        计算 FP8_E8M0FNU 格式的缩放因子

        返回:
            scale: 缩放因子 (浮点数)
            exp: 指数 (E8M0FNU 格式)
        """
        # 计算块的最大绝对值
        max_abs = torch.max(torch.abs(block_data)).item()

        if max_abs < self.epsilon:
            # 全零或接近零的块
            return 1.0, 0  # 指数0表示缩放因子2^0=1

        # 计算对数 (log2)
        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128  # 最小值

        # 四舍五入到最近的整数 (指数)
        exp = int(math.floor(log2_scale)) - self.config['emax']

        # 钳位到 E8M0FNU 范围 [-128, 127]
        exp = max(self.SCALE_FORMAT['min_exp'],
                    min(exp, self.SCALE_FORMAT['max_exp']))

        # 计算缩放因子
        scale = 2.0 ** exp

        return scale, exp

    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP8 格式

        使用查找表进行最近邻量化
        """
        original_shape = data.shape
        data_flat = data.flatten()

        # 钳位到 FP8 范围
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)

        # 使用查找表量化
        quantized_flat = torch.zeros_like(data_clamped)

        # 向量化查找（比循环快）
        # 为每个值找到最接近的 FP8 值
        for i in range(len(data_clamped)):
            val = data_clamped[i].item()

            # 计算到所有 FP8 值的距离
            distances = torch.abs(self.fp8_lut - val)
            min_idx = torch.argmin(distances).item()

            quantized_flat[i] = self.fp8_lut[min_idx]

        return quantized_flat.view(original_shape)

    def _process_block(self, block_data: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        处理单个分块

        返回:
            quantized_block: 量化后的块
            scale: 缩放因子
            exp: 指数
        """
        # 计算缩放因子
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)

        # 应用缩放
        if abs(scale) > self.epsilon:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data  # 缩放因子为0，不缩放

        # 量化到 FP8
        quantized_scaled = self._quantize_to_fp8(scaled_data)

        return quantized_scaled, scale, exp

    def quantize_matrix(self,
                       matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量化二维矩阵

        参数:
            matrix: 输入二维矩阵 [M, N]

        返回:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵 (或编码后的缩放因子矩阵)
        """
        if matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {matrix.dim()}")

        M, N = matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._quantize_by_rows(matrix, M, N)
        else:  # 列方向量化
            return self._quantize_by_cols(matrix, M, N)

    def _quantize_by_rows(self,
                         matrix: torch.Tensor,
                         M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按行方向量化

        分块: 每 block_size 行一个块
        """
        # 计算分块数量
        num_blocks = (M + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        # 处理每个行块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束行
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 提取当前块
            block_data = matrix[start_row:end_row, :]

            # 对当前块的每一列单独处理（列方向）
            for col in range(N):
                # 提取列数据
                col_data = block_data[:, col]

                # 处理该列
                quantized_col, scale, exp = self._process_block(col_data)

                # 存储结果
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def _quantize_by_cols(self,
                         matrix: torch.Tensor,
                         M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按列方向量化

        分块: 每 block_size 列一个块
        """
        # 计算分块数量
        num_blocks = (N + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        # 处理每个列块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束列
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)

            # 提取当前块
            block_data = matrix[:, start_col:end_col]

            # 对当前块的每一行单独处理（行方向）
            for row in range(M):
                # 提取行数据
                row_data = block_data[row, :]

                # 处理该行
                quantized_row, scale, exp = self._process_block(row_data)

                # 存储结果
                quantized_matrix[row, start_col:end_col] = quantized_row
                scale_matrix[row, block_idx] = scale

        return quantized_matrix, scale_matrix

    def dequantize_matrix(self,
                         quantized_matrix: torch.Tensor,
                         scale_matrix: torch.Tensor) -> torch.Tensor:
        """
        反量化矩阵

        参数:
            quantized_matrix: 量化后的矩阵
            scale_matrix: 缩放因子矩阵（或编码矩阵）

        返回:
            dequantized_matrix: 反量化后的矩阵
        """
        M, N = quantized_matrix.shape

        # 根据量化轴进行反量化
        if self.axis == 0:  # 行方向
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:  # 列方向
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_rows(self,
                           quantized_matrix: torch.Tensor,
                           scale_matrix: torch.Tensor,
                           M: int, N: int) -> torch.Tensor:
        """行方向反量化"""
        num_blocks = scale_matrix.shape[0]
        dequantized_matrix = torch.zeros_like(quantized_matrix)

        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 应用缩放因子
            for col in range(N):
                scale = scale_matrix[block_idx, col].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[start_row:end_row, col] = (
                        quantized_matrix[start_row:end_row, col] * scale
                    )
                else:
                    dequantized_matrix[start_row:end_row, col] = (
                        quantized_matrix[start_row:end_row, col]
                    )

        return dequantized_matrix

    def _dequantize_by_cols(self,
                           quantized_matrix: torch.Tensor,
                           scale_matrix: torch.Tensor,
                           M: int, N: int) -> torch.Tensor:
        """列方向反量化"""
        num_blocks = scale_matrix.shape[1]
        dequantized_matrix = torch.zeros_like(quantized_matrix)

        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)

            # 应用缩放因子
            for row in range(M):
                scale = scale_matrix[row, block_idx].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[row, start_col:end_col] = (
                        quantized_matrix[row, start_col:end_col] * scale
                    )
                else:
                    dequantized_matrix[row, start_col:end_col] = (
                        quantized_matrix[row, start_col:end_col]
                    )

        return dequantized_matrix

def gen_data_a_fp8_e4m3(matrix: torch.Tensor, row, col, axis, transpose):

    quantizer = MXFP8MatrixQuantizer(
        data_format='E4M3',
        axis=axis,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    scale_matrix = scale_matrix.reshape(scale_matrix.shape[0], scale_matrix.shape[1] // 2, 2)

    if(transpose == 1):
        quantized_matrix = quantized_matrix.t()
        scale_matrix = scale_matrix.permute(1, 0, 2)

    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix

def gen_data_b_fp8_e4m3(matrix: torch.Tensor, local_expert_num, row, col, axis, transpose):
    
    quantizer = MXFP8MatrixQuantizer(
        data_format='E4M3',
        axis=axis,
        block_size=32
    )

    quantized_tensors = torch.zeros(local_expert_num, row, col, 
                                   dtype=torch.float8_e4m3fn)
    scale_tensors = torch.zeros(local_expert_num, row // 64, 2, col,
                                dtype=torch.float8_e8m0fnu)
    if(transpose == 1):
        scale_tensors = scale_tensors.permute(0, 3, 1, 2)
    else:
        scale_tensors = scale_tensors.permute(0, 1, 3, 2)
    
    for i in range(local_expert_num):
        matrix_kn = matrix[i]
        quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix_kn)

        scale_matrix = scale_matrix.reshape(scale_matrix.shape[0] // 2, 2, scale_matrix.shape[1])
        if(transpose == 1):
            quantized_matrix = quantized_matrix.t()
            scale_matrix = scale_matrix.permute(2, 0, 1)
        else:
            scale_matrix = scale_matrix.permute(0, 2, 1)

        quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
        scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

        quantized_tensors[i] = quantized_matrix
        scale_tensors[i] = scale_matrix

    return quantized_tensors, scale_tensors

def gen_dequantized_matrix_a_fp8_e4m3(matrix: torch.Tensor, rank_size, row, col, axis):
    
    quantizer = MXFP8MatrixQuantizer(
        data_format='E4M3',
        axis=axis,
        block_size=32
    )
    dequantized_tensors = torch.zeros(rank_size, row, col, 
                                      dtype=matrix.dtype)
    
    for i in range(rank_size):
        matrix_mk = matrix[i]
        # 量化
        quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix_mk)

        # 反量化
        dequantized_matrix = quantizer.dequantize_matrix(
            quantized_matrix,
            scale_matrix
        )

        dequantized_tensors[i] = dequantized_matrix.to(matrix.dtype)

    return dequantized_tensors

def gen_dequantized_matrix_b_fp8_e4m3(matrix: torch.Tensor, rank_size, local_expert_num, row, col, axis):
    
    quantizer = MXFP8MatrixQuantizer(
        data_format='E4M3',
        axis=axis,
        block_size=32
    )

    dequantized_tensors = torch.zeros(rank_size, local_expert_num, row, col, 
                                      dtype=matrix.dtype)
    
    for i in range(rank_size):
        for j in range(local_expert_num):
            matrix_kn = matrix[i][j]
            # 量化
            quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix_kn)

            # 反量化
            dequantized_matrix = quantizer.dequantize_matrix(
                quantized_matrix,
                scale_matrix
            )
            dequantized_tensors[i][j] = dequantized_matrix.to(matrix.dtype)

    return dequantized_tensors

def random_uniform(
    low: float, high: float,
    size: Tuple,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # 计算两次采样的长度
    element_num = math.prod(size)
    l0 = math.floor(math.sqrt(element_num))
    l0 = l0 if l0 * (l0 + 1) >= element_num else l0 + 1
    l1 = l0 + 1
    # 生成两次采样在 U(0, 1) 下的随机 tensor
    sample1 = torch.rand(size=(l0,), dtype=dtype).repeat(math.ceil(element_num / l0))[:element_num]
    sample2 = torch.rand(size=(l1,), dtype=dtype).repeat(math.ceil(element_num / l1))[:element_num]
    
    # 将两次采样结果组合为 U(0, 1) 的 tensor
    sample_add = sample1 + sample2
    sample_result = sample_add - torch.floor(sample_add)
    return (sample_result * (high - low) + low).reshape(size)

def generate_constrained_row_sum_tensor(
    row_sum: int,
    size: Tuple[int, int],
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    row, col = size
    tensor = torch.empty(size=size, dtype=torch.int32)
    tensor[:, :-1] = torch.randint(low=0, high=row_sum + 1, size=(row, col - 1), dtype=dtype).sort(dim=1).values
    tensor[:, -1] = row_sum
    tensor[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    return tensor

def simulate_all_to_all_v_with_permute(
    input_list: List[torch.Tensor],
    global_tokens_per_expert: torch.Tensor,
    ep_size: int
) -> List[torch.Tensor]:
    rank_size, expert_num = global_tokens_per_expert.shape
    local_expert_num = expert_num // ep_size
    input_groups_list = [
        torch.split(input, global_tokens_per_expert[rank_idx].tolist(), dim=0)
        for rank_idx, input in enumerate(input_list)
    ]
    return [
        torch.cat([
            input_groups[(output_rank_idx % ep_size) * local_expert_num + local_expert_idx]
            for local_expert_idx in range(local_expert_num)
            for input_groups in input_groups_list
        ], dim=0)
        for output_rank_idx in range(rank_size)
    ]

def grouped_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor
) -> torch.Tensor:
    activation_list = torch.split(activation.to(torch.float32), group_list.tolist(), dim=0)
    weight_list = weight.to(torch.float32)
    results: list[torch.Tensor] = [None] * len(activation_list) # type: ignore
    for i, (activation, weight) in enumerate(zip(activation_list, weight_list)):
        results[i] = torch.mm(activation, weight)
    return torch.cat(results, dim=0)

def generate_data(args: argparse.Namespace) -> None:
    out_type = args.out_type.torch_type
    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    trans_a, trans_b = args.trans_a, args.trans_b
    expert_num = args.expert
    ep_size = args.ep
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)

    tp_size = rank_size // ep_size
    local_expert_num = expert_num // ep_size

    matrix_a_origin_world = random_uniform(-2.0, 2.0, size=(rank_size, m, k))
    matrix_b_origin_world = random_uniform(-2.0, 2.0, size=(rank_size, local_expert_num, k, n))

    matrix_a_dequantized_origin_world = gen_dequantized_matrix_a_fp8_e4m3(matrix_a_origin_world, rank_size, m, k, 1)
    matrix_b_dequantized_origin_world = gen_dequantized_matrix_b_fp8_e4m3(matrix_b_origin_world, rank_size, local_expert_num, k, n, 0)

    global_tokens_per_expert = generate_constrained_row_sum_tensor(row_sum=m, size=(rank_size, expert_num))
    global_tokens_per_local_expert_world = global_tokens_per_expert.reshape(
        shape=(rank_size, ep_size, local_expert_num)).permute(1, 0, 2).repeat(tp_size, 1, 1)

    print(global_tokens_per_expert)

    permute_a_list = simulate_all_to_all_v_with_permute(list(matrix_a_dequantized_origin_world), global_tokens_per_expert, ep_size)

    group_list_world = global_tokens_per_local_expert_world.sum(dim=1)
    output_list = [
        grouped_matmul(input, other, group_list)
        for input, other, group_list in zip(permute_a_list, list(matrix_b_dequantized_origin_world), list(group_list_world))
    ]

    matrix_a_world = matrix_a_origin_world.mT if trans_a else matrix_a_origin_world
    matrix_b_world = matrix_b_origin_world.mT if trans_b else matrix_b_origin_world

    for rank_idx in range(rank_size):
        a_fp8, a_scale = gen_data_a_fp8_e4m3(matrix_a_world[rank_idx], m, k, 1, args.trans_a)
        b_fp8, b_scale = gen_data_b_fp8_e4m3(matrix_b_world[rank_idx], local_expert_num, k, n, 0, args.trans_b)
        a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8)
        b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8)
        a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8)
        b_scale_np = torch.tensor(b_scale.flatten().untyped_storage(), dtype=torch.int8)
        tensor_to_file(a_np, os.path.join(data_dir, f"input_a_{rank_idx}.bin"))
        tensor_to_file(a_scale_np, os.path.join(data_dir, f"input_a_scale_{rank_idx}.bin"))
        tensor_to_file(b_np, os.path.join(data_dir, f"input_b_{rank_idx}.bin"))
        tensor_to_file(b_scale_np, os.path.join(data_dir, f"input_b_scale_{rank_idx}.bin"))
        tensor_to_file(global_tokens_per_expert[rank_idx], os.path.join(data_dir, f"local_tokens_per_expert_{rank_idx}.bin"))
        tensor_to_file(global_tokens_per_local_expert_world[rank_idx],
                       os.path.join(data_dir, f"global_tokens_per_local_expert_{rank_idx}.bin"))

        output = output_list[rank_idx]
        padding_output = torch.zeros(size=(rank_size * m, n), dtype=torch.float32)
        padding_output[:output.shape[0]] = output
        tensor_to_file(padding_output, os.path.join(data_dir, f"golden_{rank_idx}.bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_type', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('trans_a', type=int)
    parser.add_argument('trans_b', type=int)
    parser.add_argument('--expert', type=int, default=8)
    parser.add_argument('--ep', type=int, default=2)
    parser.add_argument('data_dir', type=str,
                        help='Directory to save the data files',
                        default="./input")
    args = parser.parse_args()

    generate_data(args)
