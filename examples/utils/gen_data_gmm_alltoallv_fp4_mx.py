#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List
import numpy as np
from utils import DataType, tensor_to_file

class MXFP4MatrixQuantizer:
    """
    二维矩阵 MXFP4 量化器
    支持:
    1. 自定义量化轴 (0: 行方向, 1: 列方向)
    2. 自定义分块大小
    3. FP8_E8M0FNU 缩放因子
    4. 输出量化矩阵和缩放矩阵
    """

    # FP4 数据格式定义
    DATA_FORMATS = {
        'E2M1': {
            'exp_bits': 2,
            'mantissa_bits': 1,
            'bias': 1,
            'emax': 2,
            'max_value': 6.0,
            'min_value': -6.0
        },
        'E1M2': {
            'exp_bits': 1,
            'mantissa_bits': 2,
            'bias': 1,
            'emax': 0,
            'max_value': 1.75,
            'min_value': -1.75
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
                 data_format: str = 'E2M1',
                 axis: int = 1,
                 block_size: int = 32,
                 epsilon: float = 1e-12):
        """
        初始化 MXFP4 矩阵量化器

        参数:
            data_format: 数据格式 'E2M1' 或 'E1M2'
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

        # 预构建 FP4 值查找表
        self._build_fp4_lookup_table()

        # 量化统计
        self.stats = {
            'total_blocks': 0,
            'scale_exponents': [],
            'quantization_errors': []
        }

    def _build_fp4_lookup_table(self):
        """构建 FP4 值查找表"""
        if self.data_format == 'E2M1':
            self._build_e2m1_lookup_table()
        else:
            self._build_e1m2_lookup_table()
            
        # 创建正向查找表
        self.fp4_lut = torch.tensor(self.value_table, dtype=torch.float32)
        self.fp4_min = self.config['min_value']
        self.fp4_max = self.config['max_value']

    def _build_e2m1_lookup_table(self):
        """构建 E2M1 值查找表"""
        # E2M1: 1位符号 + 2位指数 + 1位尾数 = 4位，共16个值
        self.value_table = []

        # 生成所有可能的4位值 (0-15)
        for i in range(16):
            # 解码4位值
            # 位布局: [符号(1位)][指数(2位)][尾数(1位)]
            sign = (i >> 3) & 0x01  # 第3位是符号位
            exp = (i >> 1) & 0x03   # 第1-2位是指数
            mantissa = i & 0x01     # 第0位是尾数

            # E2M1 格式解码
            if exp == 0:
                # 次正规数
                if mantissa == 0:
                    value = 0.0
                else:
                    # 公式: 2^(1-偏置) * (尾数/2^尾数位数)
                    value = (mantissa / 2.0) * (2.0 ** (1 - self.config['bias']))
            else:
                # 正规数
                # 公式: 2^(指数-偏置) * (1 + 尾数/2^尾数位数)
                value = (1.0 + mantissa / 2.0) * (2.0 ** (exp - self.config['bias']))

            # 应用符号
            if sign == 1:
                value = -value

            self.value_table.append(value)

    def _build_e1m2_lookup_table(self):
        """构建 E1M2 值查找表"""
        # E1M2: 1位符号 + 1位指数 + 2位尾数 = 4位，共16个值
        self.value_table = []

        for i in range(16):
            # 解码4位值
            # 位布局: [符号(1位)][指数(1位)][尾数(2位)]
            sign = (i >> 3) & 0x01  # 第3位是符号位
            exp = (i >> 2) & 0x01   # 第2位是指数
            mantissa = i & 0x03     # 第0-1位是尾数 (2位)

            # E1M2 格式解码
            if exp == 0:
                # 次正规数或零
                if mantissa == 0:
                    value = 0.0
                else:
                    # 公式: 2^(1-偏置) * (尾数/2^尾数位数)
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
            else:
                # 正规数
                # 公式: 2^(指数-偏置) * (1 + 尾数/2^尾数位数)
                value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))

            # 应用符号
            if sign == 1:
                value = -value

            self.value_table.append(value)

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

    def _quantize_to_fp4(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP4 格式

        使用查找表进行最近邻量化
        """
        original_shape = data.shape
        data_flat = data.flatten()

        # 钳位到 FP4 范围
        data_clamped = torch.clamp(data_flat, self.fp4_min, self.fp4_max)

        # 使用查找表量化
        quantized_flat = torch.zeros_like(data_clamped)

        # 向量化查找（比循环快）
        # 为每个值找到最接近的 FP4 值
        for i in range(len(data_clamped)):
            val = data_clamped[i].item()

            # 计算到所有 FP4 值的距离
            distances = torch.abs(self.fp4_lut - val)
            min_idx = torch.argmin(distances).item()

            quantized_flat[i] = self.fp4_lut[min_idx]

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

        # 记录统计
        self.stats['scale_exponents'].append(exp)

        # 应用缩放
        if abs(scale) > self.epsilon:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data  # 缩放因子为0，不缩放

        # 量化到 FP4
        quantized_scaled = self._quantize_to_fp4(scaled_data)

        # 反缩放
        if abs(scale) > self.epsilon:
            quantized_block = quantized_scaled * scale
        else:
            quantized_block = quantized_scaled

        # 计算量化误差
        error = torch.abs(block_data - quantized_block).mean().item()
        self.stats['quantization_errors'].append(error)

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

        # 重置统计
        self.stats = {
            'total_blocks': 0,
            'scale_exponents': [],
            'quantization_errors': []
        }

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
                try:
                    quantized_col, scale, _ = self._process_block(col_data)
                except (RuntimeError, ValueError, TypeError, OverflowError) as error:
                    raise RuntimeError(
                        f"Failed to quantize row block {block_idx}, column {col}"
                    ) from error

                # 存储结果
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

            self.stats['total_blocks'] += N

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
                try:
                    quantized_row, scale, _ = self._process_block(row_data)
                except (RuntimeError, ValueError, TypeError, OverflowError) as error:
                    raise RuntimeError(
                        f"Failed to quantize column block {block_idx}, row {row}"
                    ) from error

                # 存储结果
                quantized_matrix[row, start_col:end_col] = quantized_row
                scale_matrix[row, block_idx] = scale

            self.stats['total_blocks'] += M

        return quantized_matrix, scale_matrix

    def get_quantization_stats(self) -> dict:
        """获取量化统计信息"""
        if not self.stats['scale_exponents']:
            return {'message': '尚未进行量化'}

        exponents = self.stats['scale_exponents']
        errors = self.stats['quantization_errors']

        stats = {
            'total_blocks': self.stats['total_blocks'],
            'scale_exponents': {
                'min': min(exponents),
                'max': max(exponents),
                'mean': sum(exponents) / len(exponents),
                'std': np.std(exponents) if len(exponents) > 1 else 0
            },
            'quantization_errors': {
                'min': min(errors),
                'max': max(errors),
                'mean': sum(errors) / len(errors),
                'std': np.std(errors) if len(errors) > 1 else 0
            },
            'config': {
                'data_format': self.data_format,
                'axis': self.axis,
                'block_size': self.block_size
            }
        }

        # 指数分布
        unique_exps = set(exponents)
        stats['scale_exponents']['unique_count'] = len(unique_exps)
        stats['scale_exponents']['distribution'] = {
            exp: exponents.count(exp)
            for exp in sorted(unique_exps)
        }

        return stats

    def print_stats(self):
        """打印量化统计信息"""
        stats = self.get_quantization_stats()

        if 'message' in stats:
            print(stats['message'])
            return

        print("=" * 60)
        print("MXFP4 量化统计信息")
        print("=" * 60)

        print(f"数据格式: {stats['config']['data_format']}")
        print(f"量化轴: {'行方向 (axis=0)' if stats['config']['axis'] == 0 else '列方向 (axis=1)'}")
        print(f"分块大小: {stats['config']['block_size']}")
        print(f"总块数: {stats['total_blocks']}")

        print(f"\n缩放因子指数 (FP8_E8M0FNU):")
        exp_stats = stats['scale_exponents']
        print(f"  范围: [{exp_stats['min']}, {exp_stats['max']}]")
        print(f"  均值: {exp_stats['mean']:.2f}")
        print(f"  标准差: {exp_stats['std']:.2f}")
        print(f"  唯一值数量: {exp_stats['unique_count']}")

        # 显示最常见的指数
        if exp_stats['distribution']:
            print(f"  常见指数:")
            sorted_exps = sorted(exp_stats['distribution'].items(),
                               key=lambda x: x[1], reverse=True)
            for exp, count in sorted_exps[:5]:
                percentage = count / stats['total_blocks'] * 100
                print(f"    2^{exp:4d}: {count:5d} 块 ({percentage:5.1f}%)")

        print(f"\n量化误差:")
        err_stats = stats['quantization_errors']
        print(f"  范围: [{err_stats['min']:.6f}, {err_stats['max']:.6f}]")
        print(f"  均值: {err_stats['mean']:.6f}")
        print(f"  标准差: {err_stats['std']:.6f}")

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
    
    def _fp32_to_fp4(self, value: float) -> int:
        """
        将FP32转为FP4格式

        参数:
        value: 输入值

        返回:
        FP4值
        """
        exp_bits = self.config['exp_bits']
        mant_bits = self.config['mantissa_bits']
        bias = self.config['bias']

        max_exp = (1 << exp_bits) - bias
        max_mant = (1 << mant_bits) - 1

        # 处理零值
        if value == 0:
            return 0

        # 计算符号
        sign = 1 if value < 0 else 0

        abs_val = abs(value)
        # 计算以2为底的对数
        log2_val = math.log2(abs_val)
        # 计算指数
        exp = int(math.floor(log2_val))

        if exp < 0:
            #使用非规格化数表示
            exp_biased = 0
            denorm_mantissa_val = abs_val / (2.0 ** (1 - bias))
            mantissa = int(math.ceil(denorm_mantissa_val * max_mant))
        else:
            exp_biased = exp + bias
            exp_biased = min(exp_biased, max_exp)
            #计算尾数
            mantissa_val = abs_val / (2.0 ** exp)
            mantissa_frac = mantissa_val - 1.0
            mantissa = int(math.ceil(mantissa_frac * max_mant))

        mantissa = min(max(mantissa, 0), max_mant)

        # 组合成4位
        if self.data_format == 'E2M1':
            # 格式: [符号(1位) | 指数(2位) | 尾数(1位)]
            fp4_val = (sign << 3) | (exp_biased << 1) | mantissa
        else:  # e1m2
            # 格式: [符号(1位) | 指数(1位) | 尾数(2位)]
            fp4_val = (sign << 3) | (exp_biased << 2) | mantissa

        #高位为当前值的FP4，低位为0（占位符）
        packed_uint8 = (fp4_val << 4)
        return packed_uint8

    def _fp4_to_fp32(self, fp4_val: int) -> float:
        """
        将FP4值转换回fp32

        参数:
        fp4_val: FP4值

        返回:
        转换后的fp32值
        """
        if fp4_val == 0:
            return 0.0

        exp_bits = self.config['exp_bits']
        mant_bits = self.config['mantissa_bits']
        bias = self.config['bias']

        # 解析4位
        if self.data_format == 'E2M1':
            sign = (fp4_val >> 3) & 0x1
            exp_biased = (fp4_val >> 1) & 0x3
            mantissa = fp4_val & 0x1
            div = 2.0
        else:  # e1m2
            sign = (fp4_val >> 3) & 0x1
            exp_biased = (fp4_val >> 2) & 0x1
            mantissa = fp4_val & 0x3
            div = 4.0

        if exp_biased == 0:
            # 次正规数
            if mantissa == 0:
                value = 0.0
            else:
                # 公式: 2^(1-偏置) * (尾数/2^尾数位数)
                value = (mantissa / div) * (2.0 ** (1 - self.config['bias']))
        else:
            # 正规数
            # 公式: 2^(指数-偏置) * (1 + 尾数/2^尾数位数)
            value = (1.0 + mantissa / div) * (2.0 ** (exp_biased - self.config['bias']))

        # 应用符号
        if sign == 1:
            value = -value

        return value
    
    def matrix_fp32_to_fp4(self, matrix: float) -> int:
        rows, cols = matrix.shape

        matrix_uint8 = torch.zeros(rows, ((cols + 1) // 2), dtype=torch.uint8)
        for row in range(rows):
            for col in range(cols):
                idx = col // 2  # 在matrix中的索引
                pos_in_uint8 = col % 2 # 在uint8中的位置（0：低4位， 1：高4位）

                # 转换单个值
                val = float(matrix[row, col])
                packed_uint8 = self._fp32_to_fp4(val)

                #提取FP4值
                fp4_val = (packed_uint8 >> 4) & 0x0F

                # 将FP4值放入matrix的适当位置
                if pos_in_uint8 == 0:
                    # 放在低4位
                    matrix_uint8[row, idx] = (matrix_uint8[row, idx] & 0xF0) | fp4_val
                else:
                    # 放在高4位
                    matrix_uint8[row, idx] = (matrix_uint8[row, idx] & 0x0F) | (fp4_val << 4)
        return matrix_uint8

    def matrix_fp4_to_fp32(self, matrix: int) -> float:
        rows, cols = matrix.shape

        matrix_float = torch.ones(rows, cols * 2, dtype=torch.float32)
        for row in range(rows):
            for col in range(cols):
                # 获取打包的uint8值
                packed_val = matrix[row, col]

                # 解包出两个FP4值
                fp4_high = (packed_val >> 4) & 0x0F
                fp4_low = packed_val & 0x0F

                matrix_float[row, col * 2] = self._fp4_to_fp32(fp4_high)
                matrix_float[row, col * 2 + 1] = self._fp4_to_fp32(fp4_low)
        return matrix_float

def gen_data_fp4_e2m1(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)

    quantizer = MXFP4MatrixQuantizer(
        data_format='E2M1',
        axis=axis,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(
        quantized_matrix,
        scale_matrix
    )

    if trans == 1:
        quantized_matrix = quantized_matrix.t()

    quantized_matrix_uint8 = quantizer.matrix_fp32_to_fp4(quantized_matrix)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix_uint8, scale_matrix, dequantized_matrix

def gen_data_fp4_e1m2(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)

    quantizer = MXFP4MatrixQuantizer(
        data_format='E1M2',
        axis=axis,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(
        quantized_matrix,
        scale_matrix
    )

    if trans == 1:
        quantized_matrix = quantized_matrix.t()

    quantized_matrix_uint8 = quantizer.matrix_fp32_to_fp4(quantized_matrix)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix_uint8, scale_matrix, dequantized_matrix

def random_uniform(
    low: float, high: float,
    size: tuple,
    dtype: torch.dtype = torch.float
) -> torch.Tensor:
    # 计算两次采样的长度
    element_num = math.prod(size)
    l0 = math.floor(math.sqrt(element_num))
    l0 = l0 if l0 * (l0 + 1) >= element_num else l0 + 1
    l1 = l0 + 1
    # 生成两次采样在 U(0, 1) 下的随机 tensor
    sample1 = torch.rand(size=(l0,), dtype=dtype).repeat(math.ceil(element_num / l0)+1)[:element_num]
    sample2 = torch.rand(size=(l1,), dtype=dtype).repeat(math.ceil(element_num / l1)+1)[:element_num]
    # 将两次采样结果组合为 U(0, 1) 的 tensor
    sample_add = sample1 + sample2
    sample_result = sample_add - torch.floor(sample_add)
    return (sample_result * (high - low) + low).reshape(size)
 
def generate_constrained_row_sum_tensor(
    row_sum: int,
    size: tuple[int, int],
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    row, col = size
    tensor = torch.empty(size=size, dtype=torch.int32)
    tensor[:, :-1] = torch.randint(low=0, high=row_sum + 1, size=(row, col - 1), dtype=dtype).sort(dim=1).values
    tensor[:, -1] = row_sum
    tensor[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    return tensor

def generate_uniform_tokens_table(
    row_sum: int,
    size: tuple[int, int],
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    row, col = size
    tensor = torch.ones(size=size, dtype=torch.int32) * (row_sum // col)
    return tensor
 
def simulate_all_to_all_v_with_unpermute(
    output_list: list[torch.Tensor],
    global_tokens_per_expert: torch.Tensor,
    ep_size: int
) -> list[torch.Tensor]:
    rank_size, expert_num = global_tokens_per_expert.shape
    local_expert_num = expert_num // ep_size
    output_groups_list = [
        torch.split(output, global_tokens_per_expert.reshape(shape=(rank_size, ep_size, local_expert_num))[:, ep_idx, :].transpose(0, 1).reshape(-1).tolist(), dim=0)
        for ep_idx, output in enumerate(output_list)
    ]
    return [
        torch.cat([
            output_groups[local_expert_idx * rank_size + output_rank_idx]
            for output_groups in output_groups_list
            for local_expert_idx in range(local_expert_num)
        ], dim=0)
        for output_rank_idx in range(rank_size)
    ]
 
def grouped_matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    group_list: torch.Tensor
) -> torch.Tensor:
    return torch.cat([
        torch.matmul(
            input_in_group.to(torch.float32),
            other[rank_idx].to(torch.float32)
        )
        for rank_idx, input_in_group in enumerate(torch.split(input, group_list.tolist(), dim=0))
    ], dim=0)

def generate_data(args: argparse.Namespace) -> None:
    out_type = args.out_type.torch_type
    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    trans_a, trans_b = args.trans_a, args.trans_b
    expert_num = args.expert
    ep_size = args.ep
 
    if ep_size <= 0:
        raise ValueError("ep must be greater than 0")
    if rank_size % ep_size != 0:
        raise ValueError(f"rank_size ({rank_size}) must be divisible by ep ({ep_size})")
    if expert_num % ep_size != 0:
        raise ValueError(f"expert_num ({expert_num}) must be divisible by ep ({ep_size})")
    tp_size = rank_size // ep_size
    local_expert_num = expert_num // ep_size

    matrix_a_origin_world_uint8 = torch.zeros(ep_size, m * k // 2, dtype=torch.uint8)
    matrix_b_origin_world_uint8 = torch.zeros(ep_size, local_expert_num, k * n // 2, dtype=torch.uint8)

    if trans_a == 0:
        matrix_a_origin_world_uint8 = matrix_a_origin_world_uint8.reshape(ep_size, m, k // 2)
    else:
        matrix_a_origin_world_uint8 = matrix_a_origin_world_uint8.reshape(ep_size, k, m // 2)

    if trans_b == 0:
        matrix_b_origin_world_uint8 = matrix_b_origin_world_uint8.reshape(ep_size, local_expert_num, k, n // 2)
    else:
        matrix_b_origin_world_uint8 = matrix_b_origin_world_uint8.reshape(ep_size, local_expert_num, n, k // 2)

    matrix_a_scale_origin_world = torch.zeros(ep_size, m, ((k+31)//32 + 1) //2, 2, dtype=torch.float8_e8m0fnu)
    matrix_b_scale_origin_world = torch.zeros(ep_size, local_expert_num, ((k+31)//32 + 1) //2, n, 2, dtype=torch.float8_e8m0fnu)
    matrix_a_origin_world_fp32 = torch.zeros(ep_size, m, k, dtype=torch.float32)
    matrix_b_origin_world_fp32 = torch.zeros(ep_size, local_expert_num, k, n, dtype=torch.float32)

    for rank in range(ep_size):
        a_uint8, a_scale, a_fp32 = gen_data_fp4_e2m1(m, k, 1, trans_a)
        a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
        if(trans_a == 1):
            a_scale = a_scale.permute(1, 0, 2)
        matrix_a_origin_world_uint8[rank] = a_uint8
        matrix_a_scale_origin_world[rank] = a_scale
        matrix_a_origin_world_fp32[rank] = a_fp32
        
        for expert in range(local_expert_num):
            b_uint8, b_scale, b_fp32 = gen_data_fp4_e2m1(k, n, 0, trans_b)

            b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])
            if(trans_b == 1):
                b_scale = b_scale.permute(2, 0, 1)
            else:
                b_scale = b_scale.permute(0, 2, 1)
            matrix_b_origin_world_uint8[rank, expert] = b_uint8
            matrix_b_scale_origin_world[rank, expert] = b_scale
            matrix_b_origin_world_fp32[rank, expert] = b_fp32

    global_tokens_per_expert = generate_constrained_row_sum_tensor(row_sum=m // rank_size, size=(rank_size * ep_size, expert_num // ep_size)).reshape(shape=(rank_size, expert_num))
    global_tokens_per_local_expert_world = global_tokens_per_expert.reshape(shape=(rank_size, ep_size, local_expert_num)).permute(1, 0, 2).repeat(tp_size, 1, 1)
    print(f"global_tokens_per_expert={global_tokens_per_expert}")
    print(f"global_tokens_per_local_expert_world={global_tokens_per_local_expert_world}")

    group_list_world = global_tokens_per_local_expert_world.sum(dim=1)
    output_list = [
        grouped_matmul(input, other, group_list)
        for input, other, group_list in zip(list(matrix_a_origin_world_fp32), list(matrix_b_origin_world_fp32), list(group_list_world))
    ]
 
    unpermute_output_list = simulate_all_to_all_v_with_unpermute(output_list, global_tokens_per_expert, ep_size)
 
    matrix_a_world = matrix_a_origin_world_uint8 if not trans_a else matrix_a_origin_world_uint8.mT
    matrix_a_scale_world = matrix_a_scale_origin_world if not trans_a else matrix_a_scale_origin_world.mT
    matrix_b_world = matrix_b_origin_world_uint8 if not trans_b else matrix_b_origin_world_uint8.mT
    matrix_b_scale_world = matrix_b_scale_origin_world if not trans_b else matrix_b_scale_origin_world.mT

    print(f"matrix_a_world.shape={matrix_a_world.shape}")

    for rank_idx in range(rank_size):
        matrix_a_local = matrix_a_world[rank_idx].clone()
        matrix_b_local = matrix_b_world[rank_idx].clone()
        matrix_a_scale_local = matrix_a_scale_world[rank_idx].clone()
        matrix_b_scale_local = matrix_b_scale_world[rank_idx].clone()

        a_np = torch.tensor(matrix_a_local.flatten().untyped_storage(), dtype=torch.int8)
        b_np = torch.tensor(matrix_b_local.flatten().untyped_storage(), dtype=torch.int8)
        a_scale_np = torch.tensor(matrix_a_scale_local.flatten().untyped_storage(), dtype=torch.int8)
        b_scale_np = torch.tensor(matrix_b_scale_local.flatten().untyped_storage(), dtype=torch.int8)

        tensor_to_file(a_np, f"./output/a_gm_{rank_idx}.bin")
        tensor_to_file(b_np, f"./output/b_gm_{rank_idx}.bin")
        tensor_to_file(a_scale_np, f"./output/a_scale_{rank_idx}.bin")
        tensor_to_file(b_scale_np, f"./output/b_scale_{rank_idx}.bin")
        tensor_to_file(global_tokens_per_expert[rank_idx], f"./output/local_tokens_per_expert_{rank_idx}.bin")
        tensor_to_file(global_tokens_per_local_expert_world[rank_idx],
                       f"./output/global_tokens_per_local_expert_{rank_idx}.bin")
 
        output = unpermute_output_list[rank_idx]
        padding_output = torch.zeros(size=(rank_size * m, n), dtype=torch.float32)
        padding_output[:output.shape[0]] = output
        tensor_to_file(padding_output, f"./output/golden_{rank_idx}.bin")
 
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
    args = parser.parse_args()
 
    generate_data(args)
