# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import shutil
from copy import deepcopy
import torch
import torch_npu
from enum import IntEnum
import numpy
import argparse

LCAL_PATH = os.getcwd().replace("build", "")
DATA_PATH = os.path.join(LCAL_PATH, "output")
shutil.rmtree(DATA_PATH, ignore_errors=True)
os.makedirs(DATA_PATH)
print(f'Use DATA_PATH = {DATA_PATH}')
print(f'Use LCAL_PATH = {LCAL_PATH}')

class DataType(IntEnum):
    FLOAT = 0
    INT8 = 2
    FLOAT16 = 1
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


def generate_random_tensor(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.randn(size=size, dtype=dtype)
    elif dtype is torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    elif dtype is torch.int32:
        return torch.randint(-1024, 1024, size=size, dtype=dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


class QuantInfo:
    def __init__(self, rank_size, local_expert_nums, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.rank_size = rank_size
        self.expert_per_rank = local_expert_nums

        self.dequant_scale = None
        self.dequant_scale_origin = None
        self.dequant_scale_list = []
        self.dequant_scale_origin_list = []

    def get_output_dequant_tensor(self, input_info):
        shape_info = [input_info[0], input_info[2]]
        dequant_args_shape = 1, shape_info[1]
        self.dequant_scale_origin = generate_random_tensor(size=dequant_args_shape, dtype=torch.float32) / 127
        self.dequant_scale_origin = ((self.dequant_scale_origin.view(torch.int32) >> 13) << 13).view(torch.float32)
        self.dequant_scale = torch.zeros(size=dequant_args_shape, dtype=torch.int64)
        self.dequant_scale.view(torch.float32)[:, ::2] = self.dequant_scale_origin

    def get_moe_dequant_tensor(self, input_info):
        shape_info = deepcopy(input_info)
        shape_info[-1] = shape_info[-1] * self.expert_per_rank
        self.dequant_scale_list.clear()
        self.dequant_scale_origin_list.clear()
        for _ in range(self.rank_size):
            self.get_output_dequant_tensor(shape_info)
            self.dequant_scale_list.append(self.dequant_scale)
            self.dequant_scale_origin_list.append(self.dequant_scale_origin)
            self.dequant_scale = None
            self.dequant_scale_origin = None

    def get_moe_broadcast_tensor(self, matrix_a_block_list, l0c_dtype):
        broadcast_scale_list = []
        broadcast_offset_list = []
        for i in range(self.rank_size):
            if self.dequant_scale_list[i].shape != torch.Size([1, self.expert_per_rank * self.n]):
                dequant_scale = self.dequant_scale_origin_list[i].expand(1, self.n * self.expert_per_rank)
            else:
                dequant_scale = self.dequant_scale_origin_list[i]
            scale_blocks = torch.chunk(dequant_scale, self.expert_per_rank, dim=1)
            temp_list = []
            for j, block in enumerate(scale_blocks):
                expanded_block = block.unsqueeze(0).expand(matrix_a_block_list[i][j], -1, -1)
                temp_list.append(expanded_block.squeeze(1))
            broadcast_scale_list.append(torch.cat(temp_list, dim=0))
        for i in range(self.rank_size):
            broadcast_offset = torch.zeros_like(broadcast_scale_list[i], dtype=l0c_dtype)
            broadcast_offset_list.append(broadcast_offset)
        return broadcast_scale_list, broadcast_offset_list


def read_binary_file(file_path, dtype=torch.float32):
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        with open(file_path, "rb") as f:
            binary_data = f.read()
        if len(binary_data) == 0:
            print(f"文件为空: {file_path}")
            return torch.tensor([], dtype=dtype)
        writable_data = bytearray(binary_data)
        tensor = torch.frombuffer(writable_data, dtype=dtype)
        return tensor
    except FileNotFoundError:
        print(f"The file {file_path} does not exist!")
        return None


class MoeTestDate:
    def __init__(self, m, k, n, top_k, ep, expert_per_rank, quant_info: QuantInfo, out_dtype, batch_size=1):
        self.m = m
        self.k = k
        self.n = n
        self.top_k = top_k
        self.ep = ep
        self.l0c_dtype = torch.int32
        self.expert_num = expert_per_rank * ep
        self.expert_per_rank = expert_per_rank
        self.input_info = [m * top_k, k, n]
        self.batch_size = batch_size
        self.max_output_size = m * top_k * ep
        self.quant_info = quant_info

        self.matrix_a_list = []
        self.matrix_b1_list = []
        for _ in range(ep):
            self.matrix_a_list.append(generate_random_tensor(size=(m, k), dtype=out_dtype))
            self.matrix_b1_list.append(generate_random_tensor(size=(expert_per_rank, k, n), dtype=torch.int8))

        init_routing_matrix_a = []
        num_local_tokens_per_expert = []
        self.pertoken_scale_list = []
        self.expanded_row_idx_list = []

        for i in range(self.ep):
            expert_idx = torch.randint(0, self.expert_num, (m, top_k), dtype=torch.int32)
            self.write_to_bin(expert_idx, f"expert_idx_{i}")
            
            (matrix_a, expanded_row_idx,
                expert_tokens_count_or_cumsum, pertoken_scale) = torch_npu.npu_moe_init_routing_v2(
                self.matrix_a_list[i].to('npu'), expert_idx.to('npu'), scale=None, offset=None,
                active_num=m * top_k, expert_capacity=m * top_k, expert_num=self.expert_num,
                drop_pad_mode=0,
                expert_tokens_num_type=1, expert_tokens_num_flag=True,
                active_expert_range=[0, self.expert_num], quant_mode=1, row_idx_type=0)
            
            matrix_a = matrix_a.cpu().numpy()
            expanded_row_idx = expanded_row_idx.cpu().numpy()
            expert_tokens_count_or_cumsum = expert_tokens_count_or_cumsum.cpu().numpy()
            pertoken_scale = pertoken_scale.cpu().numpy()

            self.expanded_row_idx_list.append(expanded_row_idx)
            self.write_to_bin(torch.from_numpy(matrix_a).unsqueeze(0), f"matrix_a_tmp_{i}")
            self.write_to_bin(torch.from_numpy(pertoken_scale).unsqueeze(0), f"matrix_pertoken_scale1_{i}")
            init_routing_matrix_a.append(torch.from_numpy(matrix_a).unsqueeze(0))
            num_local_tokens_per_expert.append(expert_tokens_count_or_cumsum)
            self.pertoken_scale_list.append(torch.from_numpy(pertoken_scale).unsqueeze(0).unsqueeze(2))

        (self.input_splits, self.output_splits,
            self.num_local_tokens_per_expert,
            self.num_global_tokens_per_local_expert) = self.get_moe_input_output_splits(expert_per_rank, ep, num_local_tokens_per_expert)

        for i in range(self.ep):
            self.write_to_bin(torch.from_numpy(self.num_local_tokens_per_expert[i]), f"tokenPerExpert_{i}")
        self.matrix_a_i_list, self.matrix_a_block_list = self.alltoall_nopermute(init_routing_matrix_a, k, torch.int8, ep)

        if self.max_output_size > 0:
            for i in range(self.ep):
                self.matrix_a_i_list[i] = self.matrix_a_i_list[i][:, :self.max_output_size, :]

        self.dispatch_gmm_swiglu()

        for i in range(self.ep):
            self.write_to_bin(self.matrix_a_list[i], f"a_gm_{i}")
            self.write_to_bin(self.matrix_b1_list[i], f"b_gm_{i}")

    def write_to_bin(self, tensor, prefix):
        file_path = f"{DATA_PATH}/{prefix}.bin"
        if tensor is None:
            return
        untyped_dict = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.int8: torch.int8,
            torch.float32: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int64
        }
        print(tensor.shape, tensor.dtype, file_path)
        tensor.view(untyped_dict.get(tensor.dtype)).numpy().tofile(file_path)

    def get_moe_input_output_splits(self, expert_per_rank, ep, num_local_tokens_per_expert):
        all_gather_res = num_local_tokens_per_expert[0].tolist()
        for i in range(1, ep):
            all_gather_res = all_gather_res + num_local_tokens_per_expert[i].tolist()
        input_splits = [None] * ep
        for i in range(ep):
            input_splits[i] = numpy.sum(numpy.array(num_local_tokens_per_expert[i]).reshape(ep, expert_per_rank),
                                        axis=1)
        self.global_tokens_per_expert_matrix = numpy.array(num_local_tokens_per_expert).reshape(
                                               ep * ep * expert_per_rank)
        output_splits = [None] * ep
        num_global_tokens_per_expert = numpy.array(all_gather_res).reshape(ep, self.expert_num)
        num_global_tokens_per_local_expert = [None] * ep
        for i in range(ep):
            num_global_tokens_per_local_expert[i] = num_global_tokens_per_expert[:,
                                                    i * expert_per_rank:(i + 1) * expert_per_rank]
            output_splits[i] = numpy.sum(num_global_tokens_per_local_expert[i], axis=-1)
            self.write_to_bin(
                torch.tensor(num_local_tokens_per_expert[i]).reshape(1, ep * expert_per_rank).to(dtype=torch.int32),
                f"num_local_tokens_per_expert_{i}")

        self.write_to_bin(
            torch.from_numpy(numpy.array(num_local_tokens_per_expert)).reshape(ep, ep * expert_per_rank).to(
                dtype=torch.int32), "global_tokens_per_expert_matrix")
        return input_splits, output_splits, num_local_tokens_per_expert, num_global_tokens_per_local_expert

    def alltoall_nopermute(self, matrix_a, k, element_type, ep):
        m_matrix_a = [sum(self.output_splits[i]) for i in range(ep)]
        matrix_a_i_list = [torch.zeros(size=(self.batch_size, m_matrix_a[i], k), dtype=element_type) for i in range(ep)]
        matrix_a_block_list = [[] for _ in range(ep)]
        for src_ep in range(ep):
            src_offset = 0
            sum_tokens = 0
            for local_expert_idx in range(self.expert_per_rank):
                src_offset_old = src_offset
                expert_idx = local_expert_idx + src_ep * self.expert_per_rank
                for dst_ep in range(ep):
                    dst_expert_offset = 0
                    dst_expert_len = self.num_local_tokens_per_expert[dst_ep][expert_idx]
                    for i in range(expert_idx):
                        dst_expert_offset += self.num_local_tokens_per_expert[dst_ep][i]
                    matrix_a_i_list[src_ep][:, src_offset:src_offset + dst_expert_len, :] = (
                        matrix_a[dst_ep][:, dst_expert_offset:dst_expert_offset + dst_expert_len, :])
                    src_offset += dst_expert_len
                    if self.max_output_size > 0:
                        if (sum_tokens + self.global_tokens_per_expert_matrix[
                            dst_ep * self.expert_num + expert_idx]) >= self.max_output_size:
                            self.global_tokens_per_expert_matrix[
                                dst_ep * self.expert_num + expert_idx] = self.max_output_size - sum_tokens
                            sum_tokens = self.max_output_size
                        else:
                            sum_tokens += self.global_tokens_per_expert_matrix[dst_ep * self.expert_num + expert_idx]
                if self.max_output_size > 0:
                    if src_offset >= self.max_output_size and src_offset_old <= self.max_output_size:
                        src_offset = self.max_output_size
                matrix_a_block_list[src_ep].append(src_offset - src_offset_old)
        return matrix_a_i_list, matrix_a_block_list

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        x0, gate = x.chunk(2, dim=-1)
        swish = x0 * torch.sigmoid(x0)
        y = swish * gate
        return y

    def quant(self, x: torch.Tensor):
        x_row_max = torch.max(torch.abs(x), dim=-1).values
        quant_result = x * 127. / x_row_max[:, None]
        y = torch.round(quant_result).to(torch.int8)
        scale = (x_row_max / 127.).to(torch.float32)
        return y, scale

    def dispatch_gmm_swiglu(self):
        self.quant_info.get_moe_dequant_tensor(self.input_info)
        dequant_scale_list = self.quant_info.dequant_scale_list
        broadcast_scale_list, broadcast_offset_list = self.quant_info.get_moe_broadcast_tensor(
                                                        self.matrix_a_block_list, self.l0c_dtype)
        for i in range(self.ep):
            self.write_to_bin(dequant_scale_list[i], f"scale_gm_{i}")

        quant_scale_list = self.pertoken_scale_list
        quant_scale_alltoall, _ = self.alltoall_nopermute(quant_scale_list, 1, torch.float32, self.ep)
        for i in range(self.ep):
            quant_scale = quant_scale_list[i].squeeze(0)
            quant_scale_alltoall[i] = quant_scale_alltoall[i].squeeze(0)
            if self.max_output_size > 0:
                quant_scale_alltoall[i] = quant_scale_alltoall[i][:self.max_output_size, :]

        for i in range(self.ep):
            a_blocks = torch.split(self.matrix_a_i_list[i], self.matrix_a_block_list[i], dim=1)
            b_blocks = torch.unbind(self.matrix_b1_list[i], dim=0)
            result_blocks = []
            for a_block, b_block in zip(a_blocks, b_blocks):
                a_block = a_block.unsqueeze(1)
                b_block = b_block.unsqueeze(0)
                product = torch.matmul(a_block.to(self.l0c_dtype), b_block.to(self.l0c_dtype)).squeeze(1)
                result_blocks.append(product)
            matrix_c = torch.cat(result_blocks, dim=1).to(self.l0c_dtype)
            
            matrix_c = ((matrix_c + broadcast_offset_list[i]).to(torch.float32) * broadcast_scale_list[i]).to(
                torch.float16)
            self.write_to_bin(matrix_c.to(torch.float16), f"matrix_c_before_pertoken_dequant_{i}")

            broadcast_quant_scale = quant_scale_alltoall[i].expand(-1, self.input_info[2])
            matrix_c = (matrix_c.to(torch.float32) * broadcast_quant_scale)
            self.write_to_bin(matrix_c.to(torch.float32), f"matrix_c_after_pertoken_dequant_{i}")

            swiglu_out = self.swiglu(matrix_c.squeeze(0))
            padding_output = torch.zeros(size=(self.m * self.top_k * self.ep, swiglu_out.shape[1]), dtype=torch.float32)
            padding_output[:swiglu_out.shape[0]] = swiglu_out.to(torch.float32)
            self.write_to_bin(padding_output.to(torch.float32), f"golden_{i}")


def generate_data(args: argparse.Namespace) -> None:
    M, N, K = args.m, args.n, args.k
    top_k = args.top_k
    expert_num = args.expert
    ep_size = args.ep
    out_type = args.out_dtype.torch_type
    expert_per_rank = expert_num // ep_size

    quant_info = QuantInfo(ep_size, expert_per_rank, M, N, K)

    MoeTestDate(M, K, N, top_k, ep_size, expert_per_rank, quant_info, out_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--expert', type=int, default=4)
    parser.add_argument('--ep', type=int, default=2)
    
    args = parser.parse_args()
    # Example usage:
    generate_data(args)
    print("Golden Data (FP16/BF16) initialized successfully.")
