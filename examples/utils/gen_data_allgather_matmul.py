#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import torch
import os
from utils import DataType, tensor_to_file

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)

def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16, DataType.INT8])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str,
                        help='Directory to save the data files',
                        default="./out")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)

    l0c_dtype = torch.float32
    matrix_a_list = []
    matrix_c_list = []
    for i in range(args.rank_size):
        a_gm = gen_random_data([M, K], dtype=args.out_dtype.torch_type)
        matrix_a_list.append(a_gm)
        if args.transA:
            a_gm = a_gm.transpose(0, 1).contiguous()
        a_gm_path = os.path.join(data_dir, f"rank_{i}_a.bin")
        tensor_to_file(a_gm, a_gm_path)

    allgathered_a = torch.cat(matrix_a_list, dim=0)

    for i in range(args.rank_size):
        b_gm = gen_random_data([K, N], dtype=args.out_dtype.torch_type)
        if args.transB:
            b_gm = b_gm.transpose(0, 1).contiguous()
        b_gm_path = os.path.join(data_dir, f"rank_{i}_b.bin")
        tensor_to_file(b_gm, b_gm_path)

        matrix_c = torch.matmul(allgathered_a.to(l0c_dtype), b_gm.to(l0c_dtype))
        tensor_to_file(matrix_c, os.path.join(data_dir, f"golden_{i}.bin"))

if __name__ == '__main__':
    gen_golden_data()