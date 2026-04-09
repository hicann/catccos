import os
import torch
import numpy as np
import torch_npu

from utils import DataType, tensor_to_file

class MatmulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, output_dtype=torch.float16):
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            output_dtype=output_dtype)[0]

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)

def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str,
                        help='Directory to save the data files',
                        default="./output")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    out_dtype = args.out_dtype
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # This will accumulate the matmul results from all ranks
    aclnn_accumulator_ranks = torch.zeros((args.rank_size, M, N), dtype=out_dtype.torch_type)
    aclnn_accumulator = torch.zeros((M, N), dtype=out_dtype.torch_type)
    golden_accumulator = torch.zeros((M, N), dtype=torch.float32)

    matmul_model = MatmulModel()

    for i in range(args.rank_size):
        # Generate rank-specific inputs x and weight
        x_gm = gen_random_data([M, K], dtype=out_dtype.torch_type)
        weight_gm = gen_random_data([K, N], dtype=out_dtype.torch_type)

        # npu matmul
        torch.npu.synchronize()
        aclnn_accumulator_ranks[i] = matmul_model(x_gm.npu(), weight_gm.npu(), output_dtype=out_dtype.torch_type)
        torch.npu.synchronize()

        # cpu matmul_sum
        golden_accumulator += torch.matmul(x_gm.to(torch.float32), weight_gm.to(torch.float32))

        # Save rank-specific files
        x_to_save = x_gm.transpose(0, 1).contiguous() if args.transA else x_gm
        weight_to_save = weight_gm.transpose(0, 1).contiguous() if args.transB else weight_gm

        tensor_to_file(x_to_save, os.path.join(data_dir, f"rank_{i}_a.bin"))
        tensor_to_file(weight_to_save, os.path.join(data_dir, f"rank_{i}_b.bin"))
        print(f"Generated data for rank {i}:")
        print(f"  rank_{i}_a.bin: shape={x_to_save.shape}")
        print(f"  rank_{i}_b.bin: shape={weight_to_save.shape}")

    for i in range(args.rank_size):
        mblock = M // args.rank_size
        aclnn_accum_rank_i = aclnn_accumulator_ranks[i, i*mblock:(i+1)*mblock, :]
        for j in range(args.rank_size):
            if i == j:
                continue
            aclnn_accum_rank_i += aclnn_accumulator_ranks[j, i*mblock:(i+1)*mblock, :]
        aclnn_accumulator[i*mblock:(i+1)*mblock, :] = aclnn_accum_rank_i

    aclnn_result = aclnn_accumulator
    golden_result = golden_accumulator

    # Save common files
    tensor_to_file(aclnn_result, os.path.join(data_dir, "golden_aclnn.bin"))
    tensor_to_file(golden_result, os.path.join(data_dir, "golden_fp32.bin"))

    print(f"Generated common data:")
    print(f"{aclnn_result.shape=}, {aclnn_result.dtype=}")
    print(f"{golden_result.shape=}, {golden_result.dtype=}")

if __name__ == '__main__':
    gen_golden_data()
