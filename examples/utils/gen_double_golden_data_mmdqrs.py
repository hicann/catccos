import os
import torch
import numpy as np
import torch_npu

BIAS_LOW = -65536
BIAS_HIGH = 65536

from utils import DataType, tensor_to_file

class QuantMatmulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, scale, offset, bias, pertoken_scale, output_dtype=torch.bfloat16):
        return torch_npu.npu_quant_matmul(
            x1,
            x2,
            scale,
            offset=offset,
            bias=bias,
            pertoken_scale=pertoken_scale,
            output_dtype=output_dtype
        )

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        debug = int(os.getenv("debug", 0)) == 1
        if not debug:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        else:
            # for debug use all ones.
            print(f'[warning]: in debug mode, use all ones as input.')
            return torch.ones(size=size, dtype=dtype)
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
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Generate common inputs (scales, bias)
    debug = int(os.getenv("debug", 0)) == 1
    if not debug:
        scale_x1_gm = torch.randn(M, dtype=torch.float32) * 0.01
        scale_x2_gm = torch.randn(N, dtype=torch.float32) * 0.01
    else:
        # for debug
        print(f'[warning]: in debug mode, use all 0.01 as scale input.')
        scale_x1_gm = torch.ones(size=(M,), dtype=torch.float32) * 0.01
        scale_x2_gm = torch.ones(size=(N,), dtype=torch.float32) * 0.01

    scale_x1_gm_unsqueeze_1 = scale_x1_gm.unsqueeze(1)
    scale_x2_gm_unsqueeze_0 = scale_x2_gm.unsqueeze(0)

    if debug:
        print(f'[warning]: in debug mode, use {torch.arange(1, N+1, dtype=torch.int32)} as bias input.')
        bias_gm = torch.arange(1, N+1, dtype=torch.int32)
    else:
        bias_gm = torch.randint(low=BIAS_LOW, high=BIAS_HIGH+1, size=(N,), dtype=torch.int32)

    # Placeholder for output buffer
    c_gm = torch.zeros((M // args.rank_size, N), dtype=args.out_dtype.torch_type)

    # This will accumulate the matmul results from all ranks
    aclnn_accumulator_ranks = torch.zeros((args.rank_size, M, N), dtype=torch.float16)
    aclnn_accumulator = torch.zeros((M, N), dtype=torch.float16)
    golden_accumulator = torch.zeros((M, N), dtype=torch.float32)

    quant_matmul_model = QuantMatmulModel()

    for i in range(args.rank_size):
        # Generate rank-specific inputs x1 and x2
        x1_gm_rank = gen_random_data([M, K], dtype=torch.int8)
        x2_gm_rank = gen_random_data([K, N], dtype=torch.int8)

        # npu matmul_dequant
        torch.npu.synchronize()
        if i == 0:
            aclnn_accumulator_ranks[i] = quant_matmul_model(x1_gm_rank.npu(), x2_gm_rank.npu(), scale_x2_gm.npu(), None, bias_gm.npu(), scale_x1_gm.npu(), output_dtype=torch.float16)
        else:
            aclnn_accumulator_ranks[i] = quant_matmul_model(x1_gm_rank.npu(), x2_gm_rank.npu(), scale_x2_gm.npu(), None, None, scale_x1_gm.npu(), output_dtype=torch.float16)
        torch.npu.synchronize()

        # cpu matmul_dequant
        golden_matmul_rank = torch.matmul(x1_gm_rank.to(torch.float32), x2_gm_rank.to(torch.float32))
        if i == 0:
            golden_matmul_rank += bias_gm.to(torch.float32)
        golden_dequant_rank = golden_matmul_rank * scale_x1_gm_unsqueeze_1 * scale_x2_gm_unsqueeze_0
        golden_accumulator += golden_dequant_rank

        # Save rank-specific files
        x1_to_save = x1_gm_rank.transpose(0, 1).contiguous() if args.transA else x1_gm_rank
        x2_to_save = x2_gm_rank.transpose(0, 1).contiguous() if args.transB else x2_gm_rank

        tensor_to_file(x1_to_save, os.path.join(data_dir, f"x1_gm_rank{i}.bin"))
        tensor_to_file(x2_to_save, os.path.join(data_dir, f"x2_gm_rank{i}.bin"))
        print(f"Generated data for rank {i}:")
        print(f"  x1_gm_rank{i}.bin: shape={x1_to_save.shape}")
        print(f"  x2_gm_rank{i}.bin: shape={x2_to_save.shape}")

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
    tensor_to_file(scale_x1_gm, os.path.join(data_dir, "scale_x1_gm.bin"))
    tensor_to_file(scale_x2_gm, os.path.join(data_dir, "scale_x2_gm.bin"))
    tensor_to_file(bias_gm, os.path.join(data_dir, "bias_gm.bin"))
    tensor_to_file(c_gm, os.path.join(data_dir, "c_gm.bin"))
    tensor_to_file(aclnn_result, os.path.join(data_dir, "golden_aclnn.bin"))
    tensor_to_file(golden_result, os.path.join(data_dir, "golden_fp32.bin"))

    print(f"Generated common data:")
    print(f"{scale_x1_gm.shape=}")
    print(f"{scale_x2_gm.shape=}")
    print(f"{bias_gm.shape=}")
    print(f"{aclnn_result.shape=}, {aclnn_result.dtype=}")
    print(f"{golden_result.shape=}, {golden_result.dtype=}")

if __name__ == '__main__':
    gen_golden_data()
