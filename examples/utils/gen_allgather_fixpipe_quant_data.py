import torch
import os
 
from utils import DataType, tensor_to_file
 
WORKSPACE = os.getcwd()
 
os.environ["WORKSPACE"] = WORKSPACE
 
 
def gen_random_data(size, dtype, debug=False):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.empty(size=size, dtype=dtype).uniform_(-2, 2)
    elif dtype == torch.int8:
        if not debug:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        else:
            return torch.ones(size=size, dtype=dtype)
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
                        default="./output")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    rankSize = args.rank_size
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    l0c_dtype = torch.float32
 
    debug = int(os.getenv("debug", 0)) == 1
 
    per_channel_scale = torch.zeros(size=(1,N), dtype=torch.int64)
    per_channel_scale_origin = gen_random_data([1,N], dtype=torch.float32, debug=debug) / 127
    per_channel_scale_origin = ((per_channel_scale_origin.view(torch.int32) >> 13) << 13).view(torch.float32)
    per_channel_scale.view(torch.float32)[:, ::2] = per_channel_scale_origin

    broadcast_scale = per_channel_scale_origin.expand(M*rankSize, N)
    broadcast_offset = torch.zeros((1,N), dtype=l0c_dtype)
 
    d_gm = torch.zeros((M * rankSize, N), dtype=torch.float32)
 
    A = []
    for i in range(args.rank_size):
        # Generate rank-specific inputs x1 and x2,
        a_gm_rank = gen_random_data([M, K], dtype=torch.int8, debug=debug)
        tensor_to_file(a_gm_rank, os.path.join(data_dir, f"a_gm_rank{i}.bin"))
        print(f"Generated data a for rank {i}:")
        print(f"  a_gm_rank{i}.bin: shape={a_gm_rank.shape}")
        A.append(a_gm_rank)
 
    allgathered_a = torch.cat(A, dim=0)
 
    for i in range(args.rank_size):
        b_gm_rank = gen_random_data([K, N], dtype=torch.int8, debug=debug)
        tensor_to_file(b_gm_rank, os.path.join(data_dir, f"b_gm_rank{i}.bin"))
        print(f"Generated data b for rank {i}:")
        print(f"  b_gm_rank{i}.bin: shape={b_gm_rank.shape}")
 
        # Calculate this rank's contribution to the matmul sum
        accumulator_rank = torch.matmul(
            allgathered_a.to(torch.float32), b_gm_rank.to(torch.float32)
        )
        rank_golden = ((accumulator_rank + broadcast_offset).to(torch.float32) * broadcast_scale).to(torch.float32)
        tensor_to_file(rank_golden, os.path.join(data_dir, f"golden_rank{i}.bin"))
        print(f"Generated golden for rank {i}:")
        print(f"golden_rank{i}.bin: shape={rank_golden.shape}")
 
    tensor_to_file(d_gm, os.path.join(data_dir, "d_gm.bin"))
    tensor_to_file(per_channel_scale, os.path.join(data_dir, "scale_gm.bin"))
 
 
if __name__ == "__main__":
    gen_golden_data()