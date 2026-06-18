import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from utils import DataType, tensor_to_file


def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)


def gen_alltoall_matmul_data():
    """
    AlltoAll + Matmul data generation:
    - Each rank i has its own A_i of shape (M, K)
    - AlltoAll: rank j receives chunk j from each rank i, i.e. A_i[j*chunk:(j+1)*chunk, :]
    - After AlltoAll, rank j has: concat([A_0[j*chunk:..], A_1[j*chunk:..], ..., A_{R-1}[j*chunk:..]])
    - This assembled matrix has shape (M, K) = (chunk * rankSize, K)
    - Matmul: C_j = assembled_A_j @ B
    - Golden for rank j: C_j of shape (M, N)
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dtype", type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16]
    )
    parser.add_argument("rank_size", type=int)
    parser.add_argument(
        "m", type=int, help="M dimension (must be divisible by rank_size)"
    )
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("transA", type=int)
    parser.add_argument("transB", type=int)
    parser.add_argument("data_dir", type=str, default="./out")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    rank_size = args.rank_size
    chunk = M // rank_size
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)

    l0c_dtype = torch.float32

    # Generate per-rank A matrices (each rank has different data)
    a_list = [
        gen_random_data([M, K], dtype=args.out_dtype.torch_type)
        for _ in range(rank_size)
    ]
    b = gen_random_data([K, N], dtype=args.out_dtype.torch_type)

    # Save per-rank input files
    for i in range(rank_size):
        a_gm = a_list[i]
        b_gm = b.clone()

        if args.transA:
            a_gm = a_gm.transpose(0, 1).contiguous()
        if args.transB:
            b_gm = b_gm.transpose(0, 1).contiguous()

        tensor_to_file(a_gm, os.path.join(data_dir, f"rank_{i}_a.bin"))
        tensor_to_file(b_gm, os.path.join(data_dir, f"rank_{i}_b.bin"))

    # Generate golden for each rank
    for j in range(rank_size):
        # After AlltoAll, rank j assembles:
        # [A_0[j*chunk:(j+1)*chunk, :], A_1[j*chunk:(j+1)*chunk, :], ..., A_{R-1}[j*chunk:(j+1)*chunk, :]]
        chunks = [a_list[i][j * chunk : (j + 1) * chunk, :] for i in range(rank_size)]
        assembled_a = torch.cat(chunks, dim=0)  # shape: (M, K)

        # Matmul
        golden_j = torch.matmul(assembled_a.to(l0c_dtype), b.to(l0c_dtype))
        tensor_to_file(golden_j, os.path.join(data_dir, f"golden_{j}.bin"))

    print(
        f"Generated AlltoAll+Matmul data: M={M}, N={N}, K={K}, rank_size={rank_size}, chunk={chunk}"
    )


if __name__ == "__main__":
    gen_alltoall_matmul_data()
