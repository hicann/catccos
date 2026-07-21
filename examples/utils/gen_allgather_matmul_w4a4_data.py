import os
import torch
import numpy as np

from utils import DataType, tensor_to_file

WORKSPACE = os.getcwd()
os.environ["WORKSPACE"] = WORKSPACE


def gen_random_int4(size):
    return torch.randint(-8, 8, size=size, dtype=torch.int8)


def pack_4bit_to_bytes_np(data_int8: np.ndarray) -> np.ndarray:
    data = data_int8.astype(np.int8, copy=True)
    data[data < 0] += 16
    packed = (data[..., 1::2] << 4) | (data[..., ::2] & 0x0F)
    return packed.astype(np.uint8, copy=False)


def reorder_weight_int4_np(weight_int4_packed: np.ndarray, k: int, n: int, transB: int) -> np.ndarray:
    if transB:
        assert k % 64 == 0, "k must be divisible by 64 when transB==1 (nZ layout)"
        assert n % 16 == 0, "n must be divisible by 16 when transB==1 (nZ layout)"
        w = weight_int4_packed.reshape(n, -1).astype(np.uint8, copy=False)
        w = w.reshape(n // 16, 16, k // 64, 32).transpose(2, 0, 1, 3)
    else:
        assert k % 16 == 0, "k must be divisible by 16 when transB==0 (zN layout)"
        assert n % 64 == 0, "n must be divisible by 64 when transB==0 (zN layout)"
        w = weight_int4_packed.reshape(k, -1).astype(np.uint8, copy=False)
        w = w.reshape(k // 16, 16, n // 64, 32).transpose(2, 0, 1, 3)

    return w.reshape(k, -1).astype(np.int8, copy=False)


def pack_scale_per_channel_to_u64(scale_fp32: torch.Tensor) -> torch.Tensor:
    assert scale_fp32.dtype == torch.float32 and scale_fp32.ndim == 2 and scale_fp32.shape[0] == 1
    n = scale_fp32.shape[1]

    scale_u32 = scale_fp32.view(torch.uint32)
    buf_u32 = torch.zeros((1, 2 * n), dtype=torch.uint32)
    buf_u32[:, ::2] = scale_u32
    buf_u64 = buf_u32.view(torch.uint64)
    return buf_u64.squeeze(0)


def cpu_golden_like_sample(allgathered_a_int4: torch.Tensor,
                           b_int4: torch.Tensor,
                           scale_fp32: torch.Tensor,
                           per_token_fp32: torch.Tensor) -> torch.Tensor:
    acc = torch.matmul(allgathered_a_int4.float(), b_int4.float())
    acc = acc.to(torch.float16).to(torch.float32)
    out = acc * scale_fp32
    out = out * per_token_fp32
    return out.to(torch.float32)


def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str,
                        choices=[DataType.FLOAT16, DataType.BF16, DataType.INT8])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str, default="./output")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    rankSize = args.rank_size
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    assert args.transA == 0, "Transposition is not supported for matrices A."

    scale_fp32 = torch.from_numpy(np.random.normal(0, 0.01, (1, N)).astype(np.float32))
    per_token_fp32 = torch.from_numpy(np.random.normal(0, 0.01, (M * rankSize, 1)).astype(np.float32))

    scale_u64 = pack_scale_per_channel_to_u64(scale_fp32)

    d_gm = torch.zeros((M * rankSize, N), dtype=torch.float32)

    A_int4 = []
    for i in range(rankSize):
        a_rank_int4 = gen_random_int4((M, K))
        A_int4.append(a_rank_int4)

        a_packed_np = pack_4bit_to_bytes_np(a_rank_int4.cpu().numpy())
        a_packed_t = torch.from_numpy(a_packed_np.view(np.int8))
        tensor_to_file(a_packed_t, os.path.join(data_dir, f"a_gm_rank{i}.bin"))
        print(f"Generated a_gm_rank{i}.bin packed bytes={a_packed_t.numel()}")

    allgathered_a_int4 = torch.cat(A_int4, dim=0)

    for i in range(rankSize):
        b_rank_int4 = gen_random_int4((K, N))

        if args.transB:
            b_packed_np = pack_4bit_to_bytes_np(b_rank_int4.t().contiguous().cpu().numpy())
        else:
            b_packed_np = pack_4bit_to_bytes_np(b_rank_int4.cpu().numpy())

        b_reordered_np = reorder_weight_int4_np(b_packed_np, k=K, n=N, transB=args.transB)
        b_reordered_t = torch.from_numpy(b_reordered_np)
        tensor_to_file(b_reordered_t, os.path.join(data_dir, f"b_gm_rank{i}.bin"))
        print(f"Generated b_gm_rank{i}.bin shape={tuple(b_reordered_t.shape)}")

        golden_fp32 = cpu_golden_like_sample(allgathered_a_int4, b_rank_int4, scale_fp32, per_token_fp32)
        if args.out_dtype == DataType.BF16:
            golden_out = golden_fp32.to(torch.bfloat16)
        elif args.out_dtype == DataType.FLOAT16:
            golden_out = golden_fp32.to(torch.float16)
        else:
            golden_out = golden_fp32

        tensor_to_file(golden_out, os.path.join(data_dir, f"golden_rank{i}.bin"))
        print(f"Generated golden_rank{i}.bin shape={tuple(golden_out.shape)} dtype={golden_out.dtype}")

    tensor_to_file(d_gm, os.path.join(data_dir, "d_gm.bin"))
    tensor_to_file(scale_u64, os.path.join(data_dir, "scale_gm.bin"))
    tensor_to_file(per_token_fp32.squeeze(1), os.path.join(data_dir, "per_token_scale_gm.bin"))


if __name__ == "__main__":
    gen_golden_data()
