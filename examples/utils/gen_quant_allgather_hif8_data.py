import argparse
import os
import numpy as np
from en_dtypes import hifloat8
from utils import bf16_bits_to_fp32, fp32_to_bf16_bits


def generate_and_process(args):
    rank_size = args.rankSize
    output_dir = args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    elements_per_rank = args.m * args.n * args.k

    scales = np.random.uniform(0.5, 1.5, 1).astype(np.float32)
    scale_file = os.path.join(output_dir, "scale.bin")
    scales.tofile(scale_file)
    print(f"[Gen] Generated single scale file: {scale_file} (Value: {scales[0]})")

    all_ranks_hif8 = []

    for rank in range(rank_size):
        raw_fp32 = np.random.uniform(0, 5, elements_per_rank).astype(np.float32)
        input_data_bf16 = fp32_to_bf16_bits(raw_fp32)

        input_filename = os.path.join(output_dir, f"rank_{rank}_input.bin")
        input_data_bf16.tofile(input_filename)

        input_fp32 = bf16_bits_to_fp32(input_data_bf16)
        quantized_hif8 = (input_fp32 * scales[0]).astype(hifloat8)
        all_ranks_hif8.append(quantized_hif8)

        print(f"  -> Rank {rank} processed.")

    final_gathered_hif8 = np.concatenate(all_ranks_hif8)
    print(f"[Process] All-Gather complete. Total elements: {final_gathered_hif8.size}")

    for rank in range(rank_size):
        output_filename = os.path.join(output_dir, f"rank_{rank}_golden.bin")
        final_gathered_hif8.tofile(output_filename)
        print(f"  -> Rank {rank} golden saved. Size: {final_gathered_hif8.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Golden Data for Quant+AllGather")
    parser.add_argument("--rankSize", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data_allgather")

    args = parser.parse_args()
    generate_and_process(args)
