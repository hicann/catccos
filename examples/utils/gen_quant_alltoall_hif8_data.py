import argparse
import os
import numpy as np
from en_dtypes import hifloat8
from utils import bf16_bits_to_fp32, fp32_to_bf16_bits


def generate_and_process(args):
    rank_size = args.rankSize
    output_dir = args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Config] Output Directory: {output_dir}")

    total_elements = args.m * args.n * args.k
    if total_elements % rank_size != 0:
        raise ValueError(f"Total elements ({total_elements}) must be divisible by rankSize ({rank_size})")
    split_size = total_elements // rank_size
    print(f"[Config] RankSize: {rank_size}, Total Elements per Rank: {total_elements}, Split Size: {split_size}")

    scales = np.random.uniform(0.5, 1.5, rank_size).astype(np.float32)
    scale_file = os.path.join(output_dir, "scale.bin")
    scales.tofile(scale_file)
    print(f"[Gen] Generated scale file: {scale_file}")

    global_transfers = [[None for _ in range(rank_size)] for _ in range(rank_size)]
    print("[Process] Generating inputs and simulating quantization...")

    for src_rank in range(rank_size):
        raw_fp32 = np.random.uniform(0, 5, total_elements).astype(np.float32)
        input_data_bf16 = fp32_to_bf16_bits(raw_fp32)

        input_filename = os.path.join(output_dir, f"rank_{src_rank}_input.bin")
        input_data_bf16.tofile(input_filename)

        input_data_fp32 = bf16_bits_to_fp32(input_data_bf16)

        for dest_rank in range(rank_size):
            start_idx = dest_rank * split_size
            end_idx = (dest_rank + 1) * split_size
            chunk_fp32 = input_data_fp32[start_idx:end_idx]
            scaled_chunk_fp32 = chunk_fp32 * scales[dest_rank]
            chunk_hif8 = scaled_chunk_fp32.astype(hifloat8)
            global_transfers[dest_rank][src_rank] = chunk_hif8

        print(f"  -> Rank {src_rank} input processed and saved to {input_filename}")

    print("[Process] Simulating All-to-All shuffle and saving outputs...")
    for dest_rank in range(rank_size):
        received_chunks = []
        for src_rank in range(rank_size):
            received_chunks.append(global_transfers[dest_rank][src_rank])
        final_golden_hif8 = np.concatenate(received_chunks)
        output_filename = os.path.join(output_dir, f"rank_{dest_rank}_golden.bin")
        final_golden_hif8.tofile(output_filename)
        print(f"  -> Rank {dest_rank} golden output saved, shape: {final_golden_hif8.shape}")

    print(f"[Done] All files generated successfully in '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Golden Data for Quant+AllToAll")
    parser.add_argument("--rankSize", type=int, required=True, help="Number of ranks")
    parser.add_argument("-m", type=int, required=True, help="Dimension M")
    parser.add_argument("-n", type=int, required=True, help="Dimension N")
    parser.add_argument("-k", type=int, required=True, help="Dimension K")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store generated .bin files")
    args = parser.parse_args()
    generate_and_process(args)
