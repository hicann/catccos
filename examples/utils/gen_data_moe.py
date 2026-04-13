import os
import torch
import torch_npu
import numpy as np
import collections
import argparse
from utils import DataType

# Helper for saving to binary file
def write_to_bin(tensor, file_path):
    if tensor is None or file_path is None:
        return
    untyped_dict = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.int32: torch.int32,
        torch.int64: torch.int64,
        torch.int8: torch.int8
    }
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print("Output file: ", file_path)
    tensor.cpu().view(untyped_dict.get(tensor.dtype)).numpy().tofile(file_path)

def generate_random_tensor(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.randn(size=size, dtype=dtype)
    elif dtype is torch.int32:
        return torch.randint(-1024, 1024, size=size, dtype=dtype)
    raise ValueError(f"Invalid dtype: {dtype}")

#####################################################################
# Independent Operators
#####################################################################

def moe_init_routing(matrix_a, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                     matrix_a_path=None, expert_idx_path=None,
                     out_matrix_a_path=None, out_expanded_row_idx_path=None, out_expert_tokens_path=None):
    if matrix_a_path:
        write_to_bin(matrix_a, matrix_a_path)
    if expert_idx_path:
        write_to_bin(expert_idx, expert_idx_path)

    (routed_matrix_a, expanded_row_idx,
     expert_tokens, pertoken_scale) = torch_npu.npu_moe_init_routing_v2(
        matrix_a.to('npu'), expert_idx.to('npu'), scale=None, offset=None,
        active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, expert_num], quant_mode=-1, row_idx_type=0)

    routed_matrix_a = routed_matrix_a.cpu()
    expanded_row_idx = expanded_row_idx.cpu()
    expert_tokens = expert_tokens.cpu()

    if out_matrix_a_path:
        write_to_bin(routed_matrix_a, out_matrix_a_path)
    if out_expanded_row_idx_path:
        write_to_bin(expanded_row_idx, out_expanded_row_idx_path)
    if out_expert_tokens_path:
        write_to_bin(expert_tokens, out_expert_tokens_path)

    return routed_matrix_a, expanded_row_idx, expert_tokens

def alltoallv1(matrix_a_list, expert_tokens_list, ep, expert_per_rank, k, max_output_size, batch_size,
               matrix_a_paths=None, out_matrix_a_paths=None):
    if matrix_a_paths:
        for i, path in enumerate(matrix_a_paths):
            write_to_bin(matrix_a_list[i], path)
            
    ep_expert_tokens = [t.tolist() for t in expert_tokens_list]
    
    output_splits = [None] * ep
    for i in range(ep):
        num_global_tokens_per_local_expert = np.array(ep_expert_tokens)[:, i * expert_per_rank:(i + 1) * expert_per_rank]
        output_splits[i] = np.sum(num_global_tokens_per_local_expert, axis=-1).tolist()
    
    m_matrix_a = [sum(output_splits[i]) for i in range(ep)]
    matrix_a_i_list = [torch.zeros(size=(batch_size, m_matrix_a[i], k), dtype=matrix_a_list[0].dtype) for i in range(ep)]
    matrix_a_block_list = [[] for _ in range(ep)]
    
    for src_ep in range(ep):
        src_offset = 0
        for local_expert_idx in range(expert_per_rank):
            src_offset_old = src_offset
            expert_idx = local_expert_idx + src_ep * expert_per_rank
            for dst_ep in range(ep):
                dst_expert_offset = 0
                dst_expert_len = expert_tokens_list[dst_ep][expert_idx].item()
                for j in range(expert_idx):
                    dst_expert_offset += expert_tokens_list[dst_ep][j].item()
                    
                matrix_a_i_list[src_ep][:, src_offset:src_offset + dst_expert_len, :] = \
                    matrix_a_list[dst_ep][:, dst_expert_offset:dst_expert_offset + dst_expert_len, :]
                src_offset += dst_expert_len
            
            if max_output_size > 0:
                if src_offset > max_output_size and src_offset_old <= max_output_size:
                    src_offset = max_output_size
            matrix_a_block_list[src_ep].append(src_offset - src_offset_old)

    if max_output_size > 0:
        for i in range(ep):
            matrix_a_i_list[i] = matrix_a_i_list[i][:, :max_output_size, :]

    if out_matrix_a_paths:
        for i, path in enumerate(out_matrix_a_paths):
            write_to_bin(matrix_a_i_list[i], path)

    return matrix_a_i_list, matrix_a_block_list

def gmm1(matrix_a, matrix_b, a_blocks, matrix_a_path=None, matrix_b_path=None, out_matrix_c_path=None):
    if matrix_a_path:
        write_to_bin(matrix_a, matrix_a_path)
    if matrix_b_path:
        write_to_bin(matrix_b, matrix_b_path)

    a_blocks_tensors = torch.split(matrix_a, a_blocks, dim=1)
    b_blocks_tensors = torch.unbind(matrix_b, dim=0)
    result_blocks = []

    for a_block, b_block in zip(a_blocks_tensors, b_blocks_tensors):
        a_block = a_block.unsqueeze(1)
        b_block = b_block.unsqueeze(0)
        product = torch.matmul(a_block.to(torch.float32), b_block.to(torch.float32)).squeeze(1)
        result_blocks.append(product)

    matrix_c = torch.cat(result_blocks, dim=1).to(matrix_a.dtype).to(torch.float32)
    
    if out_matrix_c_path:
        write_to_bin(matrix_c, out_matrix_c_path)
        
    return matrix_c

def swiglu(matrix_c, matrix_c_path=None, out_swiglu_path=None):
    if matrix_c_path:
        write_to_bin(matrix_c, matrix_c_path)
        
    x0, gate = matrix_c.chunk(2, dim=-1)
    swish = x0 * torch.sigmoid(x0.to(torch.float32)).to(x0.dtype)
    y = swish * gate
    
    if out_swiglu_path:
        write_to_bin(y, out_swiglu_path)
        
    return y

def alltoallv2(permuted_tokens_list, expert_tokens_list, ep, expert_per_rank, k2, 
               permuted_tokens_paths=None, out_permuted_paths=None):
    if permuted_tokens_paths:
        for i, path in enumerate(permuted_tokens_paths):
            write_to_bin(permuted_tokens_list[i], path)
            
    ep_expert_tokens = [t.tolist() for t in expert_tokens_list]
    
    input_splits = [None] * ep
    for i in range(ep):
        num_global_tokens_per_local_expert = np.array(ep_expert_tokens)[:, i * expert_per_rank:(i + 1) * expert_per_rank]
        input_splits[i] = np.sum(num_global_tokens_per_local_expert, axis=0).tolist()
    
    m_matrix_a = [sum(expert_tokens_list[i]) for i in range(ep)]
    matrix_a_i_list = [torch.zeros(size=(1, m_matrix_a[i], k2), dtype=permuted_tokens_list[0].dtype) for i in range(ep)]
    
    for src_ep in range(ep):
        src_offset = 0
        for local_expert_idx in range(expert_per_rank):
            expert_idx = local_expert_idx + src_ep * expert_per_rank
            for dst_ep in range(ep):
                dst_expert_offset = 0
                dst_expert_len = expert_tokens_list[dst_ep][expert_idx].item()
                for j in range(expert_idx):
                    dst_expert_offset += expert_tokens_list[dst_ep][j].item()
                    
                matrix_a_i_list[dst_ep][:, dst_expert_offset:dst_expert_offset + dst_expert_len, :] = \
                    permuted_tokens_list[src_ep][:, src_offset:src_offset + dst_expert_len, :]
                src_offset += dst_expert_len

    if out_permuted_paths:
        for i, path in enumerate(out_permuted_paths):
            write_to_bin(matrix_a_i_list[i], path)

    return matrix_a_i_list

def gmm2(matrix_a, matrix_b, a_blocks, matrix_a_path=None, matrix_b_path=None, out_matrix_c_path=None):
    # Same grouped matmul logic as gmm1
    return gmm1(matrix_a, matrix_b, a_blocks, matrix_a_path, matrix_b_path, out_matrix_c_path)

def unpermute(permuted_tokens, sorted_indices, probs, 
              permuted_tokens_path=None, sorted_indices_path=None, probs_path=None,
              out_unpermuted_path=None):
    if permuted_tokens_path:
        write_to_bin(permuted_tokens, permuted_tokens_path)
    if sorted_indices_path:
        write_to_bin(sorted_indices, sorted_indices_path)
    if probs is not None and probs_path:
        write_to_bin(probs, probs_path)

    unpermuted_tokens = torch_npu.npu_moe_token_unpermute(
        permuted_tokens.squeeze(0).to('npu'),
        sorted_indices.to('npu'),
        probs.to('npu') if probs is not None else None).cpu()

    if out_unpermuted_path:
        write_to_bin(unpermuted_tokens, out_unpermuted_path)
        
    return unpermuted_tokens


#####################################################################
# Compositions
#####################################################################

def generate_dispatch_gmm(m, k, n, top_k, active_num, capacity, drop_pad_mode, ep, expert_per_rank, batch_size, max_output_size, dtype, data_path="./output"):
    expert_num = ep * expert_per_rank
    matrix_a_list = [generate_random_tensor((m, k), dtype) for _ in range(ep)]
    expert_idx_list = [torch.argsort(torch.rand(m, expert_num), dim=-1)[:, :top_k].to(torch.int32) for _ in range(ep)]
    matrix_b_list = [generate_random_tensor((expert_per_rank, k, n), dtype) for _ in range(ep)] 

    routed_matrix_a_list = []
    expert_tokens_list = []
    
    # 1. Routing
    for i in range(ep):
        routed_a, _, expert_tokens = moe_init_routing(
            matrix_a_list[i], expert_idx_list[i], active_num, capacity, expert_num, drop_pad_mode,
            matrix_a_path=f"{data_path}/in_routing_matrix_a_{i}.bin",
            expert_idx_path=f"{data_path}/in_routing_expert_idx_{i}.bin",
            out_matrix_a_path=f"{data_path}/out_routing_matrix_a_{i}.bin",
            out_expert_tokens_path=f"{data_path}/out_routing_tokens_{i}.bin"
        )
        routed_matrix_a_list.append(routed_a)
        expert_tokens_list.append(expert_tokens)

    # 2. AlltoAllv1
    routed_matrix_a_list = [a.unsqueeze(0) for a in routed_matrix_a_list]
    matrix_a_i_list, matrix_a_block_list = alltoallv1(
        routed_matrix_a_list, expert_tokens_list, ep, expert_per_rank, k, max_output_size, batch_size,
        matrix_a_paths=[f"{data_path}/in_alltoall_matrix_a_{i}.bin" for i in range(ep)],
        out_matrix_a_paths=[f"{data_path}/out_alltoall_matrix_a_{i}.bin" for i in range(ep)]
    )

    # 3. GMM1
    matrix_c_list = []
    for i in range(ep):
        matrix_c = gmm1(
            matrix_a_i_list[i], matrix_b_list[i], matrix_a_block_list[i],
            matrix_b_path=f"{data_path}/in_gmm_matrix_b_{i}.bin",
            out_matrix_c_path=f"{data_path}/out_gmm_matrix_c_{i}.bin"
        )
        matrix_c_list.append(matrix_c)

    return matrix_c_list


def generate_dispatch_gmm_swiglu(m, k, n, top_k, active_num, capacity, drop_pad_mode, ep, expert_per_rank, batch_size, max_output_size, dtype, data_path="./output"):
    # Reuse dispatch_gmm up to GMM1
    matrix_c_list = generate_dispatch_gmm(m, k, n, top_k, active_num, capacity, drop_pad_mode, ep, expert_per_rank, batch_size, max_output_size, dtype, data_path)
    
    # 4. SwiGLU
    swiglu_out_list = []
    for i in range(ep):
        swiglu_out = swiglu(
            matrix_c_list[i],
            matrix_c_path=f"{data_path}/in_swiglu_matrix_c_{i}.bin",
            out_swiglu_path=f"{data_path}/out_swiglu_{i}.bin"
        )
        swiglu_out_list.append(swiglu_out)
        
    return swiglu_out_list

def generate_data(args: argparse.Namespace) -> None:
    M, N, K = args.m, args.n, args.k
    top_k = args.top_k
    expert_num = args.expert
    ep_size = args.ep
    out_type = args.out_dtype.torch_type

    expert_per_rank = expert_num // ep_size
    
    if args.kernel_name == "dispatch_gmm":
        generate_dispatch_gmm(
            M, K, N, top_k, 
            active_num=M*top_k, capacity=M*top_k,
            drop_pad_mode=0,
            ep=ep_size, expert_per_rank=expert_per_rank,
            batch_size=1,
            max_output_size=M*top_k*expert_per_rank,
            dtype=out_type
        )
    elif args.kernel_name == "dispatch_gmm_swiglu":
        generate_dispatch_gmm_swiglu(
            M, K, N, top_k, 
            active_num=M*top_k, capacity=M*top_k,
            drop_pad_mode=0,
            ep=ep_size, expert_per_rank=expert_per_rank,
            batch_size=1,
            max_output_size=M*top_k*expert_per_rank,
            dtype=out_type
        )
    else:
        raise ValueError(f"Unsupported kernel name: {args.kernel_name}")

if __name__ == "__main__":
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
