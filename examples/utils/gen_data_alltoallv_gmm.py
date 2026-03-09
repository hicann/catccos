import argparse
import math
import torch
 
from utils import DataType, tensor_to_file
 
def random_uniform(
    low: float, high: float,
    size: tuple,
    dtype: torch.dtype = torch.float
) -> torch.Tensor:
    # 计算两次采样的长度
    element_num = math.prod(size)
    l0 = math.floor(math.sqrt(element_num))
    l0 = l0 if l0 * (l0 + 1) >= element_num else l0 + 1
    l1 = l0 + 1
    # 生成两次采样在 U(0, 1) 下的随机 tensor
    sample1 = torch.rand(size=(l0,), dtype=dtype).repeat(math.ceil(element_num / l0))[:element_num]
    sample2 = torch.rand(size=(l1,), dtype=dtype).repeat(math.ceil(element_num / l1))[:element_num]
    # 将两次采样结果组合为 U(0, 1) 的 tensor
    sample_add = sample1 + sample2
    sample_result = sample_add - torch.floor(sample_add)
    return (sample_result * (high - low) + low).reshape(size)
 
def generate_constrained_row_sum_tensor(
    row_sum: int,
    size: tuple[int, int],
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    row, col = size
    tensor = torch.empty(size=size, dtype=torch.int32)
    tensor[:, :-1] = torch.randint(low=0, high=row_sum + 1, size=(row, col - 1), dtype=dtype).sort(dim=1).values
    tensor[:, -1] = row_sum
    tensor[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    return tensor
 
def simulate_all_to_all_v_with_permute(
    input_list: list[torch.Tensor],
    global_tokens_per_expert: torch.Tensor,
    ep_size: int
) -> list[torch.Tensor]:
    rank_size, expert_num = global_tokens_per_expert.shape
    local_expert_num = expert_num // ep_size
    input_groups_list = [
        torch.split(input, global_tokens_per_expert[rank_idx].tolist(), dim=0)
        for rank_idx, input in enumerate(input_list)
    ]
    return [
        torch.cat([
            input_groups[(output_rank_idx % ep_size) * local_expert_num + local_expert_idx]
            for local_expert_idx in range(local_expert_num)
            for input_groups in input_groups_list
        ], dim=0)
        for output_rank_idx in range(rank_size)
    ]
 
def grouped_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor
) -> torch.Tensor:
    activation_list = torch.split(activation.to(torch.float32), group_list.tolist(), dim=0)
    weight_list = weight.to(torch.float32)
    results: list[torch.Tensor] = [None] * len(activation_list) # type: ignore
    for i, (activation, weight) in enumerate(zip(activation_list, weight_list)):
        results[i] = torch.mm(activation, weight)
    return torch.cat(results, dim=0)
 
def generate_data(args: argparse.Namespace) -> None:
    out_type = args.out_type.torch_type
    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    trans_a, trans_b = args.trans_a, args.trans_b
    expert_num = args.expert
    ep_size = args.ep
 
    assert(rank_size % ep_size == 0)
    assert(expert_num % ep_size == 0)
    tp_size = rank_size // ep_size
    local_expert_num = expert_num // ep_size
 
    matrix_a_origin_world = random_uniform(-2.0, 2.0, size=(rank_size, m, k), dtype=out_type)
    matrix_b_origin_world = random_uniform(-2.0, 2.0, size=(rank_size, local_expert_num, k, n), dtype=out_type)
 
    global_tokens_per_expert = generate_constrained_row_sum_tensor(row_sum=m, size=(rank_size, expert_num))
    global_tokens_per_local_expert_world = global_tokens_per_expert.reshape(
        shape=(rank_size, ep_size, local_expert_num)).permute(1, 0, 2).repeat(tp_size, 1, 1)
 
    permute_a_list = simulate_all_to_all_v_with_permute(list(matrix_a_origin_world), global_tokens_per_expert, ep_size)
 
    group_list_world = global_tokens_per_local_expert_world.sum(dim=1)
    output_list = [
        grouped_matmul(input, other, group_list)
        for input, other, group_list in zip(permute_a_list, list(matrix_b_origin_world), list(group_list_world))
    ]
 
    matrix_a_world = matrix_a_origin_world.mT if trans_a else matrix_a_origin_world
    matrix_b_world = matrix_b_origin_world.mT if trans_b else matrix_b_origin_world

    # for alltoallv_gmm_v2
    tensor_to_file(global_tokens_per_expert, f"./output/global_tokens_per_expert_matrix.bin")
    
    for rank_idx in range(rank_size):
        tensor_to_file(matrix_a_world[rank_idx], f"./output/a_gm_{rank_idx}.bin")
        tensor_to_file(matrix_b_world[rank_idx], f"./output/b_gm_{rank_idx}.bin")
        tensor_to_file(global_tokens_per_expert[rank_idx], f"./output/local_tokens_per_expert_{rank_idx}.bin")
        tensor_to_file(global_tokens_per_local_expert_world[rank_idx],
                       f"./output/global_tokens_per_local_expert_{rank_idx}.bin")
 
        output = output_list[rank_idx]
        padding_output = torch.zeros(size=(rank_size * m, n), dtype=torch.float32)
        padding_output[:output.shape[0]] = output
        tensor_to_file(padding_output, f"./output/golden_{rank_idx}.bin")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_type', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('trans_a', type=int)
    parser.add_argument('trans_b', type=int)
    parser.add_argument('--expert', type=int, default=8)
    parser.add_argument('--ep', type=int, default=2)
    args = parser.parse_args()
 
    generate_data(args)