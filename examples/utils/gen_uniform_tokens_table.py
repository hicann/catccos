#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import argparse
import torch
 
from utils import tensor_to_file
 
def generate_uniform_tokens_table(
    row_sum: int,
    size: tuple[int, int],
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    row, cols = size
    tensor = torch.ones(size=size, dtype=torch.int32) * (row_sum / row)
    return tensor
 
def generate_data(args: argparse.Namespace) -> None:
    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    expert_num = args.expert
    ep_size = args.ep
 
    assert(rank_size % ep_size == 0)
    assert(expert_num % ep_size == 0)
    tp_size = rank_size // ep_size
    local_expert_num = expert_num // ep_size
 
    global_tokens_per_expert = generate_uniform_tokens_table(row_sum=m, size=(rank_size, expert_num))
    global_tokens_per_expert = torch.ones_like(global_tokens_per_expert) * (m // expert_num)
    global_tokens_per_local_expert_world = global_tokens_per_expert.reshape(
        shape=(rank_size, ep_size, local_expert_num)).permute(1, 0, 2).repeat(tp_size, 1, 1)
 
    for rank_idx in range(rank_size):
        tensor_to_file(global_tokens_per_expert[rank_idx], f"./output/local_tokens_per_expert_{rank_idx}.bin")
        tensor_to_file(global_tokens_per_local_expert_world[rank_idx],
                       f"./output/global_tokens_per_local_expert_{rank_idx}.bin")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('--expert', type=int, default=8)
    parser.add_argument('--ep', type=int, default=2)
    args = parser.parse_args()
 
    generate_data(args)