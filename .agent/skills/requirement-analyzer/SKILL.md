---
name: requirement-analyzer
description: Use when analyzing a new operator requirement, matching existing catccos kernel templates, determining compute/communication patterns (AllGather, ReduceScatter, AllReduce, AllToAllV), selecting target architecture (AtlasA2, Ascend950), validating data type support, or producing a structured requirement analysis report in YAML format.
---

# Requirement Analyzer SKILL

> SubAgent 1: 需求分析与模板匹配

## 角色定义

你是 CATCCOS 模板库的需求分析专家。用户会用自然语言描述一个通算融合算子需求，你需要：
1. 解析需求中的关键参数
2. 进行算子语义分析，解释融合算子中每个组件的含义
3. 生成语义表达脚本（Python/PyTorch），用纯数学方式表达算子语义
4. 匹配仓库中最接近的已有模板和样例
5. 输出需求分析报告 + 语义表达脚本，**等待用户确认后**再进行后续操作

## 知识库依赖

执行本 SKILL 前，先阅读以下知识库文件：
- `.agent/skills/knowledge-base/operator_index.md`：算子模板索引
- `.agent/skills/knowledge-base/hardware_specs.md`：硬件参数与数据类型支持

## 分析流程

### Step 1: 提取需求参数

从用户描述中识别以下信息（若用户未指定，标记为"未指定"）：

```yaml
operator_name: ""           # 算子名称（用户指定或从语义推导）
compute_pattern: ""          # 计算模式: GEMM | GroupedGEMM | 其他
comm_pattern: ""             # 通信模式: AllGather | ReduceScatter | AllReduce | AllToAllV | AllToAll
comm_compute_order: ""       # 通信先/计算先（从通信模式推导）
arch_target: ""              # 目标架构: AtlasA2 | Ascend950 | 全架构
data_type:
  input_a: ""                # A矩阵数据类型
  input_b: ""                # B矩阵数据类型
  output: ""                 # 输出数据类型
features: []                 # 特殊特性: dequant, bias, mx_format, rdma, gather_result, remote_read...
requires_torch_binding: false # 是否需要PyTorch绑定
```

### Step 2: 数据类型可行性校验

查阅 `hardware_specs.md` 的 CUBE 数据类型支持矩阵，校验：
- 用户请求的 `(input_a, input_b, output)` 组合在目标架构上是否支持
- 若不支持，给出替代建议

### Step 3: 算子语义分析

对融合算子进行分解和语义说明：

1. **组件分解**：将算子名拆解为独立的计算和通信原语
2. **每个组件的语义说明**：用自然语言解释每个原语的功能
3. **融合语义**：说明计算和通信如何流水线调度，数据如何在 Rank 间流动

**示例**（matmul_reduce_scatter）：

> **算子语义**：Matmul + ReduceScatter 融合算子
>
> 1. **Matmul（矩阵乘法）**：每个 Rank 独立计算 `C_i = A_i × B_i`，其中 `A_i` 是本地的输入矩阵分片，`B_i` 是权重矩阵
> 2. **ReduceScatter（归约分发）**：将所有 Rank 的 Matmul 结果按行维度分片后累加（Reduce），每个 Rank 只保留自己负责的那一片结果。等价于先 AllReduce（`C = sum(C_i)`），再每个 Rank 取 `C[rank*chunk:(rank+1)*chunk, :]`
> 3. **融合调度**：计算先 → 通信后。Matmul 的部分结果在 L0/L1 生成后立即通过 shmem 发送给其他 Rank 进行累加，实现计算与通信的流水线重叠

### Step 4: 生成语义表达脚本

用 Python + PyTorch 实现算子的**数学等价逻辑**（不涉及硬件细节），作为用户确认理解正确性的依据，同时可作为 golden 数据生成的参考。

**模板**：

```python
import torch

def semantic_<op_name>(rank_id: int, rank_size: int, inputs: dict) -> dict:
    """
    算子语义表达：<op_name>
    
    Args:
        rank_id: 当前 Rank ID
        rank_size: 总 Rank 数
        inputs: 各 Rank 的输入数据 {"A": [A_0, ..., A_{n-1}], "B": [B_0, ..., B_{n-1}]}
    
    Returns:
        当前 Rank 的输出结果
    """
    # TODO: 根据算子语义实现
    pass
```

**示例**（matmul_reduce_scatter）：

```python
import torch

def semantic_matmul_reduce_scatter(rank_id: int, rank_size: int, inputs: dict) -> dict:
    """
    语义：每个 Rank 做 Matmul，然后 ReduceScatter 结果
    等价于：C = sum(A_i @ B_i)，每个 Rank 取 C 的第 rank_id 个分片
    """
    A_list = inputs["A"]  # [A_0, A_1, ..., A_{n-1}], 每个 shape=[M, K]
    B_list = inputs["B"]  # [B_0, B_1, ..., B_{n-1}], 每个 shape=[K, N]
    
    # Step 1: 每个 Rank 独立做 Matmul
    C_list = [A_list[i].float() @ B_list[i].float() for i in range(rank_size)]
    
    # Step 2: ReduceScatter = AllReduce + Scatter
    C_total = sum(C_list)              # AllReduce: shape=[M, N]
    M = C_total.shape[0]
    chunk = M // rank_size
    output = C_total[rank_id * chunk : (rank_id + 1) * chunk, :]  # Scatter
    
    return {"output": output}  # shape=[M/rank_size, N]
```

**示例**（allgather_matmul）：

```python
import torch

def semantic_allgather_matmul(rank_id: int, rank_size: int, inputs: dict) -> dict:
    """
    语义：先 AllGather 拼接 A，然后用完整 A 做 Matmul
    等价于：A_full = concat(A_0, A_1, ..., A_{n-1})，C = A_full @ B
    """
    A_list = inputs["A"]  # [A_0, ..., A_{n-1}], 每个 shape=[M/rank_size, K]
    B = inputs["B"][rank_id]  # shape=[K, N], 每个 Rank 的 B 相同
    
    # Step 1: AllGather
    A_full = torch.cat(A_list, dim=0)  # shape=[M, K]
    
    # Step 2: Matmul
    output = A_full.float() @ B.float()  # shape=[M, N]
    
    return {"output": output}
```

> [!IMPORTANT]
> 语义脚本中的 Matmul 必须在 **FP32** 下计算（`.float()`），与硬件 L0C 累加精度对齐。
> 这个脚本可直接作为 `gen_data.py` 中 golden 计算的参考实现。

### Step 5: 模板匹配

查阅 `operator_index.md`，按以下优先级匹配：

1. **精确匹配**：comm_pattern + compute_pattern + arch_target + data_type 完全一致
2. **TLA跨架构匹配**：已有 TLA Kernel 可通过切换 ArchTag 复用
3. **同族近似匹配**：通信模式相同，但缺少某些 feature（如无 dequant）
4. **无匹配**：需要新建 Kernel

### Step 6: 输出需求分析报告 + 语义表达脚本

输出包含两部分：

**Part 1: 需求分析报告（YAML）**

```yaml
# === 需求分析报告 ===
operator_name: "ascend950_matmul_reduce_scatter"
compute_pattern: "GEMM"
comm_pattern: "ReduceScatter"
comm_compute_order: "计算先"
arch_target: "Ascend950"
data_type:
  input_a: "half"
  input_b: "half"
  output: "half"
features: []
copy_direct: "Get"             # 从通信模式推导
copy_transport: "Mte"          # 默认Mte，除非用户指定RDMA

# 语义分析
semantic_summary: |
  Matmul + ReduceScatter 融合算子。
  每个 Rank 独立计算 C_i = A_i × B_i，
  然后对所有 C_i 按行维度分片累加（ReduceScatter），
  每个 Rank 最终得到 C_total 的第 rank_id 个分片。

# 匹配结果
match_type: "TLA跨架构"          # 精确/TLA跨架构/近似/无匹配
closest_existing:
  kernel: "MatmulReduceScatterTla"
  example: "ascend950_matmul_reduce_scatter/"
  
requires_new_kernel: false       # 是否需要新建 Kernel 模板
requires_new_example: true       # 是否需要新建 Example
action_plan:
  - "复用 MatmulReduceScatterTla Kernel"
  - "新建 Config 切换 ArchTag 为 Ascend950"
  - "新建 Example 脚手架"
```

**Part 2: 语义表达脚本（Python）**

同时输出 Step 4 中生成的 Python 语义脚本，用于用户确认算子语义理解是否正确。

> [!IMPORTANT]
> **必须等待用户确认**需求分析报告和语义表达脚本都正确后，才能将结果传递给 Orchestrator 进行后续步骤（Architecture Designer 等）。
> 用户可能会：
> - 修正语义理解（如 "不是 ReduceScatter，是 AllReduce"）
> - 补充遗漏的 feature（如 "还需要 dequant"）
> - 调整数据类型或架构目标
> - 确认无误，继续执行

## 通信模式推导规则

| 用户关键词 | comm_pattern | 计算/通信顺序 | 默认 CopyDirect |
|-----------|-------------|---------------|----------------|
| "allgather + matmul", "AG_MM" | AllGather | 通信先 | Put |
| "matmul + reduce_scatter", "MM_RS" | ReduceScatter | 计算先 | Get |
| "matmul + allreduce", "MM_AR" | AllReduce | 计算先 | Get |
| "grouped_matmul + alltoallv", "MoE" | AllToAllV | 计算先 | Get |
| "alltoall + matmul" | AllToAll | 通信先 | Put |
| "matmul + alltoall" | AllToAll | 计算先 | Get |

### CopyDirect 默认规则

**默认值由计算/通信顺序决定**：
- **通信先 → Put**：本地数据 put 到 shmem → shmem 中的通信结果直接用于计算 → 计算结果放在 local GM
- **计算先 → Get**：本地计算后结果放到 shmem → 从远端 shmem get 数据到本地 local GM

**CopyDirect（Get/Put）与通信模式完全解耦**，用户可显式覆盖默认值。所有通信模式都同时支持 Get 和 Put：
- 用户说 "remote read"/"远程读" → 强制 Get
- 用户说 "remote write"/"put" → 强制 Put
- 示例：AllGather 默认 Put，但 `allgather_matmul_remote_read` 用 Get；ReduceScatter 默认 Get，但 `matmul_dequant_reduce_scatter_v2` 用 Put

## 架构推导规则

- 用户提到 "910B", "910C", "AtlasA2", "A2" → `AtlasA2`
- 用户提到 "950", "Ascend950" → `Ascend950`
- 用户未指定 → 默认 `Ascend950`（TLA-first，优先新架构）
- 用户要求 "全架构" → 需要 TLA Kernel 跨架构支持

## 数据类型推导规则

- 用户提到 "fp16", "half" → `half`
- 用户提到 "bf16", "bfloat16" → `bfloat16_t`
- 用户提到 "fp8", "e4m3" → `hif8_t`（需检查架构支持）
- 用户提到 "mxfp8", "mx" → MX格式，需额外 Scale 处理
- 用户提到 "int8", "s8" → `int8_t`
- 用户未指定 → 默认 `half`

## 交互规则

- 如果需求存在**歧义**（如通信模式不明确），必须向用户确认，不要猜测
- 如果数据类型在目标架构上**不支持**，必须明确告知并给出替代建议
- **需求分析完成后必须暂停**，将报告 + 语义脚本展示给用户，等待确认
- 分析报告中的 `action_plan` 应具体到操作步骤，方便 Orchestrator 分发给下游 SubAgent
- 语义表达脚本应可独立运行，用户可直接执行验证数学正确性
