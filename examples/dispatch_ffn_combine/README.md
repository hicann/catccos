# Dispatch-FFN-Combine 示例说明

本示例用于验证 MoE 场景下的 `dispatch + FFN + combine` 融合流程。算子整体流程包括：

```text
原始 token 输入
    -> moe_init_routing_v2
    -> AllToAllV_GMM
    -> Swiglu
    -> GMM_AllToAllV
    -> MoeTokenUnpermute
    -> 最终 combine 输出
```

其中：

- `moe_init_routing_v2`：根据 expert 路由结果对 token 进行展开和重排，并生成 `expandedRowIdx`、`tokensPerExpert` 等元信息。
- `AllToAllV_GMM`：完成 dispatch 后的 token 通信，并执行第一段 GroupedMatMul。
- `Swiglu`：对第一段 FFN 输出执行 Swiglu 激活。
- `GMM_AllToAllV`：执行第二段 GroupedMatMul，并将结果按照 dispatch 的反向路径回传。
- `MoeTokenUnpermute`：根据 `expandedRowIdx` 和 `probs` 将 `M * topK` 个展开 token 聚合回原始 `M` 个 token。

---

# 使用方式

## 1. 编译项目

进入示例目录并执行编译脚本：

```bash
cd examples/dispatch_ffn_combine
bash scripts/build.sh
```

编译完成后，会在对应的 `build` 目录下生成可执行文件和算子相关产物。

如果编译失败，请优先确认 CANN 环境变量已经正确加载：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

同时确认 `ASCEND_HOME_PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` 等环境变量是否正确。

---

## 2. 运行 Dispatch-FFN-Combine 示例程序

在示例目录下执行运行脚本：

```bash
bash scripts/run.sh <device_list>
```

### 参数说明

| 参数 | 含义 | 说明 |
|---|---|---|
| `device_list` | 使用的 NPU 设备列表 | 多个设备编号使用英文逗号分隔，例如 `0,1`、`0,1,2,3,4,5,6,7` |

`ep_size` 和 `expert_num` 不再通过命令行传入，由运行脚本或配置文件内部统一配置。使用不同卡数运行时，只需要修改 `device_list`。

示例：使用第 6 和第 7 个 NPU 设备运行 2 卡场景：

```bash
bash scripts/run.sh 6,7
```

示例：使用第 0 到第 7 个 NPU 设备运行 8 卡场景：

```bash
bash scripts/run.sh 0,1,2,3,4,5,6,7
```

---

# 配置计算规模

矩阵形状参数可在配置文件中设置：

```bash
scripts/test_shapes.csv
```

该文件用于配置测试用例的主要 shape 参数：

| 参数 | 含义 |
|---|---|
| `M` | 原始 token 数 |
| `K` | 输入 hidden size |
| `N` | 第一段 FFN 输出维度，Swiglu 后通常变为 `N / 2` |

修改 `scripts/test_shapes.csv` 后，重新运行：

```bash
bash scripts/run.sh <device_list>
```

如果 shape、专家数、topK 或 EP 配置发生变化，建议同步清理旧的测试数据，避免读取到上一轮生成的输入文件。

---

# 数据生成与输入输出

运行脚本通常会完成以下步骤：

```text
1. 根据 scripts/test_shapes.csv 生成输入数据
2. 为每个 rank 生成本地输入 token、expert index、probs、权重等文件
3. 启动多卡算子运行
4. dump 每个 rank 的实际输出
5. 与 golden 输出进行精度比对
```

典型输入包括：

| 数据 | 含义 |
|---|---|
| `in_routing_matrix_a_{rank}.bin` | 当前 rank 的原始 token 输入 |
| `in_gmm_matrix_b_{rank}.bin` / `in_gmm_matrix_b2_{rank}.bin` | FFN 两段 GMM 的权重 |
| `in_routing_expert_idx_{rank}.bin` | 每个 token 的 topK expert 路由结果 |
| `in_gather_gate_weight_{rank}.bin` | 每个 token 对应 topK expert 的 combine 权重 |

典型输出包括：

| 数据 | 含义 |
|---|---|
| `output_{rank}.bin` | 当前 rank 的最终 combine 输出 |
| `golden_output_{rank}.bin` | CPU 或 PyTorch 参考结果 |
| `real_scatter_index_{rank}.bin` | 实际运行中使用的 expanded row index，调试 unpermute 时常用 |

具体文件名以当前 `scripts/run.sh` 和数据生成脚本为准。

---

# 算子流程说明

## 1. Routing 阶段

`moe_init_routing_v2` 根据 `expert_idx` 将原始输入 token 展开为 `M * topK` 个 token，并按照 expert / rank 的顺序重排。

该阶段会生成：

```text
symmetricA       : routing 后的 token 数据
expandedRowIdx   : expanded token 到原始 token/topK slot 的映射
tokensPerExpert  : 每个 rank 发送给 expert 的 token 数
```

其中 `expandedRowIdx` 会在最后的 `MoeTokenUnpermute` 阶段使用，用于将展开后的 `M * topK` 个 token 聚合回原始 `M` 个 token。

---

## 2. AllToAllV_GMM 阶段

该阶段完成 dispatch 后的跨 rank token 搬运，并执行第一段 GroupedMatMul：

```text
A: [M * topK, K]
B: [K, N]
C: [M * topK, N]
```

输出写入中间 workspace，作为后续 Swiglu 的输入。

---

## 3. Swiglu 阶段

Swiglu 对第一段 GMM 输出执行激活计算：

```text
输入 : [M * topK, N]
输出 : [M * topK, N / 2]
```

Swiglu 输出作为第二段 GMM 的输入。

当前实现中，Swiglu 与前后 GMM 之间通过 callback / cross-core flag 进行同步：

```text
GMM1 AIC 完成
    -> 通知 Swiglu AIV 开始

Swiglu AIV 完成
    -> 通知 GMM2 AIC 开始
```

---

## 4. GMM_AllToAllV 阶段

第二段 GMM 执行：

```text
A: [M * topK, N / 2]
B: [N / 2, K]
C: [M * topK, K]
```

随后通过 AllToAllV 的反向路径将 token 结果回传。

该阶段的输出会被写入 symmetric workspace，用于最后的 unpermute。

---

## 5. MoeTokenUnpermute 阶段

`MoeTokenUnpermute` 根据：

```text
expandedRowIdx
probs
GMM2 输出
```

执行加权聚合：

```text
output[token] = sum_{i=0}^{topK-1} GMM2Output[expandedRowIdx[token, i]] * probs[token, i]
```

最终输出 shape 为：

```text
[M, K]
```

---

# Workspace 说明

当前示例中，local workspace 统一由 `WorkspaceInfo` 管理，主要包括：

| 字段 | 说明 |
|---|---|
| `expandedRowIdx` | routing 阶段生成的展开 token 映射 |
| `moeInitRoutingWorkspace` | routing 阶段临时 workspace |
| `allToAllVGmmWorkspace` | 第一段 AllToAllV_GMM 使用的临时 workspace |
| `allToAllVGmmOut` | 第一段 GMM 输出 |
| `swigluOutput` | Swiglu 输出 |
| `gmmAllToAllWorkspace` | 第二段 GMM_AllToAllV 使用的临时 workspace |

symmetric workspace 主要包括：

| 字段 | 说明 |
|---|---|
| `symmetricA` | routing 后 token 数据，同时复用为 GMM2 反向回传后的结果 |
| `tokensPerExpert` | allgather 后的全局 expert token 统计 |
| `perTokenScale` | 预留给量化场景使用 |

host 侧分配 `gmWorkSpace` 时，需要保证大小覆盖 `WorkspaceInfo` 中所有 local workspace 字段。
当 `rankSize` 从 2 卡扩展到 4 卡或 8 卡时，workspace 会按 rank 数线性增长。host 侧分配不足时，可能出现 MTE 访问越界、`aclrtMemcpy` 报错、某些 rank 精度异常等问题。

---

# 精度校验

运行完成后，脚本会将实际输出与 golden 输出进行比较。

常见输出形式如下：

```text
Index | FP16 | FP32 | AbsErr | RelErr
...
error num: xxx
precision: xx.xxxxx
```

如果误差较小，通常属于 FP16 / BF16 舍入误差；如果出现整行大幅错误，建议优先检查：

```text
1. expandedRowIdx 是否正确
2. probs 是否读取正确
3. GMM2 输出是否正确
4. workspace 是否越界
5. unpermute 的输入 shape 是否为 [M * topK, K]
6. 最终输出 shape 是否为 [M, K]
```

---

# 注意事项与排查建议

## gmA 传空指针的场景

在当前 dispatch-FFN-combine 流程中，原始输入 `gmA` 已经在 `moe_init_routing_v2` 阶段被搬运并重排到 `symmetricA` 中。后续 `AllToAllV_GMM` 实际使用的是 routing 后的 `symmetricA`，因此 `AllToAllVGMMSwigluImpl` 中的 `gmA` 可以传 `nullptr`。

## unpermute 输入位置

第二段 `GMM_AllToAllV` 执行后，会通过反向 AllToAllV 将结果回传到 symmetric workspace。当前实现复用 `symmetricA` 保存回传后的 `[M * topK, K]` 结果，因此最终 `MoeTokenUnpermute` 读取：

```cpp
workspaceInfo.symmetricA
```

作为输入。

## shape 修改后的同步更新

workspace 大小依赖以下参数：

```text
M
topK
K
N
rankSize
expertNum
epSize
expertPerRank
```

修改 `topK`、`M`、`K`、`N`、`expert_num`、`ep_size` 后，需要重新生成输入数据，并同步确认 host 侧 workspace 分配大小。

## 多卡运行

多卡运行时，脚本内部使用的 `ep_size` 应与实际使用的 NPU 数量一致，`expert_num` 需要能被 `ep_size` 整除。例如：

```text
2 卡：ep_size=2，expert_num=8
4 卡：ep_size=4，expert_num=8
8 卡：ep_size=8，expert_num=8
```

当前运行脚本只接收 `device_list` 一个入参，因此切换卡数时需要确认脚本或配置文件内部的 `ep_size`、`expert_num` 与目标场景一致。当 `expert_num / ep_size` 过小或不能整除时，部分 expert 分布可能不符合当前测试脚本或算子假设。

---

# 推荐运行流程

建议按以下顺序验证：

```bash
# 1. 编译
cd examples/dispatch_ffn_combine
bash scripts/build.sh

# 2. 运行 2 卡基础场景
bash scripts/run.sh 6,7

# 3. 运行 4 卡或 8 卡场景
bash scripts/run.sh 0,1,2,3
bash scripts/run.sh 0,1,2,3,4,5,6,7

# 4. 修改 scripts/test_shapes.csv 后重新运行
bash scripts/run.sh 0,1,2,3,4,5,6,7
```

当 2 卡、4 卡、8 卡均通过精度校验后，说明 routing、dispatch、FFN、combine 和 unpermute 链路基本正确。
