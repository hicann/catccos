# CATCCOS 算子模板索引

> 自动扫描生成，供 Agent 知识库使用

## 1. Kernel 模板总览（25个）

### 1.1 AllGather + Matmul 族（通信先于计算，默认 CopyDirect::Put）

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `AllGatherMatmul` | ❌ Legacy | AtlasA2 | 基础版 | `allgather_matmul/` |
| `AllGatherDequantMatmul` | ❌ Legacy | AtlasA2 | INT8 dequant | `allgather_matmul_dequant/` |
| `AllGatherDequantMatmulBias` | ❌ Legacy | AtlasA2 | INT8 dequant + bias | `allgather_matmul_dequant_bias/` |
| `AllGatherMatmulWithGatherResult` | ❌ Legacy | AtlasA2 | 额外 gather 结果收集 | `allgather_matmul_with_gather_result/` |
| `AllGatherMatmulWithGatherResultAndLocalMm` | ❌ Legacy | AtlasA2 | gather结果 + 本地MM | — |
| `AllGatherMatmulWithLocalMmOpt` | ❌ Legacy | AtlasA2 | 本地MM优化 | — |
| `AllGatherMatmulWithRdmaWrite` | ❌ Legacy | AtlasA2 | RDMA写 | `allgather_matmul_rdma/` |
| `AllGatherMatmulWithRemoteRead` | ❌ Legacy | AtlasA2 | 远程读 | `allgather_matmul_remote_read/` |
| `AllGatherMatmulWithRemoteReadLocalMmOpt` | ❌ Legacy | AtlasA2 | 远程读+本地MM | `allgather_matmul_remote_read/` |
| `Ascend950AllGatherMatmul` | ✅ TLA | Ascend950 | 950专用 | `ascend950_allgather_matmul/` |
| `MxAllGatherMatmulTla` | ✅ TLA | Ascend950 | MX量化 | `ascend950_fp8_mx_allgather_matmul/`, `ascend950_fp4_mx_allgather_matmul/` |

### 1.2 Matmul + ReduceScatter 族（计算先于通信，默认 CopyDirect::Get）

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `MatmulReduceScatter` | ❌ Legacy | AtlasA2 | 基础版 | `matmul_reduce_scatter/` |
| `MatmulReduceScatterTla` | ✅ TLA | AtlasA2/Ascend950 | TLA跨架构 | `ascend950_matmul_reduce_scatter/` |
| `MatmulReduceScatterMxTla` | ✅ TLA | Ascend950 | MX量化+TLA | `ascend950_mxfp8_matmul_reduce_scatter/` |
| `MatmulDequantReduceScatterV2` | ❌ Legacy | AtlasA2 | INT8 dequant | `matmul_dequant_reduce_scatter_v2/` |

### 1.3 Matmul + AllReduce 族（计算先于通信，默认 CopyDirect::Get）

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `MatmulAllReduce` | ❌ Legacy | AtlasA2 | TP场景 | `matmul_allreduce/` |

### 1.4 GroupedMatmul + AllToAllV 族（MoE 场景）

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `GroupedMatmulAllToAllV` | ❌ Legacy | AtlasA2 | 基础版 | `grouped_matmul_alltoallv/` |
| `GroupedMatmulAllToAllVTla` | ✅ TLA | AtlasA2/Ascend950 | TLA跨架构 | `grouped_matmul_alltoallv_tla/`, `ascend950_grouped_matmul_alltoallv/` |
| `GroupedMatmulAllToAllVMx` | ❌ Legacy | Ascend950 | MX量化 | `ascend950_fp8_mx_grouped_matmul_alltoallv/`, `ascend950_fp4_mx_grouped_matmul_alltoallv/` |

### 1.5 AllToAllV + GMM 族（v2 架构）

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `AlltoallvGMMKernel` | ❌ Legacy | AtlasA2 | v2架构 | `alltoallv_gmm_v2/` |
| `AllToAllVGroupedMatmul` | ❌ Legacy | AtlasA2 | 反方向 | `alltoallv_grouped_matmul/` |
| `AlltoallvGMMDequantV2` | ❌ Legacy | AtlasA2 | dequant变体 | — |
| `GMMAllToAllVV2` | ❌ Legacy | AtlasA2 | GMM先行 | `gmm_alltoallv_v2/` |

### 1.6 Ascend950 AllToAll 族

| Kernel 模板 | TLA | 架构 | 特殊特性 | 对应 Example |
|-------------|-----|------|----------|-------------|
| `Ascend950AllToAllMatmul` | ✅ TLA | Ascend950 | AllToAll→Matmul | `ascend950_alltoall_matmul/` |
| `MatmulAllToAllTla` | ✅ TLA | Ascend950 | Matmul→AllToAll | `ascend950_matmul_alltoall/` |

---

## 2. 通信模式 → 代码模式速查

| 通信模式 | 默认 CopyDirect | 计算/通信顺序 | 典型Scheduler | 需要AtomicAdd |
|----------|----------------|---------------|---------------|---------------|
| AllGather | Put | 通信先 → 计算 | `BlockCommSwizzle` / `GemmBlockSwizzleAllGatherMesh` | No |
| ReduceScatter | Get | 计算先 → 通信 | `BlockCommSchedulerReduceScatter` | Yes |
| AllReduce | Get | 计算先 → 通信 | `BlockCommSwizzle` | Yes |
| AllToAllV (MoE) | Get | 计算先 → 通信 | `BlockCommSchedulerReduceScatterAllToAllV` | No |
| AllToAll | Put | 视方向而定 | `BlockCommSwizzle` | No |

> **CopyDirect 默认规则（由计算/通信顺序决定）**：
> - **通信先 → Put**：本地数据 put 到 shmem → shmem 通信结果直接用于计算 → 结果放在 local GM
> - **计算先 → Get**：本地计算后结果放到 shmem → 从远端 shmem get 数据到本地 local GM
>
> CopyDirect 与通信模式完全解耦，用户可显式覆盖。如 AllGather 可用 Get（`allgather_matmul_remote_read`）、ReduceScatter 可用 Put（`matmul_dequant_reduce_scatter_v2`）。

---

## 3. TLA 跨架构复用清单

以下 Kernel 模板已实现 TLA 跨架构支持（同一 Kernel 模板，不同 Config）：

| TLA Kernel | AtlasA2 Example | Ascend950 Example | 缺失 |
|------------|-----------------|-------------------|------|
| `GroupedMatmulAllToAllVTla` | `grouped_matmul_alltoallv_tla/` | `ascend950_grouped_matmul_alltoallv/` | — |
| `MatmulReduceScatterTla` | `matmul_reduce_scatter_tla/` | `ascend950_matmul_reduce_scatter/` | — |

以下 Kernel 仍为**架构专用**（待新建 TLA 版本）：

| Legacy Kernel (AtlasA2) | 建议新建 TLA 版本 |
|--------------------------|-------------------|
| `AllGatherMatmul` | `AllGatherMatmulTla` |
| `MatmulAllReduce` | `MatmulAllReduceTla` |

---

## 4. Gap 分析（可新建的 Kernel + Example 组合）

> **Agent 无需扫描文件系统**。下表列出了所有已有 Kernel 但**缺少对应 Example** 的组合，新建算子时优先从这些 Gap 中选择。

### 4.1 有 Kernel 模板但无 Example 的组合

| Kernel 模板 | 架构 | 缺失的 Example | 新建方式 |
|-------------|------|----------------|---------|
| `AllGatherMatmulWithGatherResultAndLocalMm` | AtlasA2 | 无 Example | 新建（参考 `allgather_matmul_with_gather_result/`） |
| `AllGatherMatmulWithLocalMmOpt` | AtlasA2 | 无 Example | 新建 |
| `AlltoallvGMMDequantV2` | AtlasA2 | 无 Example | 新建（参考 `alltoallv_gmm_v2/`） |

### 4.2 TLA Kernel 已支持但对应架构无 Example

| TLA Kernel | 已有 Example | 可新建 Example | 新建方式 |
|------------|-------------|---------------|---------|
| `MatmulReduceScatterTla` | Ascend950, AtlasA2(TLA) | — | 已补齐 |
| `GroupedMatmulAllToAllVTla` | AtlasA2, Ascend950 | — | 已补齐 |

### 4.3 可新建 TLA 版本的 Legacy Kernel

| Legacy Kernel | 通信模式 | 现有架构 | 可扩展架构 |
|---------------|---------|---------|-----------|
| `AllGatherMatmul` | AllGather | AtlasA2 only | Ascend950（通过新建 TLA） |
| `MatmulAllReduce` | AllReduce | AtlasA2 only | Ascend950（通过新建 TLA） |

---

## 5. Example 完整列表（与 CMakeLists.txt 同步）

> 以下列表与 `examples/CMakeLists.txt` 中的 `foreach(EXAMPLE ...)` 保持同步。Agent 可直接判断某个 Example 是否已存在，无需扫描文件系统。

### 5.1 AllGather + Matmul

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `allgather_matmul/` | AtlasA2 | AllGatherMatmul | FP16 | — |
| `allgather_matmul_remote_read/` | AtlasA2 | AllGatherMatmulWithRemoteRead | FP16 | CopyDirect::Get |
| `allgather_matmul_rdma/` | AtlasA2 | AllGatherMatmulWithRdmaWrite | FP16 | RDMA |
| `allgather_matmul_with_gather_result/` | AtlasA2 | AllGatherMatmulWithGatherResult | FP16 | gather 结果 |
| `allgather_matmul_dequant/` | AtlasA2 | AllGatherDequantMatmul | INT8→FP16 | dequant |
| `allgather_matmul_dequant_bias/` | AtlasA2 | AllGatherDequantMatmulBias | INT8→FP16 | dequant+bias |
| `ascend950_allgather_matmul/` | Ascend950 | Ascend950AllGatherMatmul | FP16 | TLA |
| `ascend950_fp8_mx_allgather_matmul/` | Ascend950 | MxAllGatherMatmulTla | MXFP8 | MX量化 |
| `ascend950_fp4_mx_allgather_matmul/` | Ascend950 | MxAllGatherMatmulTla | MXFP4 | MX量化 |

### 5.2 Matmul + ReduceScatter

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `matmul_reduce_scatter/` | AtlasA2 | MatmulReduceScatter | FP16 | Legacy |
| `matmul_reduce_scatter_tla/` | AtlasA2 | MatmulReduceScatterTla | FP16 | TLA |
| `ascend950_matmul_reduce_scatter/` | Ascend950 | MatmulReduceScatterTla | FP16 | TLA |
| `ascend950_mxfp8_matmul_reduce_scatter/` | Ascend950 | MatmulReduceScatterMxTla | MXFP8 | MX量化 |
| `matmul_dequant_reduce_scatter_v2/` | AtlasA2 | MatmulDequantReduceScatterV2 | INT8→FP16 | dequant, CopyDirect::Put |

### 5.3 Matmul + AllReduce

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `matmul_allreduce/` | AtlasA2 | MatmulAllReduce | FP16 | Legacy |

### 5.4 GroupedMatmul + AllToAllV（MoE）

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `grouped_matmul_alltoallv/` | AtlasA2 | GroupedMatmulAllToAllV | FP16 | Legacy |
| `grouped_matmul_alltoallv_tla/` | AtlasA2 | GroupedMatmulAllToAllVTla | FP16 | TLA |
| `ascend950_grouped_matmul_alltoallv/` | Ascend950 | GroupedMatmulAllToAllVTla | FP16 | TLA |
| `ascend950_fp8_mx_grouped_matmul_alltoallv/` | Ascend950 | GroupedMatmulAllToAllVMx | MXFP8 | MX量化 |
| `ascend950_fp4_mx_grouped_matmul_alltoallv/` | Ascend950 | GroupedMatmulAllToAllVMx | MXFP4 | MX量化 |

### 5.5 AllToAllV + GMM / GMM + AllToAllV

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `alltoallv_grouped_matmul/` | AtlasA2 | AllToAllVGroupedMatmul | FP16 | 反方向 |
| `alltoallv_gmm_v2/` | AtlasA2 | AlltoallvGMMKernel | FP16 | v2架构 |
| `gmm_alltoallv_v2/` | AtlasA2 | GMMAllToAllVV2 | FP16 | GMM先行 |

### 5.6 Ascend950 AllToAll

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `ascend950_alltoall_matmul/` | Ascend950 | Ascend950AllToAllMatmul | FP16 | AllToAll→Matmul |
| `ascend950_matmul_alltoall/` | Ascend950 | MatmulAllToAllTla | FP16 | Matmul→AllToAll |
| `ascend950_quant_alltoall/` | Ascend950 | — | INT8/FP8 | 量化 AllToAll |

### 5.7 MoE Dispatch/Combine（MegaMoE）

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `dispatch_gmm/` | AtlasA2 | — | FP16 | Dispatch + GMM |
| `dispatch_gmm_swiglu/` | AtlasA2 | — | FP16 | +SwiGLU |
| `dispatch_gmm_dequant_swiglu/` | AtlasA2 | — | INT8→FP16 | +dequant+SwiGLU |
| `dispatch_ffn_combine/` | AtlasA2 | — | FP16 | FFN+Combine |

### 5.8 Ascend950 量化通信

| Example 目录 | 架构 | Kernel | 数据类型 | 特殊特性 |
|-------------|------|--------|---------|---------|
| `ascend950_quant_allgather/` | Ascend950 | — | INT8/FP8 | 量化 AllGather |
| `ascend950_mx_quant_allgather/` | Ascend950 | — | MXFP8 | MX量化 AllGather |

---

## 6. Example 目录结构标准

每个 Example 目录包含：
```
examples/<op_name>/
├── CMakeLists.txt
├── README.md
├── <op_name>_device.h      # Config struct
├── <op_name>_host.h         # 内存管理（继承 CatccosOperator）
├── main.cpp                 # 入口
└── scripts/
    ├── build.sh
    ├── run.sh
    └── test_shapes.csv
```

