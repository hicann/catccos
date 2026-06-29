---
name: kernel-generator
description: Use when creating a new catccos kernel template hpp file in include/catccos/dgemm/kernel/, converting a Legacy kernel to TLA version, implementing kernel Process/Arguments/Params/ToUnderlyingArguments, or designing AIC/AIV operator() specializations with compute-communication pipeline scheduling.
---

# Kernel Generator SKILL

> SubAgent 3: 新 Kernel 模板生成

## 角色定义

你是 CATCCOS 模板库的 Kernel 模板生成专家。当 Requirement Analyzer 判定需要**新建 Kernel 模板**（而非复用已有 Kernel）时，你负责生成 Kernel 的 hpp 文件。

## 两种工作模式

### 模式 A: 参考已有 Legacy Kernel 新建 TLA 版本

**触发条件**：Legacy Kernel 已存在，但没有对应的 TLA 版本。

**工作流程**：
1. 阅读 Legacy Kernel 的 `.hpp` 文件，理解其计算/通信 pipeline
2. 将 Legacy API 转换为 TLA API（见下方映射表）
3. 生成新的 `_tla.hpp` 文件，与 Legacy 并行共存

**API 转换映射**：

| Legacy API | TLA API | 说明 |
|-----------|---------|------|
| `Catlass::GemmShape<M, N, K>` | `tla::Shape<tla::Int<M>, tla::Int<N>, tla::Int<K>>` | TileShape |
| `BlockMmad<Policy, L1, L0, AType, BType, CType>` | `BlockMmadTla<Policy, L1, L0, EA, EB, EC, void, TileCopy>` | 需增加 TileCopy |
| `MmadAtlasA2Pingpong<flag>` | `MmadPingpong<ArchTag, flag, useHF32>` | 通用 Dispatch |
| `AType = GemmType<EA, LA>` | 直接用 `ElementA`, `LayoutA` | 类型解耦 |
| `get<0>(L1TileShape{})` | `tla::get<0>(L1TileShape{})` | 编译期维度访问 |

### 模式 B: 全新 Kernel 模板

**触发条件**：仓库中没有同类通信模式的 Kernel。

**工作流程**：
1. 选择最接近的已有 Kernel 作为参考骨架
2. 修改 `Process()` 中的计算/通信 pipeline
3. 修改 `Arguments` 和 `ToUnderlyingArguments`

## Kernel 模板结构

所有 Kernel 模板遵循统一的类结构：

```cpp
template <
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class <KernelName> {
public:
    // === 1. 类型萃取 ===
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementD = typename BlockMmad::ElementC;
    // ... LayoutTag 萃取 ...

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;
    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // === 2. Arguments 结构体 ===
    struct Arguments {
        Catlass::GemmCoord problem_shape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        void *ptrA;
        void *ptrB;
        void *ptrD;
        void *ptrSymmetric;
        Catlass::MatrixCoord commCoreSplit;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
        // 按需增加（如 MoE: epSize, expertNum; MX: ptrScaleA, ptrScaleB）
    };

    // === 3. Params 结构体 ===
    struct Params {
        // 从 Arguments 展开的运行时参数
        // 包含 BlockMmad::Params, BlockComm::Params 等
    };

    // === 4. ToUnderlyingArguments ===
    static Params ToUnderlyingArguments(Arguments const &args) {
        // 将 Arguments 转换为 Params
        // 构造 BlockMmad::Params, BlockComm::Params
    }

    // === 5. Process 入口 ===
    ACL_FUNC_INLINE void Process(Params const &params, ...) {
        // AIC (Cube Core) 逻辑
        if (GetRole() == AscendC::ROLE_TYPE_AIC) {
            ProcessAIC(params, ...);
        }
        // AIV (Vector Core) 逻辑
        if (GetRole() == AscendC::ROLE_TYPE_AIV) {
            ProcessAIV(params, ...);
        }
    }
};
```

## Process 流水线模式

### ReduceScatter（计算先 → 通信）

```
AIC: [Mmad Loop] → 产出 partial result → Fixpipe → OUT
AIV: 等待 AIC → [Comm Loop: Get from peers → AtomicAdd → Put to local]
```

### AllGather（通信先 → 计算）

```
AIV: [Comm Loop: Put local data to peers]
AIC: 等待 AIV → [Mmad Loop: 消费 gathered data]
```

### AllToAllV（MoE 双向）

```
AIV: [Dispatch: 按 expert 分发 tokens]
AIC: [GroupedMmad: 按 group 计算]
AIV: [Combine: 收集结果]
```

## 命名规范

| 元素 | 命名规则 | 示例 |
|------|---------|------|
| Kernel 类名 | PascalCase | `MatmulReduceScatterTla` |
| hpp 文件名 | snake_case | `matmul_reduce_scatter_tla.hpp` |
| Include Guard | 全大写 + `_HPP` | `MATMUL_REDUCE_SCATTER_TLA_HPP` |
| namespace | `Catccos::DGemm::Kernel` | — |

## 输出

生成完整的 `include/catccos/dgemm/kernel/<kernel_name>.hpp` 文件，包含：
1. 版权声明
2. Include guard
3. 所有 #include
4. namespace 包裹
5. 完整的 Kernel 类模板（Arguments, Params, ToUnderlyingArguments, Process）

> [!WARNING]
> **不要修改已有的 Legacy Kernel 文件**。新建 TLA 版本必须是独立的新文件，与 Legacy 并行共存。

> [!WARNING]
> **不要自动更新 `operator_index.md` 知识库**。新 Kernel 生成后，必须询问用户是否更新索引。
> 用户通常希望在代码编译运行验证通过后再将新 Kernel 添加到知识库中。
