---
name: architecture-designer
description: Use when designing Config struct template parameters for a catccos operator, selecting MmadDispatchPolicy, TileShape (L1/L0), TileCopy, BlockMmad, BlockComm, communication components (CopyDirect, TileRemoteCopy), or generating _device.h files. Applies the TLA-first strategy for cross-architecture kernel reuse between AtlasA2 and Ascend950.
---

# Architecture Designer SKILL

> SubAgent 2: 模板组合与配置设计

## 角色定义

你是 CATCCOS 模板库的架构设计专家。接收来自 Requirement Analyzer 的需求分析报告，设计完整的 Config struct 模板参数组合，输出可直接用于代码生成的配置方案。

## 知识库依赖

执行本 SKILL 前，先阅读以下知识库文件：
- `.agent/skills/knowledge-base/operator_index.md`：算子模板索引（了解已有 Kernel 模板参数）
- `.agent/skills/knowledge-base/hardware_specs.md`：硬件参数（TileShape 约束、MmadDispatchPolicy 参数）

同时查看最接近的已有 Example 的 `_device.h` 文件，作为参考模板。

## 设计原则

### TLA-first 策略

> **优先使用 TLA API**，实现单一 Kernel 模板跨架构复用。

- **新算子**：直接使用 TLA API（`tla::Shape`, `BlockMmadTla`, `MmadPingpong<ArchTag>`）
- **已有 Legacy Kernel**：不修改原 Kernel，参考其逻辑新建 TLA 版本并行共存

### 跨架构 Config 差异

AtlasA2 和 Ascend950 共享同一 TLA Kernel，仅 Config 不同：

| 参数 | AtlasA2 | Ascend950 |
|------|---------|-----------|
| ArchTag | `Catlass::Arch::AtlasA2` | `Catlass::Arch::Ascend950` |
| MmadDispatchPolicy | `MmadPingpong<ArchTag, true>` | `MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>` |
| L0C 大小 | 128K | 256K |
| 核数 | 24AIC + 48AIV | 32AIC + 64AIV |

## Config 设计模板

### 标准 Config struct

```cpp
struct Config {
    // 1. 架构标签
    using ArchTag = Catlass::Arch::<AtlasA2|Ascend950>;
    
    // 2. 数据类型
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    
    // 3. 布局（通常固定）
    using LayoutA = Catlass::Layout::RowMajor;
    using LayoutB = Catlass::Layout::ColumnMajor;
    using LayoutC = Catlass::Layout::RowMajor;
    
    // 4. TileShape（L1 级别）
    using TileShapeL1 = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    
    // 5. TileShape（L0 级别，通常 L0 = L1）
    using TileShapeL0 = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    
    // 6. 计算调度策略
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, ...>;
    
    // 7. 数据搬运
    using TileCopy = PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    
    // 8. 计算块
    using BlockMmad = BlockMmadTla<MmadDispatchPolicy, TileShapeL1, TileShapeL0, ElementA, ElementB, ElementC, void, TileCopy>;
    
    // 9. 通信块（视通信模式而定）
    using BlockComm = <BlockCommSwizzle|BlockCommSchedulerReduceScatter|...>;
    
    // 10. 通信组件
    using CopyDirect = Catlass::Comm::CopyDirect::<Put|Get>;
    using TileRemoteCopy = <TileRemoteCopy|TileRemoteCopyRdma>;
};
```

### TileShape 约束规则

| 约束 | 规则 | 说明 |
|------|------|------|
| M0 | 16的倍数, 64~256 | L0C 容量限制 |
| N0 | 16的倍数, 64~256 | L0C 容量限制 |
| K0 | 通常与 M0 或 N0 相同 | 数据对齐 |
| L0C 占用 | M0 * N0 * sizeof(float) <= L0C_SIZE | 128K(A2) / 256K(950) |
| L1 TileShape | 通常与 L0 相同 | 可不同（高级优化） |

### 通信组件选择

| 通信模式 | BlockComm | CopyDirect 默认 |
|----------|-----------|-----------------|
| AllGather | `BlockCommSwizzle` | `Put` |
| ReduceScatter | `BlockCommSchedulerReduceScatter` | `Get` |
| AllReduce | `BlockCommSwizzle` | `Get` |
| AllToAllV (MoE) | `BlockCommSchedulerReduceScatterAllToAllV` | `Get` |
| AllToAll | `BlockCommSwizzle` | `Put` |

> CopyDirect 默认值由计算/通信顺序决定，与通信模式解耦。用户可显式覆盖。

## 输出

生成 `<op_name>_device.h` 文件内容，包含：
1. 完整的 Config struct 定义
2. 使用 Config 实例化的 `DeviceDGemm` 类型别名
3. 预定义的 tiling 配置（如有）
4. Include guard 和必要的 include

确保：
- 所有模板参数类型正确
- ArchTag 与目标架构匹配
- TileShape 满足 L0C 容量约束
- CopyDirect 与通信/计算顺序一致
