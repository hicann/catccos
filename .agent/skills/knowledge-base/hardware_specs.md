# CATCCOS 硬件参数知识库

> 供 Architecture Designer、Kernel Generator、Requirement Analyzer 使用
> 详细参数参考：[芯片参数.md](../../芯片参数.md)

## 1. ArchTag 映射

| ArchTag | 模板库支持 |
|---------|-----------|
| `Catlass::Arch::AtlasA2` | Legacy + TLA |
| `Catlass::Arch::Ascend950` | TLA only |

## 2. 存储层次容量（影响 TileShape 设计）

| 存储层级 | AtlasA2 | Ascend950 | 设计影响 |
|----------|---------|-----------|----------|
| L1 | 512K | 512K | L1TileShape 的 M×K + K×N 不超出 |
| L0A | 64K | 64K | L0TileShape 的 M×K 受限 |
| L0B | 64K | 64K | L0TileShape 的 K×N 受限 |
| L0C | **128K** | **256K** | 决定基本块大小 M0×N0 |
| UB | 192K | 256K | UB_STAGES × 通信块大小受限 |
| L2 | 192M | 128M | 多核缓存策略 |

### TileShape 约束公式

```
L0C ≥ M0 × N0 × sizeof(FP32)

AtlasA2:  128K / 4 = 32768  → M0×N0 ≤ 32768  → 128×256 或 256×128
Ascend950: 256K / 4 = 65536  → M0×N0 ≤ 65536  → 可支持更大基本块
```

## 3. 核心参数对比

| 参数 | AtlasA2 | Ascend950 |
|------|---------|-----------|
| 核数 | 24AIC + 48AIV | 32AIC + 64AIV |
| 主频 | 1.8GHz | 1.65GHz |
| CUBE FP16算力 | 353.89 TOPS | 432.54 TOPS |
| CUBE FP8算力 | N/A | 865.08 TOPS |
| DDR带宽 | 1.8TB/s | 1.6TB/s |
| L2带宽 | 7.86TB/s | 5.2TB/s |

## 4. 核内数据通路带宽（影响流水线策略）

| 通路 | 流水线 | Ascend950 (B/c) | AtlasA2 (B/c) | 备注 |
|------|--------|-----------------|---------------|------|
| L1→L0A | MTE1 | 256 | 256 | 相同 |
| L1→L0B | MTE1 | **256** | 128 | Ascend950加倍 |
| L1→UB | MTE1跨Core | 128 | N/A | Ascend950新增 |
| OUT→L1 | MTE2 | 256 | 256 | 相同 |
| OUT→UB | MTE2 | 128 | 128 | 相同 |
| UB→OUT | MTE3 | 128 | 128 | 相同 |
| L0C→OUT | Fixpipe | 128 | 128 | 相同 |
| L0C→UB | Fixpipe跨Core | 128 | N/A | Ascend950新增 |

## 5. CUBE 数据类型支持

| 数据类型 | AtlasA2 | Ascend950 | 备注 |
|----------|---------|-----------|------|
| FP16/FP16/FP32 | ✅ | ✅ | 基础类型 |
| BF16/BF16/FP32 | ✅ | ✅ | 基础类型 |
| S8/S8/S32 | ✅ | ✅ | INT8量化 |
| HF32/HF32/FP32 | ✅ | ✅ | TF32 |
| FP32/FP32/FP32 | ✅ | ✅(1/8) | Ascend950性能降 |
| E4M3/E4M3/FP32 | ❌ | ✅ | FP8 |
| E5M2/E4M3/FP32 | ❌ | ✅ | FP8训练 |
| MXFP8 | ❌ | ✅ | Scale=E8M0,blk=32 |
| MXFP4 | ❌ | ✅ | Scale=E8M0,blk=32 |

## 6. MmadDispatchPolicy 配置参考

```cpp
// AtlasA2 (简洁)
using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
    Catlass::Arch::AtlasA2,
    true  // enableUnitFlag
>;

// Ascend950 (需要显式buffer stage参数)
using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
    Catlass::Arch::Ascend950,
    true,   // enableUnitFlag
    false,  // useHF32
    1,      // l0CStages
    false,  // enableL1Resident
    2,      // l1AStages
    2,      // l1BStages
    2,      // l0AStages
    2       // l0BStages
>;
```
