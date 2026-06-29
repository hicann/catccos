---
name: test-integrator
description: Use when integrating a catccos operator into the tests/dynamic_tiling batch testing framework, modifying info.h (CocCommType enum, CommTypeMap, CommTypeOpNameMap), operator_host.h includes, implementing Launch functions in kernel_fp16.cpp/kernel_bf16.cpp, registering via REGISTER_KERNEL_FUNC in launch_map.h, or adding LUT tiling data.
---

# Test Integrator SKILL

> SubAgent 7: 批量测试框架接入

## 角色定义

你是 CATCCOS 的测试框架集成专家。当一个新算子的 Example 已生成后，你负责将其接入 `tests/dynamic_tiling` 批量测试框架。

## 知识库依赖

执行本 SKILL 前，先阅读：
- `tests/dynamic_tiling/README.md`：测试框架整体说明
- `tests/dynamic_tiling/include/launch_map.h`：注册宏机制
- `tests/dynamic_tiling/include/operator_host.h`：已注册的算子列表
- `utils/info.h`：CocCommType 枚举和映射

## 接入步骤（7步）

### Step 1: 修改 `utils/info.h` — 注册算子枚举与映射

需要修改**3处**：

**1.1 在 `CocCommType` 枚举中新增**（放在 `TYPE_NUM` 之前）：
```cpp
enum CocCommType {
    // ... 已有类型 ...
    <NEW_OPERATOR>,    // <-- 新增
    TYPE_NUM,
    UNKNOWN
};
```

**1.2 在 `CommTypeMap` 中新增缩写映射**：
```cpp
const std::map<std::string, CocCommType> CommTypeMap = {
    // ... 已有映射 ...
    {"<shortname>", CocCommType::<NEW_OPERATOR>},
};
```

**1.3 在 `CommTypeOpNameMap` 中新增名称映射**：
```cpp
const std::map<CocCommType, std::string> CommTypeOpNameMap = {
    // ... 已有映射 ...
    {<NEW_OPERATOR>, "<OpClassName>"},  // 必须与 host.h 中 REGISTER_OPERATOR 名一致
};
```

> [!WARNING]
> 三处映射的名称必须严格一致：枚举值、CommTypeMap 值、CommTypeOpNameMap 值、REGISTER_OPERATOR 宏参数。

### Step 2: 修改 `tests/dynamic_tiling/include/operator_host.h` — 引入 Host 头文件

```cpp
#include "<new_op>/<new_op>_host.h"
```

此 include 触发 `REGISTER_OPERATOR` 宏的静态注册。

### Step 3: 在 `tests/dynamic_tiling/impl/kernel_fp16.cpp` 中实现 FP16 Launch 函数

生成两个函数：

**3.1 模板 Launch 函数**（按 Config 模板参数化）：
```cpp
#include "<new_op>/<new_op>_device.h"

template <template <class, class, class, class, class, class> class ConfigAlias>
static void Launch<OpName>WithConfig(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    auto launch = [&](auto &&deviceOp) {
        using DeviceOp = std::decay_t<decltype(deviceOp)>;
        typename DeviceOp::Kernel::Arguments args{
            // 构造 Arguments（按通信模式差异化）
        };
        DeviceOp op;
        auto params = DeviceOp::Kernel::ToUnderlyingArguments(args);
        op.Initialize(params);
        op.Run((aclrtStream)stream, blockNum, fftsAddr);
    };
    
    if (!transA && !transB) {
        launch(typename ConfigAlias<half, LayoutA0, half, LayoutB0, half, LayoutC>::Device{});
    } else if (!transA && transB) {
        launch(typename ConfigAlias<half, LayoutA0, half, LayoutB1, half, LayoutC>::Device{});
    }
    // ... 其他 transA/transB 组合
}
```

**3.2 公开 Dispatch 函数**（按 m0 选择 Config 实例）：
```cpp
void Launch<OpName>FP16(
    void *stream, uint32_t blockNum, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    if (cocTiling.m0 == 128) {
        Launch<OpName>WithConfig<<OpName>Config_M0_128>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    } else {
        Launch<OpName>WithConfig<<OpName>Config_M0_256>(
            stream, blockNum, fftsAddr, kernelParams, symmetricPtr, cocTiling, transA, transB);
    }
}
```

### Step 4: 在 `tests/dynamic_tiling/impl/kernel_bf16.cpp` 中实现 BF16 Launch 函数

结构与 FP16 相同，将 `half` 替换为 `bfloat16_t`，函数名后缀改为 `BF16`。

> [!TIP]
> 如果是 Ascend950 专用算子，Launch 函数放在 `kernel_fp16_ascend950.cpp` / `kernel_bf16_ascend950.cpp` 中，包裹在 `#ifdef CATCCOS_ENABLE_A5_BUILD` 条件编译块内。

### Step 5: 修改 `tests/dynamic_tiling/include/launch_map.h` — 注册到 KernelDispatcher

```cpp
REGISTER_KERNEL_FUNC(<OpName>, <NEW_OPERATOR>, FP16);
REGISTER_KERNEL_FUNC(<OpName>, <NEW_OPERATOR>, BF16);
```

> [!IMPORTANT]
> `REGISTER_KERNEL_FUNC(KernelName, CommType, DataType)` 宏展开后：
> 1. 声明函数 `void Launch{KernelName}{DataType}(...)`
> 2. 创建静态注册对象
>
> 因此 Launch 函数命名**必须**严格匹配：`Launch` + KernelName + DataType

### Step 6:（可选）添加 LUT 性能查表数据

在 `tests/dynamic_tiling/tiling/<op>_tiling.cpp` 中添加经过搜索优化的 tiling 参数。

### Step 7:（可选）更新 CMakeLists.txt

如果新增了 `.cpp` 文件（如 Ascend950 专用 Launch 文件），需要更新 `impl/CMakeLists.txt`。

## 函数签名约束

所有 Launch 函数必须匹配 `KernelFuncPtr` 签名：

```cpp
using KernelFuncPtr = void (*)(void *, uint32_t, uint64_t,
    KernelParams &, uint8_t *, uint8_t *, CocTilingParams &,
    uint32_t, uint32_t);
// 参数：stream, blockNum, fftsAddr, kernelParams, workSpace, symmetricPtr, cocTiling, transA, transB
```

## Arguments 构造差异速查

| 通信模式 | commBlockShape.column() | 额外参数 |
|----------|------------------------|----------|
| MatmulAllReduce | `cocTiling.n0` | 无 |
| AllGatherMatmul | `UINT_MAX / 2` | workSpace |
| MatmulReduceScatter | `cocTiling.n0` | 无 |
| GroupedMatmul | `cocTiling.n0` | epSize, expertNum |

## 验证清单

完成接入后，检查：
- [ ] `info.h` 三处映射名称一致
- [ ] `operator_host.h` 中 include 路径正确
- [ ] Launch 函数命名匹配 `Launch{KernelName}{DataType}` 格式
- [ ] `REGISTER_KERNEL_FUNC` 的 KernelName 与 Launch 函数名一致
- [ ] Launch 函数签名匹配 `KernelFuncPtr` 类型
