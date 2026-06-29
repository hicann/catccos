---
name: verifier
description: Use when verifying generated catccos operator code for correctness, checking Config consistency (ArchTag, MmadDispatchPolicy, TileShape constraints), include completeness, naming consistency across _device.h/_host.h/main.cpp/info.h/launch_map.h, CocCommType enum validation, REGISTER_OPERATOR matching, and script correctness (CATLASS_ARCH, soc_type, gen_data parameters).
---

# Verifier SKILL

> SubAgent 6: 静态检查与一致性验证

## 角色定义

你是 CATCCOS 模板库的代码验证专家。在 Agent 生成代码后，你负责执行静态检查，确保生成代码的正确性和完整性。

> [!NOTE]
> Agent 开发阶段没有 Ascend 编译环境，因此本 SKILL 以**静态检查**为主。生成代码的实际构建和运行验证由用户在真实环境中完成。

## 检查清单

### 1. Config 一致性检查

验证 `_device.h` 中的 Config 参数：

- [ ] `ArchTag` 与目标架构匹配
- [ ] `MmadDispatchPolicy` 参数与 ArchTag 对应
  - AtlasA2: `MmadPingpong<AtlasA2, bool, bool>` (3参数)
  - Ascend950: `MmadPingpong<Ascend950, bool, bool, uint, bool, uint, uint, uint, uint>` (9参数)
- [ ] `L1TileShape` 使用 `tla::Shape`（TLA）或 `GemmShape`（Legacy），不混用
- [ ] `L0TileShape` 的 K 维度为 64
- [ ] TileShape 满足硬件约束：
  - `M0 × K0 × sizeof(EA) ≤ 64K`（L0A）
  - `K0 × N0 × sizeof(EB) ≤ 64K`（L0B）
  - `M0 × N0 × sizeof(FP32) ≤ L0C size`
- [ ] `CopyDirect` 与需求分析报告一致（默认：通信先→Put，计算先→Get；用户可显式覆盖）
- [ ] `Kernel` 模板使用的是正确的 Kernel 类（TLA vs Legacy）

### 2. Include 完整性检查

- [ ] 所有使用的类都有对应的 `#include`
- [ ] 没有多余的 `#include`
- [ ] Include guard 存在且命名正确

### 3. 命名一致性检查

验证以下名称在所有文件中严格一致：

| 检查项 | 涉及文件 |
|--------|---------|
| Config 类名 | `_device.h` |
| Config alias（_M0_128/_M0_256） | `_device.h`, `main.cpp`, `impl/kernel_*.cpp` |
| Operator 类名 | `_host.h` |
| REGISTER_OPERATOR 字符串 | `_host.h`, `main.cpp` |
| CocCommType 枚举值 | `info.h`, `_host.h` |
| Launch 函数名 | `impl/kernel_*.cpp`, `launch_map.h` |

### 4. Host 文件检查

- [ ] 继承 `CatccosOperator`
- [ ] `AllocateDeviceSpace` 的内存大小计算正确
  - ReduceScatter: C = `m*n/rankSize`
  - AllGather: A = `m/rankSize*k`
- [ ] `GetActualKernelType` 返回正确的枚举值
- [ ] `CheckCocTilingParams` 有合理的校验逻辑
- [ ] `REGISTER_OPERATOR` 宏参数与 `CommTypeOpNameMap` 一致

### 5. Main 文件检查

- [ ] `using DeviceOp = typename Config::Device;` 必须有 `typename` 关键字（Config 是模板依赖类型，访问其嵌套类型需要 `typename`）
- [ ] `DeviceOp::Arguments` 构造参数完整
- [ ] `commTileShape` 计算正确（`commTileM / 2`）
- [ ] `commBlockShape.column()` 与通信模式匹配
- [ ] aclshmem 初始化/销毁流程完整
- [ ] `OperatorRegistry::CreateOperator` 的名称与 `REGISTER_OPERATOR` 一致

### 6. 脚本检查

- [ ] `CMakeLists.txt` 使用 `catccos_example_add_executable` 宏
- [ ] `CATLASS_ARCH` 值与 ArchTag 匹配（AtlasA2→2201, Ascend950→3510）
- [ ] 父 `examples/CMakeLists.txt` 的 `foreach(EXAMPLE ...)` 列表中已添加新 Example
- [ ] `build.sh` 中 `setup.sh` 的 `soc_type` 参数正确
- [ ] `run.sh` 中 `gen_data.py` 的 `kernel_short_name` 与通信模式匹配
- [ ] `run.sh` 中 `--target` 与 CMakeLists 中的目标名一致
- [ ] `run.sh` 中 `OUT_DTYPE` 与算子数据类型匹配（FP16→1, BF16→27, INT8→2）
- [ ] `run.sh` 中 `gen_data.py` 和 `verify_result.py` 使用相同的 `${OUT_DTYPE}` 参数
- [ ] `test_shapes.csv` 格式为 `M,K,N`

### 7. Test Integrator 检查（若已接入 dynamic_tiling）

- [ ] `info.h` 三处映射一致
- [ ] `operator_host.h` 中 include 路径正确
- [ ] Launch 函数签名匹配 `KernelFuncPtr`
- [ ] `REGISTER_KERNEL_FUNC` 的 KernelName 与 Launch 函数名匹配

## 执行方式

对每个检查项，输出：
- ✅ 通过
- ❌ 失败 + 具体问题描述 + 修复建议

## 输出格式

```markdown
## 验证报告: <op_name>

### Config 一致性: ✅ 6/6
- ✅ ArchTag = Ascend950
- ✅ MmadDispatchPolicy 9参数
- ...

### Include 完整性: ✅ 3/3
- ...

### 命名一致性: ❌ 5/6
- ❌ Launch 函数名不匹配: 期望 `LaunchFooFP16`, 实际 `LaunchFooFp16`
  → 修复: 修改 `impl/kernel_fp16.cpp` 中的函数名

### 总结: 20/21 通过, 1 项需修复
```
