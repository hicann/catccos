---
name: torch-binding
description: Use when integrating catccos operators into PyTorch through TORCH_LIBRARY, adding wrapper functions, kernel shared libraries, Meta kernels, torch.ops.catccos runtime bindings, and Python verification scripts.
---

# catccos PyTorch TORCH_LIBRARY 接入

这个 skill 用于把 catccos 算子接入为 PyTorch 自定义算子。当前默认路线是 `TORCH_LIBRARY`，Python 调用面是 `torch.ops.catccos.*`。

不要再使用旧 custom class 路线：不要新增 `torch::jit::CustomClassHolder`、`REGISTER_CATCCOS_OPS_CLASS`、`torch_register.h` 或 `torch.classes.CatccosOps.*`。

## 核心结构

新增或改造一个算子时，通常涉及这些位置：

```text
catccos/
├── examples/
│   ├── <op>/
│   │   ├── <op>_wrapper.cpp
│   │   ├── <op>_device.h
│   │   ├── CMakeLists.txt
│   │   └── scripts/
│   │       ├── build_python.sh
│   │       ├── run_python.sh
│   │       ├── <op>.py
│   │       └── test_shapes.csv
│   ├── torch_binding/
│   │   ├── include/
│   │   │   └── catccos_torch_kernel.h
│   │   ├── src/
│   │   │   ├── torch_bindings.cpp
│   │   │   └── torch_bindings_meta.cpp
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
└── include/
```

AGMM 的现有实现是可参考样例：

- kernel wrapper: `examples/allgather_matmul/allgather_matmul_wrapper.cpp`
- wrapper 声明: `examples/torch_binding/include/catccos_torch_kernel.h`
- runtime 注册: `examples/torch_binding/src/torch_bindings.cpp`
- Meta 注册: `examples/torch_binding/src/torch_bindings_meta.cpp`
- Python 入口: `examples/allgather_matmul/scripts/allgather_matmul.py`

## 接入步骤

1. 在算子目录实现 wrapper
   - 在 `examples/<op>/<op>_wrapper.cpp` 中实现 `CatccosKernel::catccos_<op>_wrapper(...)`。
   - wrapper 负责设置 `CocTilingParams`、构造 device op arguments、选择 kernel config、申请必要 workspace，并调用 device op。
   - wrapper 和 native example 的 tiling 参数必须逐项对齐。不要从其他算子复制 `commBlockShape`、`commTileShape` 等参数后直接假设通用。

2. 在统一头文件声明 wrapper
   - 在 `examples/torch_binding/include/catccos_torch_kernel.h` 中添加 wrapper 函数声明。
   - 声明、定义、`torch_bindings.cpp` 调用点的签名必须一致。
   - 常见参数包括 `uint32_t blockDim`、`aclrtStream stream`、`uint64_t fftsAddr`、输入/输出 device pointer、symmetric workspace pointer、shape 和 rank 信息。

3. 编译 kernel wrapper so
   - 在 `examples/<op>/CMakeLists.txt` 中使用：

```cmake
if(CATCCOS_TORCH_EXTENSION)
catccos_example_add_library(
    <op>
    SOURCES <op>_wrapper.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=2201
)
endif()
```

   - 在 `examples/CMakeLists.txt` 的 `CATCCOS_TORCH_SUPPORTED` 中加入 `<op>`。
   - `catccos_example_add_library(<op>)` 会生成 `build/lib/lib<op>_kernel.so`，`catccos_torch` 会链接这些 kernel so。

4. 添加 runtime PyTorch op
   - 在 `examples/torch_binding/src/torch_bindings.cpp` 中添加 C++ runtime 函数，例如：

```cpp
namespace catccos {

at::Tensor <op>(const at::Tensor& a, const at::Tensor& b, int64_t rank_size)
{
    c10_npu::OptionalNPUGuard guard(a.device());
    // 校验 dtype/device/shape/init state
    // 创建输出 tensor
    // 获取 aclrtStream、FFTS 地址、symmetric workspace

    at_npu::native::OpCommand cmd;
    cmd.Name("catccos_<op>");
    cmd.Input(a_contig);
    cmd.Input(b_contig);
    cmd.Output(out);
    cmd.SetCustomHandler([...]() -> int {
        CatccosKernel::catccos_<op>_wrapper(...);
        return 0;
    });
    cmd.Run();
    return out;
}

}  // namespace catccos
```

   - lifecycle 通过 `torch.ops.catccos.init(...)` 和 `torch.ops.catccos.finalize()` 管理。
   - 不要把 symmetric workspace 绑到 custom class 构造/析构；`TORCH_LIBRARY` 路线没有 custom class 实例生命周期。

5. 注册 schema 和 backend impl
   - 在 `TORCH_LIBRARY(catccos, m)` 中定义 schema 并绑定实现：

```cpp
TORCH_LIBRARY(catccos, m) {
    m.def("init(int rank_id, int rank_size, int local_mem_size, str ip_port) -> int");
    m.def("<op>(Tensor a, Tensor b, int rank_size) -> Tensor");
    m.def("finalize() -> int");

    m.impl("init", &catccos::init);
    m.impl("<op>", torch::kPrivateUse1, &catccos::<op>);
    m.impl("finalize", &catccos::finalize);
}
```

   - 多个算子共享同一个 `catccos` namespace 时，不要重复定义已有 schema；按当前文件组织合并到同一个注册块。

6. 添加 Meta kernel
   - 在 `examples/torch_binding/src/torch_bindings_meta.cpp` 中为算子添加 Meta 实现。
   - 对需要入图或符号 shape 的算子，优先使用 `sym_sizes()` 和 `at::empty_symint(...)`：

```cpp
at::Tensor <op>_meta(const at::Tensor& a, const at::Tensor& b, int64_t rank_size)
{
    auto a_sizes = a.sym_sizes();
    auto b_sizes = b.sym_sizes();
    std::vector<c10::SymInt> out_shape = {
        a_sizes[0] * c10::SymInt(rank_size),
        b_sizes[1],
    };
    return at::empty_symint(out_shape, a.options());
}

TORCH_LIBRARY_IMPL(catccos, Meta, m) {
    m.impl("<op>", &catccos::meta::<op>_meta);
}
```

7. 更新 Python 验证入口
   - 正式脚本放在 `examples/<op>/scripts/` 下。
   - Python 中加载 `libcatccos_torch.so` 后，只通过 `torch.ops.catccos.*` 调用：

```python
status = torch.ops.catccos.init(rank_id, rank_size, local_mem_size, ip_port)
out = torch.ops.catccos.<op>(tensor_a_npu, tensor_b_npu, rank_size)
finalize_status = torch.ops.catccos.finalize()
```

   - 输出 tensor 推荐由 runtime op 返回，不再要求 Python 预先构造输出 tensor 传给 custom class。

8. 更新 build/run 脚本
   - `build_python.sh` 应配置 `-DCATCCOS_TORCH_EXTENSION=ON`，并构建 `<op>_kernel_build catccos_torch`。
   - `run_python.sh` 复用对应算子的 CSV、数据生成和 `verify_result.py`。
   - 临时验证脚本只放在算子目录 `scripts/` 下，正式提交前删除；正式保留 `<op>.py`、`run_python.sh`、`build_python.sh`、`test_shapes.csv`。

## 数据生成和验证

先查看算子原有 `scripts/run.sh`，确认它调用哪个 `examples/utils` 数据生成脚本。不要凭名字猜。

常见形式：

```bash
OUT_DTYPE=<OUT_DTYPE>  # 1=FP16, 27=BF16, 2=INT8
python3 ${UTILS_PATH}/gen_data.py "agmm" ${OUT_DTYPE} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}
python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin ${OUT_DTYPE} $((M * RANK_SIZE)) ${N} ${K}
```

> `gen_data.py` 和 `verify_result.py` 的 `out_dtype` 参数必须一致。BF16 算子使用 `27`，FP16 使用 `1`。

如果算子已有正式 Python 脚本，优先复用，只把 lifecycle 和核心 op 调用切到 `torch.ops.catccos.*`。

## 检查清单

- wrapper 声明、定义、调用三处签名一致。
- `catccos_torch` 能链接对应 `lib<op>_kernel.so`。
- `TORCH_LIBRARY(catccos, m)` schema 与 Python 调用一致。
- backend impl 注册到 `torch::kPrivateUse1`。
- Meta impl 注册到 `TORCH_LIBRARY_IMPL(catccos, Meta, m)`。
- Meta 输出 shape 保留符号维度，使用 `sym_sizes()` / `empty_symint(...)`。
- runtime 使用 `at_npu::native::OpCommand` + `SetCustomHandler` 调 wrapper。
- Python 验证入口不再出现旧 custom class 调用。
- 正式脚本统一放在 `examples/<op>/scripts/` 下。

## 禁用旧路线

下面这些只属于旧 custom class 接入方式，新增或清理算子时不要再引入：

- `torch::jit::CustomClassHolder`
- `REGISTER_CATCCOS_OPS_CLASS`
- `torch_register.h`
- `torch.classes.CatccosOps.*`
- 通过 custom class 构造函数申请 symmetric workspace、析构函数释放 workspace 的生命周期设计
