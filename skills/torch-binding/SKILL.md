---
name: torch-binding
description: Use when integrating catccos operators into PyTorch, implementing wrapper functions, custom classes, and binding Python to C++ kernels.
---

# 操作步骤

以 allgather_matmul 的 pytorch 算子接入为例子。整体的实现步骤如下：

1. 使用 `Config::Device` + `DeviceOp::Run()` 模式为算子实现 wrapper 函数。
2. 编译算子和 wrapper 文件为动态链接库 .so。
3. 在 `torch_bindings.cpp` 中创建算子类，继承 `torch::jit::CustomClassHolder`。
4. 把算子注册成 torch 算子，编译 `torch_binding` 为 .so，并链接之前的算子 .so。
5. Python 中加载动态链接库，调用并运行算子。
6. 测试数据生成和验证的脚本编写和调用。

以下是涉及的需要修改的关键文件的目录结构。

```
└── catccos/
    ├── examples/
    │   ├── <算子名>/
    │   │   ├── scripts/
    │   │   │   ├──  build_python.sh
    │   │   │   ├──  run_python.sh
    │   │   │   └──  <算子名>.py
    │   │   ├── <算子名>_wrapper.cpp      # 新增，wrapper 函数
    │   │   ├── <算子名>_device.h         # 包含 Config 结构体
    │   │   ├── main.cpp
    │   │   └── CMakeLists.txt
    │   ├── torch_binding/
    │   │   ├── include/
    │   │   │   ├── catccos_torch_kernel.h   # 新增 wrapper 函数声明
    │   │   │   └── torch_register.h
    │   │   ├── src/
    │   │   │   └── torch_bindings.cpp       # 算子类 + torch 注册
    │   │   └── CMakeLists.txt
    │   ├── utils/
    │   │   └── shmem_init.h                 # 包含 set_attr 函数
    │   └── CMakeLists.txt                   # catccos_example_add_library 函数
    └── CMakeLists.txt
```

---

## 前置检查：确认 main.cpp 使用新 API

在写 wrapper 之前，务必先检查该算子的 `main.cpp` 是否已经升级：

**旧 API（不应再使用）**：
```cpp
AllGatherMatmul<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
    <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, gmSymmetric, cocTiling);
```

**新 API（以此为模板写 wrapper）**：
```cpp
using Config = AllGatherMatmulConfig_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
using DeviceOp = Config::Device;

DeviceOp::Arguments args{ problemShape, rankId, rankSize, commInterval,
    aPtr, bPtr, cPtr, gmSymmetric,
    commCoreSplit, commBlockShape, commTileShape };

DeviceOp deviceOp;
deviceOp.Initialize(args);
deviceOp.Run(stream, BLOCK_NUM, fftsAddr);
```

**如果 main.cpp 还用的是旧 API，说明 kernel 头文件还没升级，wrapper 也先不要动。**

---

# 算子接入需要修改的文件清单

| 序号 | 文件路径 | 修改内容 |
|------|----------|----------|
| 1 | `examples/<算子名>/<算子名>_wrapper.cpp` | **新建**，Config::Device 模式的 wrapper 实现 |
| 2 | `examples/torch_binding/include/catccos_torch_kernel.h` | 添加 wrapper 函数声明（含 rankId，uint8_t*） |
| 3 | `examples/torch_binding/src/torch_bindings.cpp` | 添加算子类实现 + 注册 |
| 4 | `examples/CMakeLists.txt` | 在 `CATCCOS_TORCH_SUPPORTED` 中添加算子名 |
| 5 | `examples/<算子名>/CMakeLists.txt` | 添加 `catccos_example_add_library` + `COMPILE_DEFINITIONS` |
| 6 | `examples/<算子名>/scripts/<算子名>.py` | **新建**，Python 调用脚本 |
| 7 | `examples/<算子名>/scripts/build_python.sh` | **新建**，编译脚本 |
| 8 | `examples/<算子名>/scripts/run_python.sh` | **新建**，运行脚本 |

---

## 各文件修改详情

### 1. <算子名>_wrapper.cpp — 新建 wrapper 实现

使用 `Config::Device` + `DeviceOp::Run()` 模式，参考 main.cpp 中对应该算子的调用方式。

**所有指针参数类型必须为 `uint8_t*`**，和 main.cpp 保持一致。

**必须新增 `rankId` 参数**，传递给 `DeviceOp::Arguments`。

```cpp
#include "<算子名>_device.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

namespace CatccosKernel {

void catccos_<算子名>_wrapper(
    uint32_t blockDim,
    aclrtStream stream,
    uint64_t fftsAddr,
    uint8_t* aPtr,
    uint8_t* bPtr,
    uint8_t* cPtr,
    uint8_t* gmSymmetric,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    int rankId,
    int rankSize)
{
    using Config = <算子名>Config_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using DeviceOp = Config::Device;

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 20;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        aPtr, bPtr, cPtr, gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    deviceOp.Run(stream, blockDim, fftsAddr);
}

} // namespace CatccosKernel
```

**Config 类型名规则**：查看对应算子的 `<算子名>_device.h`，命名规则为 `<算子名>Config_M0_<tile大小>`。例如：
- `AllGatherMatmulConfig_M0_128`
- `MatmulReduceScatterConfig_M0_128`
- `MatmulAllReduceConfig_M0_128`

### 2. catccos_torch_kernel.h — 添加 wrapper 声明

在命名空间 `CatccosKernel` 中添加：

```cpp
/**
 * @brief <算子名> kernel wrapper
 * @param blockDim number of AICore blocks
 * @param stream ACL stream
 * @param fftsAddr FFTS configuration address
 * @param aPtr pointer to matrix A on device
 * @param bPtr pointer to matrix B on device
 * @param cPtr pointer to matrix C on device (output)
 * @param gmSymmetric pointer to symmetric memory
 * @param m rows of matrix A
 * @param n columns of matrix B
 * @param k columns of matrix A / rows of matrix B
 * @param rankId current rank index
 * @param rankSize number of ranks
 */
void catccos_<算子名>_wrapper(
    uint32_t blockDim,
    aclrtStream stream,
    uint64_t fftsAddr,
    uint8_t* aPtr,
    uint8_t* bPtr,
    uint8_t* cPtr,
    uint8_t* gmSymmetric,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    int rankId,
    int rankSize
);
```

**重要**：所有指针参数类型为 `uint8_t*`，不是 `void*`。

### 3. torch_bindings.cpp — 添加算子类

#### 3.1 在 namespace CatccosOps 中添加算子类

```cpp
class <算子类名> : public torch::jit::CustomClassHolder {
public:
    <算子类名>() : name_("<算子名>"), count_(0), fftsAddr_(shmemx_get_ffts_config()), symmPtr_(nullptr)
    {
        // 分配对称内存（shmem_malloc，用于跨 rank 通信数据交换）
        symmPtr_ = static_cast<uint8_t*>(shmem_malloc(SHMEM_BUFF_BYTES));
        aclrtMemset(symmPtr_, SHMEM_BUFF_BYTES, 0, SHMEM_BUFF_BYTES);
    }

    ~<算子类名>()
    {
        if (symmPtr_ != nullptr) {
            shmem_free(symmPtr_);
            symmPtr_ = nullptr;
        }
    }

    std::string get_name() const
    {
        return name_;
    }

    void compute(const at::Tensor& c_tensor,
                 const at::Tensor& a_tensor,
                 const at::Tensor& b_tensor)
    {
        // 1. 参数校验
        TORCH_CHECK(a_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ");
        TORCH_CHECK(b_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ");
        TORCH_CHECK(c_tensor.dtype() == at::kHalf,
                    "Compute Error: Only float16 (half) is supported! ");
        TORCH_CHECK(a_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! ");
        TORCH_CHECK(b_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! ");
        TORCH_CHECK(c_tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Compute Error: Only NPU device is supported! ");

        // 2. 形状校验
        TORCH_CHECK(a_tensor.dim() == 2, "A tensor must be 2D!");
        TORCH_CHECK(b_tensor.dim() == 2, "B tensor must be 2D!");
        TORCH_CHECK(c_tensor.dim() == 2, "C tensor must be 2D!");

        int64_t m = a_tensor.size(0);
        int64_t k = a_tensor.size(1);
        int64_t n = b_tensor.size(1);
        TORCH_CHECK(b_tensor.size(0) == k, "A/K mismatch!");

        int32_t n_pes = shmem_n_pes();

        // 根据算子类型校验输出形状：
        // - MatmulReduceScatter: c_shape = (m / rankSize, n)
        // - AllGatherMatmul:     c_shape = (m * rankSize, n)
        // - MatmulAllReduce:     c_shape = (m, n)

        // 3. Make tensors contiguous
        at::Tensor a_contig = a_tensor.contiguous();
        at::Tensor b_contig = b_tensor.contiguous();
        at::Tensor c_contig = c_tensor.contiguous();

        // 4. Get device pointers
        // 注意：必须使用 storage().data()，不能使用 data_ptr<T>()（对 NPU tensor 不兼容）
        uint8_t* aPtr = static_cast<uint8_t*>(const_cast<void*>(a_contig.storage().data()));
        uint8_t* bPtr = static_cast<uint8_t*>(const_cast<void*>(b_contig.storage().data()));
        uint8_t* cPtr = static_cast<uint8_t*>(const_cast<void*>(c_contig.storage().data()));

        // 5. Get NPU stream and rank info
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        int32_t my_pe = shmem_my_pe();   // 获取当前 rank id
        count_++;

        // 6. Call wrapper
        CatccosKernel::catccos_<算子名>_wrapper(
            BLOCK_NUM,
            stream,
            fftsAddr_,
            aPtr,
            bPtr,
            cPtr,
            symmPtr_,
            m,
            n,
            k,
            my_pe,      // rankId
            n_pes       // rankSize
        );
    }

private:
    std::string name_;
    int32_t count_;
    uint64_t fftsAddr_;
    uint8_t* symmPtr_;         
    uint32_t BLOCK_NUM = 20;
};
```

#### 3.2 注册算子

```cpp
REGISTER_CATCCOS_OPS_CLASS(<算子类名>, compute, get_name);
```

#### 3.3 其他注意事项

- **`set_attr` 函数**：不要在 torch_bindings.cpp 中内联定义。使用 `examples/utils/shmem_init.h` 中的版本（`#include "utils/shmem_init.h"`）。
- **Manager::attr_init**：必须添加错误检查（见下方 Code Review 清单）。
- **`symmPtr_` vs `workspace_ptr_`**：`symmPtr_` 用 `shmem_malloc` 分配，用于跨 rank 通信数据交换（所有通信算子都需要）；`workspace_ptr_` 用 `aclrtMalloc` 分配，用于算子内部中间计算缓冲（仅部分算子如 dequant、dispatch 需要）。两者独立，不可混淆。
- **`SHMEM_BUFF_BYTES`**：定义在 `utils/info.h` 中，建议用它替代硬编码的大小。

### 4. examples/CMakeLists.txt — 添加到白名单

在 `CATCCOS_TORCH_SUPPORTED` 中添加算子名：

```cmake
set(CATCCOS_TORCH_SUPPORTED
    allgather_matmul
    matmul_reduce_scatter    # 新增
)
```

### 5. <算子名>/CMakeLists.txt — 添加编译

```cmake
# Compile wrapper to shared library for PyTorch extension
if(CATCCOS_TORCH_EXTENSION)
catccos_example_add_library(
    <算子名>
    SOURCES <算子名>_wrapper.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=2201     # 必须加！
)
endif()
```

### 6. Python 脚本

根据算子类型确定输出规模：

| 算子类型 | 输出形状 | 输出文件写入 |
|----------|----------|-------------|
| matmul_reduce_scatter | (m/rankSize, n) | 所有 rank 按偏移写入 |
| allgather_matmul | (m*rankSize, n) | rank 0 写入 |
| matmul_allreduce | (m, n) | rank 0 写入 |

### 7. 编译脚本编写规范

编译脚本参考 `allgather_matmul` 编译脚本 `build_python.sh` 的实现。

### 8. 运行脚本编写规范

**测试参数读取**：必须从 CSV 读取测试参数，参考现有实现。

**测试数据生成**：生成测试数据时，应参考对应算子目录下的 `run.sh` 中调用的脚本。

**如果 utils 中没有对应的脚本**：
- 需要开发者手动实现测试数据生成脚本
- 参考其他 gen_*.py 脚本的格式
- 路径: `examples/utils/gen_<算子名>_data.py`

---

## Code Review 检查清单

接入新算子后，按以下清单进行检查：

### 1. wrapper 函数
- [ ] 使用 `Config::Device` + `DeviceOp::Run()` 模式（不是旧 `<<<>>>` 直接调用）
- [ ] 包含 `rankId` 参数，传递给 `DeviceOp::Arguments`
- [ ] 所有指针参数类型为 `uint8_t*`（不是 `void*`）
- [ ] CocTilingParams 参数和 main.cpp 中一致
- [ ] 头文件声明与实现签名一致

### 2. 算子类（torch_bindings.cpp）
- [ ] 继承 `torch::jit::CustomClassHolder`
- [ ] `symmPtr_` 类型为 `uint8_t*`，`shmem_malloc` 后 `static_cast<uint8_t*>`
- [ ] 构造函数中用 `shmem_malloc(SHMEM_BUFF_BYTES)` 分配对称内存，析构函数中 `shmem_free`
- [ ] 指针提取使用 `static_cast<uint8_t*>(const_cast<void*>(tensor.storage().data()))`，**不使用 `data_ptr<T>()`**
- [ ] compute() 中调用 `shmem_my_pe()` 获取 rankId 并传给 wrapper
- [ ] 输出形状校验符合算子语义

### 3. torch_bindings.cpp 全局
- [ ] 使用 `#include "utils/shmem_init.h"` 中的 `set_attr`，不在文件中内联定义
- [ ] Manager::attr_init 有错误检查（`set_conf_store_tls` 和 `init_attr` 都检查返回值）

### 4. CMake 配置
- [ ] 算子名在 `CATCCOS_TORCH_SUPPORTED` 中
- [ ] `catccos_example_add_library` 调用了 `COMPILE_DEFINITIONS CATLASS_ARCH=2201`
- [ ] `examples/CMakeLists.txt` 的 `add_library` 函数支持 `COMPILE_DEFINITIONS` 参数
- [ ] `add_library` 函数有 `ARCH` 检测逻辑（a5 → dav-c310, 其他 → dav-c220）
- [ ] `add_library` 链接了 `-lm`

### 5. 指针类型一致性
- [ ] `catccos_torch_kernel.h` 声明中指针为 `uint8_t*`
- [ ] `wrapper.cpp` 实现中指针为 `uint8_t*`
- [ ] `torch_bindings.cpp` 中成员变量 `symmPtr_`、`workspace_ptr_` 为 `uint8_t*`

### 6. Python 脚本
- [ ] 输出规模与算子语义匹配
- [ ] 数据文件路径正确

### 7. 运行脚本
- [ ] 使用正确的 gen_data 脚本
- [ ] 从 CSV 读取 MNK，不能硬编码
