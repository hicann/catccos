---
name: example-scaffolder
description: Use when generating a complete catccos operator example directory including _device.h, _host.h (CatccosOperator, REGISTER_OPERATOR, AllocateDeviceSpace, GetActualKernelType), main.cpp (aclshmem init, DeviceOp Arguments, CocTilingParams), CMakeLists.txt (catccos_example_add_executable), build.sh, run.sh, test_shapes.csv, gen_data.py, and README.md.
---

# Example Scaffolder SKILL

> SubAgent 4: Device/Host/Main 脚手架生成

## 角色定义

你是 CATCCOS 模板库的样例脚手架生成器。接收 Architecture Designer 输出的 Config 设计，生成完整的 Example 目录结构。

## 知识库依赖

执行本 SKILL 前，查看最接近的已有 Example 的完整文件作为参考模板：
- `<closest_example>/<op>_device.h`（由 Architecture Designer 已生成）
- `<closest_example>/<op>_host.h`
- `<closest_example>/main.cpp`
- `<closest_example>/scripts/`

## 生成文件清单

```
examples/<new_op>/
├── CMakeLists.txt                # 本 SKILL 生成
├── README.md                     # 本 SKILL 生成
├── <new_op>_device.h             # 来自 Architecture Designer
├── <new_op>_host.h               # 本 SKILL 生成
├── main.cpp                      # 本 SKILL 生成
└── scripts/
    ├── build.sh                  # 本 SKILL 生成
    ├── run.sh                    # 本 SKILL 生成
    ├── test_shapes.csv           # 本 SKILL 生成
    └── gen_data.py               # 本 SKILL 生成（仅全新算子需要）
```

## 生成规则

### 1. `<op>_host.h`

继承 `CatccosOperator` 基类，需要实现：

```cpp
class <OpName>Operator : public CatccosOperator {
public:
    // 必须实现的虚函数
    void AllocateDeviceSpace(KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override;
    void WriteResultFile(const KernelParams &params, const CocTilingParams &cocTiling,
        uint32_t rankId, std::string dataFile) override;
    size_t GetWorkspaceSize(const CocTilingParams &cocTiling) override;
    
    // 必须实现：返回算子的 CocCommType 枚举值
    CocCommType GetActualKernelType(const CocTilingParams &cocTiling) override {
        return CocCommType::<ENUM_NAME>;
    }
    
    // 必须实现：校验 tiling 参数合法性
    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override {
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;
        if (product % rankSize != 0) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("<OpClassName>", <OpName>Operator);
```

**AllocateDeviceSpace 规则**：
- 根据通信模式计算各 tensor 的形状和大小：
  - ReduceScatter：A = `m*k`, B = `k*n`, C = `m*n/rankSize`
  - AllGather：A = `m/rankSize*k`, B = `k*n`, C = `m*n`
  - AllToAllV：额外分配 `localTokensPerExpert` 和 `globalTokensPerLocalExpert`
- 支持 `dataFile` 条件分支：有数据文件时从文件读取，否则用默认值填充
- 调用 `params.SetKernelParams(aDevice, bDevice, cDevice)` 注册指针

**WriteResultFile 规则**：
- 从 device 内存拷贝结果到 host
- 使用 `WriteFile(path, host_ptr, size, offset)` 写入二进制文件

### 2. `main.cpp`

标准入口结构（含 aclshmem 初始化流程）：

```cpp
#include "<op>_device.h"
#include "<op>_host.h"

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using ElementA = half;
using ElementB = half;
using ElementD = half;

using Config = <OpName>Config_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>;
using DeviceOp = typename Config::Device;

int main(int argc, char **argv) {
    // 1. 解析命令行参数
    Options options;
    options.Parse(argc, argv);
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    int32_t deviceId = options.deviceIdList[rankId];

    // 2. 构造 CocTilingParams
    CocTilingParams cocTiling;
    cocTiling.m = options.m;
    cocTiling.n = options.n;
    cocTiling.k = options.k;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 20;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    // 3. 初始化设备和通信
    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, options.ipPort.c_str(), &attributes, &default_flag_uid);
    aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // 4. 创建算子实例和分配内存
    auto op = OperatorRegistry::Instance().CreateOperator("<OpClassName>");
    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void *symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    // 5. 构造 Arguments
    Catlass::GemmCoord problemShape{options.m, options.n, options.k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};  // 注意: /2

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        kernelParams.ptrA, kernelParams.ptrB, kernelParams.ptrC, symmetricPtr,
        commCoreSplit, commBlockShape, commTileShape
    };

    // 6. Initialize → Run
    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    uint64_t fftsAddr = shmemx_get_ffts_config();
    deviceOp.Run(stream, blockNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 7. 写结果和清理
    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    shmem_free(symmPtr);
    FreeDeviceSpace(kernelParams);
    shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
```

**commTileShape 计算规则**：
- `commTileShape = {commTileM / 2, n0}`
- 其中 `commTileM / 2` 是因为通信 tile 的行方向需要减半以匹配流水线 stage

**Arguments 构造差异**（按通信模式）：

| 通信模式 | commBlockShape.column() | 额外参数 |
|----------|------------------------|----------|
| ReduceScatter | `cocTiling.n0` | 无 |
| AllGather | `UINT_MAX / 2` | workspace 指针 |
| AllReduce | `cocTiling.n0` | 无 |
| AllToAllV | `cocTiling.n0` | epSize, expertNum |

### 3. `CMakeLists.txt`

使用仓库标准宏 `catccos_example_add_executable`：

```cmake
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ... (版权声明)

catccos_example_add_executable(
    <op_name>
    SOURCES main.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=<ARCH_CODE>
)
```

**ARCH_CODE 映射**：

| ArchTag | CATLASS_ARCH | build.sh soc_type |
|---------|-------------|-------------------|
| AtlasA2 | `2201` | 不指定（默认） |
| Ascend950 | `3510` | `-soc_type Ascend950` |

### 3.1 注册到父 `examples/CMakeLists.txt`

> [!IMPORTANT]
> 仅创建子目录的 `CMakeLists.txt` 不够，还需要在 `examples/CMakeLists.txt` 的 `foreach(EXAMPLE ...)` 列表中添加新 Example 名称，否则不会被编译。

在 `examples/CMakeLists.txt` 中找到 `foreach(EXAMPLE` 块，在列表末尾（`)`之前）添加新 Example：

```cmake
foreach(EXAMPLE
    matmul_allreduce
    allgather_matmul
    ...                          # 已有 Examples
    <new_op_name>                 # ← 在此添加新 Example
)
    add_subdirectory(${EXAMPLE})
endforeach()
```

**插入位置规则**：
- 按通信模式族分组，在同族的最后一个 Example 之后添加
- 例如新建 `ascend950_xxx_reduce_scatter`，则添加在 `ascend950_matmul_reduce_scatter` 之后

**特殊情况**：
- 如果 Example 需要 RDMA，不放入 `foreach` 中，而是放在末尾的 `if(RDMA_TRANSPORT)` 块中

### 4. `scripts/build.sh`

```bash
#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ... (版权声明)
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

# AtlasA2: 不传 -soc_type
# Ascend950: source ... -soc_type Ascend950
source $PROJECT_ROOT/examples/utils/setup.sh <SOC_TYPE_FLAG> || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

SOURCE_DIR=$PROJECT_ROOT
BUILD_DIR=$PROJECT_ROOT/build
mkdir -p $BUILD_DIR
cmake -B $BUILD_DIR -S $SOURCE_DIR <CMAKE_ARCH_FLAG>
cmake --build $BUILD_DIR --target <op_name> -j
```

**架构差异**：
- AtlasA2：`<SOC_TYPE_FLAG>` 为空，`<CMAKE_ARCH_FLAG>` 为空
- Ascend950：`<SOC_TYPE_FLAG>` 为 `-soc_type Ascend950`，`<CMAKE_ARCH_FLAG>` 为 `-DCATLASS_BISHENG_ARCH=a5`

### 5. `scripts/run.sh`

```bash
#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ... (版权声明)
#

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

source $PROJECT_ROOT/examples/utils/setup.sh <SOC_TYPE_FLAG> || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/<op_name>/
DATA_DIR=`realpath ./output`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/<op_name>
OUT_DTYPE=<OUT_DTYPE>  # 1=FP16, 27=BF16, 2=INT8

tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf ./output/*.bin
    python3 <GEN_DATA_SCRIPT> "<kernel_short_name>" ${OUT_DTYPE} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:8788"

    # Start Process
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    # Wait until all process exit
    wait

    # Verify output
    python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin ${OUT_DTYPE} ${M} ${N} ${K}
done

cd ${CURRENT_DIR}
```

**gen_data 脚本选择**：

| 情况 | `<GEN_DATA_SCRIPT>` | `<kernel_short_name>` |
|------|---------------------|----------------------|
| 标准 GEMM（可复用 utils/gen_data.py） | `${UTILS_PATH}/gen_data.py` | 见下表 |
| 全新算子（需自定义 golden 逻辑） | `${SCRIPT_DIR}/gen_data.py` | 自定义 |

**kernel_short_name 映射**（用于 `utils/gen_data.py`）：

| 通信模式 | kernel_short_name | golden 计算逻辑 |
|----------|-------------------|----------------|
| AllGather + Matmul | `"agmm"` | `torch.cat(C_list, dim=0)` |
| Matmul + ReduceScatter | `"mmrs"` | `sum(C_list)` |
| Matmul + AllReduce | `"mmar"` | `sum(C_list)` |
| AllGather + Matmul + GatherResult | `"agmmwg"` | `torch.cat(C_list)` + `gather_a.bin` |
| 自定义 | 自定义 | 需新建 gen_data.py |

### 6. `scripts/gen_data.py`（仅全新算子需要）

当算子的 golden 计算逻辑无法被 `utils/gen_data.py` 覆盖时（如 MoE AllToAllV、MX 量化等），需要在 `scripts/` 下新建 `gen_data.py`：

```python
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ... (版权声明)
#
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from utils import DataType, tensor_to_file

def gen_random_data(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)

def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('data_dir', type=str)
    # 按需添加算子特有参数（如 ep_size, expert_num 等）
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # === 生成输入数据 ===
    for i in range(args.rank_size):
        a = gen_random_data([args.m, args.k], dtype=torch.float16)
        b = gen_random_data([args.k, args.n], dtype=torch.float16)
        tensor_to_file(a, os.path.join(args.data_dir, f"rank_{i}_a.bin"))
        tensor_to_file(b, os.path.join(args.data_dir, f"rank_{i}_b.bin"))

    # === 计算 golden 结果 ===
    # TODO: 根据算子语义实现 golden 计算
    # 示例（ReduceScatter）：
    #   golden = sum(matmul(A_i, B_i) for i in range(rank_size))
    golden = torch.zeros([args.m, args.n], dtype=torch.float32)
    for i in range(args.rank_size):
        a = gen_random_data([args.m, args.k], dtype=torch.float16)
        b = gen_random_data([args.k, args.n], dtype=torch.float16)
        golden += torch.matmul(a.float(), b.float())
    tensor_to_file(golden, os.path.join(args.data_dir, "golden.bin"))

if __name__ == '__main__':
    gen_golden_data()
```

> [!IMPORTANT]
> 判断是否需要自定义 gen_data.py 的规则：
> - `utils/gen_data.py` 支持的模式：标准 GEMM 的 AllGather/ReduceScatter/AllReduce
> - 需要自定义的场景：MoE GroupedGEMM（需要 expert 分组）、MX 量化（需要 Scale 生成）、非 GEMM 算子

### 7. `scripts/test_shapes.csv`

```csv
M,K,N
1792,768,1280
5416,6144,1408
```

> [!NOTE]
> CSV 格式为 `M,K,N`（注意顺序是 M,K,N 不是 M,N,K），与 `gen_data.py` 的参数顺序一致。

### 8. `README.md`

```markdown
### 使用方式

1. **编译项目**  
   ```bash
   cd examples/<op_name>
   bash scripts/build.sh
   ```

2. **运行示例程序**  
   ```bash
   cd examples/<op_name>
   bash scripts/run.sh [device_list]
   ```

   - **参数说明**：
     - `device_list`：NPU设备编号列表，逗号分隔
     - 示例：`bash scripts/run.sh 0,1`

   - **配置计算规模**：
     修改 `scripts/test_shapes.csv` 设置测试用例维度。
```

---

## BF16 数据处理注意事项

> [!WARNING]
> BF16（bfloat16）在数据生成、文件 IO 和验证精度方面都需要特殊处理，与 FP16 不同。

### 1. 文件 IO：BF16 不能直接用 numpy 读写

BF16 在 numpy 中没有原生类型支持，必须通过 `uint16` 中转：

```python
# 写入（已封装在 utils.tensor_to_file）
if tensor.dtype == torch.bfloat16:
    tensor.view(torch.uint16).numpy().tofile(file_name)  # 不能直接 .numpy()

# 读取（已封装在 utils.tensor_from_file）
if dtype == torch.bfloat16:
    return torch.from_numpy(np.fromfile(file_name, dtype=np.float16)).view(torch.bfloat16)
```

**规则**：生成数据和读取结果时，**必须使用** `utils.py` 中的 `tensor_to_file` / `tensor_from_file`，不要手写 `.numpy().tofile()` 或 `np.fromfile()`。

### 2. Golden 计算：必须上转 FP32

BF16 精度只有 8 位尾数（vs FP16 的 10 位），golden 计算必须在 FP32 下进行：

```python
# ✅ 正确：先生成 BF16 数据，golden 在 FP32 下计算
a = torch.randn([M, K], dtype=torch.bfloat16)
b = torch.randn([K, N], dtype=torch.bfloat16)
golden = torch.matmul(a.to(torch.float32), b.to(torch.float32))  # FP32 计算
```

这与 FP16 处理方式相同（`utils/gen_data.py` 已实现：`l0c_dtype = torch.float32`），但 BF16 精度更低，验证阈值必须调整。

### 3. 验证精度阈值

`utils/verify_result.py` 中 BF16 使用不同于 FP16 的精度标准：

| 参数 | FP16 | BF16 | FP32 |
|------|------|------|------|
| `precision_threshold`（COMPUTE_FLOAT） | `rtol` | `rtol` | `rtol` |
| `eb_threshold`（COMPUTE_FLOAT） | `2^(-10)` | `2^(-7)` | `2^(-14)` |
| `rtol`（K<2048） | `2^(-8)` | `2^(-7)` | `2^(-11)` |
| `rtol`（K≥2048） | `2^(-7)` | `2^(-6)` | `2^(-10)` |

`verify_result.py` 根据 `out_dtype` 参数选择对应的精度阈值和数据读取方式。

### 4. run.sh 中的 OUT_DTYPE 变量

`gen_data.py` 和 `verify_result.py` 都需要正确的 `out_dtype` 参数。在 run.sh 模板中通过 `OUT_DTYPE` 变量统一控制：

```bash
OUT_DTYPE=<OUT_DTYPE>  # 1=FP16, 27=BF16, 2=INT8

# gen_data 和 verify_result 都使用 OUT_DTYPE
python3 ${GEN_DATA_SCRIPT} "<kernel_short_name>" ${OUT_DTYPE} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}
python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin ${OUT_DTYPE} ${M} ${N} ${K}
```

| 数据类型 | `OUT_DTYPE` 值 | DataType 枚举 |
|----------|---------------|---------------|
| FP16 | `1` | `DataType.FLOAT16 = 1` |
| BF16 | `27` | `DataType.BF16 = 27` |
| INT8 | `2` | `DataType.INT8 = 2` |

### 5. 自定义 gen_data.py 的 BF16 转换工具

当需要对 BF16 进行位级精确转换（如 FP8/MX 量化场景），使用 `utils.py` 中的工具函数：

```python
from utils import fp32_to_bf16_bits, bf16_bits_to_fp32

# FP32 → BF16 位表示（uint16），带 round-to-nearest-even
bf16_bits = fp32_to_bf16_bits(fp32_array)  # np.float32 → np.uint16
bf16_bits.tofile("input.bin")

# BF16 位表示 → FP32（用于 golden 计算）
fp32_array = bf16_bits_to_fp32(bf16_bits)  # np.uint16 → np.float32
```

---

## 输出

生成以上全部文件内容。确保：
- Include guard 命名正确（大写 + 下划线）
- 版权声明存在于所有文件
- REGISTER_OPERATOR 第一个参数与 `info.h` 中 `CommTypeOpNameMap` 一致
- `GetActualKernelType()` 返回正确的 `CocCommType` 枚举值
- `CheckCocTilingParams()` 实现 `blockNum * commInterval % rankSize == 0` 校验
- `CMakeLists.txt` 使用 `catccos_example_add_executable` 宏和正确的 `CATLASS_ARCH`
- **父 `examples/CMakeLists.txt`** 的 `foreach(EXAMPLE ...)` 列表中已添加新 Example 名称
- `build.sh` 包含 `setup.sh` source 和正确的 soc_type
- `run.sh` 包含完整的 gen_data → 多rank启动 → verify 流程
- 全新算子需附带自定义 `scripts/gen_data.py`
- **BF16 算子**：`gen_data` 调用使用 `27` 或 `bf16` 作为 dtype 参数，数据 IO 使用 `tensor_to_file`/`tensor_from_file`

> [!WARNING]
> **不要自动更新 `operator_index.md` 知识库**。代码生成完成后，必须询问用户是否更新知识库索引。
> 用户通常希望在代码运行验证通过后再将新算子添加到 `operator_index.md` 中。

