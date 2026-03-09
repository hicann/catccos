### 项目介绍

#### 1. 功能说明
支持多卡场景下的精度测试和批量性能测试，批量测试所需的tiling参数构造接口在include文件夹下实现，examples内的算子在impl文件夹下添加对应Kernel层API调用函数后即可支持。

#### 2. 流程简介

![dynamic_tiling](../../docs/images/dynamic_tiling.png)

### 使用方式

#### 1. 编译项目

进入示例目录并执行执行编译脚本：

```bash
cd examples/dynamic_tiling
bash scripts/build.sh
```

#### 2. 运行 Dynamic-Tiling 示例程序

进入示例目录并执行运行脚本：

```bash
cd examples/dynamic_tiling
bash scripts/run.sh [kernel_name] [data_type] [test_start_line] [test_collect_rows] [device_list]
```

##### 参数说明

| 参数 | 说明 | 取值示例 |
|------|------|---------|
| `kernel_name` | 通信-计算融合算子缩写, 在examples/dynamic_tiling/include/launch_map.h内设定 | `mmar`: MATMUL_ALLREDUCE<br>`agmm`: ALLGATHER_MATMUL<br>`mmrs`: MATMUL_REDUCE_SCATTER |
| `data_type` | 数据类型 | `1`: FP16<br>`27`: BF16 |
| `test_start_line`（可选） | 测试起始行索引（对应`test_shapes.csv`中的行号，从0开始）<br>需与 `test_collect_rows` 一同指定，用于性能测试 | `0`, `10`, `...` |
| `test_collect_rows`（可选） | 每次采集性能数据的测试用例数量 | `5`, `10`, `...` |
| `device_list` | 指定运行的设备（NPU）编号列表，以逗号分隔 | `0,1`, `4,5,6,7` |

> 📌 **注意**：  
> - `rankSize`由`device_list`中设备数量自动确定
> - 精度测试默认按顺序执行test_shapes.csv中定义的所有shape
> - 性能测试需指定test_start_line和test_collect_rows参数：从第test_start_line个shape开始，每次采集test_collect_rows个测试用例，持续执行直至文件末尾

##### 示例

- **精度测试示例**：  
  使用 NPU 0 和 1，运行 **MatMul-AllReduce** 精度测试，数据类型为FP16，`rankSize = 2`：
  ```bash
  bash scripts/run.sh "mmar" 1 0,1
  ```

- **性能测试示例**：  
  使用 NPU 4、5、6、7，运行 **AllGather-MatMul** 性能测试，数据类型为 BF16，从 `test_shapes.csv` 第0行开始，每 10 个 shape 采集一次 `msprof` 性能数据，`rankSize = 4`：
  ```bash
  bash scripts/run.sh "agmm" 27 0 10 4,5,6,7
  ```

#### 3. 配置计算规模

矩阵计算参数（包括 `M`, `K`, `N`, `Transpose A`, `Transpose B`）在配置文件中定义：

```
scripts/test_shapes.csv
```

请根据测试需求修改该文件，添加或调整测试用例的输入维度和属性。

---

✅ **提示**：  
- 确保设备编号正确且可用。  
- 建议在性能测试前清理无关进程，以保证数据准确性。  
- 性能数据默认输出至 `output/` 目录。

### 新增算子指导

#### 1.算子缩写与算子enum对照表
| 算子缩写 | 算子类型 |
|--------|------|
| mmar | MATMUL_ALLREDUCE |
| agmm | ALLGATHER_MATMUL |
| mmrs | MATMUL_REDUCE_SCATTER |
| agmmwg | ALLGATHER_MATMUL_WITH_GATHER_RESULT |
| gmmata | GROUPED_MATMUL_ALLTOALLV |
| atagmm | ALLTOALLV_GROUPED_MATMUL |
| agmmrdma | ALLGATHER_MATMUL_RDMA |
| atavgmmv2 | ALLTOALLV_GMM_V2 |

#### 2.相关目录结构说明
```plaintext
catccos/
├── include/
│   └── catccos/
│       ├── comm/
│       │   ├── block/  # block层数据搬运策略文件
│       │   ├── tile/  # tile层数据搬运模式设置
│       │   └── comm_dispatch_policy.hpp  # block层数据搬运策略文件相关数据结构
│       ├── detail/
│       │   └── remote_copy_type.hpp  # 数据搬运模式相关数据结构
│       ├── dgemm/
│       │   ├── block/  # block层矩阵乘swizzle文件
│       │   └── kernel/  # 算子kernel文件(新增算子在此处添加)
│       ├── layout/
│       ├── catccos.hpp
│       ├── dist_coord.hpp
│       └── symm_coord.hpp
└──  examples/
    ├── allgather_matmul/
    │   ├── scripts/  # 算子编译运行脚本
    │   ├── allgather_matmul_device.h  # 算子kernel调用入口(新增算子需添加)
    │   ├── allgather_matmul_host.h  # 算子执行输入输出内容生成(新增算子需添加)
    │   └── main.cpp # 算子执行用例
    ├── dynamic_tiling/
    │   ├── impl/  # 算子device模块调用入口(新增算子需添加)
    │   ├── include/  # tiling参数相关文件
    │   ├── scripts/  # 算子编译运行脚本
    │   ├── tiling/  # 决策树生成tiling参数相关文件
    │   ├── utils/  # 性能测试结果分析文件
    │   └── main.cpp  # 算子执行用例
    └── utils/  # 算子输入数据生成及结果精度校验文件
```

#### 3.新增算子流程(以matmul_allreduce为例)

- 添加算子kernel层实现<br>
1、在include/catccos/dgemm/kernel/路径下添加matmul_allreduce.hpp文件，实现通算融合算子Kernel层接口MatmulAllReduce。<br>

- 新增examples样例<br>
1、在examples目录下新建matmul_allreduce文件夹。<br>
2、在examples的CMakeLists.txt里添加matmul_allreduce的EXAMPLES。<br>

- 添加算子Host组件<br>
1、在matmul_allreduce文件夹下新建matmul_allreduce_host.h文件。<br>
2、matmul_allreduce_host.h文件里是算子通用host侧接口实现，包括内存空间申请、结果写入等。<br>
3、将matmul_allreduce_host.h头文件添加到operator_host.h，供批量测试框架调用。<br>

- 添加算子device组件<br>
1、在matmul_allreduce文件夹下新建matmul_allreduce_device.h文件。<br>
2、matmul_allreduce_device.h文件里实现了算子kernel层接口的调用入口，目前是基于动态参数策略实现的。<br>
3、将matmul_allreduce_host.h头文件添加到kernel_bf16.cpp和kernel_fp16.cpp，供批量测试框架调用。<br>
4、在kernel_bf16.cpp和kernel_fp16.cpp实现算子kernel层接口的调用函数LaunchMatmulAllReduceBF16和LaunchMatmulAllReduceFP16。<br>
5、在launch_map.h补充批量测试框架算子执行注册接口REGISTER_KERNEL_FUNC。<br>

- 在info.h补充算子类型信息<br>
1、在enum CocCommType里新增MATMUL_ALLREDUCE，用于批量测试工程区分算子类型。<br>
2、在CommTypeMap里补充算子名称缩写和CocCommType的映射关系，用于批量测试时动态选择执行的算子类型。<br>
3、在CommTypeOpNameMap添加CocCommType和算子名称的映射关系，用于查询算子host组件注册的接口。<br>