# allgather_matmul_remote_read

该样例用于验证 `allgather_matmul` 的远端读模式。

当前样例支持两个 kernel 变体：

```text
ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT=1
    调用 allgather_matmul_with_remote_read_local_mm_opt.hpp。
    本 rank 的 A * B 直接从 gmA 计算，其他 rank 的 A 通过 remote get 进入 gmSymmetric 后计算。

ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT=0
    默认值。
    调用 allgather_matmul_with_remote_read.hpp。
    所有 rank 的 A 都通过 remote get 进入 gmSymmetric 后计算。
```

## 编译算子样例

进入样例目录：

```bash
cd examples/allgather_matmul_remote_read
```

默认编译基础 `remote-read` 变体：

```bash
bash scripts/build.sh
```

等价于：

```bash
bash scripts/build.sh -DALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT=0
```

如果要编译 `remote-read + local_mm_opt` 变体：

```bash
bash scripts/build.sh -DALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT=1
```

构建时宏会通过如下链路传入 C++：

```text
命令行 -DALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT
    -> scripts/build.sh
    -> cmake configure
    -> CMakeLists.txt COMPILE_DEFINITIONS
    -> allgather_matmul_remote_read_device.h
```

## 执行算子样例

在样例目录下执行：

```bash
bash scripts/run.sh <device_list>
```

示例：

```bash
bash scripts/run.sh 6,7
```

运行时会打印当前宏值：

```text
[TEST] ALLGATHER_MATMUL_REMOTE_READ_LOCAL_MM_OPT: 0
```

其中：

```text
1 表示 remote-read + local_mm_opt
0 表示基础 remote-read
```

出现如下执行结果，说明算子运行成功，精度比较通过：

```bash
error num: 0
PASS
```

测试矩阵形状可以在 `scripts/test_shapes.csv` 中修改。
