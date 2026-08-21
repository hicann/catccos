# Ascend950 AllToAll MatMul Split-K URMA

本示例是 MXFP8 split-K AllToAll MatMul 的默认实现。目录名不体现 `B-trans`，因为 B 转置是当前
split-K URMA 系列的共同输入布局，不是区分实现变体的关键。

## 数据切分

```text
每个 rank 持有 A[M, localK]
AllToAll 后，每个 rank 得到 A[M / rankSize, localK * rankSize]
每个 rank 持有逻辑 B[localK * rankSize, N]
输出 C[M / rankSize, N]
```

B 在 GM 中按转置形态存储为 `[N, localK * rankSize]`，kernel 侧通过 column-major 布局按逻辑
`[localK * rankSize, N]` 读取。A scale 和 B scale 使用 MX packed scale，K 方向每 32 个元素共享一个
scale。

## 实现特点

这是精度优先的默认路径。AIC 先计算本 rank 对应的 local contribution，并将结果保留在 L0C；随后等待
AIV 通过 URMA 写入 remote rank 的 A/scaleA 数据，再继续在 L0C 上累加 remote contribution，最后统一写出
C。

相比 `ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic`，该实现不通过 FP16 atomic add 将 remote partial
累加到 GM 上，因此数值行为更接近完整 matmul，适合需要 strict FLOAT16 精度标准的验证或业务场景。

symmetric memory 中 remote 数据布局如下，保证每个源 rank 的通信块连续：

```text
A:      [stage][srcRank][commRows][alignedLocalK]
scaleA: [stage][srcRank][commRows][alignedLocalScaleK]
```

host 会根据每个 rank 的 M 方向 tile 数动态选择 `commInterval` 和 1 到 7 之间的预编译 swizzle offset。

## Golden 与精度

`scripts/run.sh` 使用 `scripts/gen_data.py` 生成随机 MXFP8 输入，并通过
`torch_npu.npu_quant_matmul` 生成每个 rank 的 `golden_*.bin`。`scripts/gen_data.py` 也会委托到同一实现。

默认校验复用 `examples/utils/verify_result.py`，与其他 ascend950 MXFP8 算子一致。输出按
`[CHUNK_M, N] = [M / rankSize, N]` 逐元素比对，累加 K 取 `FULL_K = localK * rankSize` 作为
`compute_times` 参与相对误差阈值 `rtol` 的计算：

```text
output = output.reshape(-1, N)
rtol = get_rtol(float16, compute_times=FULL_K)   # 2**-8 (< 2048) / 2**-7 (>= 2048)
通过条件: precision_percent == 100 (全部元素 |actual - golden| <= rtol * max(1, |golden|))
```


## 构建与运行

```bash
cd examples/ascend950_mxfp8_alltoall_matmul_split_k_urma
bash scripts/build.sh
bash scripts/run.sh 0,1,2,3
```

也可以指定其他卡，例如：

```bash
bash scripts/run.sh 12,13,14,15
```

测试 shape 配置在 `scripts/test_shapes.csv` 中，字段为 `M,localK,N`。`M` 必须能被 `rankSize` 整除，
`localK` 必须能被 512 整除。
