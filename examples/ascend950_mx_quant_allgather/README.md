# MxQuant + AllGather 融合算子示例

将 BF16 输入进行 MX 动态量化，并通过 IPC AllGather 将量化后的数据和 MxScale 广播到所有 rank。

## 支持的量化格式

| 格式 | ElementOutput 类型 | PACK_RATIO | target_emax | 编译宏 |
|------|-------------------|------------|-------------|--------|
| MXFP8 E4M3 | `float8_e4m3_t` | 1 | 8 | `MX_QUANT_DTYPE=0` |
| MXFP8 E5M2 | `float8_e5m2_t` | 1 | 15 | `MX_QUANT_DTYPE=1` |
| MXFP4 E2M1 | `float4_e2m1x2_t` | 2 | 2 | `MX_QUANT_DTYPE=2` |
| MXFP4 E1M2 | `float4_e1m2x2_t` | 2 | 0 | `MX_QUANT_DTYPE=3` |

## 构建

构建全部格式：

```bash
cmake --build build
```

构建指定格式：

```bash
cmake --build build --target ascend950_mx_quant_allgather              # fp8_e4m3 (默认)
cmake --build build --target ascend950_mx_quant_allgather_e5m2         # fp8_e5m2
cmake --build build --target ascend950_mx_quant_allgather_fp4_e2m1     # fp4_e2m1
cmake --build build --target ascend950_mx_quant_allgather_fp4_e1m2     # fp4_e1m2
```

构建产物位于 `build/bin/` 目录。

## 测试

### 运行脚本

```bash
# 用法: bash scripts/run.sh <device_id_list> [dtype]

# 测试 fp8_e4m3 (默认)
bash scripts/run.sh 0,1

# 测试 fp8_e5m2
bash scripts/run.sh 0,1 fp8_e5m2

# 测试 fp4_e2m1
bash scripts/run.sh 0,1 fp4_e2m1

# 测试 fp4_e1m2
bash scripts/run.sh 0,1 fp4_e1m2
```

`run.sh` 会根据 dtype 自动选择对应的二进制文件。

### 手动运行

```bash
# 1. 生成 golden 数据
python3 examples/utils/gen_mx_quant_allgather_data.py \
    --rankSize 2 -m 512 -n 512 -k 1 \
    --block_size 32 --dtype fp8_e4m3 --data_dir ./out

# 2. 多进程启动 (以 2 卡为例)
EXEC_BIN=build/bin/ascend950_mx_quant_allgather
IPPORT="tcp://127.0.0.1:8789"
${EXEC_BIN} 2 0 ${IPPORT} 512 512 1 ./out 0,1 &
${EXEC_BIN} 2 1 ${IPPORT} 512 512 1 ./out 0,1 &
wait

# 3. 验证结果
python3 -c "
import numpy as np
a = np.fromfile('./out/rank_0_output.bin', dtype=np.uint8)
b = np.fromfile('./out/rank_0_golden.bin', dtype=np.uint8)
diff = np.where(a != b)[0]
print(f'quant: {\"PASS\" if len(diff)==0 else \"FAIL\"} ({len(a)-len(diff)}/{len(a)} matched)')
"
```

### 测试用例配置

测试形状定义在 `scripts/test_shapes.csv`，格式为：

```
M,K,N
512,1,512
1024,1,1024
```

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  Kernel: MxQuantAllGather                                │
│                                                          │
│  Subcore 1 (Quant)          Subcore 0 (Comm)             │
│  ┌──────────────────┐       ┌──────────────────────┐     │
│  │ CommBlockMxQuant  │──────▶│ BlockAllGather       │     │
│  │ bf16 → fp8/fp4   │ IPC   │ (quant data copy)    │     │
│  │ + MxScale (E8M0) │ sync  │                      │     │
│  │                  │──────▶│ BlockScaleGather     │     │
│  │                  │       │ (scale data copy)    │     │
│  └──────────────────┘       └──────────────────────┘     │
│                                                          │
│  同步: aclshmem_fence() + aclshmemx_signal_op()         │
│  反压: aclshmem_signal_wait_until()                      │
└──────────────────────────────────────────────────────────┘
```
