# matmul_dequant_reduce_scatter_write

This example runs an int8 matmul with per-token/per-channel dequantization, optional bias on rank 0, and reduce-scatter writeback across ranks. The final output is FP16 and each rank writes its slice into `output.bin`.

## Build

From this example directory:

```bash
bash scripts/build.sh
```

The script configures the project and builds the `matmul_dequant_reduce_scatter_write` target into `build/bin/`.

## Run and verify

Pass a comma-separated device list to the run script:

```bash
bash scripts/run.sh 0,1
```

The script reads shapes from `scripts/test_shapes.csv`, generates input and golden data under `output/`, launches one process per device, and verifies `output.bin` against `golden.bin`.

For ACLNN and FP32 double-golden verification:

```bash
bash scripts/run_double_golden_verify.sh 0,1
```

## Manual launch

```bash
build/bin/matmul_dequant_reduce_scatter_write \
  <rank_size> <rank_id> <ip_port> <m> <n> <k> <data_path> [device_id_list]
```

Example:

```bash
build/bin/matmul_dequant_reduce_scatter_write 2 0 tcp://127.0.0.1:8734 512 1024 1024 output 0,1
```

Repeat the command for every `rank_id` in the job, using the same `rank_size`, `ip_port`, shape, data path, and device list.

`device_id_list` defaults to `0..rank_size-1` when omitted. `rank_id` selects the device id from this list.

## Data files

`data_path` should contain:

- `x1_gm_rank<rank>.bin`: rank-local int8 left matrix, shape `m x k`.
- `x2_gm_rank<rank>.bin`: rank-local int8 right matrix, shape `k x n`.
- `scale_x1_gm.bin`: FP32 per-token scale, shape `m`.
- `scale_x2_gm.bin`: FP32 per-channel scale, shape `n`.
- `bias_gm.bin`: int32 bias, shape `n`; consumed by rank 0.

After the kernel finishes, each rank writes its FP16 result slice into `output.bin` at rank-specific offsets.
