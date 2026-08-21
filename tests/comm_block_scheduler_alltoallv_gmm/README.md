# BlockCommSchedulerAllToAllVGmm UT

This test guards the communication-core remapping used by the Ascend950
Dispatch-FFN-Combine pipeline. It runs a single small kernel and verifies:

- `SetCommCore` updates `coreIdx` and `coreNum`;
- scheduler context is reset before the remapped core starts;
- `prevSum` is rebuilt from `tokenPerExpert`;
- communication-loop counts and expert offsets are correct for uneven core
  distribution;
- inactive cores (`coreIdx >= EP`) do not read token metadata.

Run on an Ascend950 device with:

```bash
bash tests/comm_block_scheduler_alltoallv_gmm/scripts/run.sh [device_id]
```

For example:

```bash
bash tests/comm_block_scheduler_alltoallv_gmm/scripts/run.sh 6
```
