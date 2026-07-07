#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Sequence


class CacheStatus(Enum):
    HIT = "hit"
    MISS = "miss"


class HardwareType(Enum):
    A2 = "a2"
    A3 = "a3"
    A5 = "a5"


class DataType(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass(frozen=True)
class MTECacheConfig:
    read_rtt_ns: float
    request_interval_ns: float


@dataclass(frozen=True)
class MTEConfig:
    read_otsd_bound: int = 64
    nd2nz_cmd_otsd: int = 2
    cache_miss: MTECacheConfig = MTECacheConfig(
        read_rtt_ns=235.0,
        request_interval_ns=4.5,
    )
    cache_hit: MTECacheConfig = MTECacheConfig(
        read_rtt_ns=110.0,
        request_interval_ns=1.9,
    )
    full_core_hit_efficiency: float = 0.8


class MTECostModel:
    def __init__(
        self,
        config: MTEConfig = MTEConfig(),
        full_core_num: int = 20,
    ) -> None:
        if full_core_num <= 0:
            raise ValueError("full_core_num must be positive")
        self.config = config
        self.full_core_num = full_core_num

    def _get_cache_config(self, status: CacheStatus) -> MTECacheConfig:
        return (
            self.config.cache_hit
            if status is CacheStatus.HIT
            else self.config.cache_miss
        )

    def nd2nz_continuous(
        self,
        core_num: int,
        instruction_num: int,
        n_value: int,
        d_value: int = 256,
        cache_status: CacheStatus = CacheStatus.MISS,
    ) -> float:
        """Estimate continuous ND2NZ time in microseconds."""
        if d_value != 256:
            raise ValueError("only d_value=256 is supported")
        if min(core_num, instruction_num, n_value) <= 0:
            raise ValueError("core_num, instruction_num and n_value must be positive")
        if (
            self.config.read_otsd_bound <= 0
            or self.config.nd2nz_cmd_otsd <= 0
        ):
            raise ValueError(
                "read_otsd_bound and nd2nz_cmd_otsd must be positive"
            )

        if n_value > self.config.read_otsd_bound:
            base_time = self.nd2nz_continuous(
                core_num=core_num,
                instruction_num=instruction_num,
                n_value=self.config.read_otsd_bound,
                d_value=d_value,
                cache_status=cache_status,
            )
            return round(base_time * n_value / self.config.read_otsd_bound, 3)

        cache = self._get_cache_config(cache_status)
        cmd_otsd = self.config.nd2nz_cmd_otsd
        cmd_rounds = 2

        if n_value <= 32:
            request_count = n_value * (cmd_otsd + 1)
            schedule_time = cache.request_interval_ns * request_count
            average_time_ns = (
                schedule_time + cache.read_rtt_ns * cmd_otsd
            ) / (cmd_otsd * cmd_rounds)
        else:
            request_count = n_value * cmd_otsd * cmd_rounds
            schedule_time = cache.request_interval_ns * request_count
            average_time_ns = (
                schedule_time + cache.read_rtt_ns
            ) / (cmd_otsd * cmd_rounds)

        time_us = average_time_ns * instruction_num / 1000.0
        if core_num == self.full_core_num and cache_status is CacheStatus.HIT:
            time_us /= self.config.full_core_hit_efficiency
        return time_us


@dataclass(frozen=True)
class HardwareConfig:
    element_size: int
    core_num: int = 20
    rank_size: int = 2
    write_rtt_ns: float = 123.0
    remote_read_schedule_ns: float = 20.0
    remote_read_rtt_ns: float = 500.0
    hccs_bandwidth: float = 19.5
    cube_flops_per_us: float = 13_920_336.0
    sync_time_us: float = 6.7
    launch_time_us: float = 6.47


@dataclass(frozen=True)
class HardwareProfile:
    hardware: HardwareConfig
    mte: MTEConfig


def _get_element_size(data_type: DataType) -> int:
    if data_type in (DataType.FP16, DataType.BF16):
        return 2
    if data_type is DataType.INT8:
        return 1
    raise ValueError(f"unsupported data type: {data_type}")


def get_a2_hardware_profile(
    data_type: DataType,
    rank_size: int,
) -> HardwareProfile:
    return HardwareProfile(
        hardware=HardwareConfig(
            element_size=_get_element_size(data_type),
            rank_size=rank_size,
        ),
        mte=MTEConfig(),
    )


def get_a3_hardware_profile(
    data_type: DataType,
    rank_size: int,
) -> HardwareProfile:
    del data_type, rank_size
    raise NotImplementedError("A3 cost-model parameters are not available")


def get_a5_hardware_profile(
    data_type: DataType,
    rank_size: int,
) -> HardwareProfile:
    del data_type, rank_size
    raise NotImplementedError("A5 cost-model parameters are not available")


def get_hardware_profile(
    hardware_type: HardwareType,
    data_type: DataType,
    rank_size: int,
) -> HardwareProfile:
    selectors = {
        HardwareType.A2: get_a2_hardware_profile,
        HardwareType.A3: get_a3_hardware_profile,
        HardwareType.A5: get_a5_hardware_profile,
    }
    return selectors[hardware_type](data_type, rank_size)


@dataclass(frozen=True)
class CostResult:
    m: int
    k: int
    n: int
    transpose_a: int
    transpose_b: int
    p: int
    t: int
    comm_tile_m: int
    m0: int
    aiv_core_num: int
    total_time: float
    bound_type: str
    aic_cube_time: float
    aic_nd2nz_hit_time: float
    aic_time: float
    aiv_time: float


@dataclass(frozen=True)
class AivWorkload:
    tile_num: float
    data_size_kb: float


def _validate_inputs(
    m: int,
    k: int,
    n: int,
    transpose_a: int,
    transpose_b: int,
    p_values: Sequence[int],
    m0: int,
    aiv_core_num: int,
    config: HardwareConfig,
) -> None:
    if min(m, k, n) <= 0:
        raise ValueError("M, K and N must be positive")
    if transpose_a not in (0, 1) or transpose_b not in (0, 1):
        raise ValueError("transpose_a and transpose_b must be 0 or 1")
    if not p_values or any(p <= 0 for p in p_values):
        raise ValueError("p_values must contain positive integers")
    if m0 not in (128, 256):
        raise ValueError("m0 must be 128 or 256")
    if transpose_a == 0 and m0 != 128:
        raise ValueError("m0 must be 128 when transpose_a is 0")
    if aiv_core_num not in (16, 20):
        raise ValueError("aiv_core_num must be 16 or 20")
    if config.core_num <= 0 or config.rank_size <= 0:
        raise ValueError("core_num and rank_size must be positive")


def _get_aiv_workload(
    tile_count: int,
    n: int,
    m0: int,
    n0: int,
    block_m: int,
    config: HardwareConfig,
) -> AivWorkload:
    tile_width = min(n, n0)
    element_count = tile_count * m0 * tile_width
    data_size_kb = element_count * config.element_size / 1024 / config.rank_size
    tile_num = (
        element_count
        * config.element_size
        / (n0 * block_m)
        / config.rank_size
    )
    return AivWorkload(tile_num=tile_num, data_size_kb=data_size_kb)


def _estimate_aiv_time(
    workload: AivWorkload,
    t: int,
    aiv_core_num: int,
    config: HardwareConfig,
) -> float:
    burstlen_otsd = t // 2
    write_time = 4.0 + 2.0 * burstlen_otsd + config.write_rtt_ns
    overlap_time = burstlen_otsd * config.remote_read_schedule_ns - write_time
    ping_pong_time = (
        config.remote_read_rtt_ns
        + 2.0 * burstlen_otsd * config.remote_read_schedule_ns
        + write_time
    )
    ping_pong_count = max(1, math.ceil(round(workload.tile_num / aiv_core_num, 5)))

    time_us = (
        25.0
        + ping_pong_time * ping_pong_count
        - overlap_time * (ping_pong_count - 1)
    ) / 1000.0

    bandwidth = workload.data_size_kb / time_us
    if bandwidth > config.hccs_bandwidth:
        time_us = workload.data_size_kb / config.hccs_bandwidth
    return time_us


def compute_cost_model(
    mte_model: MTECostModel,
    m: int,
    k: int,
    n: int,
    p_values: Sequence[int],
    *,
    transpose_a: int = 1,
    transpose_b: int = 1,
    m0: int = 128,
    aiv_core_num: int = 20,
    config: HardwareConfig,
) -> List[CostResult]:
    """Estimate reduce-scatter tiling costs, ordered by predicted time."""
    _validate_inputs(
        m,
        k,
        n,
        transpose_a,
        transpose_b,
        p_values,
        m0,
        aiv_core_num,
        config,
    )

    k0 = 256
    n0 = 256 if m0 == 128 else 128
    m_loops = math.ceil(m / m0)
    k_loops = math.ceil(k / k0)
    n_loops = math.ceil(n / n0)
    total_tiles = m_loops * n_loops
    p_max = math.ceil(total_tiles / config.core_num)

    left_hit_time = mte_model.nd2nz_continuous(
        core_num=config.core_num,
        instruction_num=k_loops,
        n_value=128,
        d_value=256,
        cache_status=CacheStatus.HIT,
    )
    right_hit_time = mte_model.nd2nz_continuous(
        core_num=config.core_num,
        instruction_num=k_loops,
        n_value=256,
        d_value=256,
        cache_status=CacheStatus.HIT,
    )
    nd2nz_hit_time_per_core = left_hit_time + right_hit_time
    cube_time_per_core = 2.0 * m0 * k * n0 / config.cube_flops_per_us

    results: List[CostResult] = []
    for p in p_values:
        if (config.core_num * p) % config.rank_size != 0:
            continue

        active_rounds = min(p, p_max)
        aic_cube_time = cube_time_per_core * active_rounds
        aic_nd2nz_hit_time = nd2nz_hit_time_per_core * active_rounds
        aic_time = max(aic_cube_time, aic_nd2nz_hit_time)

        tiles_per_comm = config.core_num * p
        comm_count = math.ceil(total_tiles / tiles_per_comm)
        full_round_tiles = min(total_tiles, tiles_per_comm)
        remainder_tiles = total_tiles % tiles_per_comm
        remainder_loops = math.ceil(remainder_tiles / config.core_num)

        if remainder_loops:
            last_aic_time = max(
                cube_time_per_core * remainder_loops,
                nd2nz_hit_time_per_core * remainder_loops,
            )
        else:
            last_aic_time = aic_time

        # ReduceScatter uses rank_size as the internal t value. The runtime
        # tiling and generated CSV store 2 * t as commTileM.
        t = config.rank_size
        block_m = 2 * t
        full_workload = _get_aiv_workload(
            full_round_tiles, n, m0, n0, block_m, config
        )
        aiv_time = _estimate_aiv_time(
            full_workload, t, aiv_core_num, config
        )

        if remainder_tiles:
            tail_workload = _get_aiv_workload(
                remainder_tiles, n, m0, n0, block_m, config
            )
            tail_time = _estimate_aiv_time(
                tail_workload, t, aiv_core_num, config
            )
        else:
            tail_time = aiv_time

        sync_and_launch_time = (
            config.sync_time_us * comm_count + config.launch_time_us
        )
        if aiv_time >= aic_time:
            pipeline_time = (
                aic_time
                + aiv_time * (comm_count - 1)
                + tail_time
            )
            bound_type = "AIV_bound"
        else:
            pipeline_time = (
                aic_time * (comm_count - 1)
                + max(last_aic_time, aiv_time)
                + tail_time
            )
            bound_type = "AIC_bound"

        results.append(
            CostResult(
                m=m,
                k=k,
                n=n,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                p=p,
                t=t,
                comm_tile_m=block_m,
                m0=m0,
                aiv_core_num=aiv_core_num,
                total_time=pipeline_time + sync_and_launch_time,
                bound_type=bound_type,
                aic_cube_time=aic_cube_time,
                aic_nd2nz_hit_time=aic_nd2nz_hit_time,
                aic_time=aic_time,
                aiv_time=aiv_time,
            )
        )

    return sorted(results, key=lambda result: result.total_time)


def evaluate_shapes(
    mte_model: MTECostModel,
    shapes: Iterable[Sequence[int]],
    p_values: Sequence[int],
    m0_values: Sequence[int],
    aiv_core_values: Sequence[int],
    config: HardwareConfig,
) -> List[CostResult]:
    results: List[CostResult] = []
    for shape in shapes:
        if len(shape) == 3:
            m, k, n = shape
            transpose_a, transpose_b = 1, 1
        elif len(shape) == 5:
            m, k, n, transpose_a, transpose_b = shape
        else:
            raise ValueError("each shape must contain M,K,N[,Transpose A,Transpose B]")

        valid_m0_values = (128,) if transpose_a == 0 else m0_values
        for m0 in valid_m0_values:
            for aiv_core_num in aiv_core_values:
                results.extend(
                    compute_cost_model(
                        mte_model,
                        int(m),
                        int(k),
                        int(n),
                        p_values,
                        transpose_a=int(transpose_a),
                        transpose_b=int(transpose_b),
                        m0=m0,
                        aiv_core_num=aiv_core_num,
                        config=config,
                    )
                )
    return results


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dynamic_tiling_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Evaluate cost-model tilings")
    parser.add_argument(
        "--input",
        type=Path,
        default=dynamic_tiling_dir / "scripts" / "test_shapes.csv",
        help="CSV file containing M, K and N columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sorted_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--rank-size", type=int, default=2)
    parser.add_argument(
        "--hardware",
        choices=[item.value for item in HardwareType],
        default=HardwareType.A2.value,
    )
    parser.add_argument(
        "--data-type",
        choices=[item.value for item in DataType],
        default=DataType.FP16.value,
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = _parse_args()
    shape_columns = ["M", "K", "N", "Transpose A", "Transpose B"]
    shape_frame = pd.read_csv(args.input, usecols=shape_columns)
    profile = get_hardware_profile(
        HardwareType(args.hardware),
        DataType(args.data_type),
        args.rank_size,
    )
    mte_model = MTECostModel(
        profile.mte,
        full_core_num=profile.hardware.core_num,
    )
    config = profile.hardware

    results = evaluate_shapes(
        mte_model=mte_model,
        shapes=shape_frame[shape_columns].itertuples(index=False, name=None),
        p_values=(1, 2, 4, 6, 8, 10, 12, 14),
        m0_values=(128, 256),
        aiv_core_values=(16, 20),
        config=config,
    )

    result_frame = pd.DataFrame(asdict(result) for result in results)
    result_frame.sort_values(["m", "k", "n", "total_time"], inplace=True)
    result_frame.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(result_frame.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
