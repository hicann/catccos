"""
Golden data generator for MxQuantAllGather.

Generates per-rank bf16 input data and computes:
  1. Dynamic MX quantization (scaleAlg=0) per BLOCK_SIZE group along N
  2. AllGather: concatenates all ranks' quantized data and mxscale

Supports output types: fp8_e4m3, fp8_e5m2, fp4_e2m1, fp4_e1m2

Output files per rank:
  - rank_{r}_input.bin    : bf16 input [M, N]
  - rank_{r}_golden.bin   : quantized output [M * rankSize, N] bytes
  - rank_{r}_golden_mxscale.bin : mxscale output [M * rankSize, N/BS] (uint8 E8M0)
"""

import argparse
import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

# Try to import ml_dtypes for fp8 support; fall back to manual implementation
try:
    from ml_dtypes import float8_e4m3fn, float8_e5m2
except ImportError:
    float8_e4m3fn = None
    float8_e5m2 = None
HAS_ML_DTYPES = float8_e4m3fn is not None

_QUANT_BACKEND = os.environ.get("GEN_DATA_QUANT_BACKEND", "lut").lower()
if _QUANT_BACKEND not in ("native", "lut"):
    raise ValueError(
        f"GEN_DATA_QUANT_BACKEND must be 'native' or 'lut' (got "
        f"{_QUANT_BACKEND!r})")

_FP8_FORMATS = {
    "E4M3": dict(
        torch_dtype=torch.float8_e4m3fn,
        emax=8,
        max_value=448.0,
        bias=7,
    ),
    "E5M2": dict(
        torch_dtype=torch.float8_e5m2,
        emax=15,
        max_value=57344.0,
        bias=15,
    ),
}

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
    "E1M2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
        "min_value": -1.75,
    },
}

# --------------------------------------------------------------------------- #
# FP8 LUT construction (kept identical to the baseline so that argmin returns
# exactly the same indices on identical inputs).
# --------------------------------------------------------------------------- #
def _build_e4m3_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E4M3"]["bias"]
    fp8_max = _FP8_FORMATS["E4M3"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif val == 127:
            v = sign * fp8_max
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                v = (mantissa / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 8.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


def _build_e5m2_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E5M2"]["bias"]
    fp8_max = _FP8_FORMATS["E5M2"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif 124 <= val <= 127:
            v = sign * fp8_max
        else:
            exp = (val >> 2) & 0x1F
            mantissa = val & 0x03
            if exp == 0:
                v = (mantissa / 4.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 4.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


_LUT_BUILDERS = {"E4M3": _build_e4m3_lut, "E5M2": _build_e5m2_lut}
_LUT_CACHE = {}
_LUT_POS_CACHE = {}


def _get_lut(format_name: str) -> torch.Tensor:
    if format_name not in _LUT_CACHE:
        _LUT_CACHE[format_name] = _LUT_BUILDERS[format_name]()
    return _LUT_CACHE[format_name]


def _get_lut_pos(format_name: str) -> torch.Tensor:
    """Return the positive half of the LUT (indices 0..127), sorted ascending.

    The full LUT is constructed as
        idx 0..127   – positive values, ascending by magnitude
        idx 128..255 – the same magnitudes negated (idx 128 = -0)
    so quantizing ``|x|`` against ``lut_pos`` reproduces the full-LUT
    ``argmin`` exactly when we then re-apply the sign and use lower-index
    tie-breaking. Verified ascending in the cache builder below.
    """
    if format_name not in _LUT_POS_CACHE:
        full = _get_lut(format_name)
        pos = full[:128].contiguous()
        # Sanity check on construction (no runtime cost after first call).
        diffs = pos[1:] - pos[:-1]
        if (diffs < 0).any():
            raise AssertionError(
                f"{format_name} positive LUT half is not non-decreasing")
        _LUT_POS_CACHE[format_name] = pos
    return _LUT_POS_CACHE[format_name]


# --------------------------------------------------------------------------- #
# E8M0 scale exponent (vectorized exact replacement for
# MXFP8MatrixQuantizer._compute_scale_fp8_e8m0fnu).
# --------------------------------------------------------------------------- #
def _e8m0_exp(max_abs: torch.Tensor, emax: int,
              epsilon: float = _EPSILON) -> torch.Tensor:
    """Return the per-block E8M0 exponent (``int32`` tensor, same shape as
    ``max_abs``). For each element:

        if max_abs < epsilon: exp = 0
        else: exp = clamp(floor(log2(max_abs)) - emax, -128, 127)

    ``floor(log2(x))`` for a positive finite FP32 ``x`` is exactly
    ``biased_exp(x) - 127``, so we extract it from the bit pattern.
    """
    if max_abs.dtype != torch.float32:
        raise TypeError(f"Expected torch.float32, got {max_abs.dtype}")
    zero_mask = max_abs < epsilon
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    # Reinterpret cast: same buffer, viewed as int32. .contiguous() is
    # required because .view(dtype) demands a contiguous tensor.
    bits = safe.contiguous().view(torch.int32)
    exp_bits = (bits >> 23) & 0xFF
    exp = exp_bits - 127 - emax
    exp = exp.clamp(-128, 127)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


# --------------------------------------------------------------------------- #
# Vectorized LUT quantization via searchsorted on the positive half of the
# LUT, replicating the baseline's argmin tie-breaking (lowest index wins,
# i.e. round-half-toward-zero) without ever materializing a (..., 256)
# distance tensor.
# --------------------------------------------------------------------------- #
def _vectorized_lut_quantize(scaled: torch.Tensor, format_name: str,
                             fp8_dtype: torch.dtype) -> torch.Tensor:
    """Quantize ``scaled`` to FP8 with the same semantics as the baseline
    ``MXFP8MatrixQuantizer._quantize_to_fp8`` (full-LUT ``argmin`` with
    PyTorch's lowest-index tie-breaking).

    Algorithm:
      1. Take the positive half of the LUT, ``lut_pos`` (sorted ascending).
      2. ``searchsorted`` gives the upper neighbour ``upper_idx`` such that
         ``lut_pos[upper_idx-1] <= |x| <= lut_pos[upper_idx]``.
      3. Pick the closer of the two neighbours; on a tie, pick the *lower*
         neighbour (smaller magnitude). This reproduces full-LUT argmin
         because all negative LUT entries are strictly farther from a
         positive ``x`` than the same-magnitude positive entry, and vice
         versa, and at a positive↔negative tie at exactly zero the lowest
         positive index wins.
      4. Re-apply the sign of ``x``. ``sign(0)*0 = +0``, which matches
         baseline (full argmin returns idx 0 = +0 for input 0).
    """
    lut_pos = _get_lut_pos(format_name)                              # (128,)
    last_idx = lut_pos.numel() - 1                                   # 127

    sign = torch.sign(scaled)
    mag = scaled.abs()

    upper_idx = torch.searchsorted(lut_pos, mag).clamp(max=last_idx)
    lower_idx = (upper_idx - 1).clamp(min=0)

    upper_val = lut_pos[upper_idx]
    lower_val = lut_pos[lower_idx]

    # On equal distance, baseline argmin returns the lower (smaller-magnitude)
    # index, so we use ``<=`` to break ties toward the lower neighbour.
    pick_lower = (mag - lower_val) <= (upper_val - mag)
    chosen_mag = torch.where(pick_lower, lower_val, upper_val)

    snapped_fp32 = sign * chosen_mag

    # Canonicalize -0 to +0. Baseline-LUT argmin over the full 256-entry LUT
    # has +0 at idx 0 AND -0 at idx 128 with identical absolute distances; on
    # any tie at distance |x| (which always happens when chosen_mag == 0),
    # ``argmin`` returns the lower index → +0. ``sign(-x) * 0`` here would
    # otherwise produce -0 (different byte: 0x80 vs 0x00 in float8_e4m3fn).
    zero_mask = chosen_mag == 0
    snapped_fp32 = torch.where(
        zero_mask, torch.zeros_like(snapped_fp32), snapped_fp32)

    # Cast is lossless: every value in `snapped_fp32` is an exact FP8 grid
    # point, so RNE picks itself.
    return snapped_fp32.to(fp8_dtype)


# --------------------------------------------------------------------------- #
# Per-axis MX quantization. ``_quantize_axis_last`` matches axis=1 of the
# baseline (block-along-columns); ``_quantize_axis_first`` matches axis=0
# (block-along-rows) by transposing.
# --------------------------------------------------------------------------- #
def _quantize_axis_last_fp8(matrix: torch.Tensor, format_name: str,
                        block_size: int = _BLOCK_SIZE
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize along the last axis. Returns ``(quant_fp8 (M, N),
    scale (M, padded_blocks), dequant_fp32 (M, N))``.

    ``padded_blocks`` is rounded up to an even number to match the baseline
    ``scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, ...))`` shape.
    """
    M, N = matrix.shape
    fmt = _FP8_FORMATS[format_name]
    fp8_dtype = fmt["torch_dtype"]
    fp8_emax = fmt["emax"]
    fp8_max = fmt["max_value"]

    num_blocks = (N + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != N:
        padded = torch.zeros(M, padded_n, dtype=matrix.dtype)
        padded[:, :N] = matrix
    else:
        padded = matrix

    blocks = padded.view(M, num_blocks, block_size)

    max_abs = blocks.abs().amax(dim=-1)                              # (M, NB)
    exp = _e8m0_exp(max_abs, fp8_emax)                               # (M, NB)
    scale = torch.exp2(exp.to(torch.float32))                        # (M, NB)

    scaled = blocks / scale.unsqueeze(-1)                            # (M, NB, BS)
    scaled_clamped = scaled.clamp(-fp8_max, fp8_max)

    if _QUANT_BACKEND == "native":
        quant_fp8 = scaled_clamped.to(fp8_dtype)
    else:
        quant_fp8 = _vectorized_lut_quantize(
            scaled_clamped, format_name, fp8_dtype)

    # Dequantize: cast back to fp32 (lossless, exact LUT value), multiply by
    # the (power-of-two) scale → exact reconstruction matching the baseline
    # `_dequantize_by_*` code path.
    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    # Trim padding (no-op for K%32==0 cases).
    if padded_n != N:
        quant_fp8 = quant_fp8.reshape(M, padded_n)[:, :N].contiguous()
        dequant = dequant.reshape(M, padded_n)[:, :N].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(M, N)
        dequant = dequant.reshape(M, N)

    # Pad scale matrix to even num_blocks (matches baseline shape).
    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((M, padded_blocks), dtype=torch.float32)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale, dequant


def _quantize_axis_first_fp8(matrix: torch.Tensor, format_name: str,
                         block_size: int = _BLOCK_SIZE
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize along the first axis (axis=0 in baseline)."""
    qt, st, dt = _quantize_axis_last_fp8(
        matrix.t().contiguous(), format_name, block_size)
    return (qt.t().contiguous(),
            st.t().contiguous(),
            dt.t().contiguous())


def _quantize_fp8(matrix: torch.Tensor, format_name: str, axis: int,
              block_size: int = _BLOCK_SIZE
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first_fp8(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_axis_last_fp8(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10.0 - 5.0
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E4M3", axis)
    return (quant_fp8.to(torch.float8_e4m3fn),
            scale_fp32.to(torch.float8_e8m0fnu),
            dequant_fp32)


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E5M2", axis)
    return (quant_fp8.to(torch.float8_e5m2),
            scale_fp32.to(torch.float8_e8m0fnu),
            dequant_fp32)

def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])

    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)

        if exp == 0:
            if mantissa == 0:
                value = 0.0
            else:
                value = (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (2.0 ** (float(exp) - bias))

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {
    "E2M1": _build_fp4_lut("E2M1"),
    "E1M2": _build_fp4_lut("E1M2"),
}


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)

    # 与原实现一致：按 LUT 顺序 argmin，距离相等时取更小下标。
    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    quantized = lut[indices]

    return quantized, indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )

    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    packed = low | high
    return packed.to(torch.uint8)


def _quantize_axis_last(matrix: torch.Tensor, format_name: str, block_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size

    if padded_n != n:
        padded = torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)

    exp = torch.floor(torch.log2(torch.clamp(max_abs, min=_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)

    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)

    quantized = quantized_blocks.reshape(m, padded_n)
    dequantized = dequant_blocks.reshape(m, padded_n)
    if padded_n != n:
        quantized = quantized[:, :n].contiguous()
        dequantized = dequantized[:, :n].contiguous()

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_axis_first(matrix: torch.Tensor, format_name: str, block_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_axis_last(matrix.t().contiguous(), format_name, block_size)
    return quantized_t.t().contiguous(), scale_t.t().contiguous(), dequantized_t.t().contiguous()


def _quantize(matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp4_e2m1(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E2M1", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    return quantized_matrix_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def gen_data_fp4_e1m2(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E1M2", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E1M2")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    return quantized_matrix_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def bf16_bits_to_fp32(bf16_bits):
    """Convert bf16 bit representation (uint16) to fp32."""
    fp32_bits = bf16_bits.astype(np.uint32) << 16
    return np.frombuffer(fp32_bits.tobytes(), dtype=np.float32)


def fp32_to_bf16_bits(fp32_arr):
    """Convert fp32 to bf16 bit representation (uint16) with round-to-nearest-even."""
    fp32_bits = np.frombuffer(fp32_arr.astype(np.float32).tobytes(), dtype=np.uint32)
    # Round to nearest even (add rounding bias)
    rounding_bias = ((fp32_bits >> 16) & 1) + 0x7FFF
    bf16_bits = ((fp32_bits + rounding_bias) >> 16).astype(np.uint16)
    return bf16_bits

def generate_and_process(args):
    rank_size = args.rankSize
    M = args.m
    N = args.n
    dtype = args.dtype
    output_dir = args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    all_ranks_quant = []
    all_ranks_scale = []

    for rank in range(rank_size):
        # Generate random bf16 input
        if dtype == "fp8_e5m2":
            a_quant, a_scale, a_fp32 = gen_data_fp8_e5m2(M, N, 1)
        elif dtype == "fp4_e2m1":
            a_quant, a_scale, a_fp32 = gen_data_fp4_e2m1(M, N, 1, 0)
        elif dtype == "fp4_e1m2":
            a_quant, a_scale, a_fp32 = gen_data_fp4_e1m2(M, N, 1, 0)
        else:
            a_quant, a_scale, a_fp32 = gen_data_fp8_e4m3(M, N, 1)

        a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)

        a_fp32_np = torch.tensor(a_fp32.flatten().untyped_storage(), dtype=torch.float32).numpy()
        input_bf16 = fp32_to_bf16_bits(a_fp32_np)

        # Save input
        input_filename = os.path.join(output_dir, f"rank_{rank}_input.bin")
        input_bf16.tofile(input_filename)

        # Compute MX quantization

        a_np = torch.tensor(a_quant.flatten().untyped_storage(), dtype=torch.int8).numpy()
        a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8).numpy()
        all_ranks_quant.append(a_np)
        all_ranks_scale.append(a_scale_np)
        print(f"  -> Rank {rank}: quant shape={a_np.shape}, scale shape={a_scale_np.shape}")

    # AllGather: concatenate all ranks along M dimension
    gathered_quant = np.concatenate(all_ranks_quant, axis=0)  # [M*rankSize, N/pack_ratio]
    gathered_scale = np.concatenate(all_ranks_scale, axis=0)  # [M*rankSize, N/BS]

    print(f"[Process] AllGather: quant={gathered_quant.shape}, scale={gathered_scale.shape}")

    # Save golden for each rank (all ranks see the same gathered result)
    for rank in range(rank_size):
        quant_file = os.path.join(output_dir, f"rank_{rank}_golden.bin")
        gathered_quant.tofile(quant_file)

        scale_file = os.path.join(output_dir, f"rank_{rank}_golden_mxscale.bin")
        gathered_scale.tofile(scale_file)

        print(f"  -> Rank {rank} golden saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Golden Data for MxQuant+AllGather")
    parser.add_argument("--rankSize", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=32, help="MX block size")
    parser.add_argument("--dtype", type=str, default="fp8_e4m3",
                        choices=["fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4_e1m2"],
                        help="MX quant output dtype")
    parser.add_argument("--data_dir", type=str, default="./data_mx_allgather")

    args = parser.parse_args()
    generate_and_process(args)
