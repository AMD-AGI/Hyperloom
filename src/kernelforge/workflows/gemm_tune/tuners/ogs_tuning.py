# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16 × MXFP4 matmul_ogs shapes and bounded RDNA4 flag candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile


@dataclass(frozen=True)
class OgsCase:
    """One exact routed projection shape to measure."""

    n: int
    k: int
    m: int
    tokens: int
    projection: str


_TILE_16 = 16
_TILE_32 = 32
_TILE_64 = 64
_TILE_128 = 128
_TILE_256 = 256
_TILE_512 = 512
_EIGHT = 8
_DEFAULT_WARPS = 4
_ROCM_MXFP4_ALIGNMENT = _TILE_256


def _round_up(value: int, alignment: int) -> int:
    return ceil(value / alignment) * alignment


def projection_shapes(profile: ModelProfile) -> tuple[tuple[int, int], tuple[int, int]]:
    """The padded GPT-OSS up and down projections consumed by matmul_ogs on ROCm."""
    hidden = int(profile.hidden_size)
    intermediate = int(profile.effective_moe_intermediate)
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("GPT-OSS MoE projection dimensions are unavailable")
    up_n = 2 * _round_up(intermediate, _ROCM_MXFP4_ALIGNMENT)
    up_k = _round_up(hidden, _ROCM_MXFP4_ALIGNMENT)
    down_n = up_k
    down_k = up_n // 2
    return (up_n, up_k), (down_n, down_k)


def cases(profile: ModelProfile, tokens: list[int], *, conc: int) -> list[OgsCase]:
    """Use only the token counts the caller supplied for the routed serving path."""
    counts = sorted({int(count) for count in [*tokens, conc] if int(count) > 0})
    top_k = int(profile.num_experts_per_tok)
    return [
        OgsCase(n=n, k=k, m=count * top_k, tokens=count, projection=projection)
        for projection, (n, k) in zip(("up", "down"), projection_shapes(profile), strict=True)
        for count in counts
    ]


_FAST_FLAGS = (
    {
        "block_m": _TILE_16,
        "block_n": _TILE_64,
        "block_k": _TILE_512,
        "num_warps": _DEFAULT_WARPS,
        "num_stages": 1,
        "split_k": 1,
        "xcd_swizzle": 0,
    },
    {
        "block_m": _TILE_32,
        "block_n": _TILE_64,
        "block_k": _TILE_256,
        "num_warps": _DEFAULT_WARPS,
        "num_stages": 1,
        "split_k": 1,
        "xcd_swizzle": 0,
    },
    {
        "block_m": _TILE_64,
        "block_n": _TILE_128,
        "block_k": _TILE_128,
        "num_warps": _DEFAULT_WARPS,
        "num_stages": 1,
        "split_k": 1,
        "xcd_swizzle": 0,
    },
    {
        "block_m": _TILE_64,
        "block_n": _TILE_128,
        "block_k": _TILE_256,
        "num_warps": _DEFAULT_WARPS,
        "num_stages": 2,
        "split_k": 1,
        "xcd_swizzle": 0,
    },
)
_THOROUGH_FLAGS = (
    {
        "block_m": _TILE_16,
        "block_n": _TILE_128,
        "block_k": _TILE_256,
        "num_warps": 2,
        "num_stages": 1,
        "split_k": 1,
        "xcd_swizzle": 0,
    },
    {
        "block_m": _TILE_32,
        "block_n": _TILE_128,
        "block_k": _TILE_512,
        "num_warps": _DEFAULT_WARPS,
        "num_stages": 2,
        "split_k": 1,
        "xcd_swizzle": _EIGHT,
    },
    {
        "block_m": _TILE_64,
        "block_n": _TILE_64,
        "block_k": _TILE_128,
        "num_warps": _EIGHT,
        "num_stages": 2,
        "split_k": 1,
        "xcd_swizzle": _EIGHT,
    },
)


def flag_search_space(*, thorough: bool) -> list[dict[str, int]]:
    """Return only supported, finite flag combinations; every row is timed and checked."""
    return [dict(flags) for flags in (*_FAST_FLAGS, *(_THOROUGH_FLAGS if thorough else ()))]
