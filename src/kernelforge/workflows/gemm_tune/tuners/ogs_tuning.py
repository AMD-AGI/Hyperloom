# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16 × MXFP4 matmul_ogs shapes and bounded RDNA4 flag candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile


@dataclass(frozen=True)
class OgsCase:
    """One projection and token-per-expert interval to measure."""

    n: int
    k: int
    min_m: int
    max_m: int
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
_MAX_TOKENS_PER_EXPERT = _TILE_512
_DECODE_BUCKET_END = _TILE_128


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


def token_ranges(profile: ModelProfile, tokens: list[int], *, conc: int, thorough: bool) -> list[tuple[int, int]]:
    """Cover decode and prefill with a bounded per-expert ladder."""
    experts = max(1, int(profile.num_experts))
    topk = max(1, int(profile.num_experts_per_tok))
    observed = [ceil(int(count) * topk / experts) for count in [*tokens, conc] if int(count) > 0]
    maximum = min(_MAX_TOKENS_PER_EXPERT, max([_DECODE_BUCKET_END, *observed]))
    edges = (
        [1, 2, _EIGHT, _TILE_32, _DECODE_BUCKET_END, _MAX_TOKENS_PER_EXPERT]
        if thorough
        else [1, 2, _EIGHT, _TILE_32, _DECODE_BUCKET_END]
    )
    edges = sorted({*(_edge for _edge in edges if _edge < maximum), maximum})
    return [(1 if index == 0 else edges[index - 1] + 1, edge) for index, edge in enumerate(edges)]


def cases(profile: ModelProfile, tokens: list[int], *, conc: int, thorough: bool) -> list[OgsCase]:
    return [
        OgsCase(n=n, k=k, min_m=lower, max_m=upper, projection=projection)
        for projection, (n, k) in zip(("up", "down"), projection_shapes(profile), strict=True)
        for lower, upper in token_ranges(profile, tokens, conc=conc, thorough=thorough)
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
