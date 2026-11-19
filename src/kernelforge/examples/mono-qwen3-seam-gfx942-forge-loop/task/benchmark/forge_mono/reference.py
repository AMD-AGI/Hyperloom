# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SGLang's path through the seam on gfx942, on the launch's tensors: SGLang's aiter block-FP8 linear
with its Triton GEMM, the all-reduce's numerics, and the fused add + RMSNorm SGLang calls."""

import aiter
import torch
import torch.distributed as dist
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils

EPS = 1e-6  # Qwen3's rms_norm_eps
WEIGHT_BLOCK = [128, 128]


def check_sglang_path() -> None:
    """Refuse a container where SGLang would not run o_proj through aiter's block-FP8 linear."""
    linear = fp8_utils.dispatch_w8a8_block_fp8_linear(WEIGHT_BLOCK)
    if linear is not fp8_utils.aiter_w8a8_block_fp8_linear:
        raise RuntimeError(f"SGLang dispatches a block-FP8 linear to {linear}; the task reproduces aiter's")


def o_proj(attn: torch.Tensor, weights) -> torch.Tensor:
    """This rank's o_proj partial: aiter's per-1x128 quant, then its Triton blockscale GEMM, which is
    SGLang's linear with ``SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK=0``. Without that setting SGLang on
    gfx942 runs aiter's CK GEMM, whose bits the launch does not reproduce (the example's README)."""
    with envs.SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK.override(False):
        return fp8_utils.aiter_w8a8_block_fp8_linear(attn, weights.wo, WEIGHT_BLOCK, weights.wos)


def all_reduce(partial: torch.Tensor) -> torch.Tensor:
    """The ranks' bf16 partials summed in rank order in f32 and rounded once.

    SGLang's custom all-reduce on gfx942 starts each rank's sum from its own partial, so its result
    is not one order; this fixes the order every rank agrees on."""
    parts = [torch.empty_like(partial) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, partial)
    acc = parts[0].float()
    for part in parts[1:]:
        acc += part.float()
    return acc.to(torch.bfloat16)


def add_rmsnorm(
    reduced: torch.Tensor, residual: torch.Tensor, norm_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``aiter.rmsnorm2d_fwd_with_add``: (normed, residual after the add)."""
    x, res_out = torch.empty_like(reduced), torch.empty_like(reduced)
    aiter.rmsnorm2d_fwd_with_add(x, reduced, residual, res_out, norm_w, EPS)
    return x, res_out


def seam(attn: torch.Tensor, residual: torch.Tensor, weights) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, res_out) for this rank's attention output ``attn`` and the replicated ``residual``."""
    return add_rmsnorm(all_reduce(o_proj(attn, weights)), residual, weights.norm_w)
