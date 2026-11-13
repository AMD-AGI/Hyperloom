# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A random Qwen3-14B-FP8 o_proj and post-attention norm, in the form SGLang holds them on gfx942."""

from dataclasses import dataclass

import torch
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz

HIDDEN = 5120
BLOCK = 128  # the checkpoint's weight block edge
E4M3FN_MAX = 448.0


@dataclass(frozen=True)
class LayerWeights:
    """One TP rank's share of a layer: o_proj's input columns of this rank's heads (RowParallelLinear)."""

    wo: torch.Tensor  # [HIDDEN, HIDDEN // tp] float8_e4m3fnuz
    wos: torch.Tensor  # [HIDDEN // BLOCK, HIDDEN // tp // BLOCK] fp32, the blocks' dequant scales
    norm_w: torch.Tensor  # [HIDDEN] bf16, post_attention_layernorm.weight, the same on every rank


def random_layer(tp: int, rank: int, device: torch.device, seed: int) -> LayerWeights:
    """The layer drawn from ``seed`` on the CPU, so every rank holds the same one and takes its
    shard. o_proj is block-quantized as the checkpoint is (e4m3fn, a 128x128 block's scale its
    amax / 448), then normalized to e4m3fnuz as SGLang's loader does on gfx942."""
    if HIDDEN % (tp * BLOCK):
        raise ValueError(f"TP {tp} does not split o_proj's {HIDDEN} inputs into {BLOCK}-wide blocks")
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(HIDDEN, HIDDEN, generator=gen) * 0.02
    norm_w = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
    k = HIDDEN // tp
    blocks = w[:, rank * k : (rank + 1) * k].reshape(HIDDEN // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True) / E4M3FN_MAX
    q = (blocks / scale).clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn)
    wo, wos, _ = normalize_e4m3fn_to_e4m3fnuz(
        q.reshape(HIDDEN, k).to(device), scale.reshape(HIDDEN // BLOCK, -1).to(device)
    )
    return LayerWeights(wo.contiguous(), wos.to(torch.float32).contiguous(), norm_w.to(torch.bfloat16).to(device))
