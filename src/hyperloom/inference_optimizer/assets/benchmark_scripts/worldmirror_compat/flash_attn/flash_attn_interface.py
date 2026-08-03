# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SDPA-backed stand-ins for the flash-attn entry points HY-World-2.0 calls.

Unsupported flash-attn features raise instead of silently computing something
else: a wrong baseline is worse than a failed one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["flash_attn_func", "flash_attn_qkvpacked_func"]

_NO_WINDOW = {(-1, -1), (None, None)}


def _expand_kv(x: torch.Tensor, heads_q: int) -> torch.Tensor:
    """Broadcast GQA/MQA key-value heads up to the query head count."""
    heads_kv = x.shape[2]
    if heads_kv == heads_q:
        return x
    if heads_q % heads_kv:
        raise ValueError(f"query heads {heads_q} not divisible by kv heads {heads_kv}")
    return x.repeat_interleave(heads_q // heads_kv, dim=2)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    """Attention over (batch, seqlen, heads, head_dim) tensors, as flash-attn.

    SDPA wants (batch, heads, seqlen, head_dim), so transpose on the way in and
    back out to keep the caller's layout contract intact.
    """
    if return_attn_probs:
        raise NotImplementedError("flash_attn SDPA shim cannot return attention probabilities")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn SDPA shim does not implement ALiBi slopes")
    if tuple(window_size) not in _NO_WINDOW:
        raise NotImplementedError(f"flash_attn SDPA shim does not implement sliding window {window_size}")

    heads_q = q.shape[2]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        _expand_kv(k, heads_q).transpose(1, 2),
        _expand_kv(v, heads_q).transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Packed variant; qkv is (batch, seqlen, 3, heads, head_dim)."""
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kwargs
    )
