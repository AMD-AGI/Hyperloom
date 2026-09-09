# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opt-in vLLM example for a Qwen3 assembly candidate, without installation edits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

_registered = False
_kernels = {}


def register() -> None:
    """Register through vLLM's general_plugins entry point in every worker."""
    global _registered
    if _registered:
        return
    from vllm.model_executor.models.qwen3 import Qwen3Attention

    from .kernel import QkNormRope

    @torch.library.custom_op("forge_qwen3::qk_norm_rope", mutates_args=())
    def operation(
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cache: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_heads: int,
        k_heads: int,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = qkv.device.index
        if device not in _kernels:
            with torch.cuda.device(qkv.device), tempfile.TemporaryDirectory(prefix="forge-qwen3-asm-") as build:
                _kernels[device] = QkNormRope(Path(build))
        return _kernels[device](qkv, positions, cache, q_weight, k_weight, q_heads, k_heads, epsilon)

    @operation.register_fake
    def fake(qkv, positions, cache, q_weight, k_weight, q_heads, k_heads, epsilon):
        return qkv.new_empty((qkv.shape[0], q_heads * 128)), qkv.new_empty((qkv.shape[0], k_heads * 128))

    original_forward = Qwen3Attention.forward

    def forward(self, positions, hidden_states):
        supported = (
            torch.version.hip is not None
            and self.head_dim == 128
            and self.num_heads == 32
            and self.num_kv_heads == 8
            and self.rotary_emb.rotary_dim == 128
            and self.rotary_emb.is_neox_style
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
            and hidden_states.dtype == torch.bfloat16
        )
        if not supported:
            return original_forward(self, positions, hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
        q, k = operation(
            qkv,
            positions,
            cache,
            self.q_norm.weight,
            self.k_norm.weight,
            self.num_heads,
            self.num_kv_heads,
            self.q_norm.variance_epsilon,
        )
        value = qkv[..., self.q_size + self.kv_size :]
        output, _ = self.o_proj(self.attn(q, k, value))
        return output

    Qwen3Attention.forward = forward
    _registered = True
