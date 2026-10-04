"""Fused residual-add + Gemma RMSNorm — the target forge-loop optimizes."""

from __future__ import annotations

import torch

# Gemma normalizes with a (1 + weight) scale, unlike the plain RMSNorm (weight) form.
EPS = 1e-6


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    """Residual add followed by Gemma RMSNorm. Public entry point."""
    summed = x + residual
    sf = summed.float()
    inv_rms = torch.rsqrt(sf.square().mean(-1, keepdim=True) + EPS)
    out.copy_((sf * inv_rms * (1.0 + weight.float())).to(out.dtype))
    residual_out.copy_(summed)
