"""Gemma RMSNorm — the target forge-loop optimizes."""

from __future__ import annotations

import torch

# Gemma normalizes with a (1 + weight) scale, unlike the plain RMSNorm (weight) form.
EPS = 1e-6


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    stream_handle: int | None = None,
) -> None:
    """Row-wise Gemma RMSNorm over the last dim. Public entry point."""
    del stream_handle  # Torch already runs on the current stream.
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    out.copy_((xf * inv_rms * (1.0 + weight.float())).to(out.dtype))
