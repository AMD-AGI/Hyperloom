"""Fused residual-add + Gemma RMSNorm — the target forge-loop optimizes.

This is the file forge-loop edits for this task. To play nicely with the loop it
must stay:

  * numerically correct (the driver gates it against a Torch oracle via SNR),
  * with a STABLE public entry point
    ``fused_add_rmsnorm(x, residual, weight, out, residual_out)`` — the driver
    imports this exact name and signature; do NOT rename it or change its
    arguments,
  * implemented in HIP (a JIT-compiled HIP C++ kernel driven from this file; do
    not rewrite it in Triton or as a pure-Torch composition),
  * free of AITER imports.

The shipped implementation is a deliberately unoptimized eager-Torch version: it
is correct, so the loop can measure a real baseline from it, but it materializes
full-size fp32 temporaries and launches a separate kernel per elementwise step.
Replacing it with one fused HIP kernel is the optimization to find.

Why ``residual_out`` is a separate buffer
----------------------------------------
Production code (e.g. SGLang) updates ``residual`` IN PLACE. This task writes the
summed residual to its own ``residual_out`` instead, because the benchmark captures
one invocation into a CUDA/HIP graph and replays it many times on the same memory.
An in-place update would make each replay read its own previous output, so the
values would compound across replays — corrupting the capture-validity check and
making the measurement meaningless. Keeping the inputs read-only makes one
invocation idempotent and therefore replay-safe.

A HIP implementation should JIT-compile its extension ONCE and cache it at module
scope: compilation must happen during warmup, never inside graph capture.
"""

from __future__ import annotations

import torch

# Gemma normalizes with a (1 + weight) scale, unlike the plain RMSNorm (weight)
# form. The driver's oracle uses the same constant and the same convention.
EPS = 1e-6


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    """Residual add followed by Gemma RMSNorm. Public entry point.

    ``summed = x + residual``, then
    ``out = summed * rsqrt(mean(summed^2) + EPS) * (1 + weight)``, with the
    reduction in fp32 for numerical stability. ``summed`` is also written to
    ``residual_out`` because the next transformer block consumes it.

    Args:
        x: (rows, hidden) bf16 input, read-only.
        residual: (rows, hidden) bf16 residual stream, read-only.
        weight: (hidden,) bf16 normalization weight.
        out: (rows, hidden) bf16 normalized destination, written in place.
        residual_out: (rows, hidden) bf16 destination for ``x + residual``,
            written in place.
    """
    summed = x + residual
    sf = summed.float()
    inv_rms = torch.rsqrt(sf.square().mean(-1, keepdim=True) + EPS)
    out.copy_((sf * inv_rms * (1.0 + weight.float())).to(out.dtype))
    residual_out.copy_(summed)
