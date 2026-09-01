"""Gemma RMSNorm — the target forge-loop optimizes.

This is the file forge-loop edits for this task. To play nicely with the loop it
must stay:

  * numerically correct (the driver gates it against a Torch oracle via SNR),
  * with a STABLE public entry point ``gemma_rmsnorm(x, weight, out, stream_handle=...)``
    — the driver imports this exact name and signature; do NOT rename it or
    change its arguments,
  * in FlyDSL (do not rewrite it in another framework),
  * free of AITER imports.

The shipped implementation is a deliberately unoptimized eager-Torch version: it
is correct, so the loop can measure a real baseline from it, but it materializes
full-size fp32 temporaries and launches a separate kernel per elementwise step.
Replacing it with a single fused FlyDSL kernel is the optimization to find.

Stream contract (matters for honest benchmarking)
-------------------------------------------------
``stream_handle`` is the raw HIP/CUDA stream handle of the stream the driver wants
the work on. Torch ops already run on the current stream, so this baseline ignores
it. A FlyDSL implementation launches on a stream it is given, so it MUST route this
handle into the launch, e.g.::

    import flydsl.expr as fx
    launch_fn(x, weight, out, rows, stream=fx.Stream(stream_handle))

The driver always passes the CURRENTLY ACTIVE stream. Under the CUDA/HIP graph
harness that active stream is the private capture stream, so honoring the handle is
what gets the kernel recorded into the graph. Launching on the default/NULL stream
instead produces a silently EMPTY graph whose replay is a few microseconds
regardless of problem size — a fake speedup the harness will reject.
"""

from __future__ import annotations

import torch

# Gemma normalizes with a (1 + weight) scale, unlike the plain RMSNorm (weight)
# form. The driver's oracle uses the same constant and the same convention.
EPS = 1e-6


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    stream_handle: int | None = None,
) -> None:
    """Row-wise Gemma RMSNorm over the last dim. Public entry point.

    ``out = x * rsqrt(mean(x^2) + EPS) * (1 + weight)``, reduced in fp32 for
    numerical stability and written back in the dtype of ``out``.

    Args:
        x: (rows, hidden) bf16 input.
        weight: (hidden,) bf16 normalization weight.
        out: (rows, hidden) bf16 destination, written in place.
        stream_handle: raw stream handle to launch on (see module docstring). The
            eager-Torch baseline ignores it; a FlyDSL kernel must honor it.
    """
    del stream_handle  # Torch already runs on the current stream.
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    out.copy_((xf * inv_rms * (1.0 + weight.float())).to(out.dtype))
