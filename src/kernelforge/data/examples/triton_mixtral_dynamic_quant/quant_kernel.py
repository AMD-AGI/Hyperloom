"""Dynamic per-tensor FP8 quantization — the target forge-loop optimizes.

This is the file forge-loop edits for this task. To play nicely with the loop it
must stay:

  * numerically correct (the driver gates it against a Torch oracle via SNR),
  * with a STABLE public entry point ``dynamic_quant_fp8(x, out, scale)`` — the
    driver imports this exact name and signature; do NOT rename it or change its
    arguments,
  * in Triton (do not rewrite it in another framework),
  * free of AITER imports.

The shipped implementation is a deliberately unoptimized eager-Torch version: it
is correct, so the loop can measure a real baseline from it, but it materializes
several full-size fp32 temporaries and launches one kernel per elementwise step.
Replacing it with a fused Triton kernel is the optimization the loop should find.

Replay safety: the entry point writes into caller-allocated ``out`` / ``scale``
and never syncs on device values (no ``.item()``), so one invocation can be
captured into a CUDA/HIP graph and replayed. Any scratch a Triton implementation
needs may be allocated internally — capture puts it in the graph's private pool.
"""

from __future__ import annotations

import torch

# gfx950 (MI355X) supports the OCP fp8 format torch.float8_e4m3fn, whose finfo
# max is 448.0. (gfx942/MI300 used the *fnuz* variant with max 240.0.)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def dynamic_quant_fp8(x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor) -> None:
    """Quantize ``x`` to fp8 with one scale for the WHOLE tensor. Public entry point.

    Args:
        x: (rows, cols) bf16 activations to quantize.
        out: (rows, cols) ``FP8_DTYPE`` destination, written in place.
        scale: (1,) float32 destination for ``amax(|x|) / FP8_MAX``, written in
            place. A zero input yields scale 1.0 so dequantization stays defined.
    """
    amax = x.float().abs().amax()
    s = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    out.copy_((x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE))
    scale.copy_(s.reshape(1))
