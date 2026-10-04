"""Dynamic per-tensor FP8 quantization — the target forge-loop optimizes."""

from __future__ import annotations

import torch

# gfx950 (MI355X) supports the OCP fp8 format torch.float8_e4m3fn, whose finfo max is 448.0. (gfx942/MI300 used the
# *fnuz* variant with max 240.0.)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def dynamic_quant_fp8(x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor) -> None:
    """Quantize ``x`` to fp8 with one scale for the WHOLE tensor. Public entry point."""
    amax = x.float().abs().amax()
    s = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    out.copy_((x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE))
    scale.copy_(s.reshape(1))
