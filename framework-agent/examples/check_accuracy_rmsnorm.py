# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight numerical-accuracy check for the explore --execute e2e path.

Compares sglang's RMSNorm output against a reference torch implementation
on the same input and emits accuracy.json compatible with
framework_agent.explorer's ``_evaluate_candidate``. The reference is just
a per-token L2 normalisation; we report the fraction of tokens whose
relative error stays under 1e-2 as the ``accuracy`` field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sglang.srt.layers.layernorm import RMSNorm


def main() -> int:
    """Parse args, compare RMSNorm vs reference, write accuracy.json.

    Returns:
        int: Process exit code: ``0`` on success, ``2`` when no CUDA/ROCm
            device is available.
    """
    parser = argparse.ArgumentParser(description="Tiny RMSNorm numerical-accuracy check")
    parser.add_argument("--out", required=True, help="Path to accuracy.json")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA / ROCm device unavailable", file=sys.stderr)
        return 2

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    layer = RMSNorm(args.hidden).to(dtype=torch.bfloat16)
    weight = layer.weight.data.to(torch.float32)
    x = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16)
    actual = layer(x).to(torch.float32)
    xf32 = x.to(torch.float32)
    variance = xf32.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + 1e-6)
    reference = xf32 * rsqrt * weight

    diff = (actual - reference).abs()
    denom = reference.abs().clamp_min(1e-3)
    rel = (diff / denom).max(dim=-1).values
    correct = (rel < args.rtol).sum().item()
    total = rel.numel()
    accuracy = correct / total if total else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "correct_tokens": correct,
                "total_tokens": total,
                "rtol": args.rtol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"accuracy wrote {out}: accuracy={accuracy:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
