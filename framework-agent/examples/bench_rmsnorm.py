# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight GPU micro-benchmark for the explore --execute e2e path.

Runs sglang's RMSNorm kernel on a fixed (N=4096, H=8192, bf16) input and
emits a benchmark.json compatible with framework_agent.explorer's
``_evaluate_candidate``. The numbers are real (timed via cuda.synchronize),
but identical across candidates because we don't rebuild sglang from the
PR's worktree in this sandbox - this exercises the full --execute code
path and the winner gate without needing the full framework build chain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sglang.srt.layers.layernorm import RMSNorm


def main() -> int:
    """Parse args, time RMSNorm, write benchmark.json with throughput field."""
    parser = argparse.ArgumentParser(description="Tiny RMSNorm GPU micro-bench")
    parser.add_argument("--out", required=True, help="Path to benchmark.json")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA / ROCm device unavailable", file=sys.stderr)
        return 2

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    layer = RMSNorm(args.hidden).to(dtype=torch.bfloat16)
    x = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16)
    for _ in range(args.warmup):
        _ = layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    throughput = (args.num_tokens * args.iters) / elapsed

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "throughput": throughput,
                "elapsed_sec": elapsed,
                "iters": args.iters,
                "num_tokens": args.num_tokens,
                "hidden": args.hidden,
                "dtype": "bf16",
                "completed": f"{args.iters}/{args.iters}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"benchmark wrote {out}: throughput={throughput:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
