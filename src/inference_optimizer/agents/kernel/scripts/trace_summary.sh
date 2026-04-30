#!/usr/bin/env bash
# trace_summary.sh — extract top-N hot kernels from a profiler trace.
#
# Usage: bash trace_summary.sh <trace.json.gz> [N]
#   default N = 5
#
# Output (one kernel per line):
#   gpu_pct=12.4 name=triton_red_fused_sum_42 framework=triton
#   gpu_pct=8.1  name=aiter::fmha_v3_fwd     framework=aiter
#   ...
#
# Read-only; never modifies the trace. Intended for the kernel agent's
# select_kernels subskill.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: trace_summary.sh <trace.json.gz> [N]" >&2
    exit 2
fi

TRACE_PATH="$1"
TOP_N="${2:-5}"

if [ ! -f "$TRACE_PATH" ]; then
    echo "ERROR: trace not found: $TRACE_PATH" >&2
    exit 1
fi

python3 - "$TRACE_PATH" "$TOP_N" <<'PY'
"""Parse a Chrome-trace JSON or .gz, aggregate per-kernel GPU time,
print top-N as `key=value` pairs.

This is intentionally minimal — we look at `cat="kernel"` events, sum
duration per `name`, and classify the kernel framework heuristically.
TraceLens has the proper analysis; this is a fallback for when the
agent only has the raw filtered trace and needs to pick candidates.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

trace_path, top_n_str = sys.argv[1], sys.argv[2]
top_n = max(1, int(top_n_str))

p = Path(trace_path)
opener = gzip.open if p.suffix == ".gz" else open
try:
    with opener(p, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception as exc:
    print(f"ERROR: failed to parse trace: {exc}", file=sys.stderr)
    sys.exit(1)

events = data.get("traceEvents") if isinstance(data, dict) else data
if not isinstance(events, list):
    print(f"ERROR: trace has no traceEvents list", file=sys.stderr)
    sys.exit(1)

# Aggregate kernel duration by name.
total_kernel_us = 0.0
per_name: dict[str, float] = defaultdict(float)
for ev in events:
    if not isinstance(ev, dict):
        continue
    cat = ev.get("cat", "")
    # Chrome trace format: kernel events typically have cat="kernel" or
    # the "ph" == "X" with a duration. We cast a wide net.
    name = str(ev.get("name", ""))
    dur = ev.get("dur")
    if not name or not isinstance(dur, (int, float)):
        continue
    if "kernel" in str(cat).lower() or name.startswith("triton_") \
       or "::" in name:
        per_name[name] += float(dur)
        total_kernel_us += float(dur)

if not per_name or total_kernel_us == 0:
    print("(no kernel events found)", file=sys.stderr)
    sys.exit(0)


def classify(kernel_name: str) -> str:
    n = kernel_name
    if n.startswith("triton_") or n.startswith("triton."):
        return "triton"
    if n.startswith("aiter::") or "aiter" in n:
        return "aiter"
    if n.startswith("Cijk_") or "hipBLAS" in n:
        return "hipblaslt"
    if "fmoe" in n or "moe_" in n:
        return "moe"
    if "vectorized_elementwise" in n:
        return "pytorch_elementwise"
    return "other"


sorted_kernels = sorted(per_name.items(), key=lambda x: x[1], reverse=True)
for name, us in sorted_kernels[:top_n]:
    pct = 100.0 * us / total_kernel_us
    framework = classify(name)
    # `name` may contain spaces; quote if needed.
    safe_name = name if " " not in name else f'"{name}"'
    print(f"gpu_pct={pct:.2f} name={safe_name} framework={framework}")
PY
