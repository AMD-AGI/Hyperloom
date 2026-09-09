"""Shared helpers for the diffusion per-denoise-step roofline divisor."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

_PROFILER_STEP_RE = re.compile(rb"ProfilerStep#(\d+)")
#: Streaming read chunk size.
_CHUNK_BYTES = 1 << 20  # 1 MiB
#: Overlap kept between chunks so a marker split across the boundary still matches.
_OVERLAP_BYTES = 64


def resolve_perstep_divisor(requested_steps: int | None, inferred_steps: int | None) -> int | None:
    """Return the denoise-step count to divide workload totals by for per-step."""
    req = int(requested_steps or 0)
    if req > 0:
        return req
    inf = int(inferred_steps or 0)
    return inf or None


def count_profiler_steps(trace_path: str) -> int:
    """Count distinct torch ``ProfilerStep#N`` iterations in a trace."""
    p = Path(trace_path)
    if p.is_dir():
        candidates = sorted(p.glob("*.pt.trace.json.gz")) + sorted(p.glob("*.json.gz")) + sorted(p.glob("*.json"))
        if not candidates:
            return 0
        p = candidates[0]
    steps: set[bytes] = set()
    try:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rb") as fh:
            carry = b""
            while True:
                chunk = fh.read(_CHUNK_BYTES)
                if not chunk:
                    break
                buf = carry + chunk
                for m in _PROFILER_STEP_RE.finditer(buf):
                    steps.add(m.group(1))  # set dedups matches re-seen in overlap
                carry = buf[-_OVERLAP_BYTES:]
    except OSError:
        return 0
    return len(steps)
