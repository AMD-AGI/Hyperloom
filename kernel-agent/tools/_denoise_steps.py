"""Shared helpers for the diffusion per-denoise-step roofline divisor.

The workload-level ``diffusion_roofline.json`` reports per-denoise-step timings
as ``workload_totals / num_denoise_steps``. The divisor MUST be the number of
denoise steps ACTUALLY IN the analyzed data (the steady-state window for bypass,
or the profiled iterations in the trace for the TraceLens route), NOT the
requested full sampler schedule -- otherwise the per-step figure is off by the
ratio of scheduled-to-profiled steps and is not comparable across routes.

Kept dependency-free (stdlib only) so both routes can import it freely.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

_PROFILER_STEP_RE = re.compile(rb"ProfilerStep#(\d+)")


def resolve_perstep_divisor(inferred_steps: int | None, requested_steps: int | None) -> int | None:
    """Return the denoise-step count to divide workload totals by for per-step.

    Prefers the steps ACTUALLY IN the analyzed window/trace (``inferred_steps``)
    over the requested full schedule (``requested_steps``), so per-step timings
    reflect the profiled data -- consistent with the denoise-step-mismatch
    warning the bypass route emits when the two disagree.

    Args:
        inferred_steps: Denoise steps detected in the analyzed window/trace.
        requested_steps: The requested full sampler schedule step count.

    Returns:
        The positive divisor to use, or ``None`` when neither is known (the
        caller then omits the per-step block).
    """
    inf = int(inferred_steps or 0)
    if inf > 0:
        return inf
    req = int(requested_steps or 0)
    return req or None


def count_profiler_steps(trace_path: str) -> int:
    """Count distinct torch ``ProfilerStep#N`` iterations in a trace.

    A lightweight, JSON-parse-free scan (regex over the decompressed bytes) so it
    stays cheap on large traces. Used as the per-step divisor source for the
    TraceLens deterministic route, which (unlike bypass) does not run its own
    steady-window step detection. Accepts a file or a directory (first trace).

    Args:
        trace_path: Path to a ``.json`` / ``.json.gz`` trace file or a directory.

    Returns:
        The number of distinct ProfilerStep indices, or ``0`` when none are found
        or the trace is unreadable.
    """
    p = Path(trace_path)
    if p.is_dir():
        candidates = sorted(p.glob("*.pt.trace.json.gz")) + sorted(p.glob("*.json.gz")) + sorted(p.glob("*.json"))
        if not candidates:
            return 0
        p = candidates[0]
    try:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rb") as fh:
            data = fh.read()
    except OSError:
        return 0
    return len({m.group(1) for m in _PROFILER_STEP_RE.finditer(data)})
