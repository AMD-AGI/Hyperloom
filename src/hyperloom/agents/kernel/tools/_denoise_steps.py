"""Shared helpers for the diffusion per-denoise-step roofline divisor.

The workload-level ``diffusion_roofline.json`` reports per-denoise-step timings
as ``workload_totals / num_denoise_steps``. An operator-declared
``--num-denoise-steps`` is authoritative for that divisor: Hyperloom cannot know
what a user's ``prof.step()`` brackets, so a count inferred from the trace is
only a fallback for when nothing was declared. Both the bypass and TraceLens
routes apply that same precedence, so the per-step figure depends on the
workload rather than on which route ran.

Note the fallbacks themselves still differ -- bypass uses its steady-state
window's step count, TraceLens the deduplicated ``ProfilerStep#N`` count -- so
route independence holds only while a count was requested.

Kept dependency-free (stdlib only) so both routes can import it freely.
"""

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
    """Return the denoise-step count to divide workload totals by for per-step.

    Prefers an explicitly requested count over the one inferred from the trace:
    Hyperloom cannot know what an operator's ``prof.step()`` brackets, so a
    declared ``--num-denoise-steps`` is authoritative. Both analysis routes use
    this same precedence, so a per-step figure depends on the workload rather
    than on which route ran. The bypass route warns when the two disagree.

    Args:
        requested_steps: The operator-declared denoise-step count.
        inferred_steps: Denoise steps detected in the analyzed window/trace.

    Returns:
        The positive divisor to use, or ``None`` when neither is known (the
        caller then omits the per-step block).
    """
    req = int(requested_steps or 0)
    if req > 0:
        return req
    inf = int(inferred_steps or 0)
    return inf or None


def count_profiler_steps(trace_path: str) -> int:
    """Count distinct torch ``ProfilerStep#N`` iterations in a trace.

    A lightweight, JSON-parse-free scan (regex over the decompressed bytes) so it
    stays cheap on large traces. Streams in bounded chunks (with a small overlap
    so a marker split across a chunk boundary still matches) rather than reading
    the whole decompressed trace into memory. Both trace routes use it as the
    inferred per-step divisor for the scriptable (diffusion) workload roofline,
    where an operator-declared step count wins when there is one. Accepts a file
    or a directory (first trace).

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
