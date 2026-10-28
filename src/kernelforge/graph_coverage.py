# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which steps replay a captured graph and which still launch every kernel one at a time.

A kernel replayed from a graph carries no launch call of its own; an eagerly dispatched one does, and the trace
records it. Counting which side of that line each step class falls on says where the per-launch cost is still
being paid -- and the answer is rarely uniform: a GLM-5.3 AgentX capture runs target decode 99.8% from a graph
while prefill and draft extend launch 100% of their kernels individually, 4958 of them in one prefill step.

That split is worth naming because closing it was one of the larger steps in the hand-tuned stack: putting prefill
behind a breakable graph was +8.75% on its own, and the draft extend was still eager after it because NextN has no
``layers`` for the capture to walk.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from kernelforge.serving_windows import StepWindow, step_windows
from kernelforge.trace_events import KERNEL_CAT, load_trace_events

#: Runtime calls that dispatch one kernel. A kernel whose correlation id matches one of these was launched for
#: that call; a kernel with no match came out of a graph replay.
_LAUNCH_MARKER: str = "LaunchKernel"

#: Share of a class's kernel time that must be eager before the class is called ungraphed. Well below the 100% a
#: GLM-5.3 prefill step shows, and far above the 0.2% its graphed decode leaves behind.
UNGRAPHED_SHARE: float = 0.5


def _launch_correlations(events: list[dict[str, Any]]) -> set[Any]:
    """Correlation ids that had their own kernel-launch call."""
    return {
        (event.get("args") or {}).get("correlation")
        for event in events
        if event.get("cat") == "cuda_runtime" and _LAUNCH_MARKER in str(event.get("name") or "")
    }


class _KernelTime(NamedTuple):
    kernels: int
    eager: int
    total_us: float
    eager_us: float


def _kernel_time_in(events: list[dict[str, Any]], spans: list[tuple[float, float]], launched: set[Any]) -> _KernelTime:
    """Kernel counts and time of the kernels starting inside *spans*, and the eagerly launched share of them."""
    kernels = eager = 0
    total_us = eager_us = 0.0
    for event in events:
        if event.get("cat") != KERNEL_CAT:
            continue
        start, duration = event.get("ts"), event.get("dur")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            continue
        if not any(lo <= start < hi for lo, hi in spans):
            continue
        kernels += 1
        total_us += float(duration)
        if (event.get("args") or {}).get("correlation") in launched:
            eager += 1
            eager_us += float(duration)
    return _KernelTime(kernels, eager, total_us, eager_us)


def graph_coverage(trace_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Per window class, how much of its kernel time replays a graph rather than launching per kernel."""
    if not str(trace_path or "").strip():
        return {}
    events = load_trace_events(trace_path)
    windows = step_windows(events)
    if not windows:
        return {}
    launched = _launch_correlations(events)
    if not launched:
        return {}
    by_class: dict[str, list[StepWindow]] = defaultdict(list)
    for window in windows:
        by_class[window.window_class].append(window)
    out: dict[str, dict[str, Any]] = {}
    for window_class, members in by_class.items():
        spans = sorted((w.start_us, w.end_us) for w in members)
        kernels, eager, total_us, eager_us = _kernel_time_in(events, spans, launched)
        if not kernels:
            continue
        eager_time_share = eager_us / total_us if total_us else 0.0
        out[window_class] = {
            "is_decode": members[0].is_decode,
            "kernels": kernels,
            "eager_kernels": eager,
            "eager_kernels_per_step": round(eager / max(1, len(members)), 1),
            "eager_time_pct": round(eager_time_share * 100.0, 1),
            "ungraphed": bool(eager_time_share >= UNGRAPHED_SHARE),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["eager_time_pct"], reverse=True))


def render_graph_coverage(by_class: dict[str, dict[str, Any]] | None) -> list[str]:
    """Render which step classes still launch every kernel, and what that costs them."""
    if not by_class:
        return []
    lines = [
        "| step class | phase | kernels | eager | eager launches/step |",
        "|---|---|---:|---:|---:|",
    ]
    for window_class, row in by_class.items():
        phase = "decode" if row["is_decode"] else "prefill"
        flag = " ⚠" if row["ungraphed"] else ""
        lines.append(
            f"| `{window_class}` | {phase} | {row['kernels']} | {row['eager_time_pct']:.1f}%{flag} | "
            f"{row['eager_kernels_per_step']:.0f} |"
        )
    if any(row["ungraphed"] for row in by_class.values()):
        lines.extend(
            [
                "",
                "A class marked ⚠ pays a launch for every kernel it runs. Capturing it costs nothing the kernels "
                "do, and a class that cannot be captured is worth saying why: a module with no `layers` for the "
                "capture to walk is a different problem from one the backend refuses on this platform.",
            ]
        )
    return lines


__all__ = [
    "UNGRAPHED_SHARE",
    "graph_coverage",
    "render_graph_coverage",
]
