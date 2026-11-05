# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernels that cost a launch rather than the work they do, and the neighbour each could fold into.

Folding a microsecond-scale kernel into the neighbour it always runs beside removes a launch without changing what
is computed. Candidates are found by the shape of their launches (many per step, microseconds each, the same
neighbours every time), not by name: the kernel categoriser misreads some reductions as attention.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kernelforge.evaluation.profiling.serving_windows import StepWindow, kernels_in, step_windows
from kernelforge.evaluation.profiling.trace_events import load_trace_events

from .diagnose import categorize_kernel_name

#: Above this a kernel is doing work worth its own launch: a few multiples of the ~1.5 us empty-kernel floor.
MAX_GLUE_US: float = 6.0

#: Below this the launches are not what the step is spending itself on, whatever each one costs.
MIN_LAUNCHES_PER_STEP: float = 8.0

#: Share of a kernel's launches that must sit beside the same neighbour before that neighbour is named as the one
#: to fold into.
MIN_NEIGHBOUR_SHARE: float = 0.9

#: Candidates named per window class.
TOP_N: int = 8

#: Decimal places a share or a millisecond figure is reported to.
_REPORTED_PLACES: int = 3


def _dominant(counter: Counter, total: int) -> dict[str, Any] | None:
    """The neighbour that holds at least :data:`MIN_NEIGHBOUR_SHARE` of the launches, or ``None``."""
    if not counter or total <= 0:
        return None
    name, hits = counter.most_common(1)[0]
    share = hits / total
    return {"name": name, "share": round(share, _REPORTED_PLACES)} if share >= MIN_NEIGHBOUR_SHARE else None


def foldable_in_class(events: list[dict[str, Any]], windows: list[StepWindow]) -> list[dict[str, Any]]:
    """Launch-bound kernels in one window class, costliest first."""
    ordered = sorted(kernels_in(events, windows), key=lambda e: float(e["ts"]))
    if not ordered:
        return []
    steps = max(1, len(windows))
    totals: dict[str, float] = defaultdict(float)
    counts: Counter = Counter()
    before: dict[str, Counter] = defaultdict(Counter)
    after: dict[str, Counter] = defaultdict(Counter)
    for index, event in enumerate(ordered):
        name = str(event.get("name") or "")
        if not name:
            continue
        counts[name] += 1
        totals[name] += float(event["dur"])
        if index:
            before[name][str(ordered[index - 1].get("name") or "")] += 1
        if index + 1 < len(ordered):
            after[name][str(ordered[index + 1].get("name") or "")] += 1
    rows: list[dict[str, Any]] = []
    for name, launches in counts.items():
        per_step = launches / steps
        mean_us = totals[name] / launches
        # A collective's time is the transfer and the wait for the other ranks, not a launch.
        if mean_us > MAX_GLUE_US or per_step < MIN_LAUNCHES_PER_STEP or categorize_kernel_name(name) == "collective":
            continue
        rows.append(
            {
                "kernel": name,
                "launches_per_step": round(per_step, 1),
                "us_per_call": round(mean_us, 2),
                "ms_per_step": round(totals[name] / steps / 1000.0, _REPORTED_PLACES),
                # What comparing across classes needs: a prefill step outweighs a decode step, the capture's
                # decode steps outnumber its prefill ones.
                "ms_in_capture": round(totals[name] / 1000.0, _REPORTED_PLACES),
                "after": _dominant(before[name], launches),
                "before": _dominant(after[name], launches),
            }
        )
    rows.sort(key=lambda row: row["ms_per_step"], reverse=True)
    return rows[:TOP_N]


def foldable_launches(trace_path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """Per window class, the launches worth folding away, costliest first."""
    if not str(trace_path or "").strip():
        return {}
    events = load_trace_events(trace_path)
    windows = step_windows(events)
    if not windows:
        return {}
    by_class: dict[str, list[StepWindow]] = defaultdict(list)
    for window in windows:
        by_class[window.window_class].append(window)
    out = {name: foldable_in_class(events, members) for name, members in by_class.items()}
    return {name: rows for name, rows in out.items() if rows}


def render_foldable_launches(by_class: dict[str, list[dict[str, Any]]] | None, *, name_chars: int = 70) -> list[str]:
    """Render the launch census, naming what each candidate could fold into."""
    if not by_class:
        return []

    def _short(name: str) -> str:
        return name if len(name) <= name_chars else name[: name_chars - 1] + "…"

    def _side(entry: Any, label: str) -> str:
        if not entry:
            return ""
        return f"{label} `{_short(entry['name'])}` ({entry['share']:.0%})"

    lines: list[str] = []
    for window_class, rows in by_class.items():
        lines.extend(
            [
                f"### {window_class}",
                "",
                "| ms/step | launches/step | us/call | kernel | always between |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for row in rows:
            neighbours = ", ".join(x for x in (_side(row["after"], "after"), _side(row["before"], "before")) if x)
            lines.append(
                f"| {row['ms_per_step']:.3f} | {row['launches_per_step']:.0f} | {row['us_per_call']:.1f} | "
                f"`{_short(row['kernel'])}` | {neighbours or 'no single neighbour'} |"
            )
        lines.append("")
    lines.append(
        "Each row costs a launch per call. One that always sits between the same two kernels can be folded into "
        "either of them without changing what is computed; one with no single neighbour cannot, and is listed "
        "because the launches are still what the step is spending itself on."
    )
    return lines


__all__ = [
    "MAX_GLUE_US",
    "MIN_LAUNCHES_PER_STEP",
    "MIN_NEIGHBOUR_SHARE",
    "TOP_N",
    "foldable_in_class",
    "foldable_launches",
    "render_foldable_launches",
]
