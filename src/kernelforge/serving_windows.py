# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which part of a serving trace a kernel ranking is about.

An agentic serving trace has no single steady state. One GLM-5.3 AgentX capture holds prefill steps of 660 ms
beside decode steps of 14 ms -- a factor of 46 -- so one ranking over the whole trace is a ranking of prefill, and
says nothing about the decode it is 46 times larger than. Which is why the same operator needs two patches: the
hand-tuned GLM-5.3 stack carries ``aiter_fmoe_cfg_ep4`` for decode MoE and ``aiter_fmoe_prefill_ep4`` for prefill
MoE, one tuned table each.

sglang annotates every step with its kind and batch size (``step[VERIFY bs=3]``), and the device timeline carries
the same annotations as the kernels, so the windows are already in the trace: group by what the step is, rank
inside each group, and a reader can tell a decode hotspot from a prefill one.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.trace_events import GPU_ANNOTATION_CAT, KERNEL_CAT, load_trace_events

#: ``step[KIND bs=N ...]``. The tail carries per-step quantities (``toks=``, ``c_sq=``) that make every prefill
#: step's name unique, which is why the class is built from the two leading fields rather than from the name.
_STEP_RE = re.compile(r"^step\[(?P<kind>[A-Z_]+)\s+bs=(?P<bs>\d+)")

#: Step kinds that run one decode cycle. Prefill (``EXTEND``) is kept as its own class, never merged into these:
#: it is where the indexer and prefill-MoE wins live, and it would otherwise drown them.
DECODE_KINDS: frozenset[str] = frozenset({"VERIFY", "DRAFT", "DECODE", "IDLE"})

#: Microseconds per millisecond; the traces are in us and the report is in ms.
_US_PER_MS: float = 1000.0
#: Decimal places every millisecond figure in this module is reported to.
_REPORTED_PLACES: int = 3
#: Kernels named per window class.
TOP_N: int = 15


@dataclass(frozen=True)
class StepWindow:
    """One annotated step on the device timeline."""

    kind: str
    batch_size: int
    start_us: float
    end_us: float

    @property
    def window_class(self) -> str:
        """The group this step is ranked within: what it does and how wide it is."""
        return f"{self.kind} bs={self.batch_size}"

    @property
    def is_decode(self) -> bool:
        return self.kind in DECODE_KINDS


def step_windows(events: list[dict[str, Any]]) -> list[StepWindow]:
    """Annotated steps on the device timeline, in time order.

    Device-side only: the host records the same annotations, but their duration is how long submission took, not
    how long the step ran -- on the GLM-5.3 capture a decode step reads 0.5 ms on the host and 14.3 ms on the
    device -- and only the device timeline shares a clock with the kernels.
    """
    out: list[StepWindow] = []
    for event in events:
        if event.get("cat") != GPU_ANNOTATION_CAT:
            continue
        match = _STEP_RE.match(str(event.get("name") or ""))
        if not match:
            continue
        start, duration = _number(event.get("ts")), _number(event.get("dur"))
        if start is None or duration is None or duration <= 0:
            continue
        out.append(
            StepWindow(
                kind=match.group("kind"),
                batch_size=int(match.group("bs")),
                start_us=start,
                end_us=start + duration,
            )
        )
    out.sort(key=lambda w: w.start_us)
    return out


def _number(value: Any) -> float | None:
    """Coerce a trace timestamp or duration to float, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _kernels_in(events: list[dict[str, Any]], windows: list[StepWindow]) -> dict[str, float]:
    """Total kernel microseconds per name, counting only kernels inside *windows*."""
    spans = sorted((w.start_us, w.end_us) for w in windows)
    totals: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("cat") != KERNEL_CAT:
            continue
        start, duration = _number(event.get("ts")), _number(event.get("dur"))
        name = str(event.get("name") or "")
        if start is None or duration is None or duration <= 0 or not name:
            continue
        # A kernel belongs to the step it starts in; one that outlives its step is still that step's work.
        if any(lo <= start < hi for lo, hi in spans):
            totals[name] += duration
    return totals


def hot_kernels_by_class(trace_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Per window class, the kernels that spent the most device time in it, worst first.

    Empty when the trace carries no step annotations, which is what a capture taken outside a serving loop looks
    like: the caller is told nothing rather than handed a ranking of everything.
    """
    if not str(trace_path or "").strip():
        return {}
    events = load_trace_events(trace_path)
    windows = step_windows(events)
    if not windows:
        return {}
    by_class: dict[str, list[StepWindow]] = defaultdict(list)
    for window in windows:
        by_class[window.window_class].append(window)
    out: dict[str, dict[str, Any]] = {}
    for window_class, members in by_class.items():
        totals = _kernels_in(events, members)
        if not totals:
            continue
        total_us = sum(totals.values())
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]
        out[window_class] = {
            "steps": len(members),
            "is_decode": members[0].is_decode,
            "step_ms_median": round(_median([(w.end_us - w.start_us) / _US_PER_MS for w in members]), _REPORTED_PLACES),
            "kernel_ms_total": round(total_us / _US_PER_MS, _REPORTED_PLACES),
            "kernels": [
                {"name": name, "ms": round(us / _US_PER_MS, _REPORTED_PLACES), "pct": round(us / total_us * 100.0, 2)}
                for name, us in ranked
            ],
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["kernel_ms_total"], reverse=True))


def _median(values: list[float]) -> float:
    """Median of a non-empty list."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


#: A rendered kernel name is cut here. A templated CK symbol runs past 400 characters of template arguments, and
#: this document is read by an agent on a context budget; what identifies the kernel is at the front.
_NAME_CHARS: int = 110


def _short(name: str) -> str:
    """A kernel name cut to what identifies it."""
    return name if len(name) <= _NAME_CHARS else name[: _NAME_CHARS - 1] + "…"


def render_hot_kernels_by_class(by_class: dict[str, dict[str, Any]] | None, *, top_n: int = 8) -> list[str]:
    """Render one ranking per window class, with what each class is, as report lines."""
    if not by_class:
        return []
    lines: list[str] = []
    for window_class, data in by_class.items():
        phase = "decode" if data["is_decode"] else "prefill"
        lines.extend(
            [
                f"### {window_class} ({phase})",
                "",
                f"{data['steps']} step(s), median {data['step_ms_median']} ms, "
                f"{data['kernel_ms_total']} ms of kernel time in class",
                "",
                "| share | ms | kernel |",
                "|---:|---:|---|",
            ]
        )
        lines.extend(f"| {k['pct']:.1f}% | {k['ms']:.2f} | `{_short(k['name'])}` |" for k in data["kernels"][:top_n])
        lines.append("")
    lines.append(
        "Rankings are per class because the classes are not comparable: a prefill step runs tens of times "
        "longer than a decode step, so one ranking over the whole trace is a ranking of prefill. Optimise "
        "against the class the work is meant to move."
    )
    return lines


__all__ = [
    "DECODE_KINDS",
    "TOP_N",
    "StepWindow",
    "hot_kernels_by_class",
    "render_hot_kernels_by_class",
    "step_windows",
]
