# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-level shares of a TraceLens report window, read from the window's own GPU events.

TraceLens' ``ops_summary.csv`` ranks aten ops, and on a decode window after a CUDA-graph replay its durations are not the
GPU's: the ROCm profiler stamps the eager kernels queued behind a replay (ROCm/rocm-systems#13882) as starting while the
replay still runs on their one in-order stream and ending at the window's end, ~9 ms each in a Qwen3-14B-FP8 window whose
real kernels last 50 to 80 us in all. The ops table then lists ``index_put``, ``copy_``, ``greedy_sample`` and ``add`` at
15 to 30% each, summing past 100% with the real work (paged attention, the FlyDSL GEMMs) under 6%. The window's raw
events are right: the replay's kernels keep their names, order and durations. So the shares a brief reads are computed from
them, per kernel, and sum to 100% of the window's kernel time.

An event is **excluded** when it overlaps another event of its stream by more than :data:`JITTER_US` (real windows overlap
by 0.07 to 0.63 us: the profiler's clocks, not a defect): a stream runs one kernel at a time, so one of the two has wrong
timestamps. The shorter stays, because a misreported event claims more than it ran: the one that spans hundreds of real
kernels is the one dropped, and so is a long one that overlaps a short real kernel by a few microseconds. The report states
``excluded_events`` and ``excluded_claimed_ms``, the time those events claim.

Leaving them out biases the shares only by what they really cost, which the window cannot say (its timestamps for them are
the defect), so the rest of the rank trace does: the median duration of the same kernel among the trace's events that are not
excluded, in the windows without a replay in flight. The window is **degraded**, with the reason, and gives no shares, when

* it has no GPU kernel events, or the events kept fill under :data:`MIN_GPU_BUSY_PERCENT` of its GPU span: the kernels of
  its graph replays are missing from the trace, so what is left is the eager ops around them, not the step;
* an excluded kernel has no clean instance anywhere in the trace, so what it costs is unknown; or
* the excluded events' estimated cost is over :data:`MAX_EXCLUDED_SHARE` of the kept kernel time: the shares could rank
  wrongly, because every kept share is that much too high and an excluded kernel might outrank a listed one.

A kernel is named by its function, with the template arguments and the parameter list dropped; a hipBLASLt (Tensile)
kernel by its signature, without the tile (``_MT...``) that makes every shape its own name. A name that several modules
launch (the four FlyDSL GEMMs of a layer are all ``kernel_gemm_0``) is split, ``name [Module.Submodule]``, when the events
say which module launched each (:mod:`~.kernel_launchers`); an event without a known launcher keeps the plain name.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import json
import math
import re
import statistics
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_multibranch.toolbox.tools.op_shares import NAME_COLUMN, SHARE_COLUMN

#: The ``cat`` of the trace events that run on the GPU.
GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
#: How much earlier than the previous event's end of its stream an event may start without being excluded, in microseconds.
#: The overlaps of the real windows are 0.07 to 0.63 us (a 5.4 ms GEMM that starts 0.63 us before its predecessor ended);
#: the defect's are a replay's length, 1.9 to 8 ms.
JITTER_US = 5.0
#: What real windows fill of their GPU span: a fully recorded prefill window 99%; the 32-step mixes of a Qwen3-14B-FP8 root
#: profile and of a Qwen3-8B one, which lose the kernels of 29 and 30 of their decode replays but keep a prefill step, 54% and
#: 16%; the single decode step of a Qwen3-14B-FP8 branch profile, whose one replay lost its kernels, 0.6%. Under a tenth of
#: the span the kernels are not the window's work.
MIN_GPU_BUSY_PERCENT = 10.0
#: The excluded events' estimated cost, as a share of the kept kernel time, above which the shares are not given: the
#: Qwen3-14B-FP8 decode window's six excluded kernels cost 23 us of its 7.9 ms (0.3%), a graph replay with its eager tail.
MAX_EXCLUDED_SHARE = 0.05
KERNEL_SHARES_FILE = "kernel_shares.csv"
_LABEL_CHARS = 120
_LAUNCHER_CHARS = 60
_US_PER_MS = 1000.0
_FUNCTION = re.compile(r"(?:\(anonymous namespace\)|[^<(])+")
_MANGLED = re.compile(r"_Z(?:N[KV]*)?(?=\d)")
_LENGTH = re.compile(r"\d+")
_TENSILE_TILE = re.compile(r"_MT\d.*$")
_ENCODING = "utf-8"
_UNREADABLE = (OSError, EOFError, ValueError, zlib.error)


class KernelTraceError(Exception):
    """A trace that cannot be read as Chrome trace events."""


@dataclass(frozen=True, slots=True)
class GpuEvent:
    """One GPU event of a trace: the ``(device, stream)`` that ran it, microseconds, and the kernel's raw name.

    ``correlation`` ties it to the host call that launched it (a graph launch's kernels share one); ``launcher`` is the
    module that launched it, empty while unknown.
    """

    stream: tuple[str, str]
    start: float
    end: float
    name: str
    correlation: int | None = None
    category: str = "kernel"
    launcher: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def gpu_event(raw: Any) -> GpuEvent | None:
    if not isinstance(raw, dict) or raw.get("ph") != "X" or raw.get("cat") not in GPU_CATEGORIES:
        return None
    start, duration, name = _number(raw.get("ts")), _number(raw.get("dur")), raw.get("name")
    if start is None or duration is None or duration < 0 or not isinstance(name, str):
        return None
    args = raw.get("args")
    stream = str(args.get("stream")) if isinstance(args, dict) else ""
    correlation = _number(args.get("correlation")) if isinstance(args, dict) else None
    return GpuEvent(
        (str(raw.get("pid")), stream),
        start,
        start + duration,
        name,
        int(correlation) if correlation is not None else None,
        raw["cat"],
    )


def trace_events(path: Path) -> list[Any]:
    """The ``traceEvents`` of the Chrome trace at ``path`` (``.json`` or ``.json.gz``), as the file holds them."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding=_ENCODING, errors="replace") as handle:
            payload = json.load(handle)
    except _UNREADABLE as exc:
        raise KernelTraceError(f"{path.name} is not a readable trace: {exc!r}") from None
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise KernelTraceError(f"{path.name} has no traceEvents list")
    return events


def read_gpu_events(path: Path) -> list[GpuEvent]:
    """The GPU events (kernels, copies, memsets) of the Chrome trace at ``path`` (``.json`` or ``.json.gz``)."""
    return [event for event in map(gpu_event, trace_events(path)) if event is not None]


@dataclass(frozen=True)
class InOrder:
    """The events of a trace split by the stream order: those that run one after another, and the ones that overlap them."""

    kept: list[GpuEvent]
    excluded: list[GpuEvent]


def _runs(events: list[GpuEvent]) -> Iterator[list[GpuEvent]]:
    """The events of one stream by start, cut wherever one starts no earlier than the jitter before all the ones before it
    ended: a stream in order is a series of runs of one event, and only a run of several holds a conflict."""
    run: list[GpuEvent] = []
    reach = -math.inf
    for event in sorted(events, key=lambda e: (e.start, e.end)):
        if run and event.start >= reach - JITTER_US:
            yield run
            run, reach = [], -math.inf
        run.append(event)
        reach = max(reach, event.end)
    if run:
        yield run


def _keep_shortest(run: list[GpuEvent], into: InOrder) -> None:
    """Settle one run, shortest event first: an event stays unless it overlaps one already kept."""
    kept: list[GpuEvent] = []
    for event in sorted(run, key=lambda e: (e.duration, e.start)):
        at = bisect.bisect_left(kept, event.start, key=lambda e: e.start)
        before, after = (kept[at - 1] if at else None), (kept[at] if at < len(kept) else None)
        if (before and before.end - JITTER_US > event.start) or (after and event.end - JITTER_US > after.start):
            into.excluded.append(event)
        else:
            kept.insert(at, event)
    into.kept.extend(kept)


def split_in_order(events: Iterable[GpuEvent]) -> InOrder:
    """Per stream, the events that run one after another, and the ones that overlap them by more than :data:`JITTER_US`.

    Of events that overlap, the shortest stay: a misreported event claims more than it ran, so the real kernels it spans
    outrank it however few they are (even one that ends a microsecond after it), and the one that spans hundreds is the one
    dropped.
    """
    by_stream: dict[tuple[str, str], list[GpuEvent]] = defaultdict(list)
    for event in events:
        by_stream[event.stream].append(event)
    ordered = InOrder([], [])
    for stream_events in by_stream.values():
        for run in _runs(stream_events):
            if len(run) == 1:
                ordered.kept.extend(run)
            else:
                _keep_shortest(run, ordered)
    return ordered


def kernel_label(name: str) -> str:
    """The kernel's function name: no ``void``, template arguments or parameters; a Tensile kernel without its tile."""
    text = name.strip()
    mangled = _MANGLED.match(text)
    if mangled:
        parts: list[str] = []
        rest = text[mangled.end() :]
        while (length := _LENGTH.match(rest)) is not None:
            end = length.end() + int(length.group())
            parts.append(rest[length.end() : end])
            rest = rest[end:]
        label = "::".join(parts)
    else:
        found = _FUNCTION.match(text.removeprefix("void "))
        label = found.group().strip() if found else ""
    if label.startswith("Cijk_"):
        label = _TENSILE_TILE.sub("", label)
    return (label or text)[:_LABEL_CHARS]


def _busy_us(events: Iterable[GpuEvent]) -> float:
    """The time at least one of ``events`` runs: the union of their intervals."""
    busy, reach = 0.0, -math.inf
    for event in sorted(events, key=lambda e: e.start):
        if event.end > reach:
            busy += event.end - max(event.start, reach)
            reach = event.end
    return busy


@dataclass(frozen=True)
class KernelRow:
    """One kernel of the window: its label, its share of the kept kernel time, that time in ms, and how many ran."""

    name: str
    percent: float
    time_ms: float
    count: int


@dataclass(frozen=True)
class KernelWindow:
    """What a window's GPU events make of its kernel shares; see the module docstring for the rules."""

    rows: tuple[KernelRow, ...]
    kernel_ms: float
    span_ms: float
    busy_percent: float
    excluded_events: int
    excluded_claimed_ms: float
    excluded_estimated_ms: float
    unbounded: tuple[str, ...]

    @property
    def degraded_reason(self) -> str:
        """Why the window gives no shares, empty when it does."""
        return self.starved_reason or self.excluded_reason

    @property
    def starved_reason(self) -> str:
        if not self.rows:
            return "the window holds no GPU kernel events in order, so there is no step to take shares of"
        if self.busy_percent >= MIN_GPU_BUSY_PERCENT:
            return ""
        return (
            f"the window's recorded GPU kernels fill {self.busy_percent:.1f}% of its {self.span_ms:.2f} ms GPU span (under "
            f"{MIN_GPU_BUSY_PERCENT:g}%): the kernels of its CUDA-graph replays are missing from the trace, so what is "
            "left is the eager ops around them"
        )

    @property
    def excluded_reason(self) -> str:
        if not self.excluded_events:
            return ""
        left_out = (
            f"{self.excluded_events} GPU events start before the previous event of their stream ended (they claim "
            f"{self.excluded_claimed_ms:.2f} ms), so the profiler's timestamps are wrong for them and they are left out; "
        )
        if self.unbounded:
            return f"{left_out}the trace holds no undisturbed instance of {', '.join(self.unbounded)} to say what they cost"
        if self.excluded_estimated_ms > MAX_EXCLUDED_SHARE * self.kernel_ms:
            return (
                f"{left_out}they are estimated at {self.excluded_estimated_ms:.2f} ms, over {MAX_EXCLUDED_SHARE:.0%} of "
                f"the other kernels' {self.kernel_ms:.2f} ms, so the shares could rank wrongly"
            )
        return ""

    @property
    def shares(self) -> dict[str, float]:
        """The percent of the kept kernel time of each kernel label, the largest first."""
        return {row.name: row.percent for row in self.rows}


def _launcher_split(labels: list[tuple[str, str]]) -> set[str]:
    """The kernel labels that more than one module launched: only those are split by launcher."""
    launchers: dict[str, set[str]] = defaultdict(set)
    for label, launcher in labels:
        if launcher:
            launchers[label].add(launcher)
    return {label for label, modules in launchers.items() if len(modules) > 1}


def _rows(kept: list[GpuEvent]) -> tuple[KernelRow, ...]:
    labels = [(kernel_label(event.name), event.launcher) for event in kept]
    split = _launcher_split(labels)
    time_us: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for event, (label, launcher) in zip(kept, labels, strict=True):
        key = f"{label} [{launcher[:_LAUNCHER_CHARS]}]" if launcher and label in split else label
        time_us[key] += event.duration
        count[key] += 1
    total = math.fsum(time_us.values())
    if total <= 0:
        return ()
    ranked = sorted(time_us.items(), key=lambda item: (-item[1], item[0]))
    return tuple(KernelRow(name, 100.0 * us / total, us / _US_PER_MS, count[name]) for name, us in ranked)


@dataclass(frozen=True)
class ExcludedCost:
    """What excluded events really cost, in ms, and the labels of the kernels with no undisturbed instance to say."""

    time_ms: float
    unbounded: tuple[str, ...]


def _excluded_cost(excluded: list[GpuEvent], reference: Callable[[], Iterable[GpuEvent]]) -> ExcludedCost:
    """The excluded events' cost: each at the median of its kernel's undisturbed instances in the whole rank trace."""
    durations: dict[str, list[float]] = defaultdict(list)
    for event in split_in_order(reference()).kept:
        durations[event.name].append(event.duration)
    cost_us = 0.0
    unknown: set[str] = set()
    for event in excluded:
        if event.name in durations:
            cost_us += statistics.median(durations[event.name])
        else:
            unknown.add(kernel_label(event.name))
    return ExcludedCost(cost_us / _US_PER_MS, tuple(sorted(unknown)))


def kernel_window(window: list[GpuEvent], reference: Callable[[], Iterable[GpuEvent]]) -> KernelWindow:
    """The kernel shares of the GPU events of one window; ``reference`` reads the whole rank trace's, only if needed."""
    ordered = split_in_order(window)
    rows = _rows(ordered.kept)
    span_us = max((e.end for e in window), default=0.0) - min((e.start for e in window), default=0.0)
    cost = _excluded_cost(ordered.excluded, reference) if ordered.excluded else ExcludedCost(0.0, ())
    return KernelWindow(
        rows=rows,
        kernel_ms=math.fsum(row.time_ms for row in rows),
        span_ms=span_us / _US_PER_MS,
        busy_percent=100.0 * _busy_us(ordered.kept) / span_us if span_us > 0 else 0.0,
        excluded_events=len(ordered.excluded),
        excluded_claimed_ms=math.fsum(event.duration for event in ordered.excluded) / _US_PER_MS,
        excluded_estimated_ms=cost.time_ms,
        unbounded=cost.unbounded,
    )


def write_kernel_table(window: KernelWindow, path: Path) -> None:
    """The kernels of ``window`` with their shares, the table the op shares of the tool's pointer are read from."""
    with path.open("w", encoding=_ENCODING, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((NAME_COLUMN, SHARE_COLUMN, "total_time_ms", "count"))
        writer.writerows((row.name, f"{row.percent:.6f}", f"{row.time_ms:.6f}", row.count) for row in window.rows)


__all__ = [
    "JITTER_US",
    "KERNEL_SHARES_FILE",
    "MAX_EXCLUDED_SHARE",
    "MIN_GPU_BUSY_PERCENT",
    "GpuEvent",
    "InOrder",
    "KernelRow",
    "KernelTraceError",
    "KernelWindow",
    "gpu_event",
    "kernel_label",
    "kernel_window",
    "read_gpu_events",
    "split_in_order",
    "trace_events",
    "write_kernel_table",
]
