# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The module that launched each kernel of a window, from the host calls and the graph captures its trace records.

A launcher is the innermost ``nn.Module`` python frame (``nn.Module: RowParallelLinear_3``) that was running when the kernel
was launched and the module around it, without the instance numbers: ``Qwen2MLP.RowParallelLinear`` is the MLP's down
projection, ``Qwen3Attention.RowParallelLinear`` the attention's output projection, so a layer's four GEMMs, all
``kernel_gemm_0`` to the profiler, have four launchers.

**An eager kernel** has its launch call in the trace (a ``cuda_runtime`` event with the kernel's correlation id); the
launcher is the module frames of the launching thread that span that call.

**A graph replay's kernels** have none: the window records one ``hipGraphLaunch`` call and the kernels that share its
correlation id, and no frame of the model, which ran once, when the graph was captured. SGLang's profiler writes a trace
per captured batch size into ``graph_capture_profile/`` (the folder TraceLens takes as ``--capture_folder``), indexed by
its ``execution_details.json``. It holds every kernel the capture launched, as a ``hipLaunchKernel``-family call whose
``args.kernel`` is the kernel's name, inside the module frames that called it. The replay's slots take the capture's
launches in order: a graph on one stream runs its kernels in the order they were captured, so the n-th slot is the n-th
launch. That is read only when the replay and the capture hold the same kernels, by name and count, on one stream;
anything else keeps the plain names, with the reason.

The capture's name is taken for the slot, not the window's: the window can record the names of a replay in another order
than its slots run (the Qwen3-14B-FP8 b40 decode replay has 420 of its 604 kernels under the name of a neighbour), and the
slots' times follow the capture. By the capture's names the 40 layers' copies of one GEMM take the same time within 10% and
all four stream their weights at 2.3 to 3.2 TB/s; by the window's names the same ``kernel_gemm_0`` takes 4 to 58 us and a
paged-attention kernel 55 us.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom_multibranch.toolbox.tools.kernel_shares import GpuEvent, KernelTraceError, gpu_event, trace_events

CAPTURE_MANIFEST = "execution_details.json"
#: The host call that launches a captured graph; the kernels of one replay share its correlation id.
GRAPH_LAUNCH = "hipGraphLaunch"
#: The capture mode of a graph that holds a whole step; a piecewise capture (several graphs per step) is not read.
_FULL_GRAPH = "FULL"
_LAUNCH_CALL = re.compile(r"hip\w*LaunchKernel")
_BATCH_SIZE = re.compile(r"\bbs=(\d+)")
_MODULE = re.compile(r"^nn\.Module: (.+?)(?:_\d+)?$")
#: A launcher is the innermost module and the one that holds it.
_PATH_DEPTH = 2
_ENCODING = "utf-8"

Thread = tuple[Any, Any]


@dataclass(frozen=True)
class Replay:
    """A CUDA-graph launch whose kernels the window records: their ``correlation`` id, and the batch size the step
    annotation around the launch call names (``None`` when no annotation does)."""

    correlation: int
    batch_size: int | None


@dataclass(frozen=True)
class TraceWindow:
    """The GPU events of a window (the eager kernels already with their launchers), and the graph launches among them."""

    events: list[GpuEvent]
    replays: tuple[Replay, ...]


@dataclass(frozen=True)
class Launch:
    """One kernel launch of a graph capture: the kernel's name and the module that launched it."""

    kernel: str
    launcher: str


@dataclass(frozen=True)
class Relabelled:
    """A window's events with the launchers found for them, and what finding them changed.

    ``launched`` counts the kernels with a launcher, ``renamed`` those of a replay whose recorded name is not the capture's
    at their slot, ``notes`` the replays left as recorded and why.
    """

    events: list[GpuEvent]
    launched: int
    renamed: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _Span:
    thread: Thread
    start: float
    end: float
    label: str


def _span(event: dict[str, Any], label: str) -> _Span | None:
    start, duration = event.get("ts"), event.get("dur", 0)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (start, duration)):
        return None
    return _Span((event.get("pid"), event.get("tid")), float(start), float(start) + float(duration), label)


def _module_frames(raw: list[Any]) -> dict[Thread, list[_Span]]:
    """The ``nn.Module`` frames of a trace per thread, outermost first among those that start together."""
    frames: dict[Thread, list[_Span]] = defaultdict(list)
    for event in raw:
        named = _MODULE.match(str(event.get("name"))) if isinstance(event, dict) else None
        span = _span(event, named.group(1)) if named and event.get("cat") == "python_function" else None
        if span:
            frames[span.thread].append(span)
    for spans in frames.values():
        spans.sort(key=lambda s: (s.start, s.start - s.end))
    return frames


def _launchers(calls: list[_Span], frames: dict[Thread, list[_Span]]) -> list[str]:
    """The launcher of each call, from the module frames of its thread that span it."""
    open_frames: dict[Thread, list[_Span]] = defaultdict(list)
    seen: dict[Thread, int] = defaultdict(int)
    labels = [""] * len(calls)
    for at in sorted(range(len(calls)), key=lambda i: calls[i].start):
        call = calls[at]
        around, own = open_frames[call.thread], frames[call.thread]
        while seen[call.thread] < len(own) and own[seen[call.thread]].start <= call.start:
            around.append(own[seen[call.thread]])
            seen[call.thread] += 1
        around[:] = [f for f in around if f.end >= call.start]
        labels[at] = ".".join(f.label for f in around[-_PATH_DEPTH:])
    return labels


def _correlation(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    value = args.get("correlation") if isinstance(args, dict) else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _host_calls(raw: list[Any]) -> dict[int, dict[str, Any]]:
    """The host calls of a trace (``cuda_runtime`` events) by the correlation id of what they launched."""
    calls: dict[int, dict[str, Any]] = {}
    for event in raw:
        if isinstance(event, dict) and event.get("cat") == "cuda_runtime":
            correlation = _correlation(event)
            if correlation is not None:
                calls.setdefault(correlation, event)
    return calls


def _steps(raw: list[Any]) -> list[tuple[_Span, int]]:
    """The annotations that name a batch size (``step[DECODE bs=64]``), with the size."""
    steps = []
    for event in raw:
        named = _BATCH_SIZE.search(str(event.get("name"))) if isinstance(event, dict) else None
        span = _span(event, "") if named and event.get("cat") == "user_annotation" else None
        if span and named:
            steps.append((span, int(named.group(1))))
    return steps


def _batch_size(steps: list[tuple[_Span, int]], call: dict[str, Any]) -> int | None:
    """The batch size of the smallest step annotation around ``call`` on its thread."""
    here = _span(call, "")
    around = [
        (s.end - s.start, size)
        for s, size in steps
        if here and s.thread == here.thread and s.start <= here.start <= s.end
    ]
    return min(around)[1] if around else None


def _eager_launchers(raw: list[Any], events: list[GpuEvent], calls: dict[int, dict[str, Any]]) -> list[GpuEvent]:
    """``events`` with the launcher of each kernel whose launch call the trace holds."""
    wanted = sorted({e.correlation for e in events if e.correlation in calls})
    spans = [_span(calls[correlation], "") for correlation in wanted]
    labels = _launchers([s for s in spans if s], _module_frames(raw))
    launcher = dict(zip((c for c, s in zip(wanted, spans, strict=True) if s), labels, strict=True))
    return [replace(e, launcher=launcher.get(e.correlation, "")) if e.correlation in launcher else e for e in events]


def read_window(path: Path) -> TraceWindow:
    """The GPU events of the window at ``path``: eager kernels with their launchers, graph replays listed for the capture."""
    raw = trace_events(path)
    events = [event for event in map(gpu_event, raw) if event is not None]
    calls = _host_calls(raw)
    recorded = {e.correlation for e in events}
    graphs = {c for c in recorded if c in calls and calls[c].get("name") == GRAPH_LAUNCH}
    steps = _steps(raw)
    replays = tuple(Replay(c, _batch_size(steps, calls[c])) for c in sorted(graphs))
    eager = {c: call for c, call in calls.items() if c not in graphs}
    return TraceWindow(_eager_launchers(raw, events, eager), replays)


def _capture_calls(raw: list[Any]) -> list[dict[str, Any]]:
    """The kernel launches between the capture's begin and end calls, in the order they were made."""
    calls = [e for e in raw if isinstance(e, dict) and e.get("cat") == "cuda_runtime" and _span(e, "")]
    begins = [e["ts"] for e in calls if e.get("name") == "hipStreamBeginCapture"]
    ends = [e["ts"] for e in calls if e.get("name") == "hipStreamEndCapture"]
    if not begins or not ends:
        return []
    inside = [e for e in calls if min(begins) <= e["ts"] <= max(ends) and _LAUNCH_CALL.fullmatch(str(e.get("name")))]
    named = [e for e in inside if isinstance(e.get("args"), dict) and isinstance(e["args"].get("kernel"), str)]
    return sorted(named, key=lambda e: e["ts"])


def read_launches(path: Path) -> list[Launch]:
    """The kernels the graph capture trace at ``path`` launched, in order, each with the module that launched it.

    A launch outside every ``nn.Module`` frame has an empty launcher.
    """
    raw = trace_events(path)
    calls = _capture_calls(raw)
    spans = [s for s in (_span(call, call["args"]["kernel"]) for call in calls) if s]
    labels = _launchers(spans, _module_frames(raw))
    return [Launch(span.label, label) for span, label in zip(spans, labels, strict=True)]


def _capture_file(folder: Path, batch_size: int) -> Path | None:
    """The trace of the smallest FULL capture that holds ``batch_size`` requests, as the folder's manifest lists it.

    Of captures of one size (one per rank) the first listed is read: every rank captures the same model.
    """
    try:
        entries = json.loads((folder / CAPTURE_MANIFEST).read_text(encoding=_ENCODING))
    except (OSError, ValueError):
        return None
    files: dict[int, Path] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("mode") != _FULL_GRAPH:
            continue
        size, name = entry.get("batch_size"), entry.get("file")
        if isinstance(size, int) and not isinstance(size, bool) and isinstance(name, str):
            files.setdefault(size, folder / Path(name).name)
    fits = [size for size in files if size >= batch_size]
    return files[min(fits)] if fits and files[min(fits)].is_file() else None


class _Unpaired(Exception):
    """A replay that takes no launcher from its capture; the message says why."""


def _launches_for(replay: Replay, folder: Path | None, cache: dict[Path, list[Launch]]) -> list[Launch]:
    """The launches of the capture of ``replay``'s batch size, each file read once."""
    if folder is None or replay.batch_size is None:
        raise _Unpaired("it has no capture folder, or its step names no batch size")
    path = _capture_file(folder, replay.batch_size)
    if path is None:
        raise _Unpaired(
            f"{folder.name} lists no FULL capture of {replay.batch_size} requests or more with a trace file"
        )
    if path not in cache:
        try:
            cache[path] = read_launches(path)
        except KernelTraceError as exc:
            raise _Unpaired(str(exc)) from None
    return cache[path]


def _slots(events: list[GpuEvent], replay: Replay, launches: list[Launch]) -> list[int]:
    """Where the kernels of ``replay`` sit in ``events``, in the order they ran, when ``launches`` are exactly them."""
    own = [i for i, e in enumerate(events) if e.correlation == replay.correlation and e.category == "kernel"]
    slots = sorted(own, key=lambda i: events[i].start)
    if not launches:
        raise _Unpaired("its capture holds no kernel launches")
    if len({events[i].stream for i in slots}) > 1:
        raise _Unpaired("its kernels run on more than one stream, so their order is not the capture's")
    if Counter(events[i].name for i in slots) != Counter(launch.kernel for launch in launches):
        raise _Unpaired(f"its {len(slots)} kernels are not the {len(launches)} its capture launched")
    return slots


def relabel_replays(window: TraceWindow, folder: Path | None) -> Relabelled:
    """``window``'s events, the kernels of each graph replay given the launcher the capture in ``folder`` records.

    A replay that cannot be paired with its capture stays as recorded, and ``notes`` says why.
    """
    events, notes = list(window.events), []
    cache: dict[Path, list[Launch]] = {}
    renamed = 0
    for replay in window.replays:
        try:
            launches = _launches_for(replay, folder, cache)
            slots = _slots(events, replay, launches)
        except _Unpaired as reason:
            notes.append(f"the replay of correlation {replay.correlation} is left as recorded: {reason}")
            continue
        for index, launch in zip(slots, launches, strict=True):
            renamed += events[index].name != launch.kernel
            events[index] = replace(events[index], name=launch.kernel, launcher=launch.launcher)
    return Relabelled(events, sum(1 for e in events if e.launcher), renamed, tuple(notes))


def read_launched_window(path: Path, folder: Path | None) -> Relabelled:
    """The GPU events of the window at ``path``, each with the module that launched it where the trace or the graph
    captures in ``folder`` say."""
    return relabel_replays(read_window(path), folder)


__all__ = [
    "CAPTURE_MANIFEST",
    "GRAPH_LAUNCH",
    "Launch",
    "Relabelled",
    "Replay",
    "TraceWindow",
    "read_launched_window",
    "read_launches",
    "read_window",
    "relabel_replays",
]
