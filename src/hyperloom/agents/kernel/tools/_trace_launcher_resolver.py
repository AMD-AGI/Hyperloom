###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Resolve a device kernel to its Python launcher frame, straight from the trace.

Motivation: TraceLens cannot attribute a hand-written Triton kernel. Such a
kernel is launched through ``triton.jit`` and never passes the ATen dispatcher,
so it has no ``cpu_op`` parent; TraceLens files it as a ``(Synthetic Op)`` with
``launcher_path = "Not found"``. The information is nonetheless present in the
trace: ``with_stack`` profiling records the full ``python_function`` chain down
to the launch, ending at the user frame that called the kernel.

This module recovers that frame deterministically, which is strictly better than
the name-grep fallback it precedes: it yields a line number and function name,
and it cannot produce grep's false positives (a test file or a CPU implementation
that merely mentions the kernel name).

Design constraints:

* **Streaming, two passes.** ``python_function`` dominates a trace (order 10^6
  events, ~95% of the file). Pass 1 reads only ``kernel`` and ``cuda_runtime``
  (order 10^4) to pick probe timestamps; pass 2 walks the frames but retains
  only those enclosing a probe, so peak memory stays flat.
* **Eager probes preferred.** A CUDA-graph replay launch has no per-kernel
  Python frame -- its stack ends at ``torch/cuda/graphs.py: replay`` -- so
  probes are drawn from non-graph launch APIs first. The same kernel is the same
  source on both paths, so one eager sample covers the graph launches too.
* **Fail soft.** Any error yields an empty mapping; the caller falls back to the
  existing grep resolution.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from _bypass_trace_reader import _open_trace_binary, stream_events

# Kineto event categories consulted here.
_CAT_KERNEL = "kernel"
_CAT_RUNTIME = "cuda_runtime"
_CAT_PYTHON = "python_function"

# Launch APIs that replay a captured graph: the launching Python frame belongs
# to the capture, not to this call, so these are poor probes.
_GRAPH_LAUNCH_APIS = frozenset({"hipgraphlaunch", "cudagraphlaunch"})

# A ``python_function`` name of the form "<path>(<line>): <func>".
_FRAME_RE = re.compile(r"^(?P<path>.+?)\((?P<line>\d+)\):\s*(?P<func>.+)$")

# Frames between the user's call site and the launch: launcher plumbing that is
# never the kernel's source. Walking outward, these are skipped.
#
# Every JIT toolchain contributes one of these. A JIT-built op's innermost Python
# frame is the compile-and-dispatch wrapper, not its kernel:
#   * triton  -> ``triton/runtime/jit.py: run``
#   * aiter   -> ``aiter/jit/core.py: wrapper``
#   * FlyDSL  -> ``flydsl/compiler/jit_executor.py: __call__``
# Those files are build machinery shared by every kernel the toolchain compiles,
# so admitting one both aims a backend at unpatchable code and collapses distinct
# kernels onto a single source (two MoE GEMMs both resolving to jit_executor.py).
#
# Scope matters: only ``flydsl/compiler/`` is plumbing. A kernel *written* in
# FlyDSL is a legitimate target -- ``classify_patchability`` accepts
# ``source_type="flydsl"`` -- so ``aiter/ops/flydsl/`` must stay visible.
_SKIP_FRAME_RE = re.compile(
    r"(?:"
    r"triton/runtime/|triton/backends/|triton/compiler/"
    r"|aiter/jit/"
    r"|flydsl/compiler/"
    r"|torch/nn/modules/module\.py"
    r"|torch/utils/_contextlib\.py"
    r"|torch/_dynamo/|torch/_inductor/"
    r"|/_ops\.py"
    r"|^<"
    r")"
)

# Generic decorator frames. Path-based skipping cannot catch these: a guard or
# retry decorator lives in an ordinary repo file, yet its body is plumbing. The
# function name is the stable signal (TraceLens keeps an equivalent list for its
# own entry-point search).
_WRAPPER_FUNC_NAMES = frozenset(
    {
        "call",
        "custom_wrapper",
        "decorate",
        "decorate_context",
        "handle_torch_function",
        "inner",
        "outer_wrapper",
        "wrapped",
        "wrapper",
        "wrapper_custom",
        "_fn",
        "_inner",
        "_wrapped",
        "_wrapper",
    }
)

# Reaching this frame means the launch came from a graph replay: everything
# further out belongs to the replay call site, not to the kernel.
_GRAPH_REPLAY_MARKER = "torch/cuda/graphs.py"

# Probe budget per kernel. Oversampled relative to ``max_samples_per_kernel``
# because some probes land on frames that resolve to nothing.
_PROBE_OVERSAMPLE = 4

# Shortest elided symbol prefix allowed to match. Below this a prefix carries no
# identity (``_ZN5...`` matches an entire namespace); real TraceLens elisions run
# far longer, so this only rejects degenerate input.
_MIN_ELIDED_PREFIX = 16


@dataclass(frozen=True)
class LauncherFrame:
    """A resolved Python launcher for one device kernel."""

    source_file: str
    line: int
    function: str
    sample_count: int
    launch_api: str

    @property
    def frame(self) -> str:
        """The frame rendered in TraceLens' ``<path>(<line>): <func>`` form."""
        return f"{self.source_file}({self.line}): {self.function}"


def _is_elided(name: str) -> bool:
    """Whether TraceLens shortened this symbol with a trailing ellipsis."""
    return name.rstrip(" ").endswith("...")


def _match_kernel(event_name: str, wanted: Iterable[str]) -> str | None:
    """Return the wanted kernel name contained in ``event_name``, if any.

    Substring rather than equality: a device kernel name may carry a template or
    ``.kd`` suffix around the symbol TraceLens reports.

    A long mangled symbol is additionally elided by TraceLens with a trailing
    ellipsis while the trace keeps the full name, so an elided name falls back to
    matching on its prefix. That fallback needs a floor: a stub like ``_ZN5...``
    is a prefix of every symbol in the namespace and would bind an arbitrary
    kernel. Sibling template instantiations can still both match a shared prefix,
    which is harmless -- they are the same source file.
    """
    for name in wanted:
        if not name:
            continue
        if name in event_name:
            return name
        probe = name.rstrip(". ")
        if probe != name and len(probe) >= _MIN_ELIDED_PREFIX and probe in event_name:
            return name
    return None


def _collect_probes(
    trace_file: Path,
    kernel_names: set[str],
    probe_budget: int,
) -> dict[str, list[tuple[int, float, str]]]:
    """Pass 1: pick ``(tid, ts, launch_api)`` probes for each wanted kernel.

    Reads only ``kernel`` and ``cuda_runtime`` events. Eager launches are
    preferred over graph replays (see module docstring).
    """
    corr_to_kernel: dict[Any, str] = {}
    runtimes: dict[Any, tuple[int, float, str]] = {}
    # Distinct trace symbols each requested name matched, to detect an elided
    # prefix that spans more than one kernel.
    matched_symbols: dict[str, set[str]] = defaultdict(set)
    with _open_trace_binary(trace_file) as fh:
        for ev in stream_events(fh):
            cat = ev.get("cat")
            if cat == _CAT_KERNEL:
                event_name = str(ev.get("name") or "")
                matched = _match_kernel(event_name, kernel_names)
                if matched is None:
                    continue
                matched_symbols[matched].add(event_name)
                corr = (ev.get("args") or {}).get("correlation")
                if corr is not None:
                    corr_to_kernel[corr] = matched
            elif cat == _CAT_RUNTIME:
                corr = (ev.get("args") or {}).get("correlation")
                if corr is None:
                    continue
                runtimes[corr] = (
                    int(ev.get("tid") or 0),
                    float(ev.get("ts") or 0.0),
                    str(ev.get("name") or ""),
                )

    # An elided name that spans several distinct symbols cannot be attributed:
    # the launcher of one would be reported as the launcher of another. Drop it
    # rather than pick arbitrarily; the grep tier still gets its turn.
    ambiguous = {
        name
        for name, symbols in matched_symbols.items()
        if _is_elided(name) and len(symbols) > 1
    }
    if ambiguous:
        corr_to_kernel = {c: k for c, k in corr_to_kernel.items() if k not in ambiguous}

    eager: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    graph: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for corr, kernel in corr_to_kernel.items():
        rt = runtimes.get(corr)
        if rt is None:
            continue
        bucket = graph if rt[2].strip().lower() in _GRAPH_LAUNCH_APIS else eager
        if len(bucket[kernel]) < probe_budget:
            bucket[kernel].append(rt)

    # Eager first; top up with graph probes only when a kernel has no eager call.
    probes: dict[str, list[tuple[int, float, str]]] = {}
    for kernel in kernel_names:
        picked = list(eager.get(kernel) or [])
        if not picked:
            picked = list(graph.get(kernel) or [])
        if picked:
            probes[kernel] = picked[:probe_budget]
    return probes


def _collect_enclosing_frames(
    trace_file: Path,
    probes: dict[str, list[tuple[int, float, str]]],
) -> dict[tuple[int, float], list[tuple[float, str]]]:
    """Pass 2: gather the ``python_function`` frames enclosing each probe.

    Only frames whose ``[ts, ts+dur]`` span covers a probe are retained, keeping
    memory proportional to the probe count rather than to the trace.
    """
    # Per-thread sorted probe timestamps, for a bisect membership test.
    by_tid: dict[int, list[float]] = defaultdict(list)
    for samples in probes.values():
        for tid, ts, _api in samples:
            by_tid[tid].append(ts)
    for tid in by_tid:
        by_tid[tid] = sorted(set(by_tid[tid]))

    enclosing: dict[tuple[int, float], list[tuple[float, str]]] = defaultdict(list)
    with _open_trace_binary(trace_file) as fh:
        for ev in stream_events(fh):
            if ev.get("cat") != _CAT_PYTHON:
                continue
            tid = int(ev.get("tid") or 0)
            stamps = by_tid.get(tid)
            if not stamps:
                continue
            start = float(ev.get("ts") or 0.0)
            end = start + float(ev.get("dur") or 0.0)
            lo = bisect.bisect_left(stamps, start)
            hi = bisect.bisect_right(stamps, end)
            if lo >= hi:
                continue
            name = str(ev.get("name") or "")
            for ts in stamps[lo:hi]:
                enclosing[(tid, ts)].append((start, name))
    return enclosing


def _innermost_user_frame(frames: list[tuple[float, str]]) -> tuple[str, int, str] | None:
    """Pick the innermost user ``.py`` frame from one probe's enclosing frames.

    Frames are ordered outermost-first by start time, so the walk runs in
    reverse. Launcher plumbing is skipped; hitting the graph-replay marker means
    this probe carries no per-kernel call site.
    """
    for _start, name in sorted(frames, key=lambda item: item[0], reverse=True):
        if _GRAPH_REPLAY_MARKER in name:
            return None
        if _SKIP_FRAME_RE.search(name):
            continue
        match = _FRAME_RE.match(name)
        if not match:
            continue
        path = match.group("path").strip()
        if not path.endswith(".py"):
            continue
        func = match.group("func").strip()
        if func in _WRAPPER_FUNC_NAMES:
            continue
        return path, int(match.group("line")), func
    return None


def resolve_launchers_from_trace(
    trace_files: list[Path] | list[str],
    kernel_names: set[str],
    *,
    max_samples_per_kernel: int = 3,
    log: Any = None,
) -> dict[str, LauncherFrame]:
    """Resolve device kernel names to their Python launcher frames.

    Args:
        trace_files: Candidate trace files, most representative first. Files are
            read in order and reading stops once every kernel is resolved, so a
            single rank's trace normally suffices.
        kernel_names: Device kernel symbols to resolve.
        max_samples_per_kernel: Probes to agree on per kernel; the majority frame
            wins.
        log: Optional ``callable(str)`` for diagnostics.

    Returns:
        Kernel name -> resolved launcher. Kernels with no eager sample (pure
        graph replay) or no usable frame are omitted, never guessed.
    """
    wanted = {str(k) for k in kernel_names if str(k).strip()}
    if not wanted or not trace_files:
        return {}

    resolved: dict[str, LauncherFrame] = {}
    probe_budget = max(1, int(max_samples_per_kernel)) * _PROBE_OVERSAMPLE

    for raw_path in trace_files:
        remaining = wanted - set(resolved)
        if not remaining:
            break
        trace_file = Path(raw_path)
        try:
            probes = _collect_probes(trace_file, remaining, probe_budget)
            if not probes:
                continue
            enclosing = _collect_enclosing_frames(trace_file, probes)
        except (OSError, ValueError) as exc:  # fail soft; grep fallback still runs
            if callable(log):
                log(f"trace_launcher_resolver: {trace_file.name}: {exc!r}")
            continue

        for kernel, samples in probes.items():
            votes: Counter[tuple[str, int, str]] = Counter()
            api_votes: Counter[str] = Counter()
            for tid, ts, api in samples:
                if sum(votes.values()) >= max_samples_per_kernel:
                    break
                frames = enclosing.get((tid, ts))
                if not frames:
                    continue
                found = _innermost_user_frame(frames)
                if found is None:
                    continue
                votes[found] += 1
                api_votes[api] += 1
            if not votes:
                continue
            (path, line, func), count = votes.most_common(1)[0]
            resolved[kernel] = LauncherFrame(
                source_file=path,
                line=line,
                function=func,
                sample_count=count,
                launch_api=(api_votes.most_common(1)[0][0] if api_votes else ""),
            )

    if callable(log):
        log(f"trace_launcher_resolver: resolved {len(resolved)}/{len(wanted)} kernel(s) from trace")
    return resolved


__all__ = ["LauncherFrame", "resolve_launchers_from_trace"]
