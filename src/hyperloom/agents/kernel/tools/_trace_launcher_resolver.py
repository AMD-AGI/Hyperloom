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
* **Eager probes only.** A CUDA-graph replay launch has no per-kernel
  Python frame -- its stack ends at ``torch/cuda/graphs.py: replay`` -- so
  graph-launch probes are rejected even when that marker is absent. The same
  kernel is the same source on both paths, so one eager sample covers graph
  launches too; a replay-only kernel falls through to the grep tier.
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

from _trace_reader import _open_trace_binary, stream_events

# Kineto event categories consulted here.
_CAT_KERNEL = "kernel"
_CAT_RUNTIME = "cuda_runtime"
_CAT_PYTHON = "python_function"

# Substring shared by hipGraphLaunch, cudaGraphLaunch and cuGraphLaunch,
# including decorated API names emitted by some Kineto versions.
_GRAPH_LAUNCH_MARKER = "graphlaunch"

# Substring shared by every kernel-dispatch runtime API across HIP and CUDA
# (hipModuleLaunchKernel, cudaLaunchKernel, hipGraphLaunch, ...). Correlated
# runtime events that are not launches -- memcpy, synchronize, malloc -- can
# never be a kernel's dispatch point, and keeping them would size the probe
# map by total runtime events rather than by launches.
_LAUNCH_API_MARKER = "launch"

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


def _event_pid(ev: dict) -> int:
    """Process id of a trace event; 0 when the trace omits one.

    Single-process traces frequently drop ``pid``, and a constant default keeps
    those keyed consistently while still separating ranks in a merged trace.
    """
    try:
        return int(ev.get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def _is_elided(name: str) -> bool:
    """Whether TraceLens shortened this symbol with a trailing ellipsis."""
    return name.rstrip(" ").endswith("...")


def _is_graph_launch(name: str) -> bool:
    """Whether a runtime API replays a captured GPU graph."""
    return _GRAPH_LAUNCH_MARKER in str(name or "").strip().lower()


def _has_symbol_boundaries(event_name: str, name: str) -> bool:
    """Return whether ``name`` appears as one complete symbol token."""
    start = 0
    while True:
        start = event_name.find(name, start)
        if start < 0:
            return False
        end = start + len(name)
        left_ok = start == 0 or not (event_name[start - 1].isalnum() or event_name[start - 1] == "_")
        right_ok = end == len(event_name) or not (event_name[end].isalnum() or event_name[end] == "_")
        if left_ok and right_ok:
            return True
        start += 1


def _has_elided_prefix_boundary(event_name: str, prefix: str) -> bool:
    """Return whether an elided prefix starts at a symbol boundary."""
    start = 0
    while True:
        start = event_name.find(prefix, start)
        if start < 0:
            return False
        if start == 0 or not (event_name[start - 1].isalnum() or event_name[start - 1] == "_"):
            return True
        start += 1


def _match_kernel(event_name: str, wanted: Iterable[str]) -> str | None:
    """Return the wanted kernel name contained in ``event_name``, if any.

    Non-exact matches require symbol boundaries, allowing known decorations such
    as template arguments and ``.kd`` while rejecting identifier suffixes.

    A long mangled symbol is additionally elided by TraceLens with a trailing
    ellipsis while the trace keeps the full name, so an elided name falls back to
    matching on its prefix. That fallback needs a floor: a stub like ``_ZN5...``
    is a prefix of every symbol in the namespace and would bind an arbitrary
    kernel. Sibling template instantiations can still both match a shared prefix,
    which is harmless -- they are the same source file.

    Overlapping names are decided rather than raced. ``wanted`` is a set, so
    returning the first substring hit made ``moe_gemm1`` and ``moe_gemm1_0``
    bind ``moe_gemm1_0.kd`` differently per ``PYTHONHASHSEED``, and a wrong
    binding here reaches the candidate ahead of grep. Every candidate is
    therefore collected and ranked: an exact hit wins, otherwise the longest and
    so most specific symbol does, and a genuine tie between equally specific
    symbols resolves to nothing rather than to whichever the set yielded first.
    """
    exact: list[str] = []
    decorated: list[str] = []
    elided: list[str] = []
    for name in wanted:
        if not name:
            continue
        if name == event_name:
            exact.append(name)
        elif _has_symbol_boundaries(event_name, name):
            decorated.append(name)
        # Gate on _is_elided, not on "rstrip changed something": a single
        # trailing dot also changes the string, and treating it as elided here
        # while _is_elided rejects it would let the name skip the ambiguity
        # check and bind to an arbitrary sibling symbol.
        elif _is_elided(name):
            probe = name.rstrip(". ")
            if len(probe) >= _MIN_ELIDED_PREFIX and _has_elided_prefix_boundary(event_name, probe):
                elided.append(name)
    if exact:
        return exact[0]
    # Decorated hits outrank elided-prefix hits: the latter is a fallback for a
    # symbol the trace only shows truncated. A tie inside the stronger bucket
    # must not fall through to the weaker one.
    for bucket in (decorated, elided):
        if not bucket:
            continue
        longest = max(len(name) for name in bucket)
        finalists = [name for name in bucket if len(name) == longest]
        return finalists[0] if len(finalists) == 1 else None
    return None


def _collect_probes(
    trace_file: Path,
    kernel_names: set[str],
    probe_budget: int,
    *,
    stream_errors: list[str] | None = None,
) -> dict[str, list[tuple[int, int, float, str]]]:
    """Pass 1: pick ``(tid, ts, launch_api)`` probes for each wanted kernel.

    Reads only ``kernel`` and ``cuda_runtime`` events. Eager launches are
    preferred over graph replays (see module docstring).
    """
    # Keyed by correlation alone. The two sides of a launch are recorded by
    # different processes in Kineto's model -- the kernel against the device,
    # the runtime call against the host -- so pairing on (pid, correlation)
    # never matches on a real trace. Correlation is the field designed to span
    # that boundary, and it is unique within one profiled process.
    #
    # A merged multi-process trace restarts correlation ids per rank, which a
    # bare id cannot separate. That is handled by detecting the collision
    # (below) and dropping the id rather than by keying on a pid that means
    # different things on either side of the pair.
    corr_to_kernel: dict[Any, str] = {}
    exact_corr: set[Any] = set()
    runtimes: dict[Any, tuple[int, int, float, str]] = {}
    # Correlation ids that identify more than one launch, and so identify none.
    ambiguous_corr: set[Any] = set()
    # Distinct complete trace symbols reached through a non-exact match.
    matched_symbols: dict[str, set[str]] = defaultdict(set)
    with _open_trace_binary(trace_file) as fh:
        for ev in stream_events(fh, errors=stream_errors):
            cat = ev.get("cat")
            if cat == _CAT_KERNEL:
                event_name = str(ev.get("name") or "")
                matched = _match_kernel(event_name, kernel_names)
                if matched is None:
                    continue
                if matched != event_name:
                    matched_symbols[matched].add(event_name)
                corr = (ev.get("args") or {}).get("correlation")
                if corr is None:
                    continue
                previous = corr_to_kernel.get(corr)
                if previous is not None and previous != matched:
                    # Two different kernels answer to this id: a merged trace
                    # reused it across ranks. Neither binding can be trusted.
                    ambiguous_corr.add(corr)
                else:
                    corr_to_kernel[corr] = matched
                    if matched == event_name:
                        exact_corr.add(corr)
            elif cat == _CAT_RUNTIME:
                api_name = str(ev.get("name") or "")
                # Kernel and runtime events arrive in no guaranteed order, so
                # this cannot filter on corr_to_kernel yet; gating on the launch
                # marker bounds the map by launches instead of by every
                # correlated runtime call in the trace.
                if _LAUNCH_API_MARKER not in api_name.lower():
                    continue
                corr = (ev.get("args") or {}).get("correlation")
                if corr is None:
                    continue
                pid = _event_pid(ev)
                if corr in runtimes:
                    # A second host-side launch claiming the same id: only a
                    # merged trace does this, and it makes the id useless.
                    ambiguous_corr.add(corr)
                    continue
                runtimes[corr] = (
                    pid,
                    int(ev.get("tid") or 0),
                    float(ev.get("ts") or 0.0),
                    api_name,
                )

    # Any non-exact name spanning distinct complete symbols is ambiguous: the
    # launcher of one could be reported as the launcher of another.
    ambiguous = {name for name, symbols in matched_symbols.items() if len(symbols) > 1}
    if ambiguous:
        corr_to_kernel = {
            corr: kernel for corr, kernel in corr_to_kernel.items() if kernel not in ambiguous or corr in exact_corr
        }

    eager: dict[str, list[tuple[int, int, float, str]]] = defaultdict(list)
    for key, kernel in corr_to_kernel.items():
        if key in ambiguous_corr:
            continue
        rt = runtimes.get(key)
        if rt is None:
            continue
        # A graph replay has one Python frame for the whole graph, never a
        # per-kernel launcher. API identity is stronger than a fragile stack
        # marker, which is absent for inductor and framework-native replays.
        if _is_graph_launch(rt[3]):
            continue
        if len(eager[kernel]) < probe_budget:
            eager[kernel].append(rt)

    # Replay-only kernels deliberately remain unresolved.
    probes: dict[str, list[tuple[int, int, float, str]]] = {}
    for kernel in kernel_names:
        picked = list(eager.get(kernel) or [])
        if picked:
            probes[kernel] = picked[:probe_budget]
    return probes


def _collect_enclosing_frames(
    trace_file: Path,
    probes: dict[str, list[tuple[int, int, float, str]]],
    *,
    stream_errors: list[str] | None = None,
) -> dict[tuple[int, int, float], list[tuple[float, float, str]]]:
    """Pass 2: gather the ``python_function`` frames enclosing each probe.

    Only frames whose ``[ts, ts+dur]`` span covers a probe are retained, keeping
    memory proportional to the probe count rather than to the trace.

    Probes are keyed by ``(pid, tid, ts)``. pid is part of the key because a
    merged multi-process trace reuses thread ids across ranks, and without it
    one rank's frames would answer for another's launch. Two launches from the
    same thread inside one microsecond still share a bucket -- legitimately, as
    a microsecond-granular trace holds one call-stack snapshot for both; where
    their real call sites differ the trace cannot tell them apart, and the vote
    across several probes is what keeps that from silently deciding a source.
    """
    # Per (pid, tid) sorted probe timestamps, for a bisect membership test.
    by_thread: dict[tuple[int, int], list[float]] = defaultdict(list)
    for samples in probes.values():
        for pid, tid, ts, _api in samples:
            by_thread[(pid, tid)].append(ts)
    for key in by_thread:
        by_thread[key] = sorted(set(by_thread[key]))

    enclosing: dict[tuple[int, int, float], list[tuple[float, float, str]]] = defaultdict(list)
    with _open_trace_binary(trace_file) as fh:
        for ev in stream_events(fh, errors=stream_errors):
            if ev.get("cat") != _CAT_PYTHON:
                continue
            pid = _event_pid(ev)
            tid = int(ev.get("tid") or 0)
            stamps = by_thread.get((pid, tid))
            if not stamps:
                continue
            start = float(ev.get("ts") or 0.0)
            dur = float(ev.get("dur") or 0.0)
            end = start + dur
            lo = bisect.bisect_left(stamps, start)
            hi = bisect.bisect_right(stamps, end)
            if lo >= hi:
                continue
            name = str(ev.get("name") or "")
            for ts in stamps[lo:hi]:
                # Keep ``dur``: profiler timestamps are microsecond-granular, so
                # adjacent frames on a fast call chain routinely share a ``ts``.
                # Start time alone cannot order them and the tie would resolve
                # by trace write order, picking an arbitrary nesting level.
                enclosing[(pid, tid, ts)].append((start, dur, name))
    return enclosing


def _innermost_user_frame(
    frames: list[tuple[float, float, str]],
) -> tuple[str, int, str] | None:
    """Pick the innermost user ``.py`` frame from one probe's enclosing frames.

    Ordered innermost-first by ``(start, -dur)``: a later start is deeper, and
    among frames starting in the same microsecond the narrower span is the
    nested one. Sorting on start alone would leave same-``ts`` frames in trace
    write order and collapse different kernels onto one source.

    Launcher plumbing is skipped; hitting the graph-replay marker means this
    probe carries no per-kernel call site.
    """
    ordered = sorted(frames, key=lambda item: (item[0], -item[1]), reverse=True)
    for _start, _dur, name in ordered:
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


#: Trace files to read at most. A capture folder holds one file per batch size
#: per rank (dozens), and every one of them costs two full streaming passes.
#: Kernels launched from C++ or only ever replayed from a graph never resolve,
#: so "stop once everything is resolved" alone would read the whole folder.
_MAX_TRACE_FILES = 2

#: Consecutive files that may add nothing before scanning gives up. Ranks run
#: the same Python, so a file that contributes nothing means the next one very
#: likely will not either.
_MAX_BARREN_FILES = 1


def resolve_launchers_from_trace(
    trace_files: list[Path] | list[str],
    kernel_names: set[str],
    *,
    max_samples_per_kernel: int = 3,
    max_trace_files: int = _MAX_TRACE_FILES,
    log: Any = None,
    file_errors: list[str] | None = None,
) -> dict[str, LauncherFrame]:
    """Resolve device kernel names to their Python launcher frames.

    Args:
        trace_files: Candidate trace files, most representative first. At most
            ``max_trace_files`` are read, and scanning also stops early once
            every kernel is resolved or a file adds nothing new -- a single
            rank's trace normally suffices.
        kernel_names: Device kernel symbols to resolve.
        max_samples_per_kernel: Probes to agree on per kernel; the majority frame
            wins.
        max_trace_files: Hard ceiling on files read, bounding this tier's cost
            against a capture folder holding dozens of them.
        log: Optional ``callable(str)`` for diagnostics.

    Returns:
        Kernel name -> resolved launcher. Kernels with no eager sample (pure
        graph replay) or no usable frame are omitted, never guessed.
    """
    wanted = {str(k) for k in kernel_names if str(k).strip()}
    if not wanted or not trace_files:
        return {}

    # Per-file failures are reported out, not just logged: swallowing them made
    # an unreadable trace indistinguishable from "nothing needed resolving".
    if file_errors is None:
        file_errors = []
    resolved: dict[str, LauncherFrame] = {}
    probe_budget = max(1, int(max_samples_per_kernel)) * _PROBE_OVERSAMPLE
    budget = max(1, int(max_trace_files))
    barren = 0
    scanned = 0

    def _report_stream_errors(trace_file: Path, messages: list[str]) -> None:
        """Expose structural parser failures without discarding recovered data."""
        for message in messages:
            note = f"{trace_file.name}: stream error: {message}"
            if note not in file_errors:
                file_errors.append(note)
            if callable(log):
                log(f"trace_launcher_resolver: {note}")

    for raw_path in trace_files[:budget]:
        remaining = wanted - set(resolved)
        if not remaining:
            break
        before = len(resolved)
        trace_file = Path(raw_path)
        scanned += 1
        try:
            probe_stream_errors: list[str] = []
            probes = _collect_probes(
                trace_file,
                remaining,
                probe_budget,
                stream_errors=probe_stream_errors,
            )
            _report_stream_errors(trace_file, probe_stream_errors)
            if not probes:
                if probe_stream_errors:
                    continue
                barren += 1
                if barren >= _MAX_BARREN_FILES:
                    break
                continue
            frame_stream_errors: list[str] = []
            enclosing = _collect_enclosing_frames(
                trace_file,
                probes,
                stream_errors=frame_stream_errors,
            )
            _report_stream_errors(trace_file, frame_stream_errors)
        except (EOFError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            # Fail soft per file so one malformed trace cannot take the tier
            # down: a non-dict ``args`` raises AttributeError, a truncated
            # gzip raises OSError, and either would otherwise surface as the
            # whole tier silently returning nothing.
            note = f"{trace_file.name}: {type(exc).__name__}: {exc}"
            file_errors.append(note)
            if callable(log):
                log(f"trace_launcher_resolver: {note}")
            continue

        for kernel, samples in probes.items():
            votes: Counter[tuple[str, int, str]] = Counter()
            api_votes: Counter[str] = Counter()
            for pid, tid, ts, api in samples:
                if sum(votes.values()) >= max_samples_per_kernel:
                    break
                frames = enclosing.get((pid, tid, ts))
                if not frames:
                    continue
                found = _innermost_user_frame(frames)
                if found is None:
                    continue
                votes[found] += 1
                api_votes[api] += 1
            if not votes:
                continue
            ranked = votes.most_common(2)
            (path, line, func), count = ranked[0]
            # A plurality is not agreement. With probes split across distinct
            # frames, most_common() returns whichever was inserted first, which
            # is trace order rather than evidence. Require a strict win: a tie
            # at the top means the probes disagree, and the grep tier is a
            # better answer than an arbitrary one.
            if len(ranked) > 1 and ranked[1][1] == count:
                if callable(log):
                    log(
                        f"trace_launcher_resolver: {kernel}: probes disagree "
                        f"({count} vs {ranked[1][1]}); leaving unresolved"
                    )
                continue
            resolved[kernel] = LauncherFrame(
                source_file=path,
                line=line,
                function=func,
                sample_count=count,
                launch_api=(api_votes.most_common(1)[0][0] if api_votes else ""),
            )

        if len(resolved) == before:
            if probe_stream_errors or frame_stream_errors:
                continue
            barren += 1
            if barren >= _MAX_BARREN_FILES:
                break
        else:
            barren = 0

    if callable(log):
        log(f"trace_launcher_resolver: resolved {len(resolved)}/{len(wanted)} kernel(s) from {scanned} trace file(s)")
    return resolved


__all__ = ["LauncherFrame", "resolve_launchers_from_trace"]
