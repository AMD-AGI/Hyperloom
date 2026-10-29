###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Independent, low-memory reader/aggregator for Kineto torch-profiler traces."""

from __future__ import annotations

import codecs
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Stdlib-only sibling; keeps this reader independent of TraceLens while sharing one capture-vs-workload rule with the
# TraceLens route.
from ._capture_shapes import is_capture_fragment
from ._trace_rank import trace_rank as _rank_of

# GPU device-side event categories (Kineto ``cat`` values).
_GPU_KERNEL_CAT = "kernel"
_GPU_MEMCPY_CATS = ("gpu_memcpy", "gpu_memset")
_GPU_CATS = frozenset((_GPU_KERNEL_CAT,) + _GPU_MEMCPY_CATS)

#: Per-pair overlap (us) below this is timestamp rounding, not corruption.
_STREAM_OVERLAP_EPS_US = 1.0
#: Minimum single overrun (us) before a stream is reported at all.
_STREAM_OVERLAP_MIN_WORST_US = 1000.0
#: Minimum total overrun, as a share of the stream's summed device time.
_STREAM_OVERLAP_MIN_SHARE = 0.05
#: At or above this share the kernel ranking is materially wrong.
_STREAM_OVERLAP_SEVERE_SHARE = 0.25

_TRACE_EXTS = (".trace.json.gz", ".pt.trace.json.gz", ".trace.json", ".json.gz", ".json")

_DECODER = json.JSONDecoder()
# The ``"traceEvents":`` member and, once buffered, the first character of its value.
_TRACE_EVENTS_OPENER = re.compile(r'"traceEvents"[ \t\r\n]*:[ \t\r\n]*([^ \t\r\n])?')
_ELEMENT_SEPARATORS = re.compile(r"[ \t\r\n,]*")

# Hard caps keep a corrupt trace from turning the streaming reader into an unbounded accumulator.
_MAX_TRACE_PREFIX_CHARS = 16 * 1024 * 1024
_MAX_EVENT_CHARS = 64 * 1024 * 1024
_MAX_BUFFERED_GPU_EVENTS = 100_000
_MAX_EVENT_NAME_CHARS = 256


# A graph trace is judged under-recorded when fewer than this fraction of its graph-launch correlations actually
# recorded any kernel: activity-buffer overflow drops whole replays, so recorded-launch coverage collapses toward
# ~1/launch_count, whereas a fully-recorded (merely idle) workload keeps kernels on essentially every launch.
_GRAPH_RECORDED_LAUNCH_COVERAGE_MAX = 0.5


def _file_size(fp: Path) -> int:
    """Return file size in bytes, or 0 on stat() failure."""
    try:
        return fp.stat().st_size
    except OSError:
        return 0


def _trace_candidates(root: Path) -> list[Path]:
    """Return all trace-shaped files under ``root`` (recursive)."""
    out: list[Path] = []
    for child in root.rglob("*"):
        if child.is_file() and any(child.name.lower().endswith(ext) for ext in _TRACE_EXTS):
            out.append(child)
    return out


def _main_trace_candidates(candidates: list[Path], root: str | Path | None = None) -> list[Path]:
    """Drop CUDA-graph capture shards, keeping only main workload traces."""
    main = [c for c in candidates if not is_capture_fragment(c, root)]
    return main or candidates


def _select_trace_file(candidates: list[Path], root: str | Path | None = None) -> Path | None:
    """Deterministically pick one trace file from candidates."""
    candidates = _main_trace_candidates(candidates, root)
    merged = [c for c in candidates if c.name.startswith("merged-")]
    if merged:
        return max(merged, key=lambda c: (_file_size(c), c.name))
    ranked = [c for c in candidates if _rank_of(c) is not None]
    if ranked:
        return min(ranked, key=lambda c: (_rank_of(c), c.name))
    return max(candidates, key=lambda c: (_file_size(c), c.name))


def resolve_trace_file(trace_input: str | Path) -> Path | None:
    """Resolve a trace input (file or directory) to a single trace file."""
    p = Path(trace_input)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    candidates = _trace_candidates(p)
    if not candidates:
        return None
    return _select_trace_file(candidates, p)


def _open_trace_binary(path: Path):
    """Open a trace file, transparently decompressing ``.gz``."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


class _ObjectBalance:
    """Resumable brace balancer for a single ``traceEvents`` element."""

    __slots__ = ("_depth", "_escaped", "_in_string", "_scan")

    def __init__(self, start: int) -> None:
        self._scan = start
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def advance(self, buf: str) -> int | None:
        """Consume newly buffered characters."""
        scan = self._scan
        depth = self._depth
        in_string = self._in_string
        escaped = self._escaped
        end: int | None = None
        while scan < len(buf):
            char = buf[scan]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = scan + 1
                    break
            scan += 1
        self._scan = scan
        self._depth = depth
        self._in_string = in_string
        self._escaped = escaped
        return end


class _TraceStructureError(Exception):
    """The ``traceEvents`` stream cannot be read past this point; the message says why."""


class _TraceEventStream:
    """Read state of one pass over a ``traceEvents`` array: the decoded text, the cursor in it, and the diagnostics."""

    def __init__(self, fileobj, chunk_size: int, errors: list[str]) -> None:
        self.errors = errors
        self._fileobj = fileobj
        self._chunk_size = chunk_size
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._pos = 0
        self._eof = False
        self._emitted = 0

    def _refill(self, max_bytes: int) -> None:
        """Decode one bounded chunk and turn gzip failures into stream errors."""
        try:
            chunk = self._fileobj.read(min(self._chunk_size, max_bytes))
        except (EOFError, gzip.BadGzipFile) as exc:
            self.errors.append(f"trace input error: {type(exc).__name__}: {exc}")
            chunk = b""
        self._eof = not chunk
        self._buf += self._decoder.decode(chunk, final=self._eof)

    def open_array(self) -> None:
        """Consume the trace prefix through the ``[`` that opens the ``traceEvents`` array."""
        while (opener := _TRACE_EVENTS_OPENER.search(self._buf)) is None or opener.group(1) is None:
            if self._eof:
                raise _TraceStructureError(
                    "traceEvents array not found" if opener is None else "traceEvents array opener not found"
                )
            if len(self._buf) >= _MAX_TRACE_PREFIX_CHARS:
                raise _TraceStructureError(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
            self._refill(_MAX_TRACE_PREFIX_CHARS - len(self._buf))
        if opener.group(1) != "[":
            raise _TraceStructureError("traceEvents value is not an array")
        self._buf = self._buf[opener.end() :]

    def next_element(self) -> str:
        """Skip separators to the next array element and return its first character."""
        while True:
            # Parsed input is discarded once it exceeds one refill chunk.
            if self._pos > self._chunk_size:
                self._buf = self._buf[self._pos :]
                self._pos = 0
            self._pos = _ELEMENT_SEPARATORS.match(self._buf, self._pos).end()
            if self._pos < len(self._buf):
                return self._buf[self._pos]
            if self._eof:
                raise _TraceStructureError(f"traceEvents array unterminated after {self._emitted} event(s)")
            self._refill(self._chunk_size)

    def skip_element(self) -> None:
        """Record a non-object element as malformed and skip it through the separator that ends it."""
        start = scan = self._pos
        curly_depth = square_depth = 0
        in_string = escaped = False
        boundary: int | None = None
        while boundary is None:
            buf = self._buf
            while scan < len(buf):
                char = buf[scan]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    curly_depth += 1
                elif char == "}":
                    curly_depth = max(0, curly_depth - 1)
                elif char == "[":
                    square_depth += 1
                elif char == "]":
                    if curly_depth == 0 and square_depth == 0:
                        boundary = scan
                        break
                    square_depth = max(0, square_depth - 1)
                elif char == "," and curly_depth == 0 and square_depth == 0:
                    boundary = scan + 1
                    break
                scan += 1
            if boundary is None:
                self._refill_element(start, "element", f"traceEvents array unterminated after {self._emitted} event(s)")
        self.errors.append(f"traceEvents element malformed after {self._emitted} event(s): expected an object")
        self._pos = boundary

    def read_object(self) -> Iterator[dict]:
        """Yield the object element at the cursor, or record it as malformed and skip it."""
        start = self._pos
        balance = _ObjectBalance(start)
        while True:
            try:
                obj, end = _DECODER.raw_decode(self._buf, start)
                break
            except json.JSONDecodeError as exc:
                # raw_decode cannot distinguish corrupt input from an object the buffer merely has not reached the end
                # of yet, so balance braces to tell the two apart.
                object_end = balance.advance(self._buf)
                if object_end is not None:
                    self.errors.append(f"traceEvents object malformed after {self._emitted} event(s): {exc.msg}")
                    self._pos = object_end
                    return
                self._refill_element(
                    start, "object", f"traceEvents object truncated after {self._emitted} event(s): unterminated object"
                )
        if end - start > _MAX_EVENT_CHARS:
            raise _TraceStructureError(
                f"traceEvents object exceeds {_MAX_EVENT_CHARS} characters after {self._emitted} event(s)"
            )
        yield obj
        self._emitted += 1
        self._pos = end

    def _refill_element(self, start: int, kind: str, eof_message: str) -> None:
        """Read more of the element at ``start``, refusing one longer than ``_MAX_EVENT_CHARS``."""
        buffered = len(self._buf) - start
        if buffered >= _MAX_EVENT_CHARS:
            raise _TraceStructureError(
                f"traceEvents {kind} exceeds {_MAX_EVENT_CHARS} characters after {self._emitted} event(s)"
            )
        if self._eof:
            raise _TraceStructureError(eof_message)
        self._refill(_MAX_EVENT_CHARS - buffered)


def stream_events(
    fileobj,
    bufsize: int = 8 * 1024 * 1024,
    *,
    errors: list[str] | None = None,
) -> Iterator[dict]:
    """Yield each object inside the ``traceEvents`` array, one at a time."""
    stream = _TraceEventStream(fileobj, max(1, int(bufsize)), [] if errors is None else errors)
    try:
        stream.open_array()
        while (first := stream.next_element()) != "]":
            if first == "{":
                yield from stream.read_object()
            else:
                stream.skip_element()
    except _TraceStructureError as exc:
        stream.errors.append(str(exc))


def _stream_overlap_health(stream_events: dict[Any, list[tuple[float, float, str]]]) -> dict[str, Any]:
    """Report device streams whose event durations physically cannot hold."""
    worst: dict[str, Any] = {}
    for key, events in stream_events.items():
        if len(events) < 2:
            continue
        events.sort()
        device_us = sum(end - ts for ts, end, _n in events)
        if device_us <= 0:
            continue
        span_us = max(end for _t, end, _n in events) - events[0][0]
        excess_us = 0.0
        overlapping = 0
        worst_name = ""
        worst_over = 0.0
        for i in range(len(events) - 1):
            ts, end, name = events[i]
            nxt_ts = events[i + 1][0]
            # Identical timestamps cannot contradict each other.
            if nxt_ts == ts:
                continue
            over = end - nxt_ts
            if over <= _STREAM_OVERLAP_EPS_US:
                continue
            excess_us += over
            overlapping += 1
            if over > worst_over:
                worst_name, worst_over = name, over
        if overlapping == 0 or worst_over < _STREAM_OVERLAP_MIN_WORST_US:
            continue
        share = excess_us / device_us
        if share < _STREAM_OVERLAP_MIN_SHARE:
            continue
        if not worst or excess_us > worst["excess_ms"] * 1000.0:
            pid, tid = key
            worst = {
                "pid": pid,
                "tid": tid,
                "overlapping_events": overlapping,
                "excess_ms": round(excess_us / 1000.0, 4),
                "device_ms": round(device_us / 1000.0, 4),
                "excess_share": round(share, 4),
                "span_ms": round(span_us / 1000.0, 4),
                "worst_event": worst_name,
                "worst_event_excess_ms": round(worst_over / 1000.0, 4),
                "severity": ("warning" if share >= _STREAM_OVERLAP_SEVERE_SHARE else "info"),
            }
    return worst


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Return the union length (in ms) of ``[start_us, end_us)`` intervals."""
    if not intervals:
        return 0.0
    intervals.sort(key=lambda iv: iv[0])
    total_us = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total_us += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    total_us += cur_end - cur_start
    return total_us / 1000.0


@dataclass
class _TraceBuffers:
    """Device events and attribution maps buffered in one pass over a trace."""

    # (duration, correlation, start, end) per GPU kernel; (start, end) per memcpy/memset.
    k_events: list[tuple[float, Any, float, float]] = field(default_factory=list)
    m_events: list[tuple[float, float]] = field(default_factory=list)
    # Kineto attribution chain: launch correlation -> external id -> framework op.
    corr_to_extid: dict[int, int] = field(default_factory=dict)
    extid_to_opname: dict[int, str] = field(default_factory=dict)
    # Correlations of CUDA/HIP graph-launch runtime events (no External id), so their replayed kernels are classified
    # graph-attributed, not (unlinked).
    graph_launch_corrs: set[int] = field(default_factory=set)
    # (pid, tid) -> [(ts, end, name), ...] for the duration-sanity check.
    stream_intervals: dict[Any, list[tuple[float, float, str]]] = field(default_factory=dict)
    # "gpu_events" once the GPU-event cap stopped buffering; empty while it holds.
    truncation_reason: str = ""

    def add(self, ev: dict[str, Any]) -> None:
        """Buffer one trace event, or set ``truncation_reason`` instead when buffering it would pass a cap."""
        cat = ev.get("cat", "")
        if cat == "cuda_runtime":
            args = ev.get("args") or {}
            cid = args.get("correlation")
            extid = args.get("External id")
            if cid is not None and extid is not None:
                self.corr_to_extid[cid] = extid
            if cid is not None and "GraphLaunch" in (ev.get("name", "") or ""):
                self.graph_launch_corrs.add(cid)
        elif cat == "cpu_op":
            extid = (ev.get("args") or {}).get("External id")
            if extid is not None:
                self.extid_to_opname[extid] = ev.get("name", "") or ""
        elif cat in _GPU_CATS and ev.get("ph") == "X":
            self._add_device_event(ev)

    def _add_device_event(self, ev: dict[str, Any]) -> None:
        """Buffer one GPU kernel or memcpy/memset with its stream interval, unless the GPU-event cap is reached."""
        dur = float(ev.get("dur", 0) or 0)
        ts = float(ev.get("ts", 0) or 0)
        end = ts + dur
        name = str(ev.get("name", "") or "")
        stream_key = (ev.get("pid"), ev.get("tid"))
        if len(self.k_events) + len(self.m_events) >= _MAX_BUFFERED_GPU_EVENTS:
            self.truncation_reason = "gpu_events"
            return
        self.stream_intervals.setdefault(stream_key, []).append((ts, end, name[:_MAX_EVENT_NAME_CHARS]))
        if ev.get("cat") != _GPU_KERNEL_CAT:
            self.m_events.append((ts, end))
            return
        self.k_events.append((dur, (ev.get("args") or {}).get("correlation"), ts, end))

    def graph_coverage(self, graph_kernels: int, busy_fraction: float) -> dict[str, Any]:
        """Report graph-replay recording health, flagging a trace that likely under-recorded replays.

        Under continuous graph replay roctracer's activity buffer overflows, so only ~1 replay's kernels are recorded.
        """
        launch_count = len(self.graph_launch_corrs)
        # Activity-buffer overflow drops replays across the whole run, so coverage is taken over every buffered kernel.
        launches_with_kernels = len({e[1] for e in self.k_events if e[1] in self.graph_launch_corrs})
        return {
            "graph_mode": launch_count > 0,
            "graph_launch_count": launch_count,
            "graph_launches_with_kernels": launches_with_kernels,
            "busy_fraction": busy_fraction,
            "graph_under_recorded": (
                launch_count >= 2
                and graph_kernels > 0
                and launches_with_kernels / launch_count < _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX
            ),
        }


def _finalize(buffers: _TraceBuffers) -> dict[str, Any]:
    """Build the timeline, kernel attribution and graph coverage from buffered device events."""
    kernel_intervals = [(start, end) for _dur, _corr, start, end in buffers.k_events]
    device_intervals = kernel_intervals + buffers.m_events
    total_ms = 0.0
    if device_intervals:
        total_ms = (max(end for _, end in device_intervals) - min(start for start, _ in device_intervals)) / 1000.0
    busy_ms = _union_ms(device_intervals)
    idle_ms = max(0.0, total_ms - busy_ms)

    kernel_us = attributed_us = 0.0
    attributed = graph = unlinked = 0
    for dur, corr, _start, _end in buffers.k_events:
        kernel_us += dur
        if buffers.extid_to_opname.get(buffers.corr_to_extid.get(corr)):
            attributed += 1
            attributed_us += dur
        elif corr in buffers.graph_launch_corrs:
            # Graph-internal kernels launched via a captured CUDA/HIP graph: they carry a correlation matching a
            # graph-launch runtime event (which has no External id), so they resolve to no cpu_op but are not a genuine
            # attribution failure.
            graph += 1
        else:
            unlinked += 1
    return {
        "timeline": {
            "total_time_ms": round(total_ms, 4),
            "busy_time_ms": round(busy_ms, 4),
            "idle_time_ms": round(idle_ms, 4),
            "gpu_kernel_sum_ms": round(kernel_us / 1000.0, 4),
            "idle_pct": round(idle_ms / total_ms * 100.0, 4) if total_ms > 0 else 0.0,
        },
        "attribution": {
            "kernel_count": len(buffers.k_events),
            "attributed_kernels": attributed,
            "unlinked_kernels": unlinked,
            "attributed_gpu_ms": round(attributed_us / 1000.0, 4),
            "attributed_pct": round(attributed_us / kernel_us * 100.0, 2) if kernel_us > 0 else 0.0,
            "cuda_runtime_links": len(buffers.corr_to_extid),
            "cpu_ops": len(buffers.extid_to_opname),
            "graph_attributed_kernels": graph,
        },
        "graph_coverage": buffers.graph_coverage(graph, round(busy_ms / total_ms, 4) if total_ms > 0 else 0.0),
    }


def analyze_trace(trace_file: Path) -> dict[str, Any]:
    """Stream a Kineto trace file and return its timeline, kernel attribution and graph-recording health."""
    buffers = _TraceBuffers()
    event_total = 0
    stream_errors: list[str] = []
    with _open_trace_binary(trace_file) as fobj:
        for ev in stream_events(fobj, errors=stream_errors):
            event_total += 1
            buffers.add(ev)
            if buffers.truncation_reason:
                break

    body = _finalize(buffers)
    stream_overlap = _stream_overlap_health(buffers.stream_intervals)
    if stream_overlap:
        body["timeline"]["stream_overlap"] = stream_overlap

    result: dict[str, Any] = {
        "event_total": event_total,
        "truncated": bool(buffers.truncation_reason),
        "stream_errors": stream_errors,
        "analyzed_rank": _rank_of(trace_file),
        **body,
    }
    if buffers.truncation_reason:
        result["truncation_reason"] = buffers.truncation_reason
    return result
