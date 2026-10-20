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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

# Stdlib-only sibling; keeps this reader independent of TraceLens while sharing one capture-vs-workload rule with the
# TraceLens route.
from ._capture_shapes import is_capture_fragment as _shared_is_capture_fragment
from ._trace_rank import select_primary_trace, trace_rank as _rank_of

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
_MAX_ANNOTATION_WINDOWS = 100_000
_MAX_BUFFERED_GPU_EVENTS = 100_000
_MAX_EVENT_NAME_CHARS = 256


def _backfill_shape_signature(meta: dict[str, Any]) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Hashable key for a cpu_op shape/dtype meta (for majority backfill)."""
    shapes = meta.get("shapes") or []
    dtypes = meta.get("dtypes") or []
    norm_shapes: list[tuple[int, ...]] = []
    for dims in shapes:
        if not isinstance(dims, (list, tuple)):
            continue
        try:
            norm_shapes.append(tuple(int(d) for d in dims))
        except (TypeError, ValueError):
            continue
    norm_dtypes = tuple(str(d) for d in dtypes) if isinstance(dtypes, (list, tuple)) else ()
    return (tuple(norm_shapes), norm_dtypes)


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


def _is_capture_fragment(path: str | Path, root: str | Path | None = None) -> bool:
    """True if ``path`` is a CUDA-graph capture shard, not a main trace."""
    return _shared_is_capture_fragment(path, root)


def _main_trace_candidates(candidates: list[Path], root: str | Path | None = None) -> list[Path]:
    """Drop CUDA-graph capture shards, keeping only main workload traces."""
    main = [c for c in candidates if not _is_capture_fragment(c, root)]
    return main or candidates


def _select_trace_file(
    candidates: list[Path],
    root: str | Path | None = None,
    *,
    require_single_rank: bool = False,
    preferred_rank: int = 0,
    tensor_parallel_size: int | None = None,
) -> Path | None:
    """Deterministically pick one trace file from candidates."""
    candidates = _main_trace_candidates(candidates, root)
    if require_single_rank:
        return select_primary_trace(
            candidates,
            file_size=_file_size,
            preferred_rank=preferred_rank,
            tensor_parallel_size=tensor_parallel_size,
        )
    merged = [c for c in candidates if c.name.startswith("merged-")]
    if merged:
        return max(merged, key=lambda c: (_file_size(c), c.name))
    ranked = [c for c in candidates if _rank_of(c) is not None]
    if ranked:
        return min(ranked, key=lambda c: (_rank_of(c), c.name))
    return max(candidates, key=lambda c: (_file_size(c), c.name))


def resolve_trace_file(
    trace_input: str | Path,
    *,
    require_single_rank: bool = False,
    preferred_rank: int = 0,
    tensor_parallel_size: int | None = None,
) -> Path | None:
    """Resolve a trace input (file or directory) to a single trace file."""
    p = Path(trace_input)
    if p.is_file():
        if require_single_rank:
            return select_primary_trace(
                [p],
                file_size=_file_size,
                preferred_rank=preferred_rank,
                tensor_parallel_size=tensor_parallel_size,
            )
        return p
    if not p.is_dir():
        return None
    candidates = _trace_candidates(p)
    if not candidates:
        return None
    return _select_trace_file(
        candidates,
        p,
        require_single_rank=require_single_rank,
        preferred_rank=preferred_rank,
        tensor_parallel_size=tensor_parallel_size,
    )


def _trace_rank_count(trace_input: str | Path) -> int:
    """Count distinct per-rank traces under ``trace_input``."""
    p = Path(trace_input)
    if p.is_file():
        return 1
    if not p.is_dir():
        return 0
    main = _main_trace_candidates(_trace_candidates(p), p)
    ranks = {r for c in main if (r := _rank_of(c)) is not None}
    return len(ranks) if ranks else 1


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


def _clipped(intervals: list[tuple[float, float]], window: tuple[float, float] | None) -> list[tuple[float, float]]:
    """Clip intervals to the steady window so occupancy math (busy/idle) stays within the window span."""
    if window is None:
        return intervals
    ws, we = window
    clipped = ((max(ws, start), min(we, end)) for start, end in intervals)
    return [(lo, hi) for lo, hi in clipped if hi > lo]


class _Agg:
    """Mutable aggregation accumulator for a single streaming pass."""

    __slots__ = ("dur_us", "count")

    def __init__(self) -> None:
        self.dur_us = 0.0
        self.count = 0

    def add(self, dur_us: float) -> None:
        self.dur_us += dur_us
        self.count += 1

    def row(self, name: str, denom_us: float) -> dict[str, Any]:
        """Report this aggregate as one ``ops``/``kernels`` row, with its share of ``denom_us``."""
        return {
            "name": name[:_MAX_EVENT_NAME_CHARS],
            "gpu_time_us": round(self.dur_us, 3),
            "gpu_time_ms": round(self.dur_us / 1000.0, 4),
            "count": self.count,
            "gpu_pct": round(self.dur_us / denom_us * 100.0, 4) if denom_us > 0 else 0.0,
        }


class _KernelAgg(_Agg):
    """One kernel name's aggregate, with what its launches carried to name and shape it in the ``kernels`` table."""

    __slots__ = ("op_us", "op_meta", "shape_us", "shape_meta", "launch_geom")

    def __init__(self) -> None:
        super().__init__()
        # Launching op -> attributed GPU us (the majority op names the kernel), and that op's first-seen meta.
        self.op_us: dict[str, float] = {}
        self.op_meta: dict[str, dict[str, Any]] = {}
        # Capture-time shape signature -> GPU us and first-seen meta: the majority shape is the backfill, and more than
        # one signature marks it ambiguous.
        self.shape_us: dict[tuple[tuple[tuple[int, ...], ...], tuple[str, ...]], float] = {}
        self.shape_meta: dict[tuple[tuple[tuple[int, ...], ...], tuple[str, ...]], dict[str, Any]] = {}
        # The first launch geometry (grid/block) any launch carried.
        self.launch_geom: tuple[Any, Any] | None = None

    def add_launch(
        self, dur_us: float, op_name: str, meta: dict[str, Any], launch_geom: tuple[Any, Any] | None
    ) -> None:
        """Count one launch, crediting its launching op and keeping the op meta and launch geometry it carried."""
        self.add(dur_us)
        self.op_us[op_name] = self.op_us.get(op_name, 0.0) + dur_us
        if meta and op_name != "(unlinked)":
            self.op_meta.setdefault(op_name, meta)
            sig = _backfill_shape_signature(meta)
            if sig[0]:
                self.shape_meta.setdefault(sig, meta)
                self.shape_us[sig] = self.shape_us.get(sig, 0.0) + dur_us
        self.launch_geom = self.launch_geom or launch_geom

    def row(self, name: str, denom_us: float) -> dict[str, Any]:
        """Report the kernel row, named by its majority launching op and carrying its shape fallbacks."""
        real_ops = {op: us for op, us in self.op_us.items() if us > 0.0 and op not in ("(unlinked)", "(graph)")}
        op_name = max(real_ops, key=real_ops.__getitem__, default="")
        meta = self.op_meta.get(op_name, {})
        backfill = self.shape_meta[max(self.shape_us, key=self.shape_us.__getitem__)] if self.shape_us else {}
        grid, block = self.launch_geom or (None, None)
        return {
            **super().row(name, denom_us),
            "op_name": op_name,
            "op_shapes": meta.get("shapes") or [],
            "op_dtypes": meta.get("dtypes") or [],
            "op_kernel_file": meta.get("kernel_file") or "",
            "op_kernel_backend": meta.get("kernel_backend") or "",
            # Shape fallbacks for a kernel whose own launch had no cpu_op shape: (a) same-name capture-time shape,
            # (b) launch geometry.
            "backfill_shapes": backfill.get("shapes") or [],
            "backfill_dtypes": backfill.get("dtypes") or [],
            "backfill_ambiguous": len(self.shape_us) > 1,
            "launch_grid": list(grid) if grid is not None else [],
            "launch_block": list(block) if block is not None else [],
        }


def _top_rows(agg: Mapping[str, _Agg], denom_us: float, top_k: int) -> list[dict[str, Any]]:
    """Return the rows of ``agg`` by descending GPU time, cut to the first ``top_k`` when it is positive."""
    rows = sorted((a.row(name, denom_us) for name, a in agg.items()), key=lambda r: r["gpu_time_ms"], reverse=True)
    return rows[:top_k] if top_k > 0 else rows


# Anchored so real iteration markers match but substrings like "writer" do not.
_STEP_MARKER_RE = re.compile(r"(?i)profilerstep|denoise|iteration|(?:^|[^a-z])step|step(?:$|[^a-z])")


def select_steady_window(
    annotation_windows: list[dict[str, Any]],
    *,
    framework: str = "",
    min_repeats: int = 3,
) -> dict[str, Any] | None:
    """Pick one representative steady-state iteration window from annotations."""
    if not annotation_windows:
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    for w in annotation_windows:
        name = w.get("name", "") or ""
        norm = re.sub(r"\d+$", "", name).rstrip("#_ -")
        if not norm:
            continue
        groups.setdefault(norm, []).append(w)
    if not groups:
        return None

    def _rank(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int]:
        norm, ws = item
        is_step = 1 if _STEP_MARKER_RE.search(norm) else 0
        return (is_step, len(ws))

    is_xdit = (framework or "").lower() == "xdit"
    threshold = 2 if is_xdit else min_repeats
    # Filter by threshold before ranking so a spurious low-count step-named annotation cannot win over a real
    # high-count loop and then be rejected.
    qualified = [(n, w) for n, w in groups.items() if len(w) >= threshold]
    if not qualified:
        return None
    norm, ws = max(qualified, key=_rank)

    ws_by_ts = sorted(ws, key=lambda w: float(w.get("ts", 0.0) or 0.0))
    steady = ws_by_ts[1:] if len(ws_by_ts) > 1 else ws_by_ts
    rep = sorted(steady, key=lambda w: float(w.get("dur", 0.0) or 0.0))[len(steady) // 2]
    start = float(rep.get("ts", 0.0) or 0.0)
    end = start + float(rep.get("dur", 0.0) or 0.0)
    return {
        "start_us": start,
        "end_us": end,
        "step_name": norm,
        "step_count": len(ws),
        "method": "annotation_step",
    }


# cpu_op ``args`` key -> op meta key.
_OP_META_ARGS = (("Input Dims", "shapes"), ("Input type", "dtypes"), ("kernel_file", "kernel_file"))


@dataclass
class _TraceBuffers:
    """Device events, annotations and attribution maps buffered in one pass over a trace.

    One pass serves both full-trace and steady-window aggregation without re-reading the trace.
    """

    # (name, duration, correlation, start, end) per GPU kernel; (duration, start, end) per memcpy/memset.
    k_events: list[tuple[str, float, Any, float, float]] = field(default_factory=list)
    m_events: list[tuple[float, float, float]] = field(default_factory=list)
    # Kineto attribution chain: launch correlation -> external id -> framework op.
    corr_to_extid: dict[int, int] = field(default_factory=dict)
    extid_to_opname: dict[int, str] = field(default_factory=dict)
    # Compact per-op meta (first-seen) for shape + Triton-source enrichment.
    extid_to_opmeta: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Launch geometry (grid/block) per correlation, kept so a kernel whose correlation->cpu_op shape chain is broken
    # (Triton direct-launch, graph replay) can still surface a launch-grid shape fallback.
    corr_to_launch_geom: dict[Any, tuple[Any, Any]] = field(default_factory=dict)
    # Correlations of CUDA/HIP graph-launch runtime events (no External id), so their replayed kernels are classified
    # graph-attributed, not (unlinked).
    graph_launch_corrs: set[int] = field(default_factory=set)
    # GPU user-annotation windows, the candidates for the steady-state window.
    annotation_windows: list[dict[str, Any]] = field(default_factory=list)
    # (pid, tid) -> [(ts, end, name), ...] for the duration-sanity check.
    stream_intervals: dict[Any, list[tuple[float, float, str]]] = field(default_factory=dict)
    # The cap that stopped buffering ("annotation_windows" or "gpu_events"); empty while every cap holds.
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
            self._add_cpu_op(ev)
        elif cat == "gpu_user_annotation" and ev.get("ph") == "X":
            if len(self.annotation_windows) >= _MAX_ANNOTATION_WINDOWS:
                self.truncation_reason = "annotation_windows"
                return
            ts = ev.get("ts", 0) or 0
            dur = ev.get("dur", 0) or 0
            name = str(ev.get("name", "") or "")[:_MAX_EVENT_NAME_CHARS]
            self.annotation_windows.append({"name": name, "ts": float(ts), "dur": float(dur)})
        elif cat in _GPU_CATS and ev.get("ph") == "X":
            self._add_device_event(ev)

    def _add_cpu_op(self, ev: dict[str, Any]) -> None:
        """Name the framework op behind one External id and keep its first-seen shape/dtype/Triton-source meta."""
        args = ev.get("args") or {}
        extid = args.get("External id")
        if extid is None:
            return
        self.extid_to_opname[extid] = ev.get("name", "") or ""
        if extid not in self.extid_to_opmeta:
            meta = {key: args[arg] for arg, key in _OP_META_ARGS if args.get(arg)}
            if "kernel_file" in meta:
                meta["kernel_backend"] = args.get("kernel_backend") or ""
            if meta:
                self.extid_to_opmeta[extid] = meta

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
            self.m_events.append((dur, ts, end))
            return
        kargs = ev.get("args") or {}
        corr = kargs.get("correlation")
        # Retain launch grid/block: for Triton (hipModuleLaunchKernel) and graph-replay kernels the correlation->cpu_op
        # shape chain is broken, but the launch geometry is a shape fallback.
        grid = kargs.get("grid")
        block = kargs.get("block")
        if grid is not None or block is not None:
            self.corr_to_launch_geom[corr] = (grid, block)
        self.k_events.append((name, dur, corr, ts, end))

    def launching_op(self, corr: Any) -> tuple[str, dict[str, Any]]:
        """Return the name and meta of the cpu_op that launched correlation ``corr``, both empty when unlinked."""
        extid = self.corr_to_extid.get(corr)
        return self.extid_to_opname.get(extid) or "", self.extid_to_opmeta.get(extid) or {}

    def graph_coverage(self, graph_kernels: int, busy_fraction: float) -> dict[str, Any]:
        """Report graph-replay recording health, flagging a trace that likely under-recorded replays."""
        launch_count = len(self.graph_launch_corrs)
        # Graph capture health is a WHOLE-TRACE property (activity-buffer overflow drops replays across the run), so the
        # recorded-launch coverage is computed over every buffered kernel -- never the steady window -- to stay
        # scope-consistent with the launch count (also whole-trace).
        launches_with_kernels = len({e[2] for e in self.k_events if e[2] in self.graph_launch_corrs})
        return {
            "graph_mode": launch_count > 0,
            "graph_launch_count": launch_count,
            "graph_launches_with_kernels": launches_with_kernels,
            "graph_attributed_kernels": graph_kernels,
            "busy_fraction": busy_fraction,
            "graph_under_recorded": (
                launch_count >= 2
                and graph_kernels > 0
                and launches_with_kernels / launch_count < _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX
            ),
        }

    def kernel_launches(self, k_events: list[tuple[str, float, Any, float, float]]) -> list[dict[str, Any]]:
        """Return time-ordered per-launch rows: fusion analysis needs the kernel adjacency name aggregation discards."""
        launches = []
        for name, dur, corr, ts, _end in k_events:
            op_name, meta = self.launching_op(corr)
            launches.append(
                {
                    "name": name[:_MAX_EVENT_NAME_CHARS],
                    "op_name": op_name,
                    "ts": ts,
                    "dur": dur,
                    "shapes": meta.get("shapes") or [],
                    "dtypes": meta.get("dtypes") or [],
                    "kernel_file": meta.get("kernel_file") or "",
                    "kernel_backend": meta.get("kernel_backend") or "",
                    "correlation": corr,
                }
            )
        return sorted(launches, key=lambda r: r["ts"])


def _finalize(
    buffers: _TraceBuffers,
    *,
    window: tuple[float, float] | None,
    top_k: int,
    emit_launches: bool = False,
) -> dict[str, Any]:
    """Build timeline + op/kernel aggregates from buffered device events."""
    k_events, m_events = buffers.k_events, buffers.m_events
    total_ms = 0.0
    if window is not None:
        ws, we = window
        k_events = [e for e in k_events if ws <= e[3] < we]
        m_events = [e for e in m_events if ws <= e[1] < we]
        # Steady scope: total is the representative step's wall span, so idle% reflects gaps within the step rather
        # than the active-kernel envelope.
        total_ms = max(0.0, (we - ws) / 1000.0)
    kernel_intervals = _clipped([(e[3], e[4]) for e in k_events], window)
    memcpy_intervals = _clipped([(e[1], e[2]) for e in m_events], window)
    if window is None and (envelope := kernel_intervals + memcpy_intervals):
        total_ms = (max(end for _, end in envelope) - min(start for start, _ in envelope)) / 1000.0
    kernel_union_ms = _union_ms(kernel_intervals)
    busy_ms = _union_ms(kernel_intervals + memcpy_intervals)
    idle_ms = max(0.0, total_ms - busy_ms)

    op_agg: defaultdict[str, _Agg] = defaultdict(_Agg)
    kern_agg: defaultdict[str, _KernelAgg] = defaultdict(_KernelAgg)
    attributed, graph, unlinked, memcpy = _Agg(), _Agg(), _Agg(), _Agg()
    for name, dur, corr, _ts, _end in k_events:
        op_name, meta = buffers.launching_op(corr)
        if op_name:
            attributed.add(dur)
        elif corr in buffers.graph_launch_corrs:
            op_name = "(graph)"
            graph.add(dur)
        else:
            op_name = "(unlinked)"
            unlinked.add(dur)
        op_agg[op_name].add(dur)
        kern_agg[name].add_launch(dur, op_name, meta, buffers.corr_to_launch_geom.get(corr))
    for dur, _ts, _end in m_events:
        memcpy.add(dur)
    kernel_us = sum(a.dur_us for a in kern_agg.values())
    coverage = buffers.graph_coverage(graph.count, round(busy_ms / total_ms, 4) if total_ms > 0 else 0.0)
    return {
        "kernel_launches": buffers.kernel_launches(k_events) if emit_launches else [],
        "timeline": {
            "total_time_ms": round(total_ms, 4),
            "busy_time_ms": round(busy_ms, 4),
            "idle_time_ms": round(idle_ms, 4),
            "kernel_union_ms": round(kernel_union_ms, 4),
            "gpu_memcpy_ms": round(memcpy.dur_us / 1000.0, 4),
            "gpu_kernel_sum_ms": round(kernel_us / 1000.0, 4),
            "idle_pct": round(idle_ms / total_ms * 100.0, 4) if total_ms > 0 else 0.0,
            "busy_pct": round(busy_ms / total_ms * 100.0, 4) if total_ms > 0 else 0.0,
        },
        "ops": _top_rows(op_agg, kernel_us, top_k),
        "kernels": _top_rows(kern_agg, kernel_us, top_k),
        "attribution": {
            "kernel_count": len(k_events),
            "attributed_kernels": attributed.count,
            "unlinked_kernels": unlinked.count,
            "attributed_gpu_ms": round(attributed.dur_us / 1000.0, 4),
            "unlinked_gpu_ms": round(unlinked.dur_us / 1000.0, 4),
            "attributed_pct": round(attributed.dur_us / kernel_us * 100.0, 2) if kernel_us > 0 else 0.0,
            "cuda_runtime_links": len(buffers.corr_to_extid),
            "cpu_ops": len(buffers.extid_to_opname),
            "gpu_memcpy_count": memcpy.count,
            "graph_mode": coverage["graph_mode"],
            "graph_launch_count": coverage["graph_launch_count"],
            "graph_attributed_kernels": graph.count,
            "graph_attributed_gpu_ms": round(graph.dur_us / 1000.0, 4),
        },
        "graph_coverage": coverage,
    }


@dataclass(frozen=True)
class TraceAnalysisOptions:
    """What one trace analysis keeps and which trace file it reads."""

    top_k: int = 10
    steady_state: bool = False
    framework: str = ""
    emit_launches: bool = False
    require_single_rank: bool = False
    tensor_parallel_size: int | None = None


def analyze_trace(trace_input: str | Path, options: TraceAnalysisOptions = TraceAnalysisOptions()) -> dict[str, Any]:
    """Stream a Kineto trace and return timeline + op/kernel aggregates."""
    tf = resolve_trace_file(
        trace_input,
        require_single_rank=options.require_single_rank,
        tensor_parallel_size=options.tensor_parallel_size,
    )
    if tf is None:
        return {
            "status": "failed",
            "error": f"no usable trace file at {trace_input}",
            "trace_file": "",
        }

    buffers = _TraceBuffers()
    event_total = 0
    stream_errors: list[str] = []
    with _open_trace_binary(tf) as fobj:
        for ev in stream_events(fobj, errors=stream_errors):
            event_total += 1
            buffers.add(ev)
            if buffers.truncation_reason:
                break

    steady_window: dict[str, Any] | None = None
    if options.steady_state:
        steady_window = select_steady_window(buffers.annotation_windows, framework=options.framework)
    window = None if steady_window is None else (steady_window["start_us"], steady_window["end_us"])
    body = _finalize(buffers, window=window, top_k=options.top_k, emit_launches=options.emit_launches)
    body["attribution"]["annotation_window_count"] = len(buffers.annotation_windows)
    stream_overlap = _stream_overlap_health(buffers.stream_intervals)
    if stream_overlap:
        body["timeline"]["stream_overlap"] = stream_overlap

    # Detect (relative to the input dir) whether the selected trace is a CUDA-graph capture shard, so the tool layer
    # can surface a health warning.
    _input_root = Path(trace_input)
    _input_root = _input_root if _input_root.is_dir() else None

    result: dict[str, Any] = {
        "status": "ok",
        "trace_file": str(tf),
        "event_total": event_total,
        "truncated": bool(buffers.truncation_reason),
        "stream_errors": stream_errors,
        "aggregation_scope": "full_trace" if window is None else "steady_state",
        "analyzed_rank": _rank_of(tf),
        "rank_count": _trace_rank_count(trace_input),
        "selected_capture_fragment": _is_capture_fragment(tf, _input_root),
        "annotation_windows": buffers.annotation_windows,
        **body,
    }
    if steady_window is not None:
        result["steady_window"] = steady_window
    elif options.steady_state:
        result["steady_window_status"] = "no_repeating_window_fell_back_to_full_trace"
    if buffers.truncation_reason:
        result["truncation_reason"] = buffers.truncation_reason
    return result
