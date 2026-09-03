###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Independent, low-memory reader/aggregator for Kineto torch-profiler traces.

Used by the bypass analysis backend (``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``).
It never imports or shells out to TraceLens.

Design constraints:

* **Streaming**: the ``traceEvents`` array is parsed element-by-element with
  the C-accelerated ``json.JSONDecoder.raw_decode`` so peak memory stays flat
  regardless of trace size (no full ``json.load``).
* **Attribution**: GPU kernel device time is attributed back to the launching
  ATen/framework operation via the standard Kineto correlation chain:
  ``kernel.args.correlation`` -> ``cuda_runtime.args.correlation`` ->
  ``cuda_runtime.args["External id"]`` -> ``cpu_op.args["External id"]``.
  Kernels whose op cannot be resolved are aggregated under ``(unlinked)``.
* **Aggregation scope**: whole trace, ranked by GPU-time share. Steady-state
  windowing is a separate, optional stage; ranking by share is stable
  regardless of windowing.
"""

from __future__ import annotations

import codecs
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

# Stdlib-only sibling; keeps this reader independent of TraceLens while sharing
# one capture-vs-workload rule with the TraceLens route.
from _capture_shapes import is_capture_fragment as _shared_is_capture_fragment

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

# Hard caps keep a corrupt trace from turning the streaming reader into an
# unbounded accumulator. Kineto prefixes are normally a few KiB and individual
# events a few KiB, so these retain generous headroom for embedded metadata.
_MAX_TRACE_PREFIX_CHARS = 16 * 1024 * 1024
_MAX_EVENT_CHARS = 64 * 1024 * 1024

# Per-rank trace filename pattern (e.g. ``rank_0.trace.json.gz``).
_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)


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


# A graph trace is judged under-recorded when fewer than this fraction of its
# graph-launch correlations actually recorded any kernel: activity-buffer
# overflow drops whole replays, so recorded-launch coverage collapses toward
# ~1/launch_count, whereas a fully-recorded (merely idle) workload keeps kernels
# on essentially every launch.
_GRAPH_RECORDED_LAUNCH_COVERAGE_MAX = 0.5


def _graph_under_recorded(
    *,
    graph_mode: bool,
    graph_launch_count: int,
    graph_launches_with_kernels: int,
    graph_kernels: int,
) -> bool:
    """Return whether a graph-mode trace likely under-recorded replays.

    Keyed on *recorded-launch coverage* — the share of graph-launch correlations
    that recorded at least one kernel — NOT on busy%. A low busy fraction alone
    is ambiguous: a genuinely idle / sparse / host-bound workload whose replays
    are all fully recorded also looks idle, and must still be gated by the
    high-idle suppression. Only when the profiler dropped whole replays (coverage
    far below 1.0) is idle% an unreliable capture artifact.
    """
    if not graph_mode or graph_launch_count < 2 or graph_kernels <= 0:
        return False
    coverage = graph_launches_with_kernels / graph_launch_count
    return coverage < _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX


def _file_size(fp: Path) -> int:
    """Return file size in bytes, or 0 on stat() failure."""
    try:
        return fp.stat().st_size
    except OSError:
        return 0


def _rank_of(path: str | Path) -> int | None:
    """Parse the rank index from a trace filename (``None`` if not rank-tagged)."""
    m = _RANK_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def _trace_candidates(root: Path) -> list[Path]:
    """Return all trace-shaped files under ``root`` (recursive)."""
    out: list[Path] = []
    for child in root.rglob("*"):
        if child.is_file() and any(child.name.lower().endswith(ext) for ext in _TRACE_EXTS):
            out.append(child)
    return out


def _is_capture_fragment(path: str | Path, root: str | Path | None = None) -> bool:
    """True if ``path`` is a CUDA-graph capture shard, not a main trace.

    Capture shards are device-kernel sparse and must not be mistaken for the
    content-rich main profiler trace. Delegates to
    :func:`_capture_shapes.is_capture_fragment` so this route and the TraceLens
    route classify identically; the local copy this replaced only knew the
    ``bs_<batch>_rank<n>`` / ``capture_traces/`` shapes and silently accepted a
    ``graph_capture_profile/cuda_graph_capture-*`` sidecar as a workload trace.
    """
    return _shared_is_capture_fragment(path, root)


def _main_trace_candidates(candidates: list[Path], root: str | Path | None = None) -> list[Path]:
    """Drop CUDA-graph capture shards, keeping only main workload traces.

    Falls back to the full list when every candidate is a capture shard, so a
    trace is always resolvable (the selection pool is never emptied).
    """
    main = [c for c in candidates if not _is_capture_fragment(c, root)]
    return main or candidates


def _select_trace_file(candidates: list[Path], root: str | Path | None = None) -> Path:
    """Deterministically pick one trace file from candidates.

    Capture shards (see :func:`_is_capture_fragment`) are excluded first so the
    content-rich main trace is never shadowed by a sparse ``bs_*_rank0`` shard.
    Among the remaining main traces the order is: a ``merged-*`` trace (largest,
    name tie-break) > the lowest-index rank trace (``rank_0`` first) > the
    largest file. Ties always break by name so selection is reproducible.
    """
    candidates = _main_trace_candidates(candidates, root)
    merged = [c for c in candidates if c.name.startswith("merged-")]
    if merged:
        return max(merged, key=lambda c: (_file_size(c), c.name))
    ranked = [c for c in candidates if _rank_of(c) is not None]
    if ranked:
        return min(ranked, key=lambda c: (_rank_of(c), c.name))
    return max(candidates, key=lambda c: (_file_size(c), c.name))


def resolve_trace_file(trace_input: str | Path) -> Path | None:
    """Resolve a trace input (file or directory) to a single trace file.

    For a directory (e.g. a ``torch_trace/`` capture dir), selection is
    deterministic, rank-aware, and skips sglang CUDA-graph capture shards (see
    :func:`_select_trace_file`): a merged trace wins, else the lowest-index
    per-rank trace, else the largest file.

    Args:
        trace_input: A trace file path or a directory containing trace files.

    Returns:
        The resolved trace file path, or ``None`` when nothing usable is found.
    """
    p = Path(trace_input)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    candidates = _trace_candidates(p)
    if not candidates:
        return None
    return _select_trace_file(candidates, p)


def _trace_rank_count(trace_input: str | Path) -> int:
    """Count distinct per-rank traces under ``trace_input``.

    Returns the number of distinct ``rank_N`` indices found in a directory, or
    ``1`` when the input is a single file or has no rank-tagged traces. Capture
    shards are excluded so their ``rank0`` tag is not counted.
    """
    p = Path(trace_input)
    if p.is_file():
        return 1
    if not p.is_dir():
        return 0
    main = _main_trace_candidates(_trace_candidates(p), p)
    ranks = {r for c in main if (r := _rank_of(c)) is not None}
    return len(ranks) if ranks else 1


def _open_trace_binary(path: Path):
    """Open a trace file, transparently decompressing ``.gz``.

    Returns:
        A binary file object positioned at the start of the (decompressed)
        JSON stream. Caller is responsible for closing it.
    """
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


class _ObjectBalance:
    """Resumable brace balancer for a single ``traceEvents`` element.

    Only the recovery path of :func:`stream_events` needs it, so it keeps its
    scan position across refills instead of restarting over the grown buffer.
    """

    __slots__ = ("_depth", "_escaped", "_in_string", "_scan")

    def __init__(self, start: int) -> None:
        self._scan = start
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def advance(self, buf: str) -> int | None:
        """Consume newly buffered characters.

        Returns:
            The index one past the element's closing brace once the object
            balances, or ``None`` when the buffer ends mid-object.
        """
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


def stream_events(
    fileobj,
    bufsize: int = 8 * 1024 * 1024,
    *,
    errors: list[str] | None = None,
) -> Iterator[dict]:
    """Yield each object inside the ``traceEvents`` array, one at a time.

    Locates the ``traceEvents`` array, then emits balanced ``{...}`` elements
    via ``raw_decode``. Complete leading events remain recoverable from a
    truncated file, while ``errors`` distinguishes that recovery from a clean
    end of the array.

    Args:
        fileobj: A binary, possibly-decompressing file object.
        bufsize: Read/refill chunk size in bytes (also the buffer-trim
            threshold).
        errors: Optional output list for structural stream errors.

    Yields:
        Parsed trace-event dicts.
    """

    def _record(message: str) -> None:
        """Append one structural error when the caller requested diagnostics."""
        if errors is not None:
            errors.append(message)

    chunk_size = max(1, int(bufsize))
    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    eof = False

    def _refill(max_bytes: int | None = None) -> bool:
        """Decode one bounded chunk and turn gzip failures into stream errors."""
        nonlocal buf, eof
        if eof:
            return False
        read_size = chunk_size
        if max_bytes is not None:
            read_size = max(1, min(read_size, max_bytes))
        try:
            chunk = fileobj.read(read_size)
        except (EOFError, gzip.BadGzipFile) as exc:
            _record(f"trace input error: {type(exc).__name__}: {exc}")
            eof = True
            tail = utf8_decoder.decode(b"", final=True)
            if tail:
                buf += tail
            return bool(tail)
        if not chunk:
            eof = True
            tail = utf8_decoder.decode(b"", final=True)
            if tail:
                buf += tail
            return bool(tail)
        buf += utf8_decoder.decode(chunk, final=False)
        return True

    def _trim_consumed() -> None:
        """Discard parsed input once it exceeds one refill chunk."""
        nonlocal buf, pos
        if pos > chunk_size:
            buf = buf[pos:]
            pos = 0

    key = '"traceEvents"'
    search_from = 0
    while True:
        key_pos = buf.find(key, search_from)
        if key_pos < 0:
            if eof:
                _record("traceEvents array not found")
                return
            if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                return
            search_from = max(0, len(buf) - len(key) + 1)
            _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
            continue

        pos = key_pos + len(key)
        while pos >= len(buf):
            if eof:
                break
            if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                return
            _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
            if pos == len(buf) and not eof:
                if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
                    _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
                    return
                _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
        if pos < len(buf) and buf[pos] == ":":
            break
        search_from = key_pos + len(key)

    pos += 1
    while True:
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
        if pos < len(buf):
            break
        if eof:
            _record("traceEvents array opener not found")
            return
        if len(buf) >= _MAX_TRACE_PREFIX_CHARS:
            _record(f"traceEvents prefix exceeds {_MAX_TRACE_PREFIX_CHARS} characters")
            return
        _refill(_MAX_TRACE_PREFIX_CHARS - len(buf))
    if buf[pos] != "[":
        _record("traceEvents value is not an array")
        return

    buf = buf[pos + 1 :]
    pos = 0
    emitted = 0
    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf):
                break
            if eof:
                _record(f"traceEvents array unterminated after {emitted} event(s)")
                return
            _trim_consumed()
            _refill()

        if buf[pos] == "]":
            return

        if buf[pos] != "{":
            invalid_start = pos
            scan = pos
            curly_depth = 0
            square_depth = 0
            in_string = False
            escaped = False
            boundary: int | None = None
            while boundary is None:
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
                if boundary is not None:
                    break
                buffered = len(buf) - invalid_start
                if buffered >= _MAX_EVENT_CHARS:
                    _record(f"traceEvents element exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
                    return
                if eof:
                    _record(f"traceEvents array unterminated after {emitted} event(s)")
                    return
                _refill(_MAX_EVENT_CHARS - buffered)
            _record(f"traceEvents element malformed after {emitted} event(s): expected an object")
            pos = boundary
            _trim_consumed()
            continue

        object_start = pos
        obj: dict[str, Any] | None = None
        decoded_end = object_start
        balance = _ObjectBalance(object_start)
        while True:
            try:
                obj, decoded_end = _DECODER.raw_decode(buf, object_start)
            except json.JSONDecodeError as exc:
                # raw_decode cannot distinguish corrupt input from an object the
                # buffer merely has not reached the end of yet, so balance braces
                # to tell the two apart. Only this path pays for that scan, and
                # it stops at the object's own end rather than the buffer's.
                object_end = balance.advance(buf)
                if object_end is not None:
                    _record(f"traceEvents object malformed after {emitted} event(s): {exc.msg}")
                    pos = object_end
                    break
                buffered = len(buf) - object_start
                if buffered >= _MAX_EVENT_CHARS:
                    _record(f"traceEvents object exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
                    return
                if eof:
                    _record(f"traceEvents object truncated after {emitted} event(s): unterminated object")
                    return
                _refill(_MAX_EVENT_CHARS - buffered)
                continue
            break

        if obj is None:
            _trim_consumed()
            continue
        if decoded_end - object_start > _MAX_EVENT_CHARS:
            _record(f"traceEvents object exceeds {_MAX_EVENT_CHARS} characters after {emitted} event(s)")
            return
        if not isinstance(obj, dict):
            _record(f"traceEvents object malformed after {emitted} event(s): invalid object boundary")
            pos = decoded_end
            _trim_consumed()
            continue
        yield obj
        emitted += 1
        pos = decoded_end
        _trim_consumed()


def _stream_overlap_health(stream_events: dict[Any, list[tuple[float, float, str]]]) -> dict[str, Any]:
    """Report device streams whose event durations physically cannot hold.

    A hardware stream executes serially, so within a ``(pid, tid)`` row an
    event's end must not run past the next event's start; where it does, the
    duration is corrupt. The check is pairwise and the threshold is a share of
    summed device time rather than of the wall span, since a corrupt event
    inflates its own span. A corrupt duration on a stream's final event is not
    detectable -- nothing follows it to contradict the value.

    Args:
        stream_events: ``(pid, tid) -> [(ts, end, name), ...]``, unsorted.

    Returns:
        The worst offending stream as a dict (including a ``severity`` grade),
        or ``{}`` when every stream is self-consistent.
    """
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


class _Agg:
    """Mutable aggregation accumulator for a single streaming pass."""

    __slots__ = ("dur_us", "count")

    def __init__(self) -> None:
        self.dur_us = 0.0
        self.count = 0

    def add(self, dur_us: float) -> None:
        self.dur_us += dur_us
        self.count += 1


# Anchored so real iteration markers match but substrings like "writer" do not.
_STEP_MARKER_RE = re.compile(r"(?i)profilerstep|denoise|iteration|(?:^|[^a-z])step|step(?:$|[^a-z])")


def select_steady_window(
    annotation_windows: list[dict[str, Any]],
    *,
    framework: str = "",
    min_repeats: int = 3,
) -> dict[str, Any] | None:
    """Pick one representative steady-state iteration window from annotations.

    Kineto emits ``gpu_user_annotation`` spans such as ``ProfilerStep#N`` (or, for
    diffusion, a repeated per-step marker). This groups annotations by their
    digit-stripped name, keeps the group that best looks like a repeating step
    marker, drops the first occurrence (warm-up / torch.compile capture), and
    returns the *median-duration* remaining occurrence as the steady window.

    Args:
        annotation_windows: The ``annotation_windows`` list from a trace pass.
        framework: Framework hint; ``xdit`` diffusion steps are homogeneous, so
            two repeats are enough to trust a window.
        min_repeats: Minimum occurrences before a non-``xdit`` group is trusted.

    Returns:
        ``{start_us, end_us, step_name, step_count, method}`` or ``None`` when no
        repeating steady window can be identified (caller falls back to full).
    """
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
    # Filter by threshold before ranking so a spurious low-count step-named
    # annotation cannot win over a real high-count loop and then be rejected.
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


def _finalize(
    k_events: list[tuple[str, float, Any, float, float]],
    m_events: list[tuple[float, float, float]],
    corr_to_extid: dict[int, int],
    extid_to_opname: dict[int, str],
    extid_to_opmeta: dict[int, dict[str, Any]],
    *,
    window: tuple[float, float] | None,
    top_k: int,
    emit_launches: bool = False,
    graph_launch_corrs: frozenset[int] | None = None,
    graph_launch_count: int = 0,
    corr_to_launch_geom: dict[Any, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Build timeline + op/kernel aggregates from buffered device events.

    Args:
        k_events: Buffered kernels as ``(name, dur_us, correlation, ts, end)``.
        m_events: Buffered memcpy/memset as ``(dur_us, ts, end)``.
        corr_to_extid: ``correlation -> External id`` (from cuda_runtime events).
        extid_to_opname: ``External id -> op name`` (from cpu_op events).
        extid_to_opmeta: ``External id -> {shapes, dtypes, kernel_file,
            kernel_backend}`` (from cpu_op args); powers per-kernel shape and
            Triton-source enrichment on the hot-kernel rows.
        window: Optional ``(start_us, end_us)`` steady-state filter; a device
            event is kept when its start ``ts`` falls in ``[start, end)``.
        top_k: Row cap for the returned lists (``<= 0`` keeps all).
        emit_launches: When true, also emit the time-ordered per-launch rows.
        graph_launch_corrs: Correlation ids belonging to graph replays.
        graph_launch_count: Number of observed graph launches.
        corr_to_launch_geom: ``correlation -> (grid, block)`` launch geometry.

    Returns:
        A dict with ``kernel_launches`` (when ``emit_launches``) / ``timeline`` /
        ``ops`` / ``kernels`` / ``attribution`` / ``graph_coverage``.
    """
    # Graph capture health is a WHOLE-TRACE property (activity-buffer overflow
    # drops replays across the run), so the recorded-launch coverage must be
    # computed over the full event stream -- BEFORE any steady-window filter --
    # to stay scope-consistent with ``graph_launch_count`` (also whole-trace).
    # Otherwise a window holding one replay of a fully-recorded N-launch trace
    # would read as 1/N and be misflagged as under-recorded.
    _graph_corrs_full = graph_launch_corrs or frozenset()
    graph_launch_corrs_with_kernels: set[int] = {
        e[2] for e in k_events if e[2] is not None and e[2] in _graph_corrs_full
    }

    ws = we = None
    if window is not None:
        ws, we = window
        k_events = [e for e in k_events if ws <= e[3] < we]
        m_events = [e for e in m_events if ws <= e[1] < we]

    def _clip(a: float, b: float) -> tuple[float, float] | None:
        """Clip an interval to the steady window so occupancy math (busy/idle)
        stays within the window span. Returns ``None`` for an empty result.

        Kernel/memcpy durations stay unclipped (GPU-time share ranks by full
        cost) while wall-clock occupancy must not exceed the window.
        """
        if window is None:
            return (a, b)
        lo, hi = max(ws, a), min(we, b)
        return (lo, hi) if hi > lo else None

    kern_agg: dict[str, _Agg] = {}
    kernel_intervals: list[tuple[float, float]] = []
    memcpy_intervals: list[tuple[float, float]] = []
    gpu_min_ts: float | None = None
    gpu_max_end: float | None = None
    memcpy_us = 0.0
    memcpy_count = 0

    for name, dur, _corr, ts, end in k_events:
        gpu_min_ts = ts if gpu_min_ts is None or ts < gpu_min_ts else gpu_min_ts
        gpu_max_end = end if gpu_max_end is None or end > gpu_max_end else gpu_max_end
        iv = _clip(ts, end)
        if iv is not None:
            kernel_intervals.append(iv)
        ka = kern_agg.get(name)
        if ka is None:
            ka = kern_agg[name] = _Agg()
        ka.add(dur)
    for dur, ts, end in m_events:
        gpu_min_ts = ts if gpu_min_ts is None or ts < gpu_min_ts else gpu_min_ts
        gpu_max_end = end if gpu_max_end is None or end > gpu_max_end else gpu_max_end
        iv = _clip(ts, end)
        if iv is not None:
            memcpy_intervals.append(iv)
        memcpy_us += dur
        memcpy_count += 1

    # --- op-level attribution (kernel -> cuda_runtime -> cpu_op) ---
    op_agg: dict[str, _Agg] = {}
    # Kernel name -> {launching op name -> attributed GPU us}, to pick a majority
    # op name for each hot kernel.
    kern_op: dict[str, dict[str, float]] = {}
    # Kernel name -> {launching op name -> op meta} (first-seen shape/dtype/file).
    kern_op_meta: dict[str, dict[str, dict[str, Any]]] = {}
    attributed_us = 0.0
    attributed_kernels = 0
    unlinked_us = 0.0
    unlinked_kernels = 0
    # Graph-internal kernels launched via a captured CUDA/HIP graph: they carry a
    # correlation matching a graph-launch runtime event (which has no External id),
    # so they resolve to no cpu_op but are not a genuine attribution failure.
    graph_corrs = graph_launch_corrs or frozenset()
    graph_kernels = 0
    graph_gpu_us = 0.0
    geom = corr_to_launch_geom or {}
    # Name-keyed shape backfill: accumulate GPU time per (kernel, shape) so the
    # majority capture-time shape wins; multiple distinct shapes mark ambiguous.
    kern_name_backfill_by_sig: dict[str, dict[tuple[tuple[tuple[int, ...], ...], tuple[str, ...]], dict[str, Any]]] = {}
    # Name-keyed launch geometry (grid/block): first launch that actually carries
    # geometry wins. Order-independent, so a shapeless graph-replay launch seen
    # first does not shadow a later eager launch that recorded grid/block.
    kern_name_launch_geom: dict[str, tuple[Any, Any]] = {}
    for name, dur, corr, _ts, _end in k_events:
        extid = corr_to_extid.get(corr) if corr is not None else None
        op_name = extid_to_opname.get(extid) if extid is not None else None
        if op_name:
            attributed_us += dur
            attributed_kernels += 1
        elif corr is not None and corr in graph_corrs:
            op_name = "(graph)"
            graph_kernels += 1
            graph_gpu_us += dur
        else:
            op_name = "(unlinked)"
            unlinked_us += dur
            unlinked_kernels += 1
        oa = op_agg.get(op_name)
        if oa is None:
            oa = op_agg[op_name] = _Agg()
        oa.add(dur)
        ko = kern_op.get(name)
        if ko is None:
            ko = kern_op[name] = {}
        ko[op_name] = ko.get(op_name, 0.0) + dur
        if op_name != "(unlinked)" and extid is not None:
            meta = extid_to_opmeta.get(extid)
            if meta:
                kern_op_meta.setdefault(name, {}).setdefault(op_name, meta)
                if meta.get("shapes"):
                    sig = _backfill_shape_signature(meta)
                    if sig[0]:
                        by_sig = kern_name_backfill_by_sig.setdefault(name, {})
                        entry = by_sig.get(sig)
                        if entry is None:
                            by_sig[sig] = {"meta": meta, "gpu_us": dur}
                        else:
                            entry["gpu_us"] += dur
        if name not in kern_name_launch_geom:
            g = geom.get(corr)
            if g is not None and (g[0] is not None or g[1] is not None):
                kern_name_launch_geom[name] = g

    def _majority_op(kernel_name: str) -> str:
        """Return the highest-GPU-time real launching op for a kernel name.

        Ignores the synthetic ``(unlinked)`` and ``(graph)`` buckets; returns
        ``""`` when no real op resolved.
        """
        best, best_dur = "", 0.0
        for op, d in (kern_op.get(kernel_name) or {}).items():
            if op not in ("(unlinked)", "(graph)") and d > best_dur:
                best, best_dur = op, d
        return best

    def _majority_op_meta(kernel_name: str) -> dict[str, Any]:
        """Return the op meta (shapes/dtypes/kernel_file) of the majority op."""
        op = _majority_op(kernel_name)
        if not op:
            return {}
        return (kern_op_meta.get(kernel_name) or {}).get(op, {})

    kernel_union_ms = _union_ms(kernel_intervals)
    busy_ms = _union_ms(kernel_intervals + memcpy_intervals)
    if window is not None:
        # Steady scope: total is the representative step's wall span, so idle%
        # reflects gaps within the step rather than the active-kernel envelope.
        total_ms = max(0.0, (window[1] - window[0]) / 1000.0)
    elif gpu_min_ts is not None and gpu_max_end is not None:
        total_ms = (gpu_max_end - gpu_min_ts) / 1000.0
    else:
        total_ms = 0.0
    idle_ms = max(0.0, total_ms - busy_ms)

    # Graph coverage health: under continuous graph replay roctracer's activity
    # buffer overflows, so only ~1 replay's kernels are recorded. Flag when
    # graphs are replaying yet recorded busy covers <50% of the wall span, or
    # many launches map to a single recorded replay's worth of kernels.
    graph_mode = graph_launch_count > 0
    busy_fraction = round(busy_ms / total_ms, 4) if total_ms > 0 else 0.0
    graph_launches_with_kernels = len(graph_launch_corrs_with_kernels)
    graph_under_recorded = _graph_under_recorded(
        graph_mode=graph_mode,
        graph_launch_count=graph_launch_count,
        graph_launches_with_kernels=graph_launches_with_kernels,
        graph_kernels=graph_kernels,
    )
    kern_name_backfill_meta: dict[str, dict[str, Any]] = {}
    kern_name_backfill_ambiguous: set[str] = set()
    for kern_name, by_sig in kern_name_backfill_by_sig.items():
        if len(by_sig) > 1:
            kern_name_backfill_ambiguous.add(kern_name)
        best_sig = max(by_sig, key=lambda sig: by_sig[sig]["gpu_us"])
        kern_name_backfill_meta[kern_name] = by_sig[best_sig]["meta"]

    gpu_kernel_total_us = sum(a.dur_us for a in kern_agg.values())

    def _top(agg: dict[str, _Agg], denom_us: float, *, attach_op: bool = False) -> list[dict[str, Any]]:
        rows = []
        for nm, a in agg.items():
            row: dict[str, Any] = {
                "name": nm,
                "gpu_time_us": round(a.dur_us, 3),
                "gpu_time_ms": round(a.dur_us / 1000.0, 4),
                "count": a.count,
                "gpu_pct": round(a.dur_us / denom_us * 100.0, 4) if denom_us > 0 else 0.0,
            }
            if attach_op:
                row["op_name"] = _majority_op(nm)
                meta = _majority_op_meta(nm)
                row["op_shapes"] = meta.get("shapes") or []
                row["op_dtypes"] = meta.get("dtypes") or []
                row["op_kernel_file"] = meta.get("kernel_file") or ""
                row["op_kernel_backend"] = meta.get("kernel_backend") or ""
                # Shape fallbacks for a kernel whose own launch had no cpu_op
                # shape: (a) same-name capture-time shape, (b) launch geometry.
                bf = kern_name_backfill_meta.get(nm) or {}
                row["backfill_shapes"] = bf.get("shapes") or []
                row["backfill_dtypes"] = bf.get("dtypes") or []
                row["backfill_ambiguous"] = nm in kern_name_backfill_ambiguous
                lg = kern_name_launch_geom.get(nm)
                row["launch_grid"] = list(lg[0]) if lg and lg[0] is not None else []
                row["launch_block"] = list(lg[1]) if lg and lg[1] is not None else []
            rows.append(row)
        rows.sort(key=lambda r: r["gpu_time_ms"], reverse=True)
        return rows if top_k is None or top_k <= 0 else rows[:top_k]

    # Time-ordered per-launch sequence (opt-in) for fusion analysis, which needs
    # the kernel adjacency the name-aggregation discards.
    #
    # Each record additively carries the launching op's shape/dtype/source meta
    # (when resolvable via the correlation chain). Existing consumers read only
    # ``name``/``op_name``/``ts``/``dur``; the extra keys power the variant-
    # discriminating TraceShapeManifest producer without changing that contract.
    kernel_launches: list[dict[str, Any]] = []
    if emit_launches:
        for _name, _dur, _corr, _ts, _e in k_events:
            _ex = corr_to_extid.get(_corr) if _corr is not None else None
            _op = extid_to_opname.get(_ex) if _ex is not None else None
            _meta = extid_to_opmeta.get(_ex) if _ex is not None else None
            _meta = _meta or {}
            kernel_launches.append(
                {
                    "name": _name,
                    "op_name": _op or "",
                    "ts": _ts,
                    "dur": _dur,
                    "shapes": _meta.get("shapes") or [],
                    "dtypes": _meta.get("dtypes") or [],
                    "kernel_file": _meta.get("kernel_file") or "",
                    "kernel_backend": _meta.get("kernel_backend") or "",
                    "correlation": _corr,
                }
            )
        kernel_launches.sort(key=lambda r: r["ts"])

    return {
        "kernel_launches": kernel_launches,
        "timeline": {
            "total_time_ms": round(total_ms, 4),
            "busy_time_ms": round(busy_ms, 4),
            "idle_time_ms": round(idle_ms, 4),
            "kernel_union_ms": round(kernel_union_ms, 4),
            "gpu_memcpy_ms": round(memcpy_us / 1000.0, 4),
            "gpu_kernel_sum_ms": round(gpu_kernel_total_us / 1000.0, 4),
            "idle_pct": round(idle_ms / total_ms * 100.0, 4) if total_ms > 0 else 0.0,
            "busy_pct": round(busy_ms / total_ms * 100.0, 4) if total_ms > 0 else 0.0,
        },
        "ops": _top(op_agg, gpu_kernel_total_us),
        "kernels": _top(kern_agg, gpu_kernel_total_us, attach_op=True),
        "attribution": {
            "kernel_count": len(k_events),
            "attributed_kernels": attributed_kernels,
            "unlinked_kernels": unlinked_kernels,
            "attributed_gpu_ms": round(attributed_us / 1000.0, 4),
            "unlinked_gpu_ms": round(unlinked_us / 1000.0, 4),
            "attributed_pct": round(attributed_us / gpu_kernel_total_us * 100.0, 2) if gpu_kernel_total_us > 0 else 0.0,
            "cuda_runtime_links": len(corr_to_extid),
            "cpu_ops": len(extid_to_opname),
            "gpu_memcpy_count": memcpy_count,
            "graph_mode": graph_mode,
            "graph_launch_count": graph_launch_count,
            "graph_attributed_kernels": graph_kernels,
            "graph_attributed_gpu_ms": round(graph_gpu_us / 1000.0, 4),
        },
        "graph_coverage": {
            "graph_mode": graph_mode,
            "graph_launch_count": graph_launch_count,
            "graph_launches_with_kernels": graph_launches_with_kernels,
            "graph_attributed_kernels": graph_kernels,
            "busy_fraction": busy_fraction,
            "graph_under_recorded": graph_under_recorded,
        },
    }


def analyze_trace(
    trace_input: str | Path,
    *,
    top_k: int = 10,
    steady_state: bool = False,
    framework: str = "",
    emit_launches: bool = False,
) -> dict[str, Any]:
    """Stream a Kineto trace and return timeline + op/kernel aggregates.

    Args:
        trace_input: Trace file or capture directory.
        top_k: How many top ops/kernels to include in the returned lists
            (0 or negative means "all").
        steady_state: When True, try to restrict aggregation to a single
            representative steady-state iteration window (see
            :func:`select_steady_window`); falls back to the whole trace when no
            window is found. Ranking-by-share is stable either way.
        framework: Framework hint forwarded to the steady-state window selector.

    Returns:
        A dict with keys:
          ``trace_file`` (resolved path str), ``status``,
          ``timeline`` (total/busy/idle/kernel/memcpy ms),
          ``ops`` (op-level GPU-time aggregates, desc by gpu time),
          ``kernels`` (device-kernel aggregates, desc by gpu time),
          ``attribution`` (coverage stats + graph-mode signals),
          ``graph_coverage`` (graph-mode / under-recording health signals),
          ``annotation_windows`` (gpu_user_annotation name/ts/dur),
          ``aggregation_scope`` (``steady_state`` or ``full_trace``),
          ``steady_window`` (window meta when steady state was applied),
          ``analyzed_rank`` (rank index of the selected trace, or ``None``),
          ``rank_count`` (distinct per-rank traces found in the input dir),
          ``event_total`` (events scanned).
        On resolution failure, ``status='failed'`` with an ``error`` string.
    """
    tf = resolve_trace_file(trace_input)
    if tf is None:
        return {
            "status": "failed",
            "error": f"no usable trace file at {trace_input}",
            "trace_file": "",
        }

    # Correlation maps + light buffers (see module docstring for the chain).
    corr_to_extid: dict[int, int] = {}
    extid_to_opname: dict[int, str] = {}
    # Compact per-op meta (first-seen) for shape + Triton-source enrichment.
    extid_to_opmeta: dict[int, dict[str, Any]] = {}
    # Launch geometry (grid/block) per correlation, kept so a kernel whose
    # correlation->cpu_op shape chain is broken (Triton direct-launch,
    # graph replay) can still surface a launch-grid shape fallback.
    corr_to_launch_geom: dict[Any, tuple[Any, Any]] = {}
    # Buffered device events so one pass serves both full-trace and steady-window
    # aggregation without re-reading the trace.
    k_events: list[tuple[str, float, Any, float, float]] = []
    m_events: list[tuple[float, float, float]] = []
    annotation_windows: list[dict[str, Any]] = []
    # Correlations of CUDA/HIP graph-launch runtime events (no External id), so
    # their replayed kernels are classified graph-attributed, not (unlinked).
    graph_launch_corrs: set[int] = set()
    # (pid, tid) -> [(ts, end, name), ...] for the duration-sanity check.
    stream_intervals: dict[Any, list[tuple[float, float, str]]] = {}
    event_total = 0
    stream_errors: list[str] = []

    fobj = _open_trace_binary(tf)
    try:
        for ev in stream_events(fobj, errors=stream_errors):
            event_total += 1
            cat = ev.get("cat", "")
            if cat == "cuda_runtime":
                args = ev.get("args") or {}
                cid = args.get("correlation")
                extid = args.get("External id")
                if cid is not None and extid is not None:
                    corr_to_extid[cid] = extid
                if cid is not None and "GraphLaunch" in (ev.get("name", "") or ""):
                    graph_launch_corrs.add(cid)
                continue
            if cat == "cpu_op":
                args = ev.get("args") or {}
                extid = args.get("External id")
                if extid is not None:
                    extid_to_opname[extid] = ev.get("name", "") or ""
                    if extid not in extid_to_opmeta:
                        meta: dict[str, Any] = {}
                        dims = args.get("Input Dims")
                        if dims:
                            meta["shapes"] = dims
                        itype = args.get("Input type")
                        if itype:
                            meta["dtypes"] = itype
                        kfile = args.get("kernel_file")
                        if kfile:
                            meta["kernel_file"] = kfile
                            meta["kernel_backend"] = args.get("kernel_backend") or ""
                        if meta:
                            extid_to_opmeta[extid] = meta
                continue
            if cat == "gpu_user_annotation" and ev.get("ph") == "X":
                ts = ev.get("ts", 0) or 0
                dur = ev.get("dur", 0) or 0
                annotation_windows.append({"name": ev.get("name", "") or "", "ts": float(ts), "dur": float(dur)})
                continue
            if cat in _GPU_CATS and ev.get("ph") == "X":
                dur = float(ev.get("dur", 0) or 0)
                ts = float(ev.get("ts", 0) or 0)
                end = ts + dur
                name = ev.get("name", "") or ""
                stream_intervals.setdefault((ev.get("pid"), ev.get("tid")), []).append((ts, end, name))
                if cat == _GPU_KERNEL_CAT:
                    kargs = ev.get("args") or {}
                    corr = kargs.get("correlation")
                    # Retain launch grid/block: for Triton (hipModuleLaunchKernel)
                    # and graph-replay kernels the correlation->cpu_op shape chain
                    # is broken, but the launch geometry is a shape fallback.
                    grid = kargs.get("grid")
                    block = kargs.get("block")
                    if grid is not None or block is not None:
                        corr_to_launch_geom[corr] = (grid, block)
                    k_events.append((name, dur, corr, ts, end))
                else:
                    m_events.append((dur, ts, end))
    finally:
        fobj.close()

    scope = "full_trace"
    steady_window: dict[str, Any] | None = None
    window: tuple[float, float] | None = None
    if steady_state:
        steady_window = select_steady_window(annotation_windows, framework=framework)
        if steady_window is not None:
            window = (steady_window["start_us"], steady_window["end_us"])
            scope = "steady_state"

    body = _finalize(
        k_events,
        m_events,
        corr_to_extid,
        extid_to_opname,
        extid_to_opmeta,
        window=window,
        top_k=top_k,
        emit_launches=emit_launches,
        graph_launch_corrs=frozenset(graph_launch_corrs),
        graph_launch_count=len(graph_launch_corrs),
        corr_to_launch_geom=corr_to_launch_geom,
    )
    body["attribution"]["annotation_window_count"] = len(annotation_windows)
    stream_overlap = _stream_overlap_health(stream_intervals)
    if stream_overlap:
        body["timeline"]["stream_overlap"] = stream_overlap

    # Detect (relative to the input dir) whether the selected trace is a
    # CUDA-graph capture shard, so the tool layer can surface a health warning.
    _input_root = Path(trace_input)
    _input_root = _input_root if _input_root.is_dir() else None

    result: dict[str, Any] = {
        "status": "ok",
        "trace_file": str(tf),
        "event_total": event_total,
        "stream_errors": stream_errors,
        "aggregation_scope": scope,
        "analyzed_rank": _rank_of(tf),
        "rank_count": _trace_rank_count(trace_input),
        "selected_capture_fragment": _is_capture_fragment(tf, _input_root),
        "annotation_windows": annotation_windows,
        **body,
    }
    if steady_window is not None:
        result["steady_window"] = steady_window
    elif steady_state:
        result["steady_window_status"] = "no_repeating_window_fell_back_to_full_trace"
    return result
