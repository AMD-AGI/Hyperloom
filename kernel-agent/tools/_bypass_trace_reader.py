###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Independent, low-memory reader/aggregator for Kineto torch-profiler traces.

Used by the bypass analysis backend (``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``).
It never imports or shells out to TraceLens.

Design constraints (validated on a 3GB dev box against a 560MB / 2M-event
trace):

* **Streaming**: the ``traceEvents`` array is parsed element-by-element with
  the C-accelerated ``json.JSONDecoder.raw_decode`` so peak memory stays flat
  regardless of trace size (no full ``json.load``).
* **Attribution**: GPU kernel device time is attributed back to the launching
  ATen/framework operation via the standard Kineto correlation chain:
  ``kernel.args.correlation`` -> ``cuda_runtime.args.correlation`` ->
  ``cuda_runtime.args["External id"]`` -> ``cpu_op.args["External id"]``.
  Kernels whose op cannot be resolved are aggregated under ``(unlinked)``.
* **Aggregation scope**: whole trace, ranked by GPU-time share. Steady-state
  windowing is a separate, optional stage (see the annotation windows returned
  here); ranking by share is stable regardless of windowing.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

# GPU device-side event categories (Kineto ``cat`` values).
_GPU_KERNEL_CAT = "kernel"
_GPU_MEMCPY_CATS = ("gpu_memcpy", "gpu_memset")
_GPU_CATS = frozenset((_GPU_KERNEL_CAT,) + _GPU_MEMCPY_CATS)

_TRACE_EXTS = (".trace.json.gz", ".pt.trace.json.gz", ".trace.json", ".json.gz", ".json")

_DECODER = json.JSONDecoder()

# Per-rank trace filenames from the Magpie xDiT bridge / torch profiler, e.g.
# ``rank_0.trace.json.gz`` or ``...rank-3...``. Used for deterministic rank
# selection and multi-rank provenance.
_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)


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


def _select_trace_file(candidates: list[Path]) -> Path:
    """Deterministically pick one trace file from candidates.

    Order: a ``merged-*`` trace (largest, name tie-break) > the **lowest-index
    rank** trace (``rank_0`` first; representative under sequence/tensor
    parallel where every rank runs the same kernels on sharded data) > the
    largest file. Ties always break by name so selection is reproducible across
    runs (the previous ``max(..., key=size)`` was non-deterministic on equal
    sizes, which is exactly the xDiT TP>1 case of N equal per-rank traces).
    """
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
    deterministic and rank-aware (see :func:`_select_trace_file`): a merged
    trace wins, else the lowest-index per-rank trace, else the largest file.

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
    return _select_trace_file(candidates)


def _trace_rank_count(trace_input: str | Path) -> int:
    """Count distinct per-rank traces under ``trace_input``.

    Returns the number of distinct ``rank_N`` indices found in a directory
    (e.g. 8 for xDiT TP=8), or ``1`` when the input is a single file or has no
    rank-tagged traces. Used only for provenance / the multi-rank warning.
    """
    p = Path(trace_input)
    if p.is_file():
        return 1
    if not p.is_dir():
        return 0
    ranks = {r for c in _trace_candidates(p) if (r := _rank_of(c)) is not None}
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


def stream_events(fileobj, bufsize: int = 8 * 1024 * 1024) -> Iterator[dict]:
    """Yield each object inside the ``traceEvents`` array, one at a time.

    Locates the ``traceEvents`` array, then emits balanced ``{...}`` elements
    via ``raw_decode``. Constant memory; refills the buffer on truncation.

    Args:
        fileobj: A binary, possibly-decompressing file object.
        bufsize: Read/refill chunk size in bytes (also the buffer-trim
            threshold).

    Yields:
        Parsed trace-event dicts.
    """
    buf = ""
    key = '"traceEvents"'
    while key not in buf:
        chunk = fileobj.read(bufsize)
        if not chunk:
            return
        buf += chunk.decode("utf-8", "replace")
    pos = buf.index("[", buf.index(key)) + 1
    eof = False
    while True:
        while pos < len(buf) and buf[pos] in " \t\r\n,":
            pos += 1
        if pos < len(buf) and buf[pos] == "]":
            return
        if pos >= len(buf) or buf[pos] != "{":
            if eof:
                return
            chunk = fileobj.read(bufsize)
            if not chunk:
                eof = True
            else:
                buf += chunk.decode("utf-8", "replace")
            continue
        try:
            obj, end = _DECODER.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                return
            chunk = fileobj.read(bufsize)
            if not chunk:
                eof = True
            else:
                buf += chunk.decode("utf-8", "replace")
            continue
        yield obj
        pos = end
        if pos > bufsize:
            buf = buf[pos:]
            pos = 0


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


# Anchored so real iteration markers (ProfilerStep#N, decode_step, denoise,
# iteration) match, but substrings like "writer"/"iterator" do not.
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
    # Only groups that repeat enough to be a real steady loop qualify; among
    # those prefer a genuine step/iter marker, then the highest repeat count.
    # Filtering by threshold *before* ranking is deliberate: otherwise a spurious
    # low-count step/iter-named annotation (rank (1, 1)) would win over a real
    # high-count loop (rank (0, N)) and then be rejected, dropping the loop.
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

    Returns:
        A dict with ``timeline`` / ``ops`` / ``kernels`` / ``attribution``.
    """
    ws = we = None
    if window is not None:
        ws, we = window
        k_events = [e for e in k_events if ws <= e[3] < we]
        m_events = [e for e in m_events if ws <= e[1] < we]

    def _clip(a: float, b: float) -> tuple[float, float] | None:
        """Clip an interval to the steady window so occupancy math (busy/idle)
        stays within the window span. Returns ``None`` for an empty result.

        Kernel/memcpy *durations* below intentionally stay unclipped: GPU-time
        share ranks a kernel by its full cost, while wall-clock occupancy must
        not exceed the window (otherwise busy_pct could top 100%).
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
    # Per device-kernel name -> {launching op name -> attributed GPU us}, used
    # to pick a majority op name for each hot kernel (best-effort; cudagraph
    # replay leaves many kernels ``(unlinked)`` — see module docstring).
    kern_op: dict[str, dict[str, float]] = {}
    # Per device-kernel name -> {launching op name -> op meta} (first-seen shape/
    # dtype/kernel_file), so the majority op's meta can enrich the hot kernel.
    kern_op_meta: dict[str, dict[str, dict[str, Any]]] = {}
    attributed_us = 0.0
    attributed_kernels = 0
    unlinked_us = 0.0
    unlinked_kernels = 0
    for name, dur, corr, _ts, _end in k_events:
        extid = corr_to_extid.get(corr) if corr is not None else None
        op_name = extid_to_opname.get(extid) if extid is not None else None
        if op_name:
            attributed_us += dur
            attributed_kernels += 1
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

    def _majority_op(kernel_name: str) -> str:
        """Return the highest-GPU-time real launching op for a kernel name.

        Ignores the ``(unlinked)`` bucket; returns ``""`` when no op resolved.
        """
        best, best_dur = "", 0.0
        for op, d in (kern_op.get(kernel_name) or {}).items():
            if op != "(unlinked)" and d > best_dur:
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
        # reflects gaps *within* the step rather than just the active-kernel
        # envelope (a kernel may end just past the window; idle clamps at 0).
        total_ms = max(0.0, (window[1] - window[0]) / 1000.0)
    elif gpu_min_ts is not None and gpu_max_end is not None:
        total_ms = (gpu_max_end - gpu_min_ts) / 1000.0
    else:
        total_ms = 0.0
    idle_ms = max(0.0, total_ms - busy_ms)

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
            rows.append(row)
        rows.sort(key=lambda r: r["gpu_time_ms"], reverse=True)
        return rows if top_k is None or top_k <= 0 else rows[:top_k]

    return {
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
        },
    }


def analyze_trace(
    trace_input: str | Path,
    *,
    top_k: int = 10,
    steady_state: bool = False,
    framework: str = "",
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
          ``attribution`` (coverage stats),
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
    # Compact per-op meta (first-seen) for shape + Triton-source enrichment;
    # only populated for cpu_ops that carry Input Dims / a Triton kernel_file.
    extid_to_opmeta: dict[int, dict[str, Any]] = {}
    # Buffered device events so the same pass can serve full-trace and
    # steady-window aggregation without re-reading the (possibly 500MB+) trace.
    k_events: list[tuple[str, float, Any, float, float]] = []
    m_events: list[tuple[float, float, float]] = []
    annotation_windows: list[dict[str, Any]] = []
    event_total = 0

    fobj = _open_trace_binary(tf)
    try:
        for ev in stream_events(fobj):
            event_total += 1
            cat = ev.get("cat", "")
            if cat == "cuda_runtime":
                args = ev.get("args") or {}
                cid = args.get("correlation")
                extid = args.get("External id")
                if cid is not None and extid is not None:
                    corr_to_extid[cid] = extid
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
                annotation_windows.append(
                    {"name": ev.get("name", "") or "", "ts": float(ts), "dur": float(dur)}
                )
                continue
            if cat in _GPU_CATS and ev.get("ph") == "X":
                dur = float(ev.get("dur", 0) or 0)
                ts = float(ev.get("ts", 0) or 0)
                end = ts + dur
                name = ev.get("name", "") or ""
                if cat == _GPU_KERNEL_CAT:
                    corr = (ev.get("args") or {}).get("correlation")
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
        k_events, m_events, corr_to_extid, extid_to_opname, extid_to_opmeta, window=window, top_k=top_k
    )
    body["attribution"]["annotation_window_count"] = len(annotation_windows)

    result: dict[str, Any] = {
        "status": "ok",
        "trace_file": str(tf),
        "event_total": event_total,
        "aggregation_scope": scope,
        "analyzed_rank": _rank_of(tf),
        "rank_count": _trace_rank_count(trace_input),
        "annotation_windows": annotation_windows,
        **body,
    }
    if steady_window is not None:
        result["steady_window"] = steady_window
    elif steady_state:
        result["steady_window_status"] = "no_repeating_window_fell_back_to_full_trace"
    return result
