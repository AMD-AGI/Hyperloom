# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile self-certification: can this trace be analysed, and would the answer be true?

``certify_trace_dir`` is the entry point; the caller supplies ``chunk_files`` and
the workload parameters. The profile executor calls it once the capture lands,
which is why there is no driver here: the executor already holds the session
identity and the attempt context a standalone driver would have had to rebuild.

Implements ``trace-selfcert-checklist.md``. The two questions are answered
independently and never merged into one verdict, because a trace that analyses
cleanly can still yield a false decode conclusion -- an under-recorded CUDA-graph
capture produces hot kernels drawn entirely from prefill and nothing complains.

Groups 1-5 reuse ``_bypass_trace_reader.analyze_trace()`` so the certificate and
the shipping consumer see identical numbers from identical code. Groups 6-8 --
step structure, split forecast and the idle gate -- are new, and are the reason
this exists: they answer "which ``--steady-state-mode`` will work" at capture
time instead of after an analysis has already failed.

Two streaming passes per source file. The first is ``analyze_trace`` itself; the
second collects step annotations and per-step subtree attribution, which
``analyze_trace`` does not retain. Scopes are deliberately different between the
two and must not be merged: graph coverage is a whole-trace property, idle is a
within-window one.
"""

from __future__ import annotations

import math
import os
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

try:  # in-package import
    from hyperloom.agents.kernel.tools._bypass_trace_reader import (
        _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
        _MAX_EVENT_CHARS,
        _MAX_TRACE_PREFIX_CHARS,
        _open_trace_binary,
        _rank_of,
        _select_trace_file,
        _trace_candidates,
        _union_ms,
        analyze_trace,
        resolve_trace_file,
        select_steady_window,
        stream_events,
    )
    from hyperloom.agents.kernel.tools._capture_shapes import is_capture_fragment
except ImportError:  # sys.path import
    from _bypass_trace_reader import (
        _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
        _MAX_EVENT_CHARS,
        _MAX_TRACE_PREFIX_CHARS,
        _open_trace_binary,
        _rank_of,
        _select_trace_file,
        _trace_candidates,
        _union_ms,
        analyze_trace,
        resolve_trace_file,
        select_steady_window,
        stream_events,
    )
    from _capture_shapes import is_capture_fragment

SCHEMA_VERSION = 1
PROBE_VERSION = "selfcert-1.0.0"

#: Checklist thresholds. Every one is the value the shipping consumer uses, so a
#: certificate and the real gate cannot disagree. Overridable values are read
#: from the environment and echoed into ``thresholds_effective``.
IDLE_PCT_THRESHOLD_DEFAULT = 80.0
IDLE_PCT_ENV = "HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD"
CHUNK_QUALITY_MIN_BUSY_RATIO = 0.05
CHUNK_QUALITY_ALTERNATE_MARGIN = 0.10
MIN_REPEATS_DEFAULT = 3
MIN_REPEATS_XDIT = 2

#: Splitter CLI defaults, overridden per-directory when the benchmark config
#: says otherwise.
DEFAULT_NUM_STEPS = 32
DEFAULT_R = 1.0

#: Iteration-root patterns, verbatim from ``split_inference_trace_annotation``.
#: The primary vLLM shape wins outright; the backup list (vLLM's older shapes and
#: SGLang's per-step annotations) is consulted only when the primary matches
#: nothing. Reproducing that precedence matters -- a looser or differently
#: ordered pattern set would forecast over steps the splitter never sees.
_PRIMARY_ROOT_PATTERNS = (
    re.compile(r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"),
)
_BACKUP_ROOT_PATTERNS = (
    re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"step\[(?:EXTEND|DECODE|MIXED)\b.*\]"),
)
_SGLANG_STEP_RE = re.compile(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]")
_STEP_MARKER_RE = re.compile(r"(?i)profilerstep|denoise|iteration|(?:^|[^a-z])step|step(?:$|[^a-z])")
_ROOFLINE_TASK_RE = re.compile(r"/runs/[a-z_]+/([0-9a-f]{32})/")
_SPLIT_NAME_RE = re.compile(r"^(?P<mode>.+?)_steady_state_prefill_\d+_prefilldecode_\d+_decode_\d+_bs\d+_conc\d+_")

_SCRIPTABLE_FRAMEWORKS = frozenset(("sglang", "vllm", "trtllm", "tensorrt_llm"))

#: The splitter's internal mode names differ from the value a caller sets in
#: ``INFERENCE_OPTIMIZER_STEADY_STATE_MODE``. Recommendations are reported in the
#: caller's vocabulary so the certificate is directly actionable.
_CONSUMER_MODE = {"mixed": "mixed", "decode_only": "decode_only", "max_prefilldecode": "prefilldecode"}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Scalars pulled out of the benchmark ``config.yaml``. Matched line-anchored on
#: purpose: the same key names also occur inside a quoted JSON blob elsewhere in
#: the file with different values, and that blob must not win. Read by regex
#: rather than a YAML parser to keep the certifier stdlib-only.
_CONFIG_KEYS = {
    "conc": (re.compile(r"^\s+CONC:\s*(\d+)\s*$", re.MULTILINE), int),
    "osl": (re.compile(r"^\s+OSL:\s*([\d.]+)\s*$", re.MULTILINE), float),
    "isl": (re.compile(r"^\s+ISL:\s*([\d.]+)\s*$", re.MULTILINE), float),
    "r": (re.compile(r"^\s+RANDOM_RANGE_RATIO:\s*([\d.]+)\s*$", re.MULTILINE), float),
    "num_steps": (re.compile(r"^\s+num_steps:\s*(\d+)\s*$", re.MULTILINE), int),
    "framework": (re.compile(r"^framework:\s*(\S+)\s*$", re.MULTILINE), str),
}


def read_workload_params(trace_dir: str | Path) -> dict[str, Any]:
    """Read CONC / OSL / R / num_steps / framework from the benchmark config.

    These are capture-time workload parameters, not analysis results, so reading
    them keeps the certificate independent of any analysis having been run --
    which is the point of certifying at capture time.
    """
    out: dict[str, Any] = {"source": None}
    config = Path(trace_dir).parent / "config.yaml"
    if not config.is_file():
        return out
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return out
    out["source"] = str(config)
    for key, (pattern, cast) in _CONFIG_KEYS.items():
        m = pattern.search(text)
        if m:
            try:
                out[key] = cast(m.group(1))
            except ValueError:
                pass
    return out


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return round(num / den, 6)


def effective_thresholds() -> dict[str, Any]:
    """Resolve the gate constants, honouring the environment overrides."""
    idle = IDLE_PCT_THRESHOLD_DEFAULT
    raw = os.environ.get(IDLE_PCT_ENV)
    if raw:
        try:
            idle = float(raw)
        except ValueError:
            pass
    return {
        "graph_launch_coverage_max": _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
        "idle_pct_threshold": idle,
        "idle_pct_threshold_source": "env" if raw else "default",
        "min_repeats": MIN_REPEATS_DEFAULT,
        "chunk_quality_min_busy_ratio": CHUNK_QUALITY_MIN_BUSY_RATIO,
        "chunk_quality_alternate_margin": CHUNK_QUALITY_ALTERNATE_MARGIN,
        "trace_prefix_max_chars": _MAX_TRACE_PREFIX_CHARS,
        "event_max_chars": _MAX_EVENT_CHARS,
    }


# --------------------------------------------------------------------------
# group 2: candidate inventory and selection
# --------------------------------------------------------------------------


def _role_of(path: Path, root: Path) -> str:
    if "trace_split" in path.parts:
        return "split_chunk"
    if is_capture_fragment(path, root):
        return "capture_sidecar"
    return "source"


def inventory(trace_dir: Path) -> dict[str, Any]:
    """List every trace-shaped candidate, and pick the file to certify.

    Reports two selections, because they can differ and the difference is itself a
    finding. ``production_selected_path`` is what ``resolve_trace_file`` would hand
    the analyzer. ``selected_path`` is what gets certified: the best *source*
    trace, chosen with production's own ranking but restricted to source-role
    candidates.

    They diverge when a capture directory also holds the splitter's output --
    xdit writes ``trace_split/`` inside ``torch_trace/`` -- because the live
    resolver filters capture shards but not split chunks, and a chunk can win the
    name tie-break. Certifying that chunk would be meaningless: splitting consumes
    the annotations, so the chunk shows none and the directory would be recorded
    as unable to support a steady window when the source supports one fine.
    """
    candidates = []
    for p in sorted(_trace_candidates(trace_dir)):
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        candidates.append({"path": str(p), "bytes": size, "role": _role_of(p, trace_dir)})

    production = resolve_trace_file(trace_dir)
    production_role = _role_of(production, trace_dir) if production else None

    source_paths = [Path(c["path"]) for c in candidates if c["role"] == "source"]
    selected = _select_trace_file(source_paths, trace_dir) if source_paths else production

    ranks = {_rank_of(p) for p in source_paths}
    ranks.discard(None)
    return {
        "candidates": candidates,
        "selected_path": str(selected) if selected else None,
        "selected_role": _role_of(selected, trace_dir) if selected else None,
        "production_selected_path": str(production) if production else None,
        "production_selected_role": production_role,
        "production_would_analyze_split_chunk": production_role == "split_chunk",
        "selected_capture_fragment": bool(selected) and is_capture_fragment(selected, trace_dir),
        "rank_count": len(ranks) if ranks else (1 if source_paths else 0),
        "file_count_by_role": {
            role: sum(1 for c in candidates if c["role"] == role)
            for role in ("source", "capture_sidecar", "split_chunk")
        },
    }


# --------------------------------------------------------------------------
# groups 6/7/8: the second pass
# --------------------------------------------------------------------------


class StepPass:
    """Second-pass state: step annotations and per-step subtree attribution.

    A chunk is an annotation subtree, so a step's GPU work is found by taking the
    ``cuda_runtime`` launches whose host-side timestamp falls inside the step and
    then following their correlation ids to device kernels. Selecting kernels by
    timestamp instead would sweep in work launched from sibling annotations --
    the scheduler and result-copy spans that run in the same interval -- which is
    exactly the mistake that makes an empty chunk look like splitter data loss.
    """

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.root_family: str | None = None
        self._primary_roots: list[dict[str, Any]] = []
        self._backup_roots: list[dict[str, Any]] = []
        self.gpu_annotations: list[dict[str, Any]] = []
        self.runtime_ts: list[float] = []
        self.runtime_corr: list[Any] = []
        self.graph_corrs: set[Any] = set()
        self.kernel_by_corr: dict[Any, list[tuple[float, float, float]]] = {}
        self.cpu_op_total = 0
        self.cpu_op_with_meta = 0
        self.gpu_intervals: list[tuple[float, float]] = []

    def scan(self, path: Path) -> list[str]:
        errors: list[str] = []
        runtime: list[tuple[float, Any]] = []
        fobj = _open_trace_binary(path)
        try:
            for ev in stream_events(fobj, errors=errors):
                cat = ev.get("cat", "")
                if cat == "cuda_runtime":
                    args = ev.get("args") or {}
                    corr = args.get("correlation")
                    if corr is None:
                        continue
                    runtime.append((float(ev.get("ts") or 0.0), corr))
                    if "GraphLaunch" in (ev.get("name") or ""):
                        self.graph_corrs.add(corr)
                    continue
                if cat == "cpu_op":
                    args = ev.get("args") or {}
                    self.cpu_op_total += 1
                    if args.get("Input Dims") or args.get("Input type"):
                        self.cpu_op_with_meta += 1
                    continue
                if ev.get("ph") != "X":
                    continue
                if cat == "user_annotation":
                    name = ev.get("name") or ""
                    primary = any(p.match(name) for p in _PRIMARY_ROOT_PATTERNS)
                    backup = any(p.match(name) for p in _BACKUP_ROOT_PATTERNS)
                    if primary or backup:
                        ts = float(ev.get("ts") or 0.0)
                        dur = float(ev.get("dur") or 0.0)
                        root = {"name": name, "ts": ts, "end": ts + dur, "dur_us": dur}
                        if primary:
                            self._primary_roots.append(root)
                        if backup:
                            self._backup_roots.append(root)
                    continue
                if cat == "gpu_user_annotation":
                    ts = float(ev.get("ts") or 0.0)
                    dur = float(ev.get("dur") or 0.0)
                    self.gpu_annotations.append({"name": ev.get("name") or "", "ts": ts, "dur": dur})
                    continue
                if cat == "kernel":
                    ts = float(ev.get("ts") or 0.0)
                    dur = float(ev.get("dur") or 0.0)
                    corr = (ev.get("args") or {}).get("correlation")
                    self.kernel_by_corr.setdefault(corr, []).append((ts, ts + dur, dur))
                    self.gpu_intervals.append((ts, ts + dur))
                elif cat in ("gpu_memcpy", "gpu_memset"):
                    ts = float(ev.get("ts") or 0.0)
                    dur = float(ev.get("dur") or 0.0)
                    self.gpu_intervals.append((ts, ts + dur))
        finally:
            fobj.close()
        runtime.sort(key=lambda x: x[0])
        self.runtime_ts = [t for t, _ in runtime]
        self.runtime_corr = [c for _, c in runtime]
        if self._primary_roots:
            roots, self.root_family = self._primary_roots, "primary"
        elif self._backup_roots:
            roots, self.root_family = self._backup_roots, "backup"
        else:
            roots, self.root_family = [], None
        roots.sort(key=lambda s: s["ts"])
        self.steps = [{**r, **iter_details_from_name(r["name"])} for r in roots]
        return errors

    def attribute_steps(self) -> None:
        """Fill each step with the device work its own launches produced."""
        for idx, step in enumerate(self.steps):
            lo = bisect_left(self.runtime_ts, step["ts"])
            hi = bisect_right(self.runtime_ts, step["end"])
            corrs = set(self.runtime_corr[lo:hi])
            intervals: list[tuple[float, float]] = []
            kernels = 0
            for corr in corrs:
                for k_ts, k_end, _dur in self.kernel_by_corr.get(corr, ()):
                    intervals.append((k_ts, k_end))
                    kernels += 1
            step.update(
                {
                    "index": idx,
                    "runtime_launch_count": hi - lo,
                    "graph_launch_count": sum(1 for c in corrs if c in self.graph_corrs),
                    "kernel_count": kernels,
                    "gpu_busy_us": round(_union_ms(intervals) * 1000.0, 3) if intervals else 0.0,
                }
            )

    def window_stats(self, lo_idx: int, hi_idx: int) -> dict[str, Any]:
        """Aggregate a half-open step range the way the splitter's chunk would.

        ``busy_ratio`` divides by the GPU-side extent of the attributed kernels
        rather than the steps' wall span, matching the live quality gate, which
        reads ``gpu_busy_duration / gpu_duration`` from the splitter's CSV. Using
        the host-side span instead yields ratios above 1, because a step's kernels
        keep running after its host annotation has closed.
        """
        window = self.steps[lo_idx:hi_idx]
        empty = {
            "num_gpu_events_pred": 0,
            "gpu_busy_us_pred": 0.0,
            "busy_ratio_pred": 0.0,
            "gpu_extent_us": 0.0,
            "host_span_us": 0.0,
        }
        if not window:
            return empty
        intervals: list[tuple[float, float]] = []
        kernels = 0
        for step in window:
            lo = bisect_left(self.runtime_ts, step["ts"])
            hi = bisect_right(self.runtime_ts, step["end"])
            for corr in set(self.runtime_corr[lo:hi]):
                for k_ts, k_end, _dur in self.kernel_by_corr.get(corr, ()):
                    intervals.append((k_ts, k_end))
                    kernels += 1
        host_span = round(window[-1]["end"] - window[0]["ts"], 3)
        if not intervals:
            return {**empty, "host_span_us": host_span}
        busy = _union_ms(intervals) * 1000.0
        extent = max(iv[1] for iv in intervals) - min(iv[0] for iv in intervals)
        return {
            "num_gpu_events_pred": kernels,
            "gpu_busy_us_pred": round(busy, 3),
            "busy_ratio_pred": round(min(1.0, busy / extent), 6) if extent > 0 else 0.0,
            "gpu_extent_us": round(extent, 3),
            "host_span_us": host_span,
        }


def iter_details_from_name(name: str) -> dict[str, Any]:
    """Map an iteration-root annotation to the request counts the splitter derives.

    Transcribed from ``get_iter_details_from_name``. SGLang's DECODE contributes
    generation requests only, while EXTEND and MIXED are both treated as
    prefill-bearing with ``toks`` standing in for the batch size. vLLM's
    ``execute_..._context_...`` names are decoded by collapsing the sq/sk shape
    letters to separators and picking counts positionally, which is fragile but is
    exactly what the splitter does.
    """
    m = _SGLANG_STEP_RE.match(name)
    if m:
        kind, bs = m.group(1), int(m.group(2))
        toks = int(m.group(3) or 0)
        if kind == "DECODE":
            ctx_req, ctx_sum, gen_req, gen_sum, batch = 0, 0, bs, bs, bs
        else:
            ctx_req, ctx_sum, gen_req, gen_sum, batch = bs, toks, 0, 0, (toks or bs)
        return _details(kind, batch, ctx_req, ctx_sum, gen_req, gen_sum)

    try:
        parts = re.sub(r"[sqk]+", "_", name.replace("(", "_").replace(")", "_")).split("_")
        if len(parts) < 10:
            idx = (2, 3, 6, 7)
        elif len(parts) < 12:
            idx = (2, 3, 7, 8)
        else:
            idx = (3, 5, 11, 13)
        ctx_req, ctx_sum, gen_req, gen_sum = (int(parts[i]) for i in idx)
        kind = "EXTEND" if ctx_req > 0 and gen_req == 0 else ("MIXED" if ctx_req > 0 else "DECODE")
        return _details(kind, ctx_sum + gen_sum, ctx_req, ctx_sum, gen_req, gen_sum)
    except (ValueError, IndexError):
        return _details("DECODE", 1, 0, 0, 1, 1)


def _details(kind: str, batch: int, ctx_req: int, ctx_sum: int, gen_req: int, gen_sum: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "batch_size": batch,
        "num_requests": ctx_req + gen_req,
        "context_requests": ctx_req,
        "context_sum": ctx_sum,
        "generation_requests": gen_req,
        "generation_sum": gen_sum,
        "is_prefill": ctx_req > 0,
    }


def identify_steady_state_regions(details: Sequence[dict[str, Any]], num_steps: int) -> list[tuple[int, int]]:
    """Reproduce the splitter's steady-state region detection.

    Transcribed from ``identify_steady_state_regions``, including its quirks: the
    running counter is decremented rather than reset, and the trailing region is
    closed at the last index rather than at ``len(details)``, so the final step is
    excluded. That off-by-one is load-bearing -- it is why a trace whose only
    prefill step sits at the end can still forecast as decode-only.
    """
    n = len(details)
    if not n:
        return []
    thresh = 0.1 if n >= num_steps else 0.2
    global_max = max(t["num_requests"] for t in details)

    started = False
    in_steady = 0
    start_index = 0
    regions: list[tuple[int, int]] = []
    i = 0
    for i, t in enumerate(details):
        if abs(t["num_requests"] - global_max) <= max(1, thresh * global_max):
            if not started:
                in_steady += 1
        elif started:
            in_steady -= 1

        if in_steady > 5 and not started:
            started = True
            start_index = i - in_steady + 1

        if in_steady <= 0 and started:
            regions.append((start_index, i))
            started = False
            in_steady = 0

    if started:
        regions.append((start_index, i))

    if not regions:
        delta = min(n, max(8, num_steps - n))
        start = max(0, delta // 2)
        end = max(start + 1, min(n, n - delta // 2))
        regions = [(start, end)]
    return regions


def _longest_run(details: Sequence[dict[str, Any]], lo: int, hi: int, predicate) -> tuple[int, int] | None:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(lo, hi):
        if predicate(details[idx]):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, hi))
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def forecast_split(sp: StepPass, *, num_steps: int, conc: int | None, osl: float | None, r: float) -> dict[str, Any]:
    """Predict, without running the splitter, what each mode's chunk would hold.

    A faithful transcription of ``find_steady_state_window`` for the three modes
    the analysis driver can request. Fidelity matters more than elegance here: the
    candidate windows are strided, not sliding, so a lone prefill step near the
    end of the region falls into no candidate at all and ``mixed`` degrades to a
    pure-decode window. Under an under-recorded graph capture that window is
    empty, which is the whole failure mode this forecast exists to predict.
    """
    steps = sp.steps
    total = len(steps)
    if not total:
        return {
            "per_mode": [],
            "viable_modes": [],
            "step_count": 0,
            "note": "no step annotations; the splitter would fall back to generic call-tree traversal",
        }

    regions = identify_steady_state_regions(steps, num_steps)
    region_stats = [
        {
            "start": s,
            "end": e,
            "size": e - s,
            "pd_count": sum(1 for t in steps[s:e] if t["is_prefill"]),
        }
        for s, e in regions
    ]
    largest = max(region_stats, key=lambda x: x["size"])
    region_lo, region_hi = largest["start"], largest["end"]
    reference_ratio = largest["pd_count"] / largest["size"] if largest["size"] else 0.0

    ideal = None
    effective_num_steps = num_steps
    if conc and osl and r is not None:
        r = max(0.0, min(1.0, r))
        ideal = (conc * 2.0) / (osl * (1.0 + r))
        if ideal > 0:
            effective_num_steps = max(num_steps, math.ceil(1.0 / ideal))
        reference_ratio = ideal

    divider = max(1, min(int(effective_num_steps / 2), 10))
    stride = max(1, effective_num_steps // divider)

    candidates: list[dict[str, Any]] = []
    if (region_hi - region_lo) >= effective_num_steps:
        for s1 in range(region_lo, region_hi - effective_num_steps + 1, stride):
            window = steps[s1 : s1 + effective_num_steps]
            candidates.append(
                {
                    "start": s1,
                    "end": s1 + effective_num_steps,
                    "pd_count": sum(1 for t in window if t["is_prefill"]),
                    "pd_ratio": sum(1 for t in window if t["is_prefill"]) / effective_num_steps,
                    "avg_requests": sum(t["num_requests"] for t in window) / len(window),
                }
            )
    else:
        window = steps[region_lo:region_hi]
        candidates.append(
            {
                "start": region_lo,
                "end": region_hi,
                "pd_count": sum(1 for t in window if t["is_prefill"]),
                "pd_ratio": (sum(1 for t in window if t["is_prefill"]) / len(window)) if window else 0.0,
                "avg_requests": (sum(t["num_requests"] for t in window) / len(window)) if window else 0,
            }
        )

    per_mode: list[dict[str, Any]] = []

    def emit(mode: str, window: tuple[int, int] | None, note: str) -> None:
        if window is None or window[1] <= window[0]:
            per_mode.append(
                {
                    "mode": mode,
                    "consumer_mode": _CONSUMER_MODE.get(mode, mode),
                    "window_step_range": None,
                    "chunk_count_pred": 0,
                    "num_gpu_events_pred": 0,
                    "gpu_busy_us_pred": 0.0,
                    "busy_ratio_pred": 0.0,
                    "window_contains_prefill": False,
                    "note": note,
                }
            )
            return
        lo, hi = window
        stats = sp.window_stats(lo, hi)
        per_mode.append(
            {
                "mode": mode,
                "consumer_mode": _CONSUMER_MODE.get(mode, mode),
                "window_step_range": [lo, hi],
                "chunk_count_pred": 1,
                "num_gpu_events_pred": stats["num_gpu_events_pred"],
                "gpu_busy_us_pred": stats["gpu_busy_us_pred"],
                "busy_ratio_pred": stats["busy_ratio_pred"],
                "gpu_extent_us": stats["gpu_extent_us"],
                "host_span_us": stats["host_span_us"],
                "window_contains_prefill": any(s["is_prefill"] for s in steps[lo:hi]),
                "window_prefill_step_count": sum(1 for s in steps[lo:hi] if s["is_prefill"]),
                "note": note,
            }
        )

    pd_candidates = [c for c in candidates if c["pd_count"] > 0]
    pool = pd_candidates or candidates
    best = min(pool, key=lambda c: (abs(c["pd_ratio"] - reference_ratio), -c["avg_requests"]))
    emit(
        "mixed",
        (best["start"], best["end"]),
        f"{len(pd_candidates)}/{len(candidates)} candidate windows contain a prefill step"
        if pd_candidates
        else f"none of the {len(candidates)} candidate windows contains a prefill step; fell back to the full set",
    )

    do_run = _longest_run(
        steps,
        region_lo,
        region_hi,
        lambda t: t["generation_requests"] > 0 and t["context_requests"] == 0,
    )
    if do_run:
        lo, hi = do_run
        emit("decode_only", (lo, min(hi, lo + effective_num_steps)), f"longest pure decode run [{lo}, {hi})")
    else:
        emit("decode_only", None, "no pure decode-only run in the steady-state region")

    pd_run = _longest_run(steps, region_lo, region_hi, lambda t: t["context_requests"] > 0)
    if pd_run:
        lo, hi = pd_run
        emit(
            "max_prefilldecode",
            (lo, min(hi, lo + effective_num_steps)),
            f"longest pure prefill run [{lo}, {hi})",
        )
    else:
        emit("max_prefilldecode", None, "no prefill step in the steady-state region")

    viable = [m["mode"] for m in per_mode if m["num_gpu_events_pred"] > 0 and m["gpu_busy_us_pred"] > 0]
    return {
        "per_mode": per_mode,
        "viable_modes": viable,
        "viable_consumer_modes": [_CONSUMER_MODE.get(m, m) for m in viable],
        "steady_state_regions": [[s, e] for s, e in regions],
        "largest_region": [region_lo, region_hi],
        "candidate_window_count": len(candidates),
        "candidate_windows_with_prefill": len(pd_candidates),
        "candidate_stride": stride,
        "reference_prefill_ratio": round(reference_ratio, 6),
        "ideal_prefill_ratio": round(ideal, 6) if ideal is not None else None,
        "step_count": total,
        "prefill_step_count": sum(1 for s in steps if s["is_prefill"]),
        "num_steps_param": num_steps,
        "num_steps_effective": effective_num_steps,
    }


def annotation_report(sp: StepPass, *, framework: str, min_repeats: int) -> dict[str, Any]:
    """Group 6: reproduce the window selector's view of the annotations."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for w in sp.gpu_annotations:
        norm = re.sub(r"\d+$", "", w["name"]).rstrip("#_ -")
        if norm:
            groups.setdefault(norm, []).append(w)
    rows = [
        {
            "norm_name": norm,
            "occurrences": len(ws),
            "is_step_marker": bool(_STEP_MARKER_RE.search(norm)),
            "dur_median_us": round(median([w["dur"] for w in ws]), 3),
        }
        for norm, ws in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]
    window = select_steady_window(sp.gpu_annotations, framework=framework, min_repeats=min_repeats)
    qualified = [r for r in rows if r["occurrences"] >= min_repeats]
    selected = None
    if qualified:
        selected = max(qualified, key=lambda r: (r["is_step_marker"], r["occurrences"]))["norm_name"]

    contains_prefill = None
    if window:
        overlapping = [s for s in sp.steps if s["ts"] < window["end_us"] and s["end"] > window["start_us"]]
        contains_prefill = any(s["is_prefill"] for s in overlapping)

    return {
        "annotation_window_count": len(sp.gpu_annotations),
        "groups": rows,
        "selected_group": selected,
        # Two different requirements, deliberately not merged. bypass windows over
        # gpu_user_annotation groups and falls back to the whole trace when none
        # repeats enough, so this is soft. tracelens cuts on the splitter's
        # user_annotation step roots and has no fallback, so that one is hard.
        "min_repeats_met_bypass_window": bool(qualified),
        "step_root_count": len(sp.steps),
        "step_root_pattern_family": sp.root_family,
        "step_roots_sufficient_for_tracelens": len(sp.steps) >= min_repeats,
        "steady_window": window,
        "steady_window_contains_prefill": contains_prefill,
        "steps": [
            {
                "index": s["index"],
                "kind": s["kind"],
                "is_prefill": s["is_prefill"],
                "kernel_count": s["kernel_count"],
                "graph_launch_count": s["graph_launch_count"],
                "runtime_launch_count": s["runtime_launch_count"],
                "gpu_busy_us": s["gpu_busy_us"],
                "dur_us": round(s["dur_us"], 3),
            }
            for s in sp.steps
        ],
        "steps_with_gpu": sum(1 for s in sp.steps if s["kernel_count"] > 0),
        "step_gpu_coverage": _ratio(sum(1 for s in sp.steps if s["kernel_count"] > 0), len(sp.steps)),
    }


def _idle_in_span(sp: StepPass, lo: float, hi: float, threshold: float, scope: str) -> dict[str, Any]:
    """Idle percentage over one time span, measured the way the live gate does.

    Total is the span's wall clock rather than the kernel-activity envelope, so
    idle reflects gaps inside the step. Occupancy intervals are clipped to the
    span; kernel durations themselves are not shortened.
    """
    clipped = [(max(lo, a), min(hi, b)) for a, b in sp.gpu_intervals if b > lo and a < hi]
    clipped = [iv for iv in clipped if iv[1] > iv[0]]
    span = hi - lo
    busy = _union_ms(clipped) * 1000.0 if clipped else 0.0
    idle_pct = round((1.0 - busy / span) * 100.0, 4) if span > 0 else None
    return {
        "scope": scope,
        "idle_pct_window": idle_pct,
        "busy_us_window": round(busy, 3),
        "window_span_us": round(span, 3),
        "idle_pct_threshold_effective": threshold,
        "would_trip_idle_gate": idle_pct is not None and idle_pct >= threshold,
    }


def idle_gate(
    sp: StepPass,
    forecast: dict[str, Any],
    bypass_window: dict[str, Any] | None,
    threshold: float,
    default_mode: str = "mixed",
) -> dict[str, Any]:
    """Group 8: would the idle gate fire, for each mode's forecast chunk.

    Reported per mode rather than once, because the gate runs against whichever
    chunk the analysis was asked for -- and the modes differ enormously on an
    under-recorded capture, where a pure-decode chunk is close to 100% idle while
    the prefill chunk is busy. Must be read alongside group 4: low busy on its own
    is ambiguous, since a genuinely idle or host-bound workload looks identical.
    """
    per_mode: dict[str, Any] = {}
    for entry in forecast.get("per_mode", ()):
        rng = entry.get("window_step_range")
        if not rng:
            continue
        lo_idx, hi_idx = rng
        window = sp.steps[lo_idx:hi_idx]
        if not window:
            continue
        per_mode[entry["mode"]] = _idle_in_span(
            sp, window[0]["ts"], window[-1]["end"], threshold, f"forecast_chunk:{entry['mode']}"
        )

    selected = per_mode.get(default_mode)
    if selected is None:
        viable = forecast.get("viable_modes") or []
        selected = per_mode.get(viable[0]) if viable else None
    if selected is None and bypass_window is not None:
        selected = _idle_in_span(
            sp, bypass_window["start_us"], bypass_window["end_us"], threshold, "bypass_steady_window"
        )
    if selected is None:
        selected = {
            "scope": "no_window",
            "idle_pct_window": None,
            "idle_pct_threshold_effective": threshold,
            "would_trip_idle_gate": None,
        }

    out = dict(selected)
    out["per_mode"] = per_mode
    if bypass_window is not None:
        out["bypass_steady_window"] = _idle_in_span(
            sp, bypass_window["start_us"], bypass_window["end_us"], threshold, "bypass_steady_window"
        )
    return out


# --------------------------------------------------------------------------
# chunk level
# --------------------------------------------------------------------------


def certify_chunks(chunk_files: Iterable[Path], source_kernel_corrs: set[Any]) -> list[dict[str, Any]]:
    """Check each existing chunk for kernels it should have carried but did not.

    ``lost_by_correlation`` is the only sound test: it asks whether the chunk kept
    a host-side launch whose device kernel exists in the source. Comparing time
    spans instead charges the splitter for sibling annotations' kernels.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(chunk_files):
        mode_match = _SPLIT_NAME_RE.match(path.name)
        retained: set[Any] = set()
        kernels = 0
        annotation_names: set[str] = set()
        errors: list[str] = []
        try:
            fobj = _open_trace_binary(path)
        except OSError as exc:
            out.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            for ev in stream_events(fobj, errors=errors):
                cat = ev.get("cat", "")
                if cat == "cuda_runtime":
                    corr = (ev.get("args") or {}).get("correlation")
                    if corr is not None:
                        retained.add(corr)
                elif cat == "kernel" and ev.get("ph") == "X":
                    kernels += 1
                elif cat == "user_annotation" and ev.get("ph") == "X":
                    name = ev.get("name") or ""
                    if name.startswith("step["):
                        annotation_names.add(name)
        finally:
            fobj.close()
        intersect = len(retained & source_kernel_corrs)
        out.append(
            {
                "path": str(path),
                "mode": mode_match.group("mode") if mode_match else None,
                "annotation_name": sorted(annotation_names)[:3],
                "kernel_count": kernels,
                "retained_runtime_correlations": len(retained),
                "corr_intersect_source": intersect,
                "lost_by_correlation": kernels == 0 and intersect > 0,
                "stream_errors": errors[:5],
            }
        )
    return out


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def build_verdict(
    *,
    parse_ok: bool,
    kernel_count: int,
    capture_fragment: bool,
    attributed_pct: float | None,
    step_roots_sufficient: bool,
    forecast_modelled: bool,
    viable_modes: Sequence[str],
    idle: dict[str, Any],
    graph_under_recorded: bool | None,
    thresholds: dict[str, Any],
    production_pick: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer the two questions separately, per the checklist's hard-requirement table."""
    blocking: list[str] = []
    bypass_ok = True

    # Checked before the file's own health, because it decides which file the
    # question is even about. The live resolver filters CUDA-graph capture shards
    # but not the splitter's own output, so a capture directory that also holds
    # trace_split/ can hand the analyzer a chunk instead of the source. Splitting
    # consumes the annotations and leaves the kernels behind, so such a chunk
    # carries a few events and no GPU work while the source beside it is intact.
    # Both consumers resolve the file the same way, so neither can use the
    # directory however healthy the capture is.
    if production_pick and (production_pick.get("kernel_count") or 0) <= 0:
        blocking.append(
            f"the live resolver selects {production_pick.get('role') or 'another file'} "
            f"'{Path(production_pick['path']).name}' from this directory, which carries "
            f"{production_pick.get('event_total')} events and no GPU kernels, so both "
            f"consumers see an empty trace; the source beside it recorded {kernel_count} kernels"
        )
        bypass_ok = False

    if not parse_ok:
        blocking.append("trace did not parse cleanly")
        bypass_ok = False
    if kernel_count <= 0:
        blocking.append("source recorded zero GPU kernels")
        bypass_ok = False
    if capture_fragment:
        blocking.append("selected file is a CUDA-graph capture sidecar")
        bypass_ok = False

    recommended = viable_modes[0] if viable_modes else None
    warnings: list[str] = []
    tracelens_ok = bypass_ok
    if bypass_ok:
        if not attributed_pct:
            blocking.append("correlation chain resolves no kernel to an op")
            tracelens_ok = False
        # The split forecast only models the two annotation families the splitter
        # matches by pattern. When neither is present it falls back to generic
        # call-tree traversal, which is not modelled here, so tracelens usability
        # is unknown rather than false -- typical of diffusion captures, which
        # have no prefill/decode steps to begin with. Withholding the claim says
        # that; a blocking reason would instead assert a failure never observed.
        if not forecast_modelled:
            tracelens_ok = False
            warnings.append(
                "no pattern-matched iteration roots; the splitter would fall back to generic "
                "call-tree traversal, which this probe does not model, so tracelens usability "
                "is unverified here rather than ruled out"
            )
        elif not step_roots_sufficient:
            blocking.append("too few step annotations for the splitter to cut a steady window")
            tracelens_ok = False
        elif not viable_modes:
            blocking.append("every steady-state mode would produce an empty chunk")
            tracelens_ok = False

    # Only worth saying when the divergence did not already block above: the
    # resolver opens a different file, but one with GPU work in it, so the
    # measurements here describe a different object than production would see.
    if production_pick and (production_pick.get("kernel_count") or 0) > 0:
        warnings.append(
            f"the live resolver would select "
            f"{production_pick.get('role') or 'a different file'} rather than the source trace; "
            f"the measurements in this certificate describe the source "
            f"({Path(production_pick['certified_path']).name})"
        )

    # The idle gate is a warning, not a blocker: the live gate suppresses the
    # hot-kernel list and routes the session to parameter tuning, but the analysis
    # still completes. Treating it as blocking mislabels five runs in this cohort
    # that finished fine at 82-91% predicted idle.
    rec_idle = (idle.get("per_mode") or {}).get(recommended) if recommended else None
    suppressed = bool(rec_idle and rec_idle.get("would_trip_idle_gate"))
    if suppressed:
        warnings.append(
            f"idle {rec_idle['idle_pct_window']}% in the {recommended} chunk exceeds the "
            f"{rec_idle['idle_pct_threshold_effective']}% gate; the hot-kernel list would be "
            "suppressed and the session routed to parameter tuning"
        )

    usable = []
    if bypass_ok:
        usable.append("bypass")
    if tracelens_ok:
        usable.append("tracelens")

    # Taken from the reader rather than recomputed from the raw ratio, because the
    # ratio alone cannot tell an under-recorded graph capture from a trace that
    # never used graphs. An eager-mode capture has no graph launches, so the
    # coverage denominator is zero and any threshold test on it reads as a
    # failure -- which would report a healthy trace as silently wrong. The
    # reader's own preconditions (graph_mode, launch_count >= 2, graph_kernels)
    # already draw that distinction, and deferring to them is also what keeps the
    # certificate and the live gate from disagreeing.
    #
    # ``is False`` reads an unmeasured trace as invalid rather than unknown, and
    # that is deliberate rather than a collapsed tri-state. The reader returns a
    # bool on every path it reaches, so ``None`` means it produced no coverage
    # block at all -- and that same absence leaves ``attribution`` empty, which
    # blocks on zero kernels above and keeps ``tracelens`` out of ``usable``. So
    # the case where an unknown could be mistaken for a proven-wrong answer
    # cannot arise: ``silently_wrong`` requires a usable trace.
    valid = graph_under_recorded is False
    if tracelens_ok and graph_under_recorded:
        warnings.append(
            "graph launches are under-recorded, so a decode conclusion drawn from this "
            "trace would be wrong even though the analysis runs clean"
        )

    # ``usable_by`` means usable *via the recommended mode*. Naming the modes that
    # would fail keeps that from being read as "any mode works" -- three runs in
    # this cohort failed by asking for mixed on a trace whose only viable mode was
    # prefilldecode. Left null when the forecast could not run, so an unmodelled
    # annotation shape is not reported as three modes proven to fail.
    failing = (
        [_CONSUMER_MODE.get(m, m) for m in ("mixed", "decode_only", "max_prefilldecode") if m not in viable_modes]
        if forecast_modelled and tracelens_ok
        else None
    )
    return {
        "usable_by": usable,
        "decode_conclusions_valid": valid,
        "silently_wrong": "tracelens" in usable and not valid,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "hot_kernel_list_would_be_suppressed": suppressed,
        "recommended_steady_state_mode": _CONSUMER_MODE.get(recommended, recommended) if recommended else None,
        "recommended_splitter_mode": recommended,
        "modes_that_would_fail": failing,
        "thresholds_effective": thresholds,
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def certify_trace_dir(
    trace_dir: str | Path,
    *,
    chunk_files: Iterable[str | Path] = (),
    framework: str = "",
    num_steps: int = DEFAULT_NUM_STEPS,
    conc: int | None = None,
    osl: float | None = None,
    r: float = DEFAULT_R,
) -> dict[str, Any]:
    """Produce the self-certification record for one capture directory."""
    tdir = Path(trace_dir)
    thresholds = effective_thresholds()
    min_repeats = MIN_REPEATS_XDIT if framework.lower() == "xdit" else MIN_REPEATS_DEFAULT
    thresholds["min_repeats"] = min_repeats

    inv = inventory(tdir)
    task = _ROOFLINE_TASK_RE.search(str(tdir))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probe_version": PROBE_VERSION,
        "probed_at": _iso_now(),
        "trace_dir": str(tdir),
        "roofline_task_id": task.group(1) if task else None,
        "framework": framework or None,
        "has_steady_state_semantics": (framework or "").lower() in _SCRIPTABLE_FRAMEWORKS if framework else None,
        "trace_dir_level": inv,
        "rank_level": [],
        "chunk_level": [],
    }

    selected = inv["selected_path"]
    if not selected:
        # An empty capture directory still deserves a full record rather than a
        # hole: the profiler created the directory and wrote nothing, which is a
        # finding in its own right and distinct from a trace that parsed badly.
        # The counts are stated as zero -- nothing was read, so nothing was seen --
        # and the blocking reason carries why, so a reader cannot mistake this for
        # a trace that was measured and found empty.
        record["rank_level"].append(
            {
                "rank": None,
                "path": None,
                "parse": {"parse_ok": False, "stream_errors": [], "event_total": 0, "bytes": None, "gz": False},
                "attribution": {},
                "density": {
                    "kernel_count": 0,
                    "graph_mode": False,
                    "graph_launch_count": 0,
                    "graph_launches_with_kernels": 0,
                    "graph_launch_coverage": None,
                    "graph_under_recorded": False,
                },
                "time_structure": {},
                "annotations": {},
                "split_forecast": {},
                "idle_gate": {},
            }
        )
        record["verdict"] = build_verdict(
            parse_ok=False,
            kernel_count=0,
            capture_fragment=False,
            attributed_pct=None,
            step_roots_sufficient=False,
            forecast_modelled=True,
            viable_modes=[],
            idle={},
            graph_under_recorded=None,
            thresholds=thresholds,
        )
        record["verdict"]["blocking_reasons"] = ["no usable trace file in the directory"]
        return record

    sel = Path(selected)

    # When the live resolver would open a different file, measure that file too.
    # It is cheap -- a split chunk is orders of magnitude smaller than the source
    # -- and it turns "production picks something else" from an observation into
    # evidence about whether production can actually read GPU work here.
    production_pick = None
    if inv["production_selected_path"] and inv["production_selected_path"] != selected:
        probe = analyze_trace(inv["production_selected_path"], top_k=1, steady_state=False)
        # ``analyze_trace`` reports the kernel count inside ``attribution`` and has
        # no ``parse_ok`` key at all -- parse health is the emptiness of
        # ``stream_errors``. Reading either name off the top level yields None,
        # which the blocking test below would read as "no GPU kernels" for every
        # directory where the resolver diverges.
        probe_errors = probe.get("stream_errors") or []
        production_pick = {
            "path": inv["production_selected_path"],
            "role": inv["production_selected_role"],
            "certified_path": selected,
            "kernel_count": (probe.get("attribution") or {}).get("kernel_count") or 0,
            "event_total": probe.get("event_total"),
            "parse_ok": not probe_errors,
        }
        record["trace_dir_level"]["production_pick_probe"] = production_pick

    # Pass 1: the shipping reader, whole-trace scope. Groups 1, 3, 4, 5.
    analysis = analyze_trace(sel, top_k=1, steady_state=False, framework=framework)
    # Pass 2: step structure and subtree attribution. Groups 6, 7, 8.
    sp = StepPass()
    pass2_errors = sp.scan(sel)
    sp.attribute_steps()

    attribution = analysis.get("attribution") or {}
    coverage_block = analysis.get("graph_coverage") or {}
    timeline = analysis.get("timeline") or {}
    stream_errors = list(analysis.get("stream_errors") or []) + pass2_errors

    ann = annotation_report(sp, framework=framework, min_repeats=min_repeats)
    forecast = forecast_split(sp, num_steps=num_steps, conc=conc, osl=osl, r=r)
    gate = idle_gate(sp, forecast, ann["steady_window"], thresholds["idle_pct_threshold"])

    kernel_count = attribution.get("kernel_count", 0)
    graph_launch_count = coverage_block.get("graph_launch_count", 0)
    graph_with_kernels = coverage_block.get("graph_launches_with_kernels", 0)
    coverage = _ratio(graph_with_kernels, graph_launch_count)

    try:
        size = sel.stat().st_size
    except OSError:
        size = None

    rank_record = {
        "rank": analysis.get("analyzed_rank"),
        "path": str(sel),
        "parse": {
            "parse_ok": not stream_errors,
            "stream_errors": stream_errors,
            "event_total": analysis.get("event_total", 0),
            "bytes": size,
            "gz": sel.name.lower().endswith(".gz"),
        },
        "attribution": {
            "cuda_runtime_links": attribution.get("cuda_runtime_links"),
            "cpu_ops": attribution.get("cpu_ops"),
            "attributed_kernels": attribution.get("attributed_kernels"),
            "unlinked_kernels": attribution.get("unlinked_kernels"),
            "graph_attributed_kernels": attribution.get("graph_attributed_kernels"),
            "attributed_pct": attribution.get("attributed_pct"),
            "op_meta_coverage": _ratio(sp.cpu_op_with_meta, sp.cpu_op_total),
        },
        "density": {
            "kernel_count": kernel_count,
            "graph_mode": coverage_block.get("graph_mode"),
            "graph_launch_count": graph_launch_count,
            "graph_launches_with_kernels": graph_with_kernels,
            "graph_launch_coverage": coverage,
            "graph_under_recorded": coverage_block.get("graph_under_recorded"),
            "busy_fraction": coverage_block.get("busy_fraction"),
            "runtime_launch_count": len(sp.runtime_ts),
            "kernel_per_launch": _ratio(kernel_count, len(sp.runtime_ts)),
        },
        "time_structure": {
            "total_ms": timeline.get("total_time_ms"),
            "busy_ms": timeline.get("busy_time_ms"),
            "idle_ms": timeline.get("idle_time_ms"),
            "idle_pct_full_trace": timeline.get("idle_pct"),
            "stream_overlap": timeline.get("stream_overlap") or {},
        },
        "annotations": ann,
        "split_forecast": forecast,
        "idle_gate": gate,
    }
    record["rank_level"].append(rank_record)

    source_kernel_corrs = {c for c in sp.kernel_by_corr if c is not None}
    record["chunk_level"] = certify_chunks([Path(c) for c in chunk_files], source_kernel_corrs)

    record["verdict"] = build_verdict(
        parse_ok=not stream_errors,
        kernel_count=kernel_count,
        capture_fragment=inv["selected_capture_fragment"],
        attributed_pct=attribution.get("attributed_pct"),
        step_roots_sufficient=ann["step_roots_sufficient_for_tracelens"],
        forecast_modelled=bool(forecast.get("per_mode")),
        production_pick=production_pick,
        viable_modes=forecast["viable_modes"],
        idle=gate,
        graph_under_recorded=coverage_block.get("graph_under_recorded"),
        thresholds=thresholds,
    )
    return record
