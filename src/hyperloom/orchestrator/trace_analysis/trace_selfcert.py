# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile self-certification: can this trace be analysed, and would the answer be true?"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, NamedTuple, Sequence

from ._bypass_trace_reader import (
    _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
    TraceAnalysisOptions,
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
from ._capture_shapes import is_capture_fragment
from ._idle_gate import resolve_idle_pct_threshold
from ._trace_gates import CHUNK_QUALITY_ALTERNATE_MARGIN, resolve_min_busy_ratio

SCHEMA_VERSION = 1
PROBE_VERSION = "selfcert-1.0.0"

MIN_REPEATS_DEFAULT = 3
MIN_REPEATS_XDIT = 2

#: Iteration-root patterns by family, verbatim from ``split_trace.main`` (the backup
#: family's three as alternatives of one expression).
#: The primary vLLM shape wins outright; the backup family (vLLM's older shapes and
#: SGLang's per-step annotations) is consulted only when the primary matches
#: nothing. Reproducing that precedence matters -- a looser or differently
#: ordered pattern set would count steps the splitter never sees.
_ROOT_PATTERNS = {
    "primary": re.compile(
        r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
    ),
    "backup": re.compile(
        r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"
        r"|execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"
        r"|step\[(?:EXTEND|DECODE|MIXED)\b.*\]"
    ),
}
_SGLANG_STEP_RE = re.compile(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]")
_STEP_MARKER_RE = re.compile(r"(?i)profilerstep|denoise|iteration|(?:^|[^a-z])step|step(?:$|[^a-z])")
_ROOFLINE_TASK_RE = re.compile(r"/runs/[a-z_]+/([0-9a-f]{32})/")

_SCRIPTABLE_FRAMEWORKS = frozenset(("sglang", "vllm", "trtllm", "tensorrt_llm"))


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: The framework name in the benchmark ``config.yaml``. Matched line-anchored on
#: purpose: the same key occurs inside a quoted JSON blob elsewhere in the file
#: with a different value, and that blob must not win. Read by regex rather than a
#: YAML parser to keep the certifier stdlib-only.
_FRAMEWORK_RE = re.compile(r"^framework:\s*(\S+)\s*$", re.MULTILINE)


def read_framework(trace_dir: str | Path) -> str:
    """Read the framework name from the sibling benchmark ``config.yaml``."""
    config = Path(trace_dir).parent / "config.yaml"
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return ""
    m = _FRAMEWORK_RE.search(text)
    return m.group(1) if m else ""


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return round(num / den, 6)


def effective_thresholds() -> dict[str, Any]:
    """Resolve the thresholds the shipping gates apply, through the gates' own resolvers."""
    return {
        "graph_launch_coverage_max": _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
        "idle_pct_threshold": resolve_idle_pct_threshold(),
        "min_repeats": MIN_REPEATS_DEFAULT,
        "chunk_quality_min_busy_ratio": resolve_min_busy_ratio(),
        "chunk_quality_alternate_margin": CHUNK_QUALITY_ALTERNATE_MARGIN,
        "trace_prefix_max_chars": _MAX_TRACE_PREFIX_CHARS,
        "event_max_chars": _MAX_EVENT_CHARS,
    }


# group 2: candidate inventory and selection


def _role_of(path: Path, root: Path) -> str:
    if "trace_split" in path.parts:
        return "split_chunk"
    if is_capture_fragment(path, root):
        return "capture_sidecar"
    return "source"


def inventory(trace_dir: Path) -> dict[str, Any]:
    """List every trace-shaped candidate, and pick the file to certify."""
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


#: Sidecars parsed in full per directory. A graph capture writes one per captured batch size, so a representative
#: slice characterises the capture; the count present is reported separately so a truncated scan cannot read as a
#: complete one.
CAPTURE_SIDECAR_SCAN_LIMIT = 24


def _scan_capture_sidecar(path: Path) -> dict[str, Any]:
    """Measure one capture sidecar's operator metadata coverage, full-stream."""
    errors: list[str] = []
    cpu_op_total = 0
    cpu_op_with_meta = 0
    kernel_count = 0
    event_total = 0
    try:
        fobj = _open_trace_binary(path)
    except OSError as exc:
        return {"path": str(path), "error": repr(exc)}
    try:
        for ev in stream_events(fobj, errors=errors):
            event_total += 1
            cat = ev.get("cat", "")
            if cat == "cpu_op":
                cpu_op_total += 1
                # The same two keys ``StepPass`` uses on the source trace, so the two coverages are comparable.
                if (ev.get("args") or {}).get("Input Dims") or (ev.get("args") or {}).get("Input type"):
                    cpu_op_with_meta += 1
            elif cat == "kernel":
                kernel_count += 1
    finally:
        fobj.close()
    return {
        "path": str(path),
        "event_total": event_total,
        "cpu_op_total": cpu_op_total,
        "cpu_op_with_meta": cpu_op_with_meta,
        "op_meta_coverage": _ratio(cpu_op_with_meta, cpu_op_total),
        "kernel_count": kernel_count,
        "stream_errors": errors,
    }


def certify_capture_sidecars(
    paths: Sequence[Path],
    *,
    limit: int = CAPTURE_SIDECAR_SCAN_LIMIT,
) -> dict[str, Any]:
    """Measure whether the capture sidecars carry the operator metadata that graph attribution depends on.

    Nothing read these files before: the inventory classified them by name and counted them, and the only other
    look was a substring count over the first couple of megabytes of the single heaviest one. When the
    shape-discovery pass does not land, the kernels captured inside a CUDA graph cannot be attributed afterwards,
    and that degradation left no measurement anywhere. This reports the coverage; it does not gate on it.
    """
    ordered = sorted(paths)
    scanned = [_scan_capture_sidecar(p) for p in ordered[: max(0, limit)]]
    measured = [f for f in scanned if "error" not in f]
    cpu_op_total = sum(f["cpu_op_total"] for f in measured)
    cpu_op_with_meta = sum(f["cpu_op_with_meta"] for f in measured)
    return {
        "files_present": len(ordered),
        "files_scanned": len(scanned),
        "scan_limit": limit,
        "truncated_scan": len(ordered) > len(scanned),
        "event_total": sum(f["event_total"] for f in measured),
        "cpu_op_total": cpu_op_total,
        "cpu_op_with_meta": cpu_op_with_meta,
        "op_meta_coverage": _ratio(cpu_op_with_meta, cpu_op_total),
        "kernel_count": sum(f["kernel_count"] for f in measured),
        "files_with_errors": len(scanned) - len(measured),
        "files": scanned,
    }


# groups 6/7/8: the second pass


def _span(ev: dict[str, Any]) -> tuple[float, float]:
    """An X event's start and duration in microseconds; a missing or empty value reads as zero."""
    return float(ev.get("ts") or 0.0), float(ev.get("dur") or 0.0)


class StepPass:
    """Second-pass state: step annotations and per-step subtree attribution."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.root_family: str | None = None
        self.gpu_annotations: list[dict[str, Any]] = []
        self.runtime_ts: list[float] = []
        self.runtime_corr: list[Any] = []
        self.graph_corrs: set[Any] = set()
        self.kernel_by_corr: dict[Any, list[tuple[float, float]]] = {}
        self.cpu_op_total = 0
        self.cpu_op_with_meta = 0

    def scan(self, path: Path) -> list[str]:
        errors: list[str] = []
        runtime: list[tuple[float, Any]] = []
        roots: dict[str, list[dict[str, Any]]] = {}
        fobj = _open_trace_binary(path)
        try:
            for ev in stream_events(fobj, errors=errors):
                cat = ev.get("cat", "")
                args = ev.get("args") or {}
                if cat == "cuda_runtime" and (corr := args.get("correlation")) is not None:
                    runtime.append((float(ev.get("ts") or 0.0), corr))
                    if "GraphLaunch" in (ev.get("name") or ""):
                        self.graph_corrs.add(corr)
                elif cat == "cpu_op":
                    self.cpu_op_total += 1
                    self.cpu_op_with_meta += bool(args.get("Input Dims") or args.get("Input type"))
                elif ev.get("ph") != "X":
                    continue
                elif cat == "user_annotation":
                    name = ev.get("name") or ""
                    family = next((family for family, pattern in _ROOT_PATTERNS.items() if pattern.match(name)), None)
                    if family is not None:
                        ts, dur = _span(ev)
                        root = {"name": name, "ts": ts, "end": ts + dur, "dur_us": dur, **iter_details_from_name(name)}
                        roots.setdefault(family, []).append(root)
                elif cat == "gpu_user_annotation":
                    ts, dur = _span(ev)
                    self.gpu_annotations.append({"name": ev.get("name") or "", "ts": ts, "dur": dur})
                elif cat == "kernel":
                    ts, dur = _span(ev)
                    self.kernel_by_corr.setdefault(args.get("correlation"), []).append((ts, ts + dur))
        finally:
            fobj.close()
        self._index_scan(runtime, roots)
        return errors

    def _index_scan(self, runtime: list[tuple[float, Any]], roots: dict[str, list[dict[str, Any]]]) -> None:
        """Order the runtime launches by time and keep the steps of the highest-precedence root family found."""
        runtime.sort(key=lambda x: x[0])
        self.runtime_ts = [t for t, _ in runtime]
        self.runtime_corr = [c for _, c in runtime]
        self.root_family = next((family for family in _ROOT_PATTERNS if family in roots), None)
        self.steps = sorted(roots.get(self.root_family, ()), key=lambda s: s["ts"])

    def attribute_steps(self) -> None:
        """Fill each step with the device work its own launches produced."""
        for idx, step in enumerate(self.steps):
            lo = bisect_left(self.runtime_ts, step["ts"])
            hi = bisect_right(self.runtime_ts, step["end"])
            corrs = set(self.runtime_corr[lo:hi])
            intervals: list[tuple[float, float]] = []
            for corr in corrs:
                intervals.extend(self.kernel_by_corr.get(corr, ()))
            step.update(
                {
                    "index": idx,
                    "runtime_launch_count": hi - lo,
                    "graph_launch_count": sum(1 for c in corrs if c in self.graph_corrs),
                    "kernel_count": len(intervals),
                    "gpu_busy_us": round(_union_ms(intervals) * 1000.0, 3) if intervals else 0.0,
                }
            )


def iter_details_from_name(name: str) -> dict[str, Any]:
    """Map an iteration-root annotation to the request counts the splitter derives."""
    m = _SGLANG_STEP_RE.match(name)
    if m:
        kind, bs = m.group(1), int(m.group(2))
        toks = int(m.group(3) or 0)
        if kind == "DECODE":
            return _details(kind, bs, _RequestCounts(0, 0), _RequestCounts(bs, bs))
        return _details(kind, toks or bs, _RequestCounts(bs, toks), _RequestCounts(0, 0))

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
        return _details(kind, ctx_sum + gen_sum, _RequestCounts(ctx_req, ctx_sum), _RequestCounts(gen_req, gen_sum))
    except (ValueError, IndexError):
        return _details("DECODE", 1, _RequestCounts(0, 0), _RequestCounts(1, 1))


class _RequestCounts(NamedTuple):
    """The requests of one iteration phase (context or generation) and their summed length."""

    requests: int
    total: int


def _details(kind: str, batch: int, context: _RequestCounts, generation: _RequestCounts) -> dict[str, Any]:
    return {
        "kind": kind,
        "batch_size": batch,
        "num_requests": context.requests + generation.requests,
        "context_requests": context.requests,
        "context_sum": context.total,
        "generation_requests": generation.requests,
        "generation_sum": generation.total,
        "is_prefill": context.requests > 0,
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
        # Two different requirements, deliberately not merged. bypass windows over gpu_user_annotation groups and
        # falls back to the whole trace when none repeats enough, so this is soft. tracelens cuts on the splitter's
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


# verdict


@dataclass(frozen=True)
class TraceHealth:
    """What the certificate's categorical answers are decided from."""

    parse_ok: bool
    kernel_count: int
    capture_fragment: bool
    attributed_pct: float | None
    graph_under_recorded: bool | None = None


@dataclass(frozen=True)
class CoverageMeasures:
    """Coverage ratios reported next to the verdict, never compared in it."""

    graph_launch_coverage: float | None = None
    op_meta_coverage: float | None = None
    capture_op_meta_coverage: float | None = None


def build_verdict(
    health: TraceHealth,
    *,
    thresholds: dict[str, Any],
    production_pick: dict[str, Any] | None = None,
    coverage: CoverageMeasures = CoverageMeasures(),
) -> dict[str, Any]:
    """Answer the two questions separately, per the checklist's hard-requirement table.

    ``coverage`` feeds ``measures`` only. It is reported next to the categorical answer because
    a boolean cannot distinguish a trace that just missed a gate from one that missed it by an order of
    magnitude, and the two call for different responses -- but choosing the response is not this function's job.
    """
    blocking: list[str] = []
    warnings: list[str] = []
    # Checked before the file's own health, because it decides which file the question is even about.
    if production_pick and production_pick["kernel_count"] <= 0:
        blocking.append(
            f"the live resolver selects {production_pick['role']} '{Path(production_pick['path']).name}' from this "
            f"directory, which carries {production_pick['event_total']} events and no GPU kernels, so both "
            f"consumers see an empty trace; the source beside it recorded {health.kernel_count} kernels"
        )
    elif production_pick:
        # The resolver opens a different file, but one with GPU work in it, so the measurements here describe a
        # different object than production would see.
        warnings.append(
            f"the live resolver would select {production_pick['role']} rather than the source trace; the measurements "
            f"in this certificate describe the source ({Path(production_pick['certified_path']).name})"
        )
    if not health.parse_ok:
        blocking.append("trace did not parse cleanly")
    if health.kernel_count <= 0:
        blocking.append("source recorded zero GPU kernels")
    if health.capture_fragment:
        blocking.append("selected file is a CUDA-graph capture sidecar")
    bypass_ok = not blocking
    if bypass_ok and not health.attributed_pct:
        blocking.append("correlation chain resolves no kernel to an op")
    tracelens_ok = not blocking

    # Taken from the reader rather than recomputed from the raw ratio, because the ratio alone cannot tell an
    # under-recorded graph capture from a trace that never used graphs.
    valid = health.graph_under_recorded is False
    if tracelens_ok and health.graph_under_recorded:
        warnings.append(
            "graph launches are under-recorded, so a decode conclusion drawn from this "
            "trace would be wrong even though the analysis runs clean"
        )
    silently_wrong = tracelens_ok and not valid
    # A restatement of the answers above on one ordered scale, so a fleet-wide query can rank outcomes without
    # re-deriving the precedence. ``silently_wrong`` outranks a plain warning: an analysis that runs clean and
    # concludes wrongly is worse than one that announces its own doubt.
    if blocking:
        severity = "blocked"
    elif silently_wrong:
        severity = "silently_wrong"
    elif warnings:
        severity = "warn"
    else:
        severity = "ok"
    return {
        "usable_by": [consumer for consumer, ok in (("bypass", bypass_ok), ("tracelens", tracelens_ok)) if ok],
        "decode_conclusions_valid": valid,
        "silently_wrong": silently_wrong,
        "severity": severity,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "thresholds_effective": thresholds,
        # The continuous quantities the categorical answers were derived from. Reported, never compared here.
        "measures": {
            "kernel_count": health.kernel_count,
            "attributed_pct": health.attributed_pct,
            "graph_launch_coverage": coverage.graph_launch_coverage,
            "graph_launch_coverage_max": thresholds.get("graph_launch_coverage_max"),
            "op_meta_coverage": coverage.op_meta_coverage,
            "capture_op_meta_coverage": coverage.capture_op_meta_coverage,
        },
    }


# entry point


def certify_trace_dir(trace_dir: str | Path, *, framework: str = "") -> dict[str, Any]:
    """Produce the self-certification record for one capture directory."""
    tdir = Path(trace_dir)
    thresholds = effective_thresholds()
    thresholds["min_repeats"] = MIN_REPEATS_XDIT if framework.lower() == "xdit" else MIN_REPEATS_DEFAULT

    inv = inventory(tdir)
    task = _ROOFLINE_TASK_RE.search(str(tdir))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probe_version": PROBE_VERSION,
        "probed_at": _iso_now(),
        "trace_dir": str(tdir),
        "roofline_task_id": task.group(1) if task else None,
        "framework": framework or None,
        "has_steady_state_semantics": framework.lower() in _SCRIPTABLE_FRAMEWORKS if framework else None,
        "trace_dir_level": inv,
        "rank_level": [],
    }

    # Before the early return below: a directory that produced only capture sidecars and no usable trace is
    # exactly the case where knowing what the sidecars contain matters most.
    capture_paths = [Path(c["path"]) for c in inv["candidates"] if c["role"] == "capture_sidecar"]
    if capture_paths:
        inv["capture_sidecar_probe"] = certify_capture_sidecars(capture_paths)

    selected = inv["selected_path"]
    if not selected:
        # An empty capture directory still deserves a full record rather than a hole: the profiler created the
        # directory and wrote nothing, which is a finding in its own right and distinct from a trace that parsed
        # badly.
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
            }
        )
        record["verdict"] = build_verdict(
            TraceHealth(parse_ok=False, kernel_count=0, capture_fragment=False, attributed_pct=None),
            thresholds=thresholds,
        )
        record["verdict"]["blocking_reasons"] = ["no usable trace file in the directory"]
        return record

    if inv["production_selected_path"] and inv["production_selected_path"] != selected:
        inv["production_pick_probe"] = _probe_production_pick(inv)
    rank = _certify_rank(Path(selected), framework=framework, thresholds=thresholds)
    record["rank_level"].append(rank)
    record["verdict"] = build_verdict(
        TraceHealth(
            parse_ok=rank["parse"]["parse_ok"],
            kernel_count=rank["density"]["kernel_count"],
            capture_fragment=inv["selected_capture_fragment"],
            attributed_pct=rank["attribution"]["attributed_pct"],
            graph_under_recorded=rank["density"]["graph_under_recorded"],
        ),
        thresholds=thresholds,
        production_pick=inv.get("production_pick_probe"),
        coverage=CoverageMeasures(
            rank["density"]["graph_launch_coverage"],
            rank["attribution"]["op_meta_coverage"],
            inv.get("capture_sidecar_probe", {}).get("op_meta_coverage"),
        ),
    )
    return record


def _probe_production_pick(inv: dict[str, Any]) -> dict[str, Any]:
    """Measure the file the live resolver would open in place of the selected one."""
    probe = analyze_trace(inv["production_selected_path"], TraceAnalysisOptions(top_k=1, steady_state=False))
    # ``analyze_trace`` reports the kernel count inside ``attribution`` and has no ``parse_ok`` key at all --
    # parse health is the emptiness of ``stream_errors``.
    return {
        "path": inv["production_selected_path"],
        "role": inv["production_selected_role"],
        "certified_path": inv["selected_path"],
        "kernel_count": (probe.get("attribution") or {}).get("kernel_count") or 0,
        "event_total": probe.get("event_total"),
        "parse_ok": not probe.get("stream_errors") and not probe.get("truncated"),
        "truncated": bool(probe.get("truncated")),
    }


def _certify_rank(path: Path, *, framework: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    """The rank-level row for one trace file: the shipping reader's pass, then the step pass."""
    # Pass 1: the shipping reader, whole-trace scope. Groups 1, 3, 4, 5.
    analysis = analyze_trace(path, TraceAnalysisOptions(top_k=1, steady_state=False, framework=framework))
    # Pass 2: step structure and subtree attribution. Groups 6, 7, 8.
    sp = StepPass()
    pass2_errors = sp.scan(path)
    sp.attribute_steps()
    attribution = analysis.get("attribution") or {}
    coverage_block = analysis.get("graph_coverage") or {}
    timeline = analysis.get("timeline") or {}
    stream_errors = list(analysis.get("stream_errors") or []) + pass2_errors
    if analysis.get("truncated"):
        stream_errors.append(
            f"trace aggregation truncated at safety cap ({analysis.get('truncation_reason', 'unknown')})"
        )
    kernel_count = attribution.get("kernel_count", 0)
    graph_launch_count = coverage_block.get("graph_launch_count", 0)
    graph_with_kernels = coverage_block.get("graph_launches_with_kernels", 0)
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {
        "rank": analysis.get("analyzed_rank"),
        "path": str(path),
        "parse": {
            "parse_ok": not stream_errors,
            "stream_errors": stream_errors,
            "event_total": analysis.get("event_total", 0),
            "bytes": size,
            "gz": path.name.lower().endswith(".gz"),
            # A ratio computed over a truncated aggregation is not the same measurement as one computed over the
            # whole trace, so the record says which one it carries.
            "aggregation_scope": analysis.get("aggregation_scope"),
            "truncated": bool(analysis.get("truncated")),
            "truncation_reason": analysis.get("truncation_reason") or "",
        },
        "attribution": {
            "cuda_runtime_links": attribution.get("cuda_runtime_links"),
            "cpu_ops": attribution.get("cpu_ops"),
            "attributed_kernels": attribution.get("attributed_kernels"),
            "unlinked_kernels": attribution.get("unlinked_kernels"),
            "graph_attributed_kernels": attribution.get("graph_attributed_kernels"),
            "attributed_pct": attribution.get("attributed_pct"),
            # The terms the percentage is a ratio of. Without them a reader cannot tell 90% of a millisecond
            # from 90% of a minute, and the two warrant opposite reactions.
            "attributed_gpu_ms": attribution.get("attributed_gpu_ms"),
            "gpu_kernel_sum_ms": timeline.get("gpu_kernel_sum_ms"),
            "op_meta_coverage": _ratio(sp.cpu_op_with_meta, sp.cpu_op_total),
            "op_meta_basis": {"cpu_op_total": sp.cpu_op_total, "cpu_op_with_meta": sp.cpu_op_with_meta},
        },
        "density": {
            "kernel_count": kernel_count,
            "graph_mode": coverage_block.get("graph_mode"),
            "graph_launch_count": graph_launch_count,
            "graph_launches_with_kernels": graph_with_kernels,
            "graph_launch_coverage": _ratio(graph_with_kernels, graph_launch_count),
            "graph_under_recorded": coverage_block.get("graph_under_recorded"),
            # The boolean above is a comparison, so it travels with the number it was compared against.
            "graph_under_recorded_threshold": thresholds["graph_launch_coverage_max"],
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
        "annotations": annotation_report(sp, framework=framework, min_repeats=thresholds["min_repeats"]),
    }
