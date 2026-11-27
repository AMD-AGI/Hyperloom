# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile self-certification: can this trace be analysed, and would the answer be true?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ._capture_shapes import is_capture_fragment
from ._idle_gate import resolve_idle_pct_threshold
from ._trace_gates import CHUNK_QUALITY_ALTERNATE_MARGIN, resolve_min_busy_ratio
from ._trace_inputs import discover_trace_inputs
from ._trace_reader import (
    _GRAPH_RECORDED_LAUNCH_COVERAGE_MAX,
    _MAX_EVENT_CHARS,
    _MAX_TRACE_PREFIX_CHARS,
    _open_trace_binary,
    _rank_of,
    _trace_candidates,
    analyze_trace,
    stream_events,
)

SCHEMA_VERSION = 1
PROBE_VERSION = "selfcert-1.0.0"

_ROOFLINE_TASK_RE = re.compile(r"/runs/[a-z_]+/([0-9a-f]{32})/")


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


def inventory(trace_dir: Path, trace_input: Path) -> dict[str, Any]:
    """List every trace-shaped candidate, and pick the file the analysis reads from ``trace_input``."""
    candidates = []
    for p in sorted(_trace_candidates(trace_dir)):
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        candidates.append({"path": str(p), "bytes": size, "role": _role_of(p, trace_dir)})

    selected = discover_trace_inputs(trace_input).files[0]
    source_paths = [Path(c["path"]) for c in candidates if c["role"] == "source"]
    ranks = {_rank_of(p) for p in source_paths}
    ranks.discard(None)
    return {
        "candidates": candidates,
        "selected_path": str(selected),
        "selected_role": _role_of(selected, trace_dir),
        "selected_capture_fragment": is_capture_fragment(selected, trace_dir),
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
                # The same two keys ``_count_launches`` uses on the source trace, so the two coverages are comparable.
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


# the second pass


@dataclass
class _LaunchCounts:
    """What the trace reader's pass does not count: device launches, and the operators that carry input metadata."""

    runtime_launch_count: int = 0
    cpu_op_total: int = 0
    cpu_op_with_meta: int = 0
    stream_errors: list[str] = field(default_factory=list)


def _count_launches(path: Path) -> _LaunchCounts:
    """Count every correlated runtime launch and every operator of the whole trace, with no safety cap."""
    counts = _LaunchCounts()
    fobj = _open_trace_binary(path)
    try:
        for ev in stream_events(fobj, errors=counts.stream_errors):
            cat = ev.get("cat", "")
            args = ev.get("args") or {}
            if cat == "cuda_runtime" and args.get("correlation") is not None:
                counts.runtime_launch_count += 1
            elif cat == "cpu_op":
                counts.cpu_op_total += 1
                counts.cpu_op_with_meta += bool(args.get("Input Dims") or args.get("Input type"))
    finally:
        fobj.close()
    return counts


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
    coverage: CoverageMeasures = CoverageMeasures(),
) -> dict[str, Any]:
    """Answer the two questions separately, per the checklist's hard-requirement table.

    ``coverage`` feeds ``measures`` only. It is reported next to the categorical answer because
    a boolean cannot distinguish a trace that just missed a gate from one that missed it by an order of
    magnitude, and the two call for different responses -- but choosing the response is not this function's job.
    """
    blocking: list[str] = []
    warnings: list[str] = []
    if not health.parse_ok:
        blocking.append("trace did not parse cleanly")
    if health.kernel_count <= 0:
        blocking.append("source recorded zero GPU kernels")
    if health.capture_fragment:
        blocking.append("selected file is a CUDA-graph capture sidecar")
    if not blocking and not health.attributed_pct:
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
        "usable_by": ["tracelens"] if tracelens_ok else [],
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


def certify_trace_dir(trace_dir: str | Path, trace_input: str | Path, *, framework: str = "") -> dict[str, Any]:
    """Produce the self-certification record for one capture directory.

    ``trace_input`` is what the analysis is handed; the certified file is the one it reads from it.
    """
    tdir = Path(trace_dir)
    thresholds = effective_thresholds()

    inv = inventory(tdir, Path(trace_input))
    task = _ROOFLINE_TASK_RE.search(str(tdir))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probe_version": PROBE_VERSION,
        "probed_at": _iso_now(),
        "trace_dir": str(tdir),
        "roofline_task_id": task.group(1) if task else None,
        "framework": framework or None,
        "trace_dir_level": inv,
        "rank_level": [],
    }

    capture_paths = [Path(c["path"]) for c in inv["candidates"] if c["role"] == "capture_sidecar"]
    if capture_paths:
        inv["capture_sidecar_probe"] = certify_capture_sidecars(capture_paths)

    rank = _certify_rank(Path(inv["selected_path"]), thresholds=thresholds)
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
        coverage=CoverageMeasures(
            rank["density"]["graph_launch_coverage"],
            rank["attribution"]["op_meta_coverage"],
            inv.get("capture_sidecar_probe", {}).get("op_meta_coverage"),
        ),
    )
    return record


def _rank_attribution_block(
    attribution: dict[str, Any], timeline: dict[str, Any], counts: _LaunchCounts
) -> dict[str, Any]:
    """The rank row's kernel-attribution block, from the trace reader's pass and the launch count."""
    return {
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
        "op_meta_coverage": _ratio(counts.cpu_op_with_meta, counts.cpu_op_total),
        "op_meta_basis": {"cpu_op_total": counts.cpu_op_total, "cpu_op_with_meta": counts.cpu_op_with_meta},
    }


def _certify_rank(path: Path, *, thresholds: dict[str, Any]) -> dict[str, Any]:
    """The rank-level row for one trace file: the trace reader's pass, then the launch count."""
    analysis = analyze_trace(path)
    counts = _count_launches(path)
    attribution = analysis.get("attribution") or {}
    coverage_block = analysis.get("graph_coverage") or {}
    timeline = analysis.get("timeline") or {}
    stream_errors = list(analysis.get("stream_errors") or []) + counts.stream_errors
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
            # whole trace, so the record says whether it was truncated.
            "truncated": bool(analysis.get("truncated")),
            "truncation_reason": analysis.get("truncation_reason") or "",
        },
        "attribution": _rank_attribution_block(attribution, timeline, counts),
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
            "runtime_launch_count": counts.runtime_launch_count,
            "kernel_per_launch": _ratio(kernel_count, counts.runtime_launch_count),
        },
        "time_structure": {
            "total_ms": timeline.get("total_time_ms"),
            "busy_ms": timeline.get("busy_time_ms"),
            "idle_ms": timeline.get("idle_time_ms"),
            "idle_pct_full_trace": timeline.get("idle_pct"),
            "stream_overlap": timeline.get("stream_overlap") or {},
        },
    }
