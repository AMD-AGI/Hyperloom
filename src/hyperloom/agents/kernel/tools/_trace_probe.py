###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Semantic validity probe for torch-profiler (Kineto) traces.

The checks that already exist ask whether a trace file is *structurally* sound:
does it open, does it parse, does it hold at least one GPU kernel event. Across
seven production sessions every trace that went on to break the KERNEL phase
answered yes to all three -- files intact, ``stream_errors`` empty, ``cpu_op``
counts identical across ranks, kernel events present on every rank. Structural
checks have no discriminating power over the failures we actually see, because
those failures are semantic: the GPU side of the recording does not match the
work the CPU side declares, or the timeline is dominated by one event that is a
measurement artifact rather than work.

This module answers the semantic question, in three families:

* **GPU-record completeness** (:data:`GRAPH_REPLAY_UNDER_RECORDED`,
  :data:`ANNOTATED_STEPS_WITHOUT_GPU_WORK`, :data:`KERNEL_LAUNCH_RATIO_COLLAPSED`)
  -- every graph replay, annotated step, and runtime launch declares GPU work
  that should have landed in the trace. When the profiler drops it, the trace
  still looks healthy by volume while describing a fraction of the run. Measured
  spread on the reference incident: 1/128 replays recorded on the bad ranks
  against 128/128 on the good ones, from the same profile.
* **Timeline exclusivity** (:data:`SINGLE_EVENT_DOMINATES_WINDOW`) -- a single
  device event covering most of the window is a rank-arrival barrier or a
  profiler-start skew charged as GPU-busy, not steady-state cost. Measured at
  73.9-97.2% of the window on six of six reference runs; a clean decode window
  puts its largest single kernel under 1%.
* **Cross-rank consistency** (:data:`RANK_KERNEL_COUNT_IMBALANCE`,
  :data:`RANK_PROFILER_START_SKEW`) -- tensor-parallel ranks execute in lockstep,
  so their kernel counts must agree and their capture windows must open
  together. Measured 87x kernel-count spread and 30.88 s of start skew against
  0.5 s of stop skew on the reference incident.

Findings are advisory. :meth:`TraceProbeResult.to_health_warnings` caps severity
at ``warning`` by default so a probe can be wired into a route without changing
what that route decides -- the point of the first deployment is to establish the
distribution of these metrics on real captures, not to start rejecting work on
thresholds calibrated against seven sessions.

Kept stdlib-only (the streaming reader is imported lazily, see :func:`_reader`)
so both the orchestrator, which imports this as
``hyperloom.agents.kernel.tools._trace_probe``, and the standalone kernel-agent
tools, which import it as a bare ``_trace_probe``, get the same answers.
"""

from __future__ import annotations

import os
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Finding codes
# --------------------------------------------------------------------------- #

#: A graph-mode capture recorded kernels for only a fraction of its replays.
GRAPH_REPLAY_UNDER_RECORDED = "graph_replay_under_recorded"
#: Per-iteration annotations exist but enclose no GPU work.
ANNOTATED_STEPS_WITHOUT_GPU_WORK = "annotated_steps_without_gpu_work"
#: Runtime launches were recorded without the kernels they launched.
KERNEL_LAUNCH_RATIO_COLLAPSED = "kernel_launch_ratio_collapsed"
#: One device event covers most of the analysed window.
SINGLE_EVENT_DOMINATES_WINDOW = "single_event_dominates_window"
#: Python-function events crowd out device events (``with_stack`` overhead).
PYTHON_FUNCTION_FLOOD = "python_function_flood"
#: Lockstep ranks disagree on how many kernels they executed.
RANK_KERNEL_COUNT_IMBALANCE = "rank_kernel_count_imbalance"
#: Ranks opened their capture windows far apart.
RANK_PROFILER_START_SKEW = "rank_profiler_start_skew"
#: The probe could not read the trace at all.
TRACE_PROBE_UNREADABLE = "trace_probe_unreadable"

#: Every code this module can emit, so a consumer can allow-list them.
KNOWN_CODES: frozenset[str] = frozenset(
    {
        GRAPH_REPLAY_UNDER_RECORDED,
        ANNOTATED_STEPS_WITHOUT_GPU_WORK,
        KERNEL_LAUNCH_RATIO_COLLAPSED,
        SINGLE_EVENT_DOMINATES_WINDOW,
        PYTHON_FUNCTION_FLOOD,
        RANK_KERNEL_COUNT_IMBALANCE,
        RANK_PROFILER_START_SKEW,
        TRACE_PROBE_UNREADABLE,
    }
)

VERDICT_USABLE = "usable"
VERDICT_DEGRADED = "degraded"
VERDICT_UNUSABLE = "unusable"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_BLOCKING = "blocking"

_SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_WARNING: 1, SEVERITY_BLOCKING: 2}

# --------------------------------------------------------------------------- #
# Thresholds
#
# Every default below is the value that cleanly separates the reference sample
# (7 sessions, 3 models, sglang + TP8 + MI355X). None of them is derived from
# first principles, and the sample carries no vLLM, single-GPU, or diffusion
# capture -- hence the env overrides and the advisory default severity.
# --------------------------------------------------------------------------- #

#: Recorded-replay coverage at or below this is under-recording. Measured
#: 0.008 (1/128) on the bad ranks, 1.0 on the good ones in the same profile.
GRAPH_REPLAY_COVERAGE_MIN_DEFAULT = 0.5
GRAPH_REPLAY_COVERAGE_MIN_ENV = "HYPERLOOM_TRACE_PROBE_GRAPH_COVERAGE_MIN"

#: Share of step annotations that must enclose at least one kernel. The
#: reference incident recorded 0 of 128.
ANNOTATED_STEP_GPU_COVERAGE_MIN_DEFAULT = 0.5
ANNOTATED_STEP_GPU_COVERAGE_MIN_ENV = "HYPERLOOM_TRACE_PROBE_STEP_COVERAGE_MIN"

#: Kernels per recorded runtime launch. Deliberately near zero: an eager trace
#: sits around 1.0 and a graph-replay trace an order of magnitude above, so this
#: is only decisive when the device side is missing outright (measured 0/640 on
#: the MiniMax steady-state chunk).
KERNEL_PER_LAUNCH_RATIO_MIN_DEFAULT = 0.01
KERNEL_PER_LAUNCH_RATIO_MIN_ENV = "HYPERLOOM_TRACE_PROBE_KERNEL_LAUNCH_RATIO_MIN"

#: Largest single device event as a share of the window span. Measured
#: 0.739-0.972 on the six reference runs carrying a rank-skew barrier; a clean
#: decode window stays under 0.01.
SINGLE_EVENT_WINDOW_SHARE_MAX_DEFAULT = 0.30
SINGLE_EVENT_WINDOW_SHARE_MAX_ENV = "HYPERLOOM_TRACE_PROBE_SINGLE_EVENT_SHARE_MAX"

#: ``python_function`` share of all events. A cost signal rather than a validity
#: one (it tracks ``with_stack``), so it is reported at ``info``.
PYTHON_FUNCTION_SHARE_MAX_DEFAULT = 0.95
PYTHON_FUNCTION_SHARE_MAX_ENV = "HYPERLOOM_TRACE_PROBE_PYTHON_SHARE_MAX"

#: max/min kernel count across lockstep ranks. Measured 87x on the incident.
RANK_KERNEL_SPREAD_MAX_DEFAULT = 2.0
RANK_KERNEL_SPREAD_MAX_ENV = "HYPERLOOM_TRACE_PROBE_RANK_SPREAD_MAX"

#: Capture-open skew across ranks, as a share of the median rank window span.
#: Measured above 1.0 on the incident (30.88 s of skew over a ~21 s window)
#: against 0.5 s of stop skew.
RANK_START_SKEW_SHARE_MAX_DEFAULT = 0.10
RANK_START_SKEW_SHARE_MAX_ENV = "HYPERLOOM_TRACE_PROBE_RANK_SKEW_SHARE_MAX"

#: Events decoded before the single-file probe stops early. A capped probe still
#: reports, with ``truncated=True`` on the metrics, because a partial answer on
#: the leading events is worth more than no answer on a 300 MB trace.
MAX_EVENTS_DEFAULT = 4_000_000
MAX_EVENTS_ENV = "HYPERLOOM_TRACE_PROBE_MAX_EVENTS"

#: Master switch. Unset means on; set to a falsey value to skip probing.
ENABLED_ENV = "HYPERLOOM_TRACE_PROBE"

_FALSEY = frozenset({"0", "false", "no", "off"})

# Kineto categories.
_CAT_KERNEL = "kernel"
_CAT_RUNTIME = "cuda_runtime"
_CAT_PYTHON = "python_function"
_GPU_ANNOTATION_CATS = frozenset({"gpu_user_annotation", "user_annotation"})
_DEVICE_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})


def probe_enabled() -> bool:
    """Whether trace probing is switched on (default yes).

    Returns:
        ``False`` only when :data:`ENABLED_ENV` is set to a falsey value.
    """
    return os.environ.get(ENABLED_ENV, "").strip().lower() not in _FALSEY


def _resolve_float(env_name: str, default: float) -> float:
    """Read a non-negative float override from ``env_name``.

    Args:
        env_name: Environment variable holding the override.
        default: Value used when unset, unparseable, or negative.

    Returns:
        The resolved value.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _resolve_int(env_name: str, default: int) -> int:
    """Read a positive int override from ``env_name``.

    Args:
        env_name: Environment variable holding the override.
        default: Value used when unset, unparseable, or non-positive.

    Returns:
        The resolved value.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _reader():
    """Return the streaming Kineto reader module.

    Imported lazily rather than at module scope because
    ``_bypass_trace_reader`` resolves its own siblings by bare name, so it is
    not importable through the package path until this directory is on
    ``sys.path``. Doing the insert here keeps the cost on the one call that
    needs it and leaves this module importable either way -- which is the whole
    point of keeping it stdlib-only.

    Returns:
        The ``_bypass_trace_reader`` module.
    """
    try:
        import _bypass_trace_reader as reader  # type: ignore[import-not-found]
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import _bypass_trace_reader as reader  # type: ignore[import-not-found]
    return reader


@dataclass(frozen=True)
class TraceFinding:
    """One semantic defect observed in a trace.

    Attributes:
        code: One of :data:`KNOWN_CODES`.
        severity: ``info`` / ``warning`` / ``blocking`` -- the severity the
            evidence warrants, before any caller-side capping.
        message: Human-readable statement of what was measured.
        evidence: The measurements behind the finding, so a reader can judge it
            without re-opening the trace.
    """

    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render as a ``trace_health_warnings`` row.

        Returns:
            A dict carrying ``code`` / ``severity`` / ``message`` plus the
            evidence fields flattened alongside them, matching the shape the
            existing warnings use.
        """
        row: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        row.update(self.evidence)
        return row


@dataclass(frozen=True)
class TraceProbeResult:
    """Outcome of probing one trace file, or one set of per-rank traces.

    Attributes:
        target: The file or directory probed.
        findings: Every defect observed, most severe first.
        metrics: Raw measurements, emitted whether or not anything fired, so a
            passing probe still contributes a data point.
    """

    target: str
    findings: list[TraceFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """The overall judgement implied by :attr:`findings`.

        Returns:
            :data:`VERDICT_UNUSABLE` when any finding is blocking,
            :data:`VERDICT_DEGRADED` when any is a warning, else
            :data:`VERDICT_USABLE`.
        """
        worst = max((_SEVERITY_RANK.get(f.severity, 0) for f in self.findings), default=-1)
        if worst >= _SEVERITY_RANK[SEVERITY_BLOCKING]:
            return VERDICT_UNUSABLE
        if worst >= _SEVERITY_RANK[SEVERITY_WARNING]:
            return VERDICT_DEGRADED
        return VERDICT_USABLE

    def to_health_warnings(self, *, observation_only: bool = True) -> list[dict[str, Any]]:
        """Render the findings as ``trace_health_warnings`` rows.

        Args:
            observation_only: When True (the default) a ``blocking`` finding is
                emitted as a ``warning``. The thresholds behind these codes are
                calibrated against seven sessions on one framework and one
                platform; until the fleet distribution is known, a probe must be
                able to say what it saw without that statement being read as an
                instruction to stop.

        Returns:
            One row per finding, most severe first.
        """
        rows = []
        for finding in sorted(
            self.findings,
            key=lambda f: -_SEVERITY_RANK.get(f.severity, 0),
        ):
            row = finding.to_dict()
            if observation_only and row.get("severity") == SEVERITY_BLOCKING:
                row["severity"] = SEVERITY_WARNING
                row["probe_severity"] = SEVERITY_BLOCKING
            rows.append(row)
        return rows

    def summary_line(self) -> str:
        """One-line rendering for the tool log.

        Returns:
            ``trace_probe: <verdict> <target> [<code>=<value> ...]``, listing
            every finding's headline measurement.
        """
        parts = [f"trace_probe: {self.verdict} {self.target}"]
        for finding in self.findings:
            parts.append(f"{finding.code}({finding.severity})")
        metric_keys = (
            "kernel_events",
            "graph_replay_coverage",
            "annotated_step_gpu_coverage",
            "single_event_window_share",
            "rank_kernel_spread",
            "rank_start_skew_share",
        )
        measured = [f"{k}={self.metrics[k]}" for k in metric_keys if self.metrics.get(k) is not None]
        if measured:
            parts.append("| " + " ".join(measured))
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Single-file probe
# --------------------------------------------------------------------------- #


def _scan(events: Iterable[dict], *, max_events: int) -> dict[str, Any]:
    """Fold one event stream into the raw counters every check reads.

    One pass, because the traces this runs on reach hundreds of megabytes
    decompressed and each additional pass costs the same again.

    Args:
        events: Decoded Kineto trace events.
        max_events: Stop after this many events; the result is flagged
            ``truncated``.

    Returns:
        A dict of raw counters and buffers for :func:`_evaluate_file`.
    """
    cat_counts: dict[str, int] = {}
    kernel_count = 0
    kernel_dur_us = 0.0
    max_event = ("", 0.0, 0.0)  # name, dur_us, ts
    ts_min: float | None = None
    ts_max: float | None = None
    # Latest *end*, not the latest start. The window closes when the last event
    # finishes, and on a capture whose final event is the long one -- exactly the
    # rank-skew barrier this module looks for -- a span measured to the last
    # start omits that event's own duration and inflates its share past 1.0.
    end_max: float | None = None
    launch_correlations = 0
    graph_launch_corrs: set[Any] = set()
    kernel_corrs: set[Any] = set()
    kernel_starts: list[float] = []
    step_spans: list[tuple[float, float, str]] = []
    scanned = 0
    truncated = False

    for ev in events:
        scanned += 1
        if scanned > max_events:
            truncated = True
            break
        cat = ev.get("cat") or ""
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        ts = ev.get("ts")
        if isinstance(ts, (int, float)):
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            dur_any = ev.get("dur")
            end = float(ts) + (float(dur_any) if isinstance(dur_any, (int, float)) else 0.0)
            end_max = end if end_max is None else max(end_max, end)

        if cat == _CAT_RUNTIME:
            args = ev.get("args") or {}
            corr = args.get("correlation")
            if corr is not None:
                launch_correlations += 1
                if "GraphLaunch" in (ev.get("name") or ""):
                    graph_launch_corrs.add(corr)
            continue

        if cat in _GPU_ANNOTATION_CATS and ev.get("ph") == "X":
            # Only spans that look like a per-iteration step marker; a bare
            # region annotation covers the whole capture and would make the
            # coverage check trivially pass.
            name = ev.get("name") or ""
            dur = ev.get("dur")
            if isinstance(ts, (int, float)) and isinstance(dur, (int, float)) and dur > 0:
                step_spans.append((float(ts), float(ts) + float(dur), str(name)))
            continue

        if cat in _DEVICE_CATS and ev.get("ph") == "X":
            dur = float(ev.get("dur") or 0)
            start = float(ts) if isinstance(ts, (int, float)) else 0.0
            if dur > max_event[1]:
                max_event = (str(ev.get("name") or ""), dur, start)
            if cat == _CAT_KERNEL:
                kernel_count += 1
                kernel_dur_us += dur
                kernel_starts.append(start)
                corr = (ev.get("args") or {}).get("correlation")
                if corr is not None:
                    kernel_corrs.add(corr)

    return {
        "cat_counts": cat_counts,
        "kernel_count": kernel_count,
        "kernel_dur_us": kernel_dur_us,
        "max_event": max_event,
        "ts_min": ts_min,
        "ts_max": ts_max if end_max is None else max(ts_max or end_max, end_max),
        "launch_correlations": launch_correlations,
        "graph_launch_corrs": graph_launch_corrs,
        "kernel_corrs": kernel_corrs,
        "kernel_starts": kernel_starts,
        "step_spans": step_spans,
        "event_total": scanned - (1 if truncated else 0),
        "truncated": truncated,
    }


def _steps_with_gpu_work(step_spans: list[tuple[float, float, str]], kernel_starts: list[float]) -> int:
    """Count step annotations enclosing at least one kernel start.

    Args:
        step_spans: ``(start_us, end_us, name)`` per annotation.
        kernel_starts: Kernel start timestamps, any order.

    Returns:
        How many spans contain at least one kernel start.
    """
    if not step_spans or not kernel_starts:
        return 0
    ordered = sorted(kernel_starts)
    hits = 0
    for start, end, _name in step_spans:
        idx = bisect_left(ordered, start)
        if idx < len(ordered) and ordered[idx] < end:
            hits += 1
    return hits


def _evaluate_file(scan: dict[str, Any], target: str) -> TraceProbeResult:
    """Turn raw counters into findings and metrics.

    Args:
        scan: Output of :func:`_scan`.
        target: The probed path, recorded on the result.

    Returns:
        The probe result for one trace file.
    """
    findings: list[TraceFinding] = []
    cat_counts: dict[str, int] = scan["cat_counts"]
    event_total = int(scan["event_total"])
    kernel_count = int(scan["kernel_count"])
    ts_min, ts_max = scan["ts_min"], scan["ts_max"]
    span_us = float(ts_max - ts_min) if ts_min is not None and ts_max is not None and ts_max > ts_min else 0.0

    metrics: dict[str, Any] = {
        "event_total": event_total,
        "kernel_events": kernel_count,
        "kernel_busy_ms": round(scan["kernel_dur_us"] / 1000.0, 3),
        "window_span_ms": round(span_us / 1000.0, 3),
        "cpu_op_events": cat_counts.get("cpu_op", 0),
        "truncated": bool(scan["truncated"]),
        # Underscore-prefixed: the absolute capture-open timestamp is meaningless
        # on its own (it is a raw device clock) and is only ever read by
        # ``probe_rank_set`` to measure skew between ranks of one capture.
        "_ts_min": ts_min,
    }

    # --- GPU-record completeness: graph replays -------------------------- #
    graph_launches = len(scan["graph_launch_corrs"])
    replays_with_kernels = len(scan["graph_launch_corrs"] & scan["kernel_corrs"])
    metrics["graph_launch_count"] = graph_launches
    metrics["graph_replays_with_kernels"] = replays_with_kernels
    if graph_launches >= 2 and kernel_count > 0:
        coverage = replays_with_kernels / graph_launches
        metrics["graph_replay_coverage"] = round(coverage, 4)
        floor = _resolve_float(GRAPH_REPLAY_COVERAGE_MIN_ENV, GRAPH_REPLAY_COVERAGE_MIN_DEFAULT)
        if coverage < floor:
            findings.append(
                TraceFinding(
                    code=GRAPH_REPLAY_UNDER_RECORDED,
                    severity=SEVERITY_BLOCKING,
                    message=(
                        f"only {replays_with_kernels} of {graph_launches} graph replays recorded any "
                        f"kernel ({coverage:.1%} < {floor:.0%}). The kernels inside the remaining "
                        "replays are absent from this trace, so every share computed from it -- idle, "
                        "compute, per-kernel GPU% -- is taken over a fraction of the run, and the "
                        "missing kernels carry no enclosing cpu_op, hence no source file and no "
                        "operand shapes."
                    ),
                    evidence={
                        "graph_launch_count": graph_launches,
                        "graph_replays_with_kernels": replays_with_kernels,
                        "graph_replay_coverage": round(coverage, 4),
                        "threshold": floor,
                    },
                )
            )

    # --- GPU-record completeness: annotated steps ------------------------- #
    step_spans = scan["step_spans"]
    metrics["annotated_steps"] = len(step_spans)
    if step_spans:
        with_work = _steps_with_gpu_work(step_spans, scan["kernel_starts"])
        coverage = with_work / len(step_spans)
        metrics["annotated_steps_with_gpu_work"] = with_work
        metrics["annotated_step_gpu_coverage"] = round(coverage, 4)
        floor = _resolve_float(ANNOTATED_STEP_GPU_COVERAGE_MIN_ENV, ANNOTATED_STEP_GPU_COVERAGE_MIN_DEFAULT)
        if coverage < floor:
            findings.append(
                TraceFinding(
                    code=ANNOTATED_STEPS_WITHOUT_GPU_WORK,
                    severity=SEVERITY_BLOCKING,
                    message=(
                        f"{len(step_spans) - with_work} of {len(step_spans)} annotated steps enclose no "
                        f"GPU kernel ({coverage:.1%} coverage < {floor:.0%}). The CPU side declared the "
                        "work and the device side did not record it."
                    ),
                    evidence={
                        "annotated_steps": len(step_spans),
                        "annotated_steps_with_gpu_work": with_work,
                        "annotated_step_gpu_coverage": round(coverage, 4),
                        "threshold": floor,
                    },
                )
            )

    # --- GPU-record completeness: launches vs kernels --------------------- #
    launches = int(scan["launch_correlations"])
    metrics["runtime_launches"] = launches
    if launches > 0:
        ratio = kernel_count / launches
        metrics["kernel_per_launch_ratio"] = round(ratio, 4)
        floor = _resolve_float(KERNEL_PER_LAUNCH_RATIO_MIN_ENV, KERNEL_PER_LAUNCH_RATIO_MIN_DEFAULT)
        if ratio < floor:
            findings.append(
                TraceFinding(
                    code=KERNEL_LAUNCH_RATIO_COLLAPSED,
                    severity=SEVERITY_BLOCKING,
                    message=(
                        f"{launches} runtime launches produced {kernel_count} recorded kernels "
                        f"(ratio {ratio:.4f} < {floor}). The host side of the trace is intact and the "
                        "device side is missing."
                    ),
                    evidence={
                        "runtime_launches": launches,
                        "kernel_events": kernel_count,
                        "kernel_per_launch_ratio": round(ratio, 4),
                        "threshold": floor,
                    },
                )
            )

    # --- Timeline exclusivity --------------------------------------------- #
    name, dur_us, start_us = scan["max_event"]
    if span_us > 0 and dur_us > 0:
        share = dur_us / span_us
        metrics["single_event_window_share"] = round(share, 4)
        metrics["single_event_name"] = name
        metrics["single_event_ms"] = round(dur_us / 1000.0, 3)
        ceiling = _resolve_float(SINGLE_EVENT_WINDOW_SHARE_MAX_ENV, SINGLE_EVENT_WINDOW_SHARE_MAX_DEFAULT)
        if share > ceiling:
            offset_ms = round((start_us - float(ts_min)) / 1000.0, 3) if ts_min is not None else None
            findings.append(
                TraceFinding(
                    code=SINGLE_EVENT_DOMINATES_WINDOW,
                    severity=SEVERITY_BLOCKING,
                    message=(
                        f"a single device event covers {share:.1%} of the {span_us / 1000.0:.1f} ms window "
                        f"({name[:80]!r}, {dur_us / 1000.0:.1f} ms, starting {offset_ms} ms into the "
                        f"capture; threshold {ceiling:.0%}). A collective this long is peer-arrival skew "
                        "charged as GPU-busy, not communication cost: it leaves idle% near zero while "
                        "diluting every percent-of-window metric, compute% included. Percentages taken "
                        "over this window describe the stall, not the workload."
                    ),
                    evidence={
                        "single_event_name": name,
                        "single_event_ms": round(dur_us / 1000.0, 3),
                        "single_event_offset_ms": offset_ms,
                        "window_span_ms": round(span_us / 1000.0, 3),
                        "single_event_window_share": round(share, 4),
                        "threshold": ceiling,
                    },
                )
            )

    # --- Recording overhead ------------------------------------------------ #
    python_events = cat_counts.get(_CAT_PYTHON, 0)
    if event_total > 0:
        share = python_events / event_total
        metrics["python_function_share"] = round(share, 4)
        ceiling = _resolve_float(PYTHON_FUNCTION_SHARE_MAX_ENV, PYTHON_FUNCTION_SHARE_MAX_DEFAULT)
        if share > ceiling and python_events > 0:
            findings.append(
                TraceFinding(
                    code=PYTHON_FUNCTION_FLOOD,
                    severity=SEVERITY_INFO,
                    message=(
                        f"{share:.1%} of events are python_function ({python_events} of {event_total}); "
                        "with_stack capture is crowding out the device side and may be starving the "
                        "profiler's activity buffer."
                    ),
                    evidence={
                        "python_function_events": python_events,
                        "event_total": event_total,
                        "python_function_share": round(share, 4),
                        "threshold": ceiling,
                    },
                )
            )

    return TraceProbeResult(target=target, findings=findings, metrics=metrics)


def probe_file(path: str | Path, *, max_events: int | None = None) -> TraceProbeResult:
    """Probe one trace file for semantic defects.

    Never raises: an unreadable trace comes back as a
    :data:`TRACE_PROBE_UNREADABLE` finding, because a probe that can break the
    run it is observing is worse than no probe.

    Args:
        path: The trace file to read (``.json`` or ``.json.gz``).
        max_events: Cap on events decoded; defaults to :data:`MAX_EVENTS_ENV`
            or :data:`MAX_EVENTS_DEFAULT`.

    Returns:
        The probe result, with ``metrics`` populated even when nothing fired.
    """
    target = str(path)
    cap = max_events if max_events is not None else _resolve_int(MAX_EVENTS_ENV, MAX_EVENTS_DEFAULT)
    try:
        reader = _reader()
        stream_errors: list[str] = []
        fobj = reader._open_trace_binary(Path(path))
        try:
            scan = _scan(reader.stream_events(fobj, errors=stream_errors), max_events=cap)
        finally:
            fobj.close()
    except Exception as exc:  # noqa: BLE001 - the probe must never break its caller
        return TraceProbeResult(
            target=target,
            findings=[
                TraceFinding(
                    code=TRACE_PROBE_UNREADABLE,
                    severity=SEVERITY_WARNING,
                    message=f"could not read {target}: {type(exc).__name__}: {exc}",
                    evidence={"trace_file": target, "error": f"{type(exc).__name__}: {exc}"},
                )
            ],
        )
    result = _evaluate_file(scan, target)
    result.metrics["stream_errors"] = len(stream_errors)
    if stream_errors:
        result.metrics["stream_error_sample"] = stream_errors[:3]
    return result


# --------------------------------------------------------------------------- #
# Rank-set probe
# --------------------------------------------------------------------------- #


def probe_rank_set(results: dict[Any, TraceProbeResult]) -> TraceProbeResult:
    """Cross-check per-rank probe results for lockstep consistency.

    Tensor-parallel ranks run the same graph on the same shapes, so a
    disagreement between them is a property of the *recording*, not of the
    workload -- which makes this the one check that needs no threshold on
    absolute values.

    Args:
        results: ``identity -> probe result`` for the ranks of one capture. The
            key is whatever names the rank to a reader -- a rank index or a file
            name -- and is echoed back in the findings.

    Returns:
        A result whose ``target`` names the rank set, carrying only the
        cross-rank findings. Returns an empty result for fewer than two ranks.
    """
    target = f"rank_set[{','.join(str(r) for r in sorted(results, key=str))}]"
    findings: list[TraceFinding] = []
    metrics: dict[str, Any] = {"rank_count": len(results)}
    if len(results) < 2:
        return TraceProbeResult(target=target, findings=findings, metrics=metrics)

    counts = {rank: int(res.metrics.get("kernel_events") or 0) for rank, res in results.items()}
    metrics["rank_kernel_events"] = dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
    lo, hi = min(counts.values()), max(counts.values())
    if lo > 0:
        spread = hi / lo
        metrics["rank_kernel_spread"] = round(spread, 2)
        ceiling = _resolve_float(RANK_KERNEL_SPREAD_MAX_ENV, RANK_KERNEL_SPREAD_MAX_DEFAULT)
        if spread > ceiling:
            starved = sorted((r for r, c in counts.items() if c * ceiling < hi), key=str)
            findings.append(
                TraceFinding(
                    code=RANK_KERNEL_COUNT_IMBALANCE,
                    severity=SEVERITY_BLOCKING,
                    message=(
                        f"lockstep ranks recorded between {lo} and {hi} kernels ({spread:.1f}x spread, "
                        f"threshold {ceiling}x). Ranks {starved} under-recorded; a conclusion drawn from "
                        "one of them describes the profiler, not the workload."
                    ),
                    evidence={
                        "rank_kernel_events": metrics["rank_kernel_events"],
                        "rank_kernel_spread": round(spread, 2),
                        "starved_ranks": starved,
                        "threshold": ceiling,
                    },
                )
            )
    elif hi > 0:
        metrics["rank_kernel_spread"] = None
        findings.append(
            TraceFinding(
                code=RANK_KERNEL_COUNT_IMBALANCE,
                severity=SEVERITY_BLOCKING,
                message=(
                    f"at least one rank recorded zero kernels while another recorded {hi}; "
                    "the capture is not consistent across ranks."
                ),
                evidence={"rank_kernel_events": metrics["rank_kernel_events"]},
            )
        )

    starts = {r: res.metrics.get("_ts_min") for r, res in results.items()}
    spans = [float(res.metrics.get("window_span_ms") or 0.0) for res in results.values()]
    known = {r: v for r, v in starts.items() if isinstance(v, (int, float))}
    if len(known) >= 2 and spans:
        skew_ms = (max(known.values()) - min(known.values())) / 1000.0
        ordered_spans = sorted(spans)
        median_span_ms = ordered_spans[len(ordered_spans) // 2]
        metrics["rank_start_skew_ms"] = round(skew_ms, 3)
        metrics["rank_median_span_ms"] = round(median_span_ms, 3)
        if median_span_ms > 0:
            share = skew_ms / median_span_ms
            metrics["rank_start_skew_share"] = round(share, 4)
            ceiling = _resolve_float(RANK_START_SKEW_SHARE_MAX_ENV, RANK_START_SKEW_SHARE_MAX_DEFAULT)
            if share > ceiling:
                latest = max(known, key=lambda r: known[r])
                findings.append(
                    TraceFinding(
                        code=RANK_PROFILER_START_SKEW,
                        severity=SEVERITY_BLOCKING,
                        message=(
                            f"ranks opened their capture windows {skew_ms:.1f} ms apart against a median "
                            f"window of {median_span_ms:.1f} ms ({share:.1%}, threshold {ceiling:.0%}); "
                            f"rank {latest} opened last. Every earlier rank spends the difference inside "
                            "the first collective of the window, which is recorded as GPU-busy time and "
                            "dilutes every share computed from that window."
                        ),
                        evidence={
                            "rank_start_skew_ms": round(skew_ms, 3),
                            "rank_median_span_ms": round(median_span_ms, 3),
                            "rank_start_skew_share": round(share, 4),
                            "last_rank_to_open": latest,
                            "threshold": ceiling,
                        },
                    )
                )
    return TraceProbeResult(target=target, findings=findings, metrics=metrics)


def first_event_ts(path: str | Path) -> float | None:
    """Timestamp of the first event carrying one, or ``None``.

    Reads only until that event, so this costs a few KB per file rather than a
    full decode. That matters because the skew check below is the one cross-rank
    signal worth paying for on every analysis, and paying for it the way
    :func:`probe_paths` does -- a complete pass over every rank -- would add
    minutes to a run for two numbers per file.

    Args:
        path: The trace file to peek at.

    Returns:
        The first ``ts`` seen, or ``None`` when the file is unreadable or holds
        no timestamped event.
    """
    try:
        reader = _reader()
        fobj = reader._open_trace_binary(Path(path))
        try:
            for ev in reader.stream_events(fobj):
                ts = ev.get("ts")
                if isinstance(ts, (int, float)):
                    return float(ts)
        finally:
            fobj.close()
    except Exception:  # noqa: BLE001 - the probe must never break its caller
        return None
    return None


def probe_start_skew(paths: Iterable[str | Path]) -> TraceProbeResult:
    """Check whether per-rank captures opened together, reading only file heads.

    The cheap half of the cross-rank story. Ranks decide to start profiling in
    the same instant but call into the profiler independently, and on the
    reference incident the calls landed 30.88 s apart while the stops landed
    within 0.5 s. Every rank that opened early spends the difference inside the
    window's first collective, which is recorded as GPU-busy time and dilutes
    every share taken over that window.

    Unlike :func:`probe_rank_set` this needs no kernel counts, so it compares
    the skew against the *shortest* window rather than the median: with only
    head timestamps the span of each rank is unknown, and the earliest-opening
    rank's window is the one the skew consumes the largest fraction of.

    Args:
        paths: Trace files belonging to one capture, one per rank.

    Returns:
        A result carrying at most the :data:`RANK_PROFILER_START_SKEW` finding.
    """
    starts: dict[str, float] = {}
    for path in paths:
        ts = first_event_ts(path)
        if ts is not None:
            starts[Path(path).name] = ts
    metrics: dict[str, Any] = {"rank_count": len(starts)}
    target = f"start_skew[{len(starts)} rank(s)]"
    if len(starts) < 2:
        return TraceProbeResult(target=target, metrics=metrics)

    latest_name = max(starts, key=lambda k: starts[k])
    skew_ms = (starts[latest_name] - min(starts.values())) / 1000.0
    metrics["rank_start_skew_ms"] = round(skew_ms, 3)
    metrics["last_rank_to_open"] = latest_name
    findings: list[TraceFinding] = []
    # Absolute floor: sub-second skew is start-up jitter on any real capture and
    # cannot swallow a decode window. The reference healthy case measured under
    # 5 ms; the incident measured 30,884 ms.
    if skew_ms >= 1000.0:
        findings.append(
            TraceFinding(
                code=RANK_PROFILER_START_SKEW,
                severity=SEVERITY_BLOCKING,
                message=(
                    f"per-rank captures opened {skew_ms:.1f} ms apart ({latest_name} opened last). "
                    "Every earlier rank waits out the difference inside the window's first collective, "
                    "which the profiler records as GPU-busy time: idle% stays near zero while compute% "
                    "and every per-kernel GPU% are diluted by the stall."
                ),
                evidence={
                    "rank_start_skew_ms": round(skew_ms, 3),
                    "last_rank_to_open": latest_name,
                    "rank_first_event_ts": {k: round(v, 1) for k, v in sorted(starts.items())},
                },
            )
        )
    return TraceProbeResult(target=target, findings=findings, metrics=metrics)


def probe_paths(paths: Iterable[str | Path], *, max_events: int | None = None) -> TraceProbeResult:
    """Probe several per-rank traces and fold in the cross-rank checks.

    Args:
        paths: Trace files belonging to one capture, one per rank.
        max_events: Forwarded to :func:`probe_file`.

    Returns:
        A combined result: the union of every per-file finding plus the
        cross-rank ones, with per-file metrics nested under ``per_file``.
    """
    per_file: dict[str, TraceProbeResult] = {}
    for path in paths:
        per_file[str(path)] = probe_file(path, max_events=max_events)
    if not per_file:
        return TraceProbeResult(target="<no traces>")

    combined = probe_rank_set(per_file)
    findings = list(combined.findings)
    for name, res in per_file.items():
        for finding in res.findings:
            findings.append(
                TraceFinding(
                    code=finding.code,
                    severity=finding.severity,
                    message=f"[{Path(name).name}] {finding.message}",
                    evidence={**finding.evidence, "trace_file": name},
                )
            )
    metrics = dict(combined.metrics)
    metrics["per_file"] = {Path(n).name: r.metrics for n, r in per_file.items()}
    return TraceProbeResult(
        target=f"{len(per_file)} trace file(s)",
        findings=findings,
        metrics=metrics,
    )


__all__ = [
    "ANNOTATED_STEPS_WITHOUT_GPU_WORK",
    "GRAPH_REPLAY_UNDER_RECORDED",
    "KERNEL_LAUNCH_RATIO_COLLAPSED",
    "KNOWN_CODES",
    "PYTHON_FUNCTION_FLOOD",
    "RANK_KERNEL_COUNT_IMBALANCE",
    "RANK_PROFILER_START_SKEW",
    "SINGLE_EVENT_DOMINATES_WINDOW",
    "TRACE_PROBE_UNREADABLE",
    "VERDICT_DEGRADED",
    "VERDICT_UNUSABLE",
    "VERDICT_USABLE",
    "TraceFinding",
    "TraceProbeResult",
    "first_event_ts",
    "probe_enabled",
    "probe_file",
    "probe_paths",
    "probe_rank_set",
    "probe_start_skew",
]
