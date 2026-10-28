#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens analysis tool for the resident Kernel-agent skill.

Conservative: records every step, writes a stable artifact set, supports TraceLens
capture directories, and has a dry-run path that works without TraceLens installed.
Two routes produce the report: ``agent`` runs the TraceLens skill under a model,
``non-llm`` maps TraceLens's own perf report.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.coerce import first_str
from hyperloom.common.io import JsonFormat, atomic_write_json
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.framework_paths import (
    FRAMEWORK_SOURCE_PACKAGES as _FRAMEWORK_SOURCE_PACKAGES,
)


from .tracelens_arch_benchmark import internal_extension_arch_json, normalize_platform, populate_gpu_arch_json
from .tracelens_skill_runner import (
    TraceLensSkillRequest,
    TraceLensSkillRunResult,
    discover_capture_folder,
    extract_compute_pct_from_analysis_md,
    extract_exposed_comm_pct_from_analysis_md,
    extract_idle_pct_from_analysis_md,
    run_tracelens_skill,
    write_run_config,
)
from hyperloom.inference_optimizer.framework_registry import (
    tracelens_analysis_mode,
)
from TraceLens.Agent.Analysis.category_analyses.analysis_utils import perf_report_csv_dir
from TraceLens.Agent.Analysis.post_processing import render_analysis_json

from ._analysis_json import load_report_tasks
from ._io_utils import append_log
from ._analysis_reports import OrchestratorOutcome, write_reports
from .analysis_route import AnalysisRoute
from ._candidate_assembly import _finalize_candidates, analyze_trace_files, write_source_resolution_artifact
from ._collective_candidates_inject import _inject_collective_candidates
from ._kernel_roofline import default_roofline_fields
from ._kernel_search import refresh_kernel_search_roots
from ._perf_report_analysis import analyze_perf_report
from ._run_status import RunStatus
from ._trace_gates import WorkloadShares, _evaluate_idle_gate_with_graph_guard, _evaluate_low_compute_gate
from ._trace_inputs import (
    _capture_classification_root,
    _count_kernels_if_readable,
    _is_capture_fragment,
    _trace_file_size,
    discover_trace_inputs,
)
from ._trace_split import TraceSplitJob, _default_workspace_path, _run_trace_split
from ._tracelens_checkout import (
    _ensure_tracelens_checkout,
    _is_default_tracelens_root,
    _resolve_arch_benchmark_timeout_s,
    _resolve_tracelens_model,
    _tracelens_checkout_complete,
    run_command,
)

from hyperloom.common import kernel_source_contract as _KSC

log = logging.getLogger(__name__)

#: How many discovered files the CPU-only preflight will open before giving up.
#: A cost ceiling, not the thing that makes the preflight land on the capture --
#: size ordering does that, and it holds whatever the fragments are called or
#: where they sit. Reaching this limit means every large candidate was empty,
#: which is a real capture problem rather than a selection one.
_KERNEL_PROBE_LIMIT = 8

#: Cumulative bytes of candidate traces the preflight will deserialise before
#: giving up. Only the failing path spends this: with size ordering a healthy
#: capture answers on the first probe. It exists because production rank traces
#: reach hundreds of megabytes, and eight of those would turn a failure that
#: used to take a second into one that takes minutes or exhausts memory.
_KERNEL_PROBE_BYTE_BUDGET = 512 * 1024 * 1024


# Candidate building keeps a broad pool; dispatch grouping owns the real budget gate.
# Override with HYPERLOOM_KERNEL_CANDIDATES_TOP_K; non-positive means unbounded.
_DEFAULT_KERNEL_CANDIDATES_TOP_K = 100


def _default_top_k() -> int:
    """Resolve the default kernel-candidate pool size.

    Reads ``HYPERLOOM_KERNEL_CANDIDATES_TOP_K`` when set (``0``/negative =>
    unbounded pool, represented internally as a very large cap), otherwise
    falls back to :data:`_DEFAULT_KERNEL_CANDIDATES_TOP_K`.

    Returns:
        The candidate-build-time cap (a large number when unbounded).
    """
    raw = os.environ.get("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "").strip()
    if not raw:
        return _DEFAULT_KERNEL_CANDIDATES_TOP_K
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_KERNEL_CANDIDATES_TOP_K
    # 0 / negative => no build-time cap (dispatch layer owns the budget).
    return val if val > 0 else 1_000_000


# HIGH_IDLE_PCT_THRESHOLD_* and the idle-gate helpers now live in _idle_gate
# (imported above) as the shared single source of truth across trace routes.


# TRACELENS_ROOT comes from env / --tracelens-root (fail loudly if absent);
# the internal extension is opt-in via TRACELENS_INTERNAL_ROOT.
DEFAULT_TRACELENS_INTERNAL_ROOT = ""

# The analysis-orchestrator skill entry point, relative to the TraceLens root.
_ORCHESTRATOR_SKILL = "TraceLens/Agent/Analysis/skills/analysis-orchestrator/SKILL.md"


def _trajectory_scope(args: argparse.Namespace) -> contextlib.AbstractContextManager[Any]:
    """Scope the SDK run to the launching session's trajectory ledger, when the launcher named one.

    ``asyncio.run`` copies the calling context into its main task, so the scope reaches the run's coroutine.
    """
    if not args.trajectory_session_dir:
        return contextlib.nullcontext()
    from hyperloom.inference_optimizer.trace.trajectory_trace import trajectory_scope

    return trajectory_scope(
        session_dir=Path(args.trajectory_session_dir),
        component="tracelens",
        agent="tracelens",
        phase=args.trajectory_phase or None,
        tick=args.trajectory_tick,
        task_id=args.trajectory_task_id or None,
        parent_span_id=args.trajectory_parent_span_id or None,
    )


def _gpu_arch_json(
    args: argparse.Namespace, *, tl_root: Path, tl_internal_root: Path | None, log_path: Path
) -> Path | None:
    """The arch JSON roofline reads: the bundled spec under the internal extension, else bundled, measured or fallback."""
    arch_benchmark_timeout_s = _resolve_arch_benchmark_timeout_s()
    if tl_internal_root is not None:
        return internal_extension_arch_json(args.target_platform, log=lambda msg: append_log(log_path, msg))
    return populate_gpu_arch_json(
        tracelens_root=tl_root,
        platform=args.target_platform,
        log=lambda msg: append_log(log_path, msg),
        run_command=lambda cmd, *, cwd, timeout_s, env=None: run_command(
            cmd,
            cwd=cwd,
            log_path=log_path,
            timeout_s=timeout_s,
            env=env,
        ),
        timeout_s=arch_benchmark_timeout_s,
    )


class _TraceLensCheckout(NamedTuple):
    """A validated TraceLens checkout: its root, the internal-extension root and the orchestrator skill."""

    root: Path
    internal_root: Path | None
    skill: Path


def _check_tracelens_checkout(args: argparse.Namespace, log_path: Path) -> _TraceLensCheckout:
    """Return the usable TraceLens root, internal-extension root (None when open-source-only) and skill.

    Raises when the root is missing or incomplete, TraceLens and its splitter do not import, or the
    analysis-orchestrator skill is absent.
    """
    tl_root_arg = args.tracelens_root.strip()
    if not tl_root_arg:
        raise SystemExit(
            "TraceLens root not provided: set TRACELENS_ROOT in env "
            "(src/hyperloom/inference_optimizer/assets/install.sh writes it to "
            "kernel-agent.env.sh) or pass --tracelens-root."
        )
    tl_root = Path(tl_root_arg)
    # Internal extension is opt-in (non-empty --tracelens-internal-root / env).
    internal_root_arg = args.tracelens_internal_root.strip()
    tl_internal_root: Path | None = Path(internal_root_arg) if internal_root_arg else None
    if not _tracelens_checkout_complete(tl_root) and _is_default_tracelens_root(tl_root):
        # The installer-managed pod-local checkout can vanish mid-run;
        # self-heal the default path. An operator override fails fast below.
        _ensure_tracelens_checkout(tl_root, log_path=log_path)
    if not tl_root.exists():
        raise FileNotFoundError(f"TraceLens root not found: {tl_root} (set TRACELENS_ROOT or pass --tracelens-root)")
    # A dir that exists but has neither git metadata nor the TraceLens
    # skill tree is unusable; fail fast.
    if not _tracelens_checkout_complete(tl_root):
        raise FileNotFoundError(
            f"TraceLens root incomplete (not a git checkout): {tl_root} "
            "(set TRACELENS_ROOT or pass --tracelens-root to a valid checkout)"
        )
    if tl_internal_root is not None and not tl_internal_root.exists():
        append_log(
            log_path,
            f"TraceLens-internal root not found: {tl_internal_root}; "
            "falling back to open-source-only "
            "(provide an existing internal checkout to enable)",
        )
        tl_internal_root = None
    if tl_internal_root is None:
        append_log(
            log_path,
            "TraceLens-internal: not provided (open-source-only; set TRACELENS_INTERNAL_ROOT to enable)",
        )
        os.environ.pop("TL_EXTENSION", None)
    _check_tracelens_importable(tl_root, args.budget_minutes, log_path)
    # Read and follow the analysis-orchestrator skill entry point.
    skill = tl_root / _ORCHESTRATOR_SKILL
    if not skill.exists():
        raise FileNotFoundError(f"TraceLens analysis-orchestrator skill not found: {skill}")
    append_log(log_path, f"TraceLens skill: {skill}")
    return _TraceLensCheckout(tl_root, tl_internal_root, skill)


def _check_tracelens_importable(tl_root: Path, budget_minutes: float, log_path: Path) -> None:
    """Raise when TraceLens and its splitter do not import in this interpreter."""
    dependency_rc = run_command(
        [
            sys.executable,
            "-c",
            "import TraceLens; import TraceLens.TraceUtils.split_trace.main",
        ],
        cwd=tl_root,
        log_path=log_path,
        timeout_s=max(60, int(budget_minutes * 60)),
    )
    if dependency_rc != 0:
        raise RuntimeError(
            f"tracelens_dependency_error: TraceLens/splitter import failed in {sys.executable} "
            f"(exit {dependency_rc}); install TraceLens dependencies in this interpreter before analysis. "
            f"See {log_path} for subprocess output."
        )


class _ReportGates(NamedTuple):
    """The idle and low-compute gate verdicts on one TraceLens report."""

    idle_pct: float | None
    idle_threshold: float
    high_idle: dict[str, Any] | None
    compute_pct: float | None
    compute_threshold: float
    low_compute: dict[str, Any] | None

    @property
    def suppressed(self) -> bool:
        return self.high_idle is not None or self.low_compute is not None


def _evaluate_report_gates(
    shares: WorkloadShares, report_path: Path, raw_trace_path: Path, warnings: list[dict[str, Any]], log_path: Path
) -> _ReportGates:
    """Evaluate the idle and low-compute gates on the report's shares, recording a graph under-recording guard.

    Logs the idle share when it clears the gates.
    """
    idle_pct_threshold, high_idle_warning, graph_under_recorded_warning = _evaluate_idle_gate_with_graph_guard(
        shares.idle_pct, report_path, raw_trace_path
    )
    compute_pct_threshold, low_compute_warning = _evaluate_low_compute_gate(
        shares.compute_pct, shares.exposed_communication_pct, report_path
    )
    if graph_under_recorded_warning is not None:
        # Under-recording deflates every recorded share alike, so
        # the compute share is as unreliable as idle% here.
        low_compute_warning = None
        warnings.append(graph_under_recorded_warning)
        append_log(
            log_path,
            "TraceLens high idle% is a graph under-recording "
            "artifact (profiler captured ~1 of N graph replays); "
            "skipping the idle/compute gates and keeping hot_kernels[].",
        )
    gates = _ReportGates(
        shares.idle_pct,
        idle_pct_threshold,
        high_idle_warning,
        shares.compute_pct,
        compute_pct_threshold,
        low_compute_warning,
    )
    if not gates.suppressed and shares.idle_pct is not None and graph_under_recorded_warning is None:
        append_log(
            log_path,
            f"TraceLens Executive Summary: "
            f"Idle % = {shares.idle_pct:.2f}% "
            f"(threshold {idle_pct_threshold:.2f}%) — "
            "below gate, continuing with kernel "
            "candidate extraction",
        )
    return gates


def _record_gate_suppression(gates: _ReportGates, warnings: list[dict[str, Any]], log_path: Path) -> None:
    """Record each gate that suppressed hot_kernels[]; both can fire on one window (95% idle AND 3% compute)."""
    if gates.high_idle is not None:
        assert gates.idle_pct is not None
        warnings.append(gates.high_idle)
        append_log(
            log_path,
            f"TraceLens Executive Summary reports "
            f"Idle % = {gates.idle_pct:.2f}% (threshold "
            f"{gates.idle_threshold:.2f}%); suppressing "
            "hot_kernels[] — kernel rewriting cannot move "
            "end-to-end latency in the high-idle regime. "
            "Coordinator will see this in "
            "trace_health_warnings[] and route to "
            "parameter optimization.",
        )
    if gates.low_compute is not None:
        assert gates.compute_pct is not None
        warnings.append(gates.low_compute)
        append_log(
            log_path,
            f"TraceLens Executive Summary reports "
            f"Compute % = {gates.compute_pct:.2f}% "
            f"(threshold {gates.compute_threshold:.2f}%); "
            "suppressing hot_kernels[] — a kernel rewrite "
            "is bounded by the compute share and cannot "
            "move end-to-end latency here. Coordinator "
            "will see this in trace_health_warnings[] and "
            "route to comm/parameter optimization.",
        )


class RouteOutcome(NamedTuple):
    """What a route hands the shared tail: its report, how it ran and the candidates it kept."""

    report_path: Path
    orchestrator_mode: str
    raw_candidates: list[dict[str, Any]]


def _run_agent_skill(
    request: TraceLensSkillRequest, args: argparse.Namespace, log_path: Path
) -> TraceLensSkillRunResult:
    """Run the TraceLens skill under a model."""
    with _trajectory_scope(args):
        return asyncio.run(
            run_tracelens_skill(
                request,
                budget_minutes=args.budget_minutes,
                model=_resolve_tracelens_model(),
                log=lambda msg: append_log(log_path, msg),
            )
        )


def _agent_route_outcome(
    skill_result: TraceLensSkillRunResult,
    args: argparse.Namespace,
    raw_trace_path: Path,
    warnings: list[dict[str, Any]],
    log_path: Path,
) -> RouteOutcome:
    """Gate on the skill's ``analysis.md``, read candidates from its ``analysis.json``."""
    report_path = skill_result.report_path
    shares = WorkloadShares(
        idle_pct=extract_idle_pct_from_analysis_md(report_path),
        compute_pct=extract_compute_pct_from_analysis_md(report_path),
        exposed_communication_pct=extract_exposed_comm_pct_from_analysis_md(report_path),
    )
    gates = _evaluate_report_gates(shares, report_path, raw_trace_path, warnings, log_path)
    raw_candidates: list[dict[str, Any]] = []
    if gates.suppressed:
        _record_gate_suppression(gates, warnings, log_path)
    else:
        # TraceLens owns the parse/group/prose/impact/order; render its structured analysis.json beside the md.
        report_cands = load_report_tasks(render_analysis_json(report_path), framework=args.framework)
        raw_candidates = _inject_collective_candidates(
            skill_result.output_dir,
            report_cands,
            log_path=log_path,
            health_warnings=warnings,
        )
    return RouteOutcome(
        report_path=report_path,
        orchestrator_mode=skill_result.runner,
        raw_candidates=raw_candidates,
    )


def _run_non_llm_route(
    run_config_path: Path,
    framework: str,
    raw_trace_path: Path,
    warnings: list[dict[str, Any]],
    log_path: Path,
) -> RouteOutcome:
    """Map TraceLens's perf report to ``analysis.json`` with no model, gate on its shares and read candidates from it.

    Collective candidates are not injected: they come from ``category_data/multi_kernel_metrics.json``,
    which only the agent writes.
    """
    analysis = analyze_perf_report(run_config_path)
    report_path = analysis.report_path
    warnings.extend(
        {"code": "trace_quality_poor", "severity": "warning", "message": text, "source": str(report_path)}
        for text in analysis.warnings
    )
    gates = _evaluate_report_gates(analysis.shares, report_path, raw_trace_path, warnings, log_path)
    raw_candidates: list[dict[str, Any]] = []
    if gates.suppressed:
        _record_gate_suppression(gates, warnings, log_path)
    else:
        raw_candidates = load_report_tasks(report_path, framework=framework)
    return RouteOutcome(
        report_path=report_path,
        orchestrator_mode=AnalysisRoute.NON_LLM.value,
        raw_candidates=raw_candidates,
    )


def _reject_capture_only_input(trace_input: Path, trace_files: list[Path]) -> None:
    """Raise when every discovered trace is a CUDA-graph capture sidecar."""
    capture_root = _capture_classification_root(trace_input)
    if all(_is_capture_fragment(p, capture_root) for p in trace_files):
        raise RuntimeError(
            "trace_input_capture_only: "
            f"all {len(trace_files)} trace file(s) under {trace_input} "
            "are CUDA-graph capture sidecars. "
            "A capture records graph construction rather than "
            "execution, so it holds no per-iteration annotations for "
            "the steady-state splitter to cut on. "
            "The upstream profile produced no annotated workload trace: "
            "re-profile so the run writes one beside the sidecars. "
            "Sidecars are a supported auxiliary input -- pass them as "
            "--capture-folder alongside a real trace, never as "
            "--trace-input."
        )


class _KernelProbe(NamedTuple):
    """The trace the CPU-only preflight settled on, and what it read to get there."""

    analysed: Path
    kernel_event_count: int
    probed: list[str]
    spent_bytes: int


def _probe_gpu_kernels(trace_files: list[Path]) -> _KernelProbe:
    """Probe discovered traces in order for GPU kernel events, within the file-count and byte budgets.

    ``analysed`` is the first candidate carrying kernels, else the leading one.
    """
    analysis_trace_path = trace_files[0]
    kernel_event_count = 0
    probed: list[str] = []
    spent_bytes = 0
    for candidate in trace_files[:_KERNEL_PROBE_LIMIT]:
        # Each probe deserialises the whole file, so a directory of large
        # empty captures could otherwise turn a fast failure into a slow
        # one. Ordering puts the likeliest candidate first, so stopping
        # on a byte budget costs the unlikely tail, not the answer.
        if probed and spent_bytes >= _KERNEL_PROBE_BYTE_BUDGET:
            probed.append(f"(stopped after {spent_bytes} bytes probed)")
            break
        spent_bytes += _trace_file_size(candidate)
        readable, kernel_event_count = _count_kernels_if_readable(candidate)
        if not readable:
            # Distinct from an empty trace on purpose: "unreadable" sends
            # a reader to the file, "no kernels" sends them to the
            # profiler, and conflating them is the misdirection this
            # whole change exists to remove.
            probed.append(f"{candidate.name}=unreadable")
            continue
        probed.append(f"{candidate.name}={kernel_event_count}")
        if kernel_event_count:
            analysis_trace_path = candidate
            break
    return _KernelProbe(analysis_trace_path, kernel_event_count, probed, spent_bytes)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add the report-naming, capture-folder and budget flags; their env defaults are read now."""
    parser.add_argument(
        "--roofline-output-name",
        default="kernel_roofline.json",
        help="Report file name under reports/. Pass kernel_roofline_opt.json "
        "for the post-kernel-opt snapshot so it does not overwrite baseline.",
    )
    parser.add_argument(
        "--capture-folder",
        default=os.environ.get("TRACELENS_CAPTURE_FOLDER", ""),
        help=(
            "Optional graph-capture folder for TraceLens inference graph "
            "replay analysis. Also accepts env TRACELENS_CAPTURE_FOLDER."
        ),
    )
    parser.add_argument("--budget-minutes", type=float, default=60.0)


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    """Add the steady-state splitter window flags; their env defaults are read now."""
    parser.add_argument(
        "--split-num-steps",
        type=int,
        default=int(os.environ.get("TRACELENS_SPLIT_NUM_STEPS", "32") or 32),
        help=(
            "Number of steady-state iterations for the splitter to extract "
            "(#127). Maps to --num-steps on TraceLens.TraceUtils.split_trace.main."
        ),
    )
    parser.add_argument(
        "--split-conc",
        default=os.environ.get("TRACELENS_SPLIT_CONC", "") or os.environ.get("CONC", ""),
        help=("Expected peak concurrency for the splitter (#127). Maps to --CONC. Defaults to $CONC when set."),
    )
    parser.add_argument(
        "--split-osl",
        default=os.environ.get("TRACELENS_SPLIT_OSL", "") or os.environ.get("OSL", ""),
        help=("Maximum output sequence length hint for the splitter (#127). Maps to --OSL. Defaults to $OSL when set."),
    )
    parser.add_argument(
        "--split-r",
        default=(os.environ.get("TRACELENS_SPLIT_R", "") or os.environ.get("RANDOM_RANGE_RATIO", "")),
        help=(
            "OSL window ratio R for the splitter (#194 §3). Maps to "
            "--R on TraceLens.TraceUtils.split_trace.main. "
            "Pairs with --CONC / --OSL so mixed-window selection uses the "
            "benchmark-contract PD ratio instead of an empirical default. "
            "Defaults to $RANDOM_RANGE_RATIO when set; leave empty to let "
            "the splitter fall back to its built-in heuristic."
        ),
    )


def _add_chunk_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the splitter phase-classification and steady-state chunk flags; their env defaults are read now."""
    parser.add_argument(
        "--split-llm-inference",
        action="store_true",
        help=(
            "Pass --llm-inference to the TraceLens splitter. Enables "
            "shape-based phase classification for traces without serving "
            "annotations. Set automatically for SERVING frameworks."
        ),
    )
    parser.add_argument(
        "--steady-state-mode",
        choices=("mixed", "decode_only", "prefilldecode", "generic"),
        default=(os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "").strip() or "mixed"),
        help=(
            "Which steady-state chunk to consume for the perf report. "
            "For LLM inference (vLLM/SGLang), the splitter produces three "
            "phase-specific chunks (mixed / decode_only / prefilldecode); "
            "for scriptable frameworks (xDiT/custom), 'generic' selects "
            "a single steady_state chunk. "
            "Defaults to 'mixed' (representative DO:PD mix at ~max "
            "concurrency) which matches roofline-v2's default profiling "
            "intent. Switch to 'prefilldecode' when the workload is "
            "short / batched (NUM_PROMPTS << CONC*OSL) so prefill is "
            "burst-shaped and the mixed window degenerates to PD=0 -- "
            "TP=1 + CUDA-graph traces frequently hit this corner case "
            "because the decode region's GPU work is fully inside the "
            "graph replay and rocprofiler-sdk doesn't emit aggregate "
            "Dispatch Task events outside TP-multi-stream contexts, so "
            "the mixed chunk looks 99%% idle while the prefilldecode "
            "chunk carries the real GEMM / attention kernels. "
            "Switch to 'decode_only' when you specifically want the "
            "longest pure-decode region (decode-perf comparison runs). "
            "May also be set via env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE so the coordinator "
            "can re-issue roofline with a different mode after a "
            "steady_state_chunk_low_quality warning lands."
        ),
    )


def main() -> int:
    """CLI entry point for the TraceLens analysis tool.

    Parses arguments, optionally splits the trace into a steady-state chunk,
    runs the selected analysis route, extracts hot-kernel candidates,
    merges roofline data, and writes the run's report sidecars and status.

    Returns:
        int: ``0`` on success, ``1`` when the run failed (the error is also
            written to status and printed as JSON).
    """
    parser = argparse.ArgumentParser(description="Kernel-agent TraceLens analysis tool")
    parser.add_argument("--trace-input", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--target-platform", default="MI355X")
    parser.add_argument(
        "--workspace-path",
        default=_default_workspace_path(),
        help=(
            "Root the tool writes under (output lands at "
            "<workspace_path>/kernel-agent/runs/<session_id>/...). "
            "Defaults to $USER_DATA_PATH so every kernel-agent artefact "
            "stays inside the session dir; falls back to $WORKSPACE_PATH "
            "for legacy launchers, then to /workspace/hyperloom."
        ),
    )
    parser.add_argument(
        "--tracelens-root",
        default=os.environ.get("TRACELENS_ROOT", ""),
        help="TraceLens public checkout (TRACELENS_ROOT). Required: "
        "src/hyperloom/inference_optimizer/assets/install.sh exports it from "
        "kernel-agent.env.sh; pass --tracelens-root only when "
        "running outside the installer-managed env.",
    )
    parser.add_argument(
        "--tracelens-internal-root",
        default=os.environ.get("TRACELENS_INTERNAL_ROOT", DEFAULT_TRACELENS_INTERNAL_ROOT),
        help="Optional TraceLens-internal checkout (TRACELENS_INTERNAL_ROOT). "
        "Rehydration module; plumbed to run_tracelens_skill. "
        "Leave empty for the open-source-only report.",
    )
    _add_run_args(parser)
    # Join keys of the launching Hyperloom session's trajectory ledger; unset outside a session.
    parser.add_argument("--trajectory-session-dir", default="")
    parser.add_argument("--trajectory-phase", default="")
    parser.add_argument("--trajectory-tick", type=int, default=None)
    parser.add_argument("--trajectory-task-id", default="")
    parser.add_argument("--trajectory-parent-span-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--route",
        choices=[route.value for route in AnalysisRoute],
        default=AnalysisRoute.AGENT.value,
        help=(
            "agent runs the TraceLens analysis-orchestrator skill through claude_agent_sdk; "
            "non-llm maps TraceLens's perf report with no model."
        ),
    )
    _add_split_args(parser)
    _add_chunk_selection_args(parser)
    args = parser.parse_args()
    route = AnalysisRoute(args.route)

    args.target_platform = normalize_platform(args.target_platform)

    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"tl-{uuid.uuid4().hex[:8]}"
    started_at = now_iso(timespec="auto")
    # Keep each TraceLens invocation's artifacts in its own run subdirectory.
    # now_iso renders UTC as "+00:00", never "Z", so the suffix is always added.
    ts_compact = started_at.replace("-", "").replace(":", "").split(".")[0] + "Z"
    sub_dir = f"{ts_compact}_{run_id}"
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id / sub_dir
    log_path = run_dir / "logs" / "tracelens_analysis" / f"{run_id}.log"
    status_path = run_dir / "status" / "tracelens_analysis" / f"{run_id}.json"
    artifacts: dict[str, str] = {}
    run_status = RunStatus(status_path, log_path, artifacts, run_id, started_at)
    route_candidates: list[dict[str, Any]] | None = None
    route_report_path: Path | None = None
    orchestrator_mode = "inline"
    # Structured trace-health findings surfaced to the Coordinator.
    trace_health_warnings: list[dict[str, Any]] = []
    # What this run saw and decided, returned to the caller instead of only
    # reaching the CLI log. The caller records it on the SBD V6 roofline event as
    # the run happens, so "which file was analysed, how was the steady-state
    # window picked, which gate refused" survives without re-reading the log.
    run_meta: dict[str, Any] = {
        "preflight": {},
        "split": {},
        "selection": {},
        "steps": [],
        "route_ext": {},
    }

    def _note_step(
        step_id: str,
        *,
        category: str,
        status: str,
        skip_reason: str | None = None,
        **detail: Any,
    ) -> None:
        """Append one step to the run's ladder.

        Args:
            step_id: Stable step name.
            category: Ladder phase (``preflight`` / ``install`` / ``split`` /
                ``select`` / ``analyze`` / ``emit``).
            status: ``ok`` / ``failed`` / ``skipped``.
            skip_reason: Why the step did not run, when skipped.
            **detail: Measured values for the step.
        """
        run_meta["steps"].append(
            {
                "step_id": step_id,
                "order": len(run_meta["steps"]) + 1,
                "category": category,
                "status": status,
                "skip_reason": skip_reason,
                "detail": detail,
            }
        )

    # Discovery is cached per run, so re-run it here: a process that outlives a
    # single run would otherwise keep the roots it saw when it first imported
    # this module, and miss a framework installed since.
    search_roots = refresh_kernel_search_roots()
    # Without a single searchable root every kernel resolves to "" and the whole
    # run reports zero routable candidates -- a host misconfiguration that reads
    # exactly like a trace with nothing worth optimizing. Say so up front.
    if not search_roots:
        trace_health_warnings.append(
            {
                "code": "no_framework_source_root",
                "severity": "error",
                "packages": list(_FRAMEWORK_SOURCE_PACKAGES),
                "message": (
                    "No framework source root exists on this host (looked for "
                    f"{', '.join(_FRAMEWORK_SOURCE_PACKAGES)}). Source resolution "
                    "cannot grep anything, so every hot kernel will be reported "
                    "as non-routable and kernel-opt will have nothing to "
                    "dispatch. Install the framework in this interpreter's "
                    "environment or point $FRAMEWORK_REPO_PATH at its checkout."
                ),
            }
        )

    try:
        run_status.write(
            state="running",
            current_step="discover_trace_input",
        )
        trace_input = Path(args.trace_input).expanduser().resolve()
        trace_inputs = discover_trace_inputs(trace_input)
        trace_input_type, trace_files = trace_inputs
        append_log(log_path, f"trace_input_type={trace_input_type}")
        append_log(log_path, f"trace_files={len(trace_files)}")
        run_meta["preflight"].update(
            {
                "trace_input": str(trace_input),
                "trace_input_type": trace_input_type,
                "trace_file_count": len(trace_files),
            }
        )
        _note_step(
            "discover_inputs",
            category="preflight",
            status="ok",
            trace_input_type=trace_input_type,
            trace_file_count=len(trace_files),
        )
        # The file the analysis will actually read. Discovery order picks the
        # default; the preflight below promotes whichever candidate it proves
        # carries GPU kernels, because passing the check on one file and then
        # analysing another is how an empty trace reaches TraceLens silently.
        #
        # Unconditional: discover_trace_inputs returns [trace_input] for a file
        # and raises FileNotFoundError for a directory with no traces, so the
        # list is never empty. Guarding it would type this as Path | None and
        # push that None through every downstream call for a branch that cannot
        # be taken.
        analysis_trace_path = trace_files[0]

        # Fail-fast when the input is nothing but CUDA-graph capture sidecars.
        #
        # Ahead of the kernel probe below because the two answer different
        # questions and only this one is reliable here. A capture records the
        # graph being built, so it carries a handful of stray kernels rather
        # than none -- 2 out of 1.49M events on the run that prompted this --
        # and a probe that asks "are there any kernels" therefore passes it and
        # hands the splitter a file with no iteration loop in it. The splitter
        # then fails with ``trace_split_no_steady_state``, which reads as "the
        # window was too short" and sends the next person to lengthen a capture
        # that was never a workload timeline to begin with.
        #
        # ``all`` rather than ``any``: a healthy profile writes the annotated
        # trace *beside* its sidecars, and discovery already sorts those first.
        # Only an input with nothing else in it is unanalysable.
        if not args.dry_run:
            _reject_capture_only_input(trace_input, trace_files)

            # Fail-fast on CPU-only traces.
            #
            # Probes candidates in discovery order rather than only the first. A
            # single-file probe reports the capture directory as CPU-only whenever
            # the leading file happens to be a fragment with no kernels, and the
            # error it raises then blames the profiler for a capture that is sitting
            # in the same directory with thirty thousand kernel events in it.
            #
            # With size ordering the first candidate is normally the capture, so this
            # loop exits on one probe and the promotion below stays quiet. It earns
            # its keep on the layouts where it does not: a multi-rank capture whose
            # leading rank recorded nothing.
            probe = _probe_gpu_kernels(trace_files)
            analysis_trace_path = probe.analysed
            kernel_event_count, probed, spent_bytes = probe.kernel_event_count, probe.probed, probe.spent_bytes
            append_log(
                log_path,
                f"trace_gpu_kernel_events={kernel_event_count} (probed={', '.join(probed)})",
            )
            run_meta["preflight"].update(
                {
                    "probe_gpu_kernel_events": kernel_event_count,
                    "probed_files": list(probed),
                    "probed_bytes": spent_bytes,
                    "analysed_file": analysis_trace_path.name,
                    "input_promoted": analysis_trace_path != trace_files[0],
                }
            )
            _note_step(
                "probe_gpu_kernels",
                category="preflight",
                status="ok" if kernel_event_count else "failed",
                gpu_kernel_events=kernel_event_count,
                probed_file_count=len(probed),
                analysed_file=analysis_trace_path.name,
            )
            if analysis_trace_path != trace_files[0]:
                promotion_warning: dict[str, Any] = {
                    "code": "trace_analysis_input_promoted",
                    "severity": "info",
                    "leading_candidate": trace_files[0].name,
                    "analysed": analysis_trace_path.name,
                    "probed": list(probed),
                    "detail": (
                        "the leading candidate carried no GPU kernel events or "
                        "could not be read; the analysis ran on the first "
                        "candidate that did"
                    ),
                }
                # Structured as well as logged: a run that changed its own input
                # has to be explicable from the artifacts, not only from a tool
                # log nobody keeps.
                trace_health_warnings.append(promotion_warning)
                artifacts["tracelens_analysis_input"] = str(analysis_trace_path)
                append_log(
                    log_path,
                    "trace_analysis_input promoted from "
                    f"{trace_files[0].name} to {analysis_trace_path.name} "
                    f"(probed={', '.join(probed)})",
                )
            if kernel_event_count == 0:
                raise RuntimeError(
                    "Trace contains zero GPU kernel events in any of "
                    f"{len(probed)} probed file(s) under {trace_input}: "
                    f"{', '.join(probed)}. Either the upstream profile run "
                    "captured CPU-only activity -- re-run profile with the "
                    "torch.profiler GPU activities enabled (no LD_PRELOAD "
                    "competing for ROCprofiler-SDK) -- or the traces listed as "
                    "'unreadable' above are truncated or corrupt, which is a "
                    "different problem in the same place."
                )

            run_status.write(
                state="running",
                current_step="check_tracelens_dependencies",
            )
            tl_root, tl_internal_root, skill = _check_tracelens_checkout(args, log_path)

            tracelens_dir = run_dir / "tracelens"
            tracelens_dir.mkdir(parents=True, exist_ok=True)

            # Backfill MAF for the open-source TraceLens path: measure the
            # missing arch/MAF spec on an idle GPU before report generation,
            # unless the internal extension (which backfills MAF) is enabled.
            run_status.write(
                state="running",
                current_step="populate_gpu_arch_json",
            )
            gpu_arch_path = _gpu_arch_json(args, tl_root=tl_root, tl_internal_root=tl_internal_root, log_path=log_path)
            if gpu_arch_path is not None:
                artifacts["tracelens_gpu_arch_json"] = str(gpu_arch_path)

            # Split the full-window filtered trace into steady-state chunks via
            # TraceLens's own splitter, since the perf report expects a single
            # steady-state chunk.
            # Whichever candidate the preflight proved has GPU kernels, which is
            # trace_files[0] unless it was promoted. Analysing a different file
            # from the one that passed the check would let an empty rank through
            # on a sibling's evidence.
            cli_trace_path = analysis_trace_path
            # The un-split source trace: analysis runs on the steady-state chunk
            # (cli_trace_path is reassigned below), but graph-capture health is a
            # whole-run property and must be read from the original trace -- the
            # chunk may drop the graph-launch runtime events the detector needs.
            raw_trace_path = analysis_trace_path
            run_status.write(
                state="running",
                current_step="split_trace",
            )
            split_dir = tracelens_dir / "trace_split"
            cli_trace_path, split_meta = _run_trace_split(
                args,
                TraceSplitJob(analysis_trace_path, split_dir, tl_root),
                log_path,
                trace_health_warnings,
            )
            run_meta["split"].update(split_meta)
            run_meta["selection"].update(
                {
                    "requested_mode": args.steady_state_mode,
                    "chunk_label": split_meta.get("chunk_label"),
                    "selected_chunk": str(cli_trace_path),
                    "selected_chunk_count": split_meta.get("selected_chunk_count"),
                    "available_modes": split_meta.get("available_modes"),
                    "fell_back_to_full_trace": False,
                }
            )
            _note_step(
                "split_trace",
                category="split",
                status="ok",
                returncode=split_meta.get("returncode"),
                num_steps=split_meta.get("num_steps"),
                chunks_extracted=split_meta["chunks_extracted"],
            )
            _note_step(
                "select_chunk",
                category="select",
                status="ok",
                requested_mode=args.steady_state_mode,
                selected_chunk=cli_trace_path.name,
                selected_chunk_count=split_meta.get("selected_chunk_count"),
            )
            artifacts["tracelens_trace_split_dir"] = str(split_dir)
            artifacts["tracelens_steady_state_trace"] = str(cli_trace_path)
            append_log(
                log_path,
                f"trace split OK: "
                f"{split_meta['chunks_by_mode']}; "
                f"--steady-state-mode={args.steady_state_mode} -> "
                f"using {cli_trace_path.name} for perf report",
            )

            # Graph-capture folder for TraceLens inference graph replay analysis.
            capture_folder: Path | None = (
                Path(args.capture_folder).expanduser().resolve()
                if args.capture_folder
                # The analysed trace, not the leading candidate: the helper looks
                # for capture_traces/ beside the file it is given, and after a
                # cross-directory promotion those are different places.
                else discover_capture_folder(trace_input, [analysis_trace_path])
            )
            if capture_folder:
                _note_step(
                    "discover_capture_folder",
                    category="preflight",
                    status="ok",
                    skip_reason=None,
                    capture_folder=str(capture_folder),
                    exists=capture_folder.is_dir(),
                )
                append_log(
                    log_path,
                    f"capture_folder resolved: {capture_folder} (exists={capture_folder.is_dir()})",
                )
            else:
                _note_step(
                    "discover_capture_folder",
                    category="preflight",
                    status="skipped",
                    skip_reason="no capture_traces/ beside the analysed trace",
                    capture_folder="",
                    exists=False,
                )

            run_status.write(
                state="running",
                current_step="run_trace_analysis",
            )
            run_config_path = write_run_config(
                output_dir=tracelens_dir,
                trace_path=cli_trace_path,
                platform=args.target_platform,
                analysis_mode=tracelens_analysis_mode(args.framework),
                capture_folder=capture_folder,
            )
            if route is AnalysisRoute.AGENT:
                skill_result = _run_agent_skill(
                    TraceLensSkillRequest(
                        skill_path=skill,
                        output_dir=tracelens_dir,
                        tracelens_root=tl_root,
                        tracelens_internal_root=tl_internal_root,
                        capture_folder=capture_folder,
                    ),
                    args,
                    log_path,
                )
                artifacts.update(skill_result.artifact_paths)
                outcome = _agent_route_outcome(skill_result, args, raw_trace_path, trace_health_warnings, log_path)
            else:
                outcome = _run_non_llm_route(
                    run_config_path, args.framework, raw_trace_path, trace_health_warnings, log_path
                )
            route_report_path = outcome.report_path
            orchestrator_mode = outcome.orchestrator_mode
            route_candidates = []
            if outcome.raw_candidates:
                route_candidates = _finalize_candidates(
                    outcome.raw_candidates,
                    perf_report_csv_dir=perf_report_csv_dir(str(tracelens_dir)),
                )
                write_source_resolution_artifact(
                    route_candidates,
                    run_dir / _KSC.SOURCE_RESOLUTION_FILENAME,
                    framework=args.framework,
                    model_name=args.model_name,
                    log_path=log_path,
                )
        else:
            append_log(log_path, "[dry-run] skipping TraceLens install and external CLI")

        run_status.write(
            state="running",
            current_step="extract_hot_kernels",
        )
        # Production candidates come only from the route's analysis.json.
        candidates = route_candidates
        if candidates:
            append_log(
                log_path,
                f"hot kernels from the {route.value} route ({len(candidates)})",
            )
        elif args.dry_run:
            # Dry-run only.
            append_log(log_path, "dry-run: parsing raw trace for hot kernels")
            candidates = analyze_trace_files(trace_files, _default_top_k())
        else:
            # Routing signal (high idle / TraceLens failure): keep candidates
            # empty so the Coordinator pivots to params/backends.
            append_log(
                log_path,
                "TraceLens produced no kernel candidates; returning "
                "empty hot_kernels[] without fallback so params/backends "
                "optimization can continue.",
            )
        default_roofline_fields(candidates)
        source_resolution_path = run_dir / _KSC.SOURCE_RESOLUTION_FILENAME
        if not source_resolution_path.is_file():
            write_source_resolution_artifact(
                candidates,
                source_resolution_path,
                framework=args.framework,
                model_name=args.model_name,
                log_path=log_path,
            )
        if source_resolution_path.is_file():
            artifacts["kernel_source_resolution"] = str(source_resolution_path)
        artifacts.update(
            write_reports(
                run_dir,
                trace_inputs,
                candidates,
                args,
                OrchestratorOutcome(route_report_path, trace_health_warnings),
            )
        )
        artifacts["cli_log_path"] = str(log_path)
        artifacts["status_path"] = str(status_path)

        # Surface the contracted ``analysis.md`` exit path alongside hot_kernels.
        analysis_report_path = first_str(artifacts.get("tracelens_agent_report"), artifacts.get("trace_report_path"))

        result = {
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "trace_input_type": trace_input_type,
            "hot_kernels": candidates,
            "trace_report_path": artifacts["trace_report_path"],
            "analysis_report_path": analysis_report_path,
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "artifact_paths": artifacts,
            "orchestrator_mode": orchestrator_mode,
            # Trace-quality findings surfaced to the Coordinator (empty = nothing wrong).
            "trace_health_warnings": trace_health_warnings,
            # What this run saw and decided, for the caller's timeline event.
            "run_meta": run_meta,
        }
        _note_step(
            "extract_hot_kernels",
            category="emit",
            status="ok",
            hot_kernel_count=len(candidates),
            orchestrator_mode=orchestrator_mode,
        )
        atomic_write_json(
            run_dir / "session_state.json",
            {
                "session_id": session_id,
                "last_tool": "tracelens_analysis",
                "last_run_id": run_id,
                "updated_at": now_iso(timespec="auto"),
                "model_name": args.model_name,
                "framework": args.framework,
            },
            fmt=JsonFormat(trailing_newline=True),
        )
        run_status.write(
            state="succeeded",
            current_step="done",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # top-level barrier; logged and reported
        log.warning("tracelens analysis failed: %s", exc, exc_info=True)
        append_log(log_path, f"[error] {type(exc).__name__}: {exc}")
        run_status.write(
            state="failed",
            current_step="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        # Include trace_health_warnings accumulated pre-exception for auto-recovery.
        # ``run_meta`` goes out on this path too: the ladder up to the failure is
        # what says which rung refused, so dropping it here would leave the
        # failure cases -- the ones worth explaining -- as the only blind spot.
        print(
            json.dumps(
                {
                    "tool": "tracelens_analysis",
                    "session_id": session_id,
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cli_log_path": str(log_path),
                    "status_path": str(status_path),
                    "trace_health_warnings": trace_health_warnings,
                    "run_meta": run_meta,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
