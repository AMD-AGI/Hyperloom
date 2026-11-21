# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One TraceLens job per PD server node, for a multi-trace analysis aggregated with ``combine_by_node``.

Each server contributes the first GPU's trace of each node. Decode traces go through the TraceLens splitter and the
decode-only steady-state chunk (else the generic one), stream-overlap clamped, is the job's trace; prefill traces are
analysed whole. Decode jobs come first, so the first decode node is the focus job. Every job uses the graph-capture
folder found beside its own trace.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

from TraceLens.Agent.Analysis.utils.config_utils import SCOPE_MULTI_TRACE

from ._analysis_run import AnalysisRun
from ._io_utils import append_log
from ._run_status import note_step
from ._stream_overlap_repair import clamp_stream_overlaps
from ._trace_split import (
    DECODE_ONLY_OR_GENERIC_MODE,
    UNSPLIT_MODE,
    _build_split_cmd,
    _collect_split_chunks,
    _validate_selected_chunk,
    copy_unsplit_trace,
)
from ._tracelens_checkout import run_command
from .tracelens_skill_runner import TraceLensSkillRequest, discover_capture_folder

_SPLIT_NUM_STEPS = 32
_PD_SPLIT_ARGS = argparse.Namespace(split_llm_inference=False)
_PHASE_ORDER = ("decode", "prefill")
_SPLIT_TIMEOUT_S = 1800
_SPLIT_CATEGORY = "split"
_SKIPPED = "skipped"


@dataclasses.dataclass(frozen=True)
class PdJobSettings:
    """Where and how the PD jobs of one analysis are built.

    Attributes:
        output_dir: The TraceLens output directory; decode chunks go under its ``trace_split``.
        platform: The target platform every job names.
        tracelens_root: The TraceLens checkout the splitter runs from.
        log_path: The run log.
        focus_mode: The steady-state mode of the first decode node's chunk.
        health_warnings: Where the focus chunk's quality refusals are recorded.
    """

    output_dir: Path
    platform: str
    tracelens_root: Path
    log_path: Path
    focus_mode: str
    health_warnings: list[dict[str, Any]]


class PdAnalysisPlan(NamedTuple):
    """The jobs of a PD multi-trace analysis, each node's raw trace and the skill request focused on the first job."""

    jobs: list[dict[str, Any]]
    raw_traces: dict[str, Path]
    request: TraceLensSkillRequest


def pd_node_name(phase: str, node: int) -> str:
    """The TraceLens node id of one PD job, e.g. ``decode_0``."""
    return f"{phase}_{node}"


def _decode_chunk(
    trace_path: Path,
    split_dir: Path,
    settings: PdJobSettings,
    mode: str,
    health_warnings: list[dict[str, Any]] | None,
) -> Path:
    """Split one decode node's trace and return its stream-overlap-clamped chunk for ``mode``.

    With ``health_warnings`` the chunk also passes the low-quality gate, and every refusal is recorded there.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_split_cmd(_PD_SPLIT_ARGS, trace_path, split_dir, _SPLIT_NUM_STEPS, settings.log_path)
    rc = run_command(cmd, cwd=settings.tracelens_root, log_path=settings.log_path, timeout_s=_SPLIT_TIMEOUT_S)
    if rc != 0:
        raise RuntimeError(f"trace_split_failed: TraceLens splitter exited with code {rc} for {trace_path}")
    warnings = health_warnings if health_warnings is not None else []
    chunk, _meta, mode_to_chunks = _collect_split_chunks(split_dir, mode, trace_path, warnings)
    if health_warnings is not None:
        _validate_selected_chunk(split_dir, chunk, mode, mode_to_chunks, health_warnings)
    _clamp_and_log(chunk, settings.log_path)
    return chunk


def _clamp_and_log(chunk_path: Path, log_path: Path) -> dict[str, Any]:
    """Clamp ``chunk_path``'s stream overlaps and log the outcome; return the summary, or ``skip_reason`` on failure."""
    try:
        summary = clamp_stream_overlaps(chunk_path)
    except (OSError, ValueError) as exc:
        append_log(log_path, f"stream overlap clamp skipped for {chunk_path.name}: {exc}")
        return {"skip_reason": str(exc)}
    append_log(
        log_path,
        f"stream overlap clamp: {summary['clamped_events']} event(s), {summary['removed_ms']} ms removed "
        f"(worst {summary['worst_event'] or '-'} {summary['worst_removed_ms']} ms) in {chunk_path.name}",
    )
    return summary


def pd_report_jobs(node_traces: dict[str, list[str]], settings: PdJobSettings) -> list[dict[str, Any]]:
    """One job per server node, decode first: decode nodes as split chunks, prefill nodes as captured.

    The first decode node's chunk is selected with ``settings.focus_mode`` and gated for quality, its refusals recorded
    in ``settings.health_warnings``; every other decode node takes the decode-only or generic chunk.
    """
    jobs: list[dict[str, Any]] = []
    for phase in _PHASE_ORDER:
        for node, trace in enumerate(node_traces.get(phase) or []):
            trace_path = Path(trace)
            capture_folder = discover_capture_folder(trace_path, [trace_path])
            if phase == "decode":
                focus = node == 0
                trace_path = _decode_chunk(
                    trace_path,
                    settings.output_dir / "trace_split" / pd_node_name(phase, node),
                    settings,
                    settings.focus_mode if focus else DECODE_ONLY_OR_GENERIC_MODE,
                    settings.health_warnings if focus else None,
                )
            job: dict[str, Any] = {
                "phase": phase,
                "node": node,
                "trace_path": str(trace_path),
                "platform": settings.platform,
            }
            if capture_folder:
                job["capture_folder"] = str(capture_folder)
            jobs.append(job)
    return jobs


def plan_pd_analysis(run: AnalysisRun, pd_input: dict[str, Any], request: TraceLensSkillRequest) -> PdAnalysisPlan:
    """Build one job per PD node of ``pd_input`` and focus ``request`` on the first decode job, under multi-trace scope.

    The first decode job's chunk takes ``--steady-state-mode`` and its graph-capture folder feeds the request; the
    jobs and that chunk are recorded on ``run``. Raises ``RuntimeError`` when there is no decode node.
    """
    node_traces = {role: list(paths or []) for role, paths in (pd_input.get("role_node_traces") or {}).items()}
    if not node_traces.get("decode"):
        raise RuntimeError("pd_multi_trace_without_decode: no decode node trace to focus the analysis on")
    args = run.args
    settings = PdJobSettings(
        output_dir=request.output_dir,
        platform=args.target_platform,
        tracelens_root=request.tracelens_root,
        log_path=run.log_path,
        focus_mode=args.steady_state_mode,
        health_warnings=run.trace_health_warnings,
    )
    jobs = pd_report_jobs(node_traces, settings)
    run.run_meta["selection"].update({"requested_mode": settings.focus_mode, "pd_jobs": jobs})
    run.artifacts["tracelens_steady_state_trace"] = jobs[0]["trace_path"]
    raw_traces = {
        pd_node_name(phase, node): Path(trace)
        for phase, paths in node_traces.items()
        for node, trace in enumerate(paths)
    }
    capture = jobs[0].get("capture_folder")
    focused = dataclasses.replace(request, capture_folder=Path(capture) if capture else None, scope=SCOPE_MULTI_TRACE)
    return PdAnalysisPlan(jobs, raw_traces, focused)


def use_unsplit_trace(run: AnalysisRun, trace_path: Path, split_dir: Path) -> Path:
    """Hand the whole captured trace to the perf report, as a copy under ``split_dir``."""
    copied = copy_unsplit_trace(trace_path, split_dir)
    run.run_meta["selection"].update(
        {
            "requested_mode": UNSPLIT_MODE,
            "chunk_label": UNSPLIT_MODE,
            "selected_chunk": str(copied),
            "selected_chunk_count": 1,
            "fell_back_to_full_trace": False,
        }
    )
    note_step(run.run_meta, "split_trace", category=_SPLIT_CATEGORY, status=_SKIPPED, skip_reason=UNSPLIT_MODE)
    run.artifacts["tracelens_trace_split_dir"] = str(split_dir)
    run.artifacts["tracelens_steady_state_trace"] = str(copied)
    append_log(run.log_path, f"--steady-state-mode={UNSPLIT_MODE} -> using {copied.name} for perf report")
    return copied


def clamp_chunk_stream_overlaps(run: AnalysisRun, chunk_path: Path) -> None:
    """Truncate same-stream overrunning GPU events in the steady-state chunk when ``--clamp-stream-overlaps`` is set."""
    if not run.args.clamp_stream_overlaps:
        return
    summary = _clamp_and_log(chunk_path, run.log_path)
    status = _SKIPPED if "skip_reason" in summary else "ok"
    note_step(run.run_meta, "clamp_stream_overlaps", category=_SPLIT_CATEGORY, status=status, **summary)


def add_pd_args(parser: argparse.ArgumentParser) -> None:
    """Register the PD multi-trace and stream-overlap clamp arguments."""
    parser.add_argument(
        "--pd-node-traces",
        default="",
        help=(
            "JSON object with a PD profile's role_node_traces; analyses every node in one "
            "combine_by_node multi-trace run focused on the first decode node."
        ),
    )
    parser.add_argument(
        "--clamp-stream-overlaps",
        action="store_true",
        help="Truncate same-stream overrunning GPU events in the steady-state chunk (set for PD profiles).",
    )
