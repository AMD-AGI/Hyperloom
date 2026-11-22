# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One TraceLens job per PD server node and the ``combine_by_node`` multi-trace run config they form.

Each node contributes its first GPU's trace: decode traces as their decode-only (else generic) steady-state chunk,
prefill traces whole. Decode jobs come first, so the first decode node is the focus job.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any, NamedTuple

from TraceLens.Agent.Analysis.utils.config_utils import SCOPE_MULTI_TRACE
from TraceLens.Reporting.aggregate_perf_reports import POLICY_BY_NODE

from ._analysis_run import AnalysisRun
from ._trace_split import DECODE_ONLY_OR_GENERIC_MODE, _build_split_cmd, _validate_selected_chunk, collect_split_chunks
from ._tracelens_checkout import run_command
from .tracelens_skill_runner import TraceLensSkillRequest, discover_capture_folder, write_checked_config

_SPLIT_NUM_STEPS = 32
_PD_SPLIT_ARGS = argparse.Namespace(split_llm_inference=False)
_PHASE_ORDER = ("decode", "prefill")
_SPLIT_TIMEOUT_S = 1800
_TRACE_TP_RANK_RE = re.compile(r"-TP-(\d+)(?:\D|$)")
_CAPTURE_RANK_RE = re.compile(r"_rank(\d+)\.json(?:\.gz)?$")


@dataclasses.dataclass(frozen=True)
class PdJobSettings:
    """Where the PD jobs are built (``output_dir``, ``tracelens_root``, ``log_path``) and how: every job's
    ``platform``, the first decode chunk's ``focus_mode``, and ``health_warnings`` for that chunk's refusals."""

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


def _decode_chunk(trace_path: Path, split_dir: Path, settings: PdJobSettings, focus: bool) -> Path:
    """Split one decode node's trace and return its decode-only or generic chunk; the focus node's chunk is selected
    with ``settings.focus_mode`` and also passes the low-quality gate, every refusal recorded in its health warnings."""
    split_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_split_cmd(_PD_SPLIT_ARGS, trace_path, split_dir, _SPLIT_NUM_STEPS, settings.log_path)
    rc = run_command(cmd, cwd=settings.tracelens_root, log_path=settings.log_path, timeout_s=_SPLIT_TIMEOUT_S)
    if rc != 0:
        raise RuntimeError(f"trace_split_failed: TraceLens splitter exited with code {rc} for {trace_path}")
    mode = settings.focus_mode if focus else DECODE_ONLY_OR_GENERIC_MODE
    warnings = settings.health_warnings if focus else []
    chunk, _meta, mode_to_chunks = collect_split_chunks(split_dir, mode, trace_path, warnings)
    if focus:
        _validate_selected_chunk(split_dir, chunk, mode, mode_to_chunks, warnings)
    return chunk


def pd_report_jobs(node_traces: dict[str, list[str]], settings: PdJobSettings) -> list[dict[str, Any]]:
    """One job per server node, decode first, each with the graph-capture folder found beside its trace.

    The first decode node's chunk is selected with ``settings.focus_mode`` and gated for quality; every other decode
    node takes the decode-only or generic chunk.
    """
    jobs: list[dict[str, Any]] = []
    for phase in _PHASE_ORDER:
        for node, trace in enumerate(node_traces.get(phase) or []):
            trace_path = Path(trace)
            capture_folder = discover_capture_folder(trace_path, [trace_path])
            if phase == "decode":
                split_dir = settings.output_dir / "trace_split" / pd_node_name(phase, node)
                trace_path = _decode_chunk(trace_path, split_dir, settings, focus=node == 0)
            job: dict[str, Any] = {"phase": phase, "node": node, "trace_path": str(trace_path)}
            job["platform"] = settings.platform
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
    raw_traces = {pd_node_name(phase, n): Path(t) for phase, paths in node_traces.items() for n, t in enumerate(paths)}
    capture = jobs[0].get("capture_folder")
    focused = dataclasses.replace(request, capture_folder=Path(capture) if capture else None, scope=SCOPE_MULTI_TRACE)
    return PdAnalysisPlan(jobs, raw_traces, focused)


def _rank_scoped_capture_folder(output_dir: Path, trace_path: Path, capture_folder: Path, label: str) -> Path:
    """Narrow ``capture_folder`` to symlinks of the captures of the TP rank an SGLang trace names (``-TP-<n>``), under
    ``output_dir/cache/<label>_rank<n>``.

    TraceLens matches a replay's captures by the ``rank_<n>`` of their file names only. ``capture_folder`` itself is
    returned when the trace names no TP rank, or the folder holds captures of fewer than two ranks or none of its rank.
    """
    match = _TRACE_TP_RANK_RE.search(trace_path.name)
    if match is None or not capture_folder.is_dir():
        return capture_folder
    by_rank: dict[int, list[Path]] = {}
    for entry in capture_folder.iterdir():
        rank_match = _CAPTURE_RANK_RE.search(entry.name)
        if rank_match and entry.is_file():
            by_rank.setdefault(int(rank_match.group(1)), []).append(entry)
    rank = int(match.group(1))
    if len(by_rank) < 2 or rank not in by_rank:
        return capture_folder
    scoped = output_dir / "cache" / f"{label}_rank{rank}"
    if scoped.exists():
        shutil.rmtree(scoped)
    scoped.mkdir(parents=True)
    for entry in by_rank[rank]:
        (scoped / entry.name).symlink_to(entry.resolve())
    return scoped


def write_multi_trace_run_config(*, output_dir: Path, jobs: list[dict[str, Any]], analysis_mode: str) -> Path:
    """Write the ``combine_by_node`` multi-trace run config for ``jobs``, the first job being the focus; job paths are
    made absolute, and a config TraceLens's ``check_config`` rejects raises ``ConfigError``."""
    resolved = []
    for index, job in enumerate(jobs):
        entry = {**job, "trace_path": str(Path(job["trace_path"]).resolve())}
        if job.get("capture_folder"):
            scoped = _rank_scoped_capture_folder(
                output_dir, Path(job["trace_path"]), Path(job["capture_folder"]).resolve(), f"capture_{index}"
            )
            entry["capture_folder"] = str(scoped.resolve())
        resolved.append(entry)
    config = {
        "analysis_mode": analysis_mode,
        "output_dir": str(output_dir.resolve()),
        "aggregation": {"policy": POLICY_BY_NODE},
        "jobs": resolved,
    }
    return write_checked_config(config, SCOPE_MULTI_TRACE, output_dir)


def add_pd_args(parser: argparse.ArgumentParser) -> None:
    """Register the PD multi-trace argument."""
    parser.add_argument(
        "--pd-node-traces",
        default="",
        help="JSON role_node_traces of a PD profile, analysed in one combine_by_node multi-trace run.",
    )
