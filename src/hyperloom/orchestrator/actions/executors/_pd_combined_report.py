# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One TraceLens perf report over every PD server node, aggregated with ``combine_by_node``.

Each server contributes the first GPU's trace of each node. Each node's trace is prepared as the per-role analysis
prepares it: decode traces go through the TraceLens splitter and the decode-only steady-state chunk (else the generic
one) is reported; prefill traces are reported whole, as a copy. Every node's chunk is then clamped of same-stream
overlapping GPU events, one node per thread. Every job uses its own server's graph-capture folder when that folder holds
files. The config is written to ``pd-run.yaml`` and run once with ``TraceLens.Reporting.generate_perf_report_from_config``,
which writes one report per job and ``perf_report_aggregated_csvs/``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hyperloom.orchestrator.trace_analysis import (
    DECODE_ONLY_OR_GENERIC_MODE,
    TraceSplitJob,
    clamp_stream_overlaps,
    copy_unsplit_trace,
    run_command,
    run_trace_split,
)

log = logging.getLogger(__name__)

PD_REPORT_CONFIG_NAME = "pd-run.yaml"
AGGREGATED_CSVS_DIR = "perf_report_aggregated_csvs"
AGGREGATION_POLICY = "combine_by_node"
SPLIT_NUM_STEPS = 32
_CAPTURE_DIR_NAMES = ("capture_traces", "graph_capture_profile", "graph_capture")
_PHASE_ORDER = ("decode", "prefill")
_SPLIT_BUDGET_MINUTES = 30
_REPORT_TIMEOUT_S = 3600


@dataclass(frozen=True)
class PdReportRun:
    """Where a combined PD report reads its traces, writes its output and finds TraceLens."""

    trace_root: Path
    output_dir: Path
    platform: str
    analysis_mode: str
    tracelens_root: Path


def role_capture_folder(trace_root: Path, role: str) -> str:
    """The graph-capture folder ``role``'s server wrote under ``trace_root``; '' when none holds files."""
    for name in _CAPTURE_DIR_NAMES:
        folder = trace_root / role / name
        if folder.is_dir() and any(path.is_file() for path in folder.iterdir()):
            return str(folder)
    return ""


def _split_options() -> SimpleNamespace:
    """The splitter options of a decode node: the decode-only or generic steady state, without LLM phase hints."""
    return SimpleNamespace(
        split_num_steps=SPLIT_NUM_STEPS,
        steady_state_mode=DECODE_ONLY_OR_GENERIC_MODE,
        budget_minutes=_SPLIT_BUDGET_MINUTES,
        split_llm_inference=False,
        split_conc="",
        split_osl="",
        split_r="",
    )


def _clamp_chunk(chunk: Path) -> None:
    """Truncate the same-stream overrunning GPU events of ``chunk``; a chunk the clamp cannot read is kept as it is."""
    try:
        summary = clamp_stream_overlaps(chunk)
    except (OSError, ValueError) as exc:
        log.warning("pd combined report: stream overlap clamp skipped for %s: %s", chunk, exc)
        return
    log.info("pd combined report: clamped %d event(s) in %s", summary["clamped_events"], chunk)


def _node_chunk(trace_path: Path, split_dir: Path, phase: str, *, tracelens_root: Path, log_path: Path) -> Path:
    """The steady-state chunk one node's trace contributes: a decode trace's split chunk, a prefill trace's copy,
    clamped of same-stream overlaps."""
    split_dir.mkdir(parents=True, exist_ok=True)
    if phase == "prefill":
        chunk = copy_unsplit_trace(trace_path, split_dir)
    else:
        job = TraceSplitJob(analysis_trace_path=trace_path, split_dir=split_dir, tl_root=tracelens_root)
        chunk, _meta = run_trace_split(_split_options(), job, log_path, [])
    _clamp_chunk(chunk)
    return chunk


def pd_report_config(jobs: list[dict[str, Any]], *, output_dir: Path, analysis_mode: str) -> dict[str, Any]:
    """The ``combine_by_node`` run config for ``jobs``."""
    return {
        "analysis_mode": analysis_mode,
        "output_dir": str(output_dir),
        "aggregation": {"policy": AGGREGATION_POLICY},
        "jobs": jobs,
    }


def _report_jobs(node_traces: dict[str, list[str]], run: PdReportRun, log_path: Path) -> list[dict[str, Any]]:
    """One job per server node, its trace the node's steady-state chunk; the nodes are prepared concurrently."""
    nodes = [
        (phase, node, Path(trace)) for phase in _PHASE_ORDER for node, trace in enumerate(node_traces.get(phase) or [])
    ]
    prepare = functools.partial(_prepare_node, run=run, log_path=log_path)
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        chunks = list(pool.map(prepare, nodes))
    captures = {phase: role_capture_folder(run.trace_root, phase) for phase in _PHASE_ORDER}
    jobs: list[dict[str, Any]] = []
    for (phase, node, _trace), chunk in zip(nodes, chunks):
        job: dict[str, Any] = {"phase": phase, "node": node, "trace_path": str(chunk), "platform": run.platform}
        if captures[phase]:
            job["capture_folder"] = captures[phase]
        jobs.append(job)
    return jobs


def _prepare_node(item: tuple[str, int, Path], *, run: PdReportRun, log_path: Path) -> Path:
    """The chunk of one ``(phase, node, trace)`` item, under the run's ``trace_split/<phase>_<node>`` directory."""
    phase, node, trace_path = item
    split_dir = run.output_dir / "trace_split" / f"{phase}_{node}"
    return _node_chunk(trace_path, split_dir, phase, tracelens_root=run.tracelens_root, log_path=log_path)


def run_pd_combined_report(node_traces: dict[str, list[str]], run: PdReportRun) -> dict[str, Any]:
    """Write ``pd-run.yaml`` for ``node_traces`` under ``run.output_dir``, run it once, and summarise the outcome.

    Never raises: a failed split or report comes back as ``status="failed"`` with the error.
    """
    output_dir = run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pd_report.log"
    config_path = output_dir / PD_REPORT_CONFIG_NAME
    result: dict[str, Any] = {"status": "failed", "output_dir": str(output_dir), "config": str(config_path)}
    try:
        jobs = _report_jobs(node_traces, run, log_path)
        if not jobs:
            raise RuntimeError("no PD node traces to report")
        config_path.write_text(
            yaml.safe_dump(
                pd_report_config(jobs, output_dir=output_dir, analysis_mode=run.analysis_mode), sort_keys=False
            ),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            "-m",
            "TraceLens.Reporting.generate_perf_report_from_config",
            "--config",
            str(config_path),
        ]
        rc = run_command(cmd, cwd=run.tracelens_root, log_path=log_path, timeout_s=_REPORT_TIMEOUT_S)
        if rc != 0:
            raise RuntimeError(f"perf_report_failed: TraceLens report exited with code {rc}; see {log_path}")
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    result.update(
        {
            "status": "ok",
            "jobs": jobs,
            "aggregated_csvs": str(output_dir / AGGREGATED_CSVS_DIR),
        }
    )
    return result


async def report_pd_profile(profile_result: dict[str, Any], *, framework: str, platform: str) -> dict[str, Any]:
    """Run the combined report for a PD profile's ``role_node_traces`` and return its summary.

    The report lands in ``<trace_dir parent>/tracelens/<UTC timestamp>/``; a profile without node traces is skipped.
    """
    from hyperloom.inference_optimizer.framework_registry import tracelens_analysis_mode
    from hyperloom.inference_optimizer.session import paths

    node_traces = profile_result.get("role_node_traces") or {}
    trace_dir = str(profile_result.get("trace_dir") or "")
    if not node_traces or not trace_dir:
        return {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = await asyncio.to_thread(
        run_pd_combined_report,
        node_traces,
        PdReportRun(
            trace_root=Path(trace_dir),
            output_dir=Path(trace_dir).parent / "tracelens" / stamp,
            platform=(platform or os.environ.get("GPU_TYPE", "")).strip().upper(),
            analysis_mode=tracelens_analysis_mode(framework),
            tracelens_root=paths.tracelens_root(),
        ),
    )
    if summary.get("status") == "ok":
        log.info("pd combined report: %d job(s) -> %s", len(summary["jobs"]), summary["aggregated_csvs"])
    else:
        log.warning("pd combined report failed: %s (config %s)", summary.get("error"), summary.get("config"))
    return summary
