# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One TraceLens perf report over every PD server node, aggregated with ``combine_by_node``.

Each server contributes the first GPU's trace of each node. Decode traces go through the TraceLens splitter and the
decode-only steady-state chunk (else the generic one) is reported; prefill traces are reported whole. Every job uses
its own server's graph-capture folder when that folder holds files. The config is written to ``pd-run.yaml`` and run
once with ``TraceLens.Reporting.generate_perf_report_from_config``, which writes one report per job and
``perf_report_aggregated_csvs/``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hyperloom.orchestrator.trace_analysis._trace_split import DECODE_ONLY_OR_GENERIC_MODE, _collect_split_chunks
from hyperloom.orchestrator.trace_analysis._tracelens_checkout import run_command

log = logging.getLogger(__name__)

PD_REPORT_CONFIG_NAME = "pd-run.yaml"
AGGREGATED_CSVS_DIR = "perf_report_aggregated_csvs"
AGGREGATION_POLICY = "combine_by_node"
SPLIT_NUM_STEPS = 32
_CAPTURE_DIR_NAMES = ("capture_traces", "graph_capture_profile", "graph_capture")
_PHASE_ORDER = ("decode", "prefill")
_SPLIT_TIMEOUT_S = 1800
_REPORT_TIMEOUT_S = 3600


def role_capture_folder(trace_root: Path, role: str) -> str:
    """The graph-capture folder ``role``'s server wrote under ``trace_root``; '' when none holds files."""
    for name in _CAPTURE_DIR_NAMES:
        folder = trace_root / role / name
        if folder.is_dir() and any(path.is_file() for path in folder.iterdir()):
            return str(folder)
    return ""


def _decode_chunk(trace_path: Path, split_dir: Path, *, tracelens_root: Path, log_path: Path) -> Path:
    """Split one decode node's trace and return its decode-only or generic steady-state chunk."""
    split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "TraceLens.TraceUtils.split_trace.main",
        str(trace_path),
        "-o",
        str(split_dir),
        "--find-steady-state",
        "--num-steps",
        str(SPLIT_NUM_STEPS),
    ]
    rc = run_command(cmd, cwd=tracelens_root, log_path=log_path, timeout_s=_SPLIT_TIMEOUT_S)
    if rc != 0:
        raise RuntimeError(f"trace_split_failed: TraceLens splitter exited with code {rc} for {trace_path}")
    chunk, _meta, _modes = _collect_split_chunks(split_dir, DECODE_ONLY_OR_GENERIC_MODE, trace_path, [])
    return chunk


def pd_report_config(jobs: list[dict[str, Any]], *, output_dir: Path, analysis_mode: str) -> dict[str, Any]:
    """The ``combine_by_node`` run config for ``jobs``."""
    return {
        "analysis_mode": analysis_mode,
        "output_dir": str(output_dir),
        "aggregation": {"policy": AGGREGATION_POLICY},
        "jobs": jobs,
    }


def _report_jobs(
    node_traces: dict[str, list[str]],
    *,
    trace_root: Path,
    output_dir: Path,
    platform: str,
    tracelens_root: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    """One job per server node: decode nodes as split chunks, prefill nodes as captured."""
    jobs: list[dict[str, Any]] = []
    for phase in _PHASE_ORDER:
        capture_folder = role_capture_folder(trace_root, phase)
        for node, trace in enumerate(node_traces.get(phase) or []):
            trace_path = Path(trace)
            if phase == "decode":
                split_dir = output_dir / "trace_split" / f"{phase}_{node}"
                trace_path = _decode_chunk(trace_path, split_dir, tracelens_root=tracelens_root, log_path=log_path)
            job: dict[str, Any] = {"phase": phase, "node": node, "trace_path": str(trace_path), "platform": platform}
            if capture_folder:
                job["capture_folder"] = capture_folder
            jobs.append(job)
    return jobs


def run_pd_combined_report(
    node_traces: dict[str, list[str]],
    *,
    trace_root: Path,
    output_dir: Path,
    platform: str,
    analysis_mode: str,
    tracelens_root: Path,
) -> dict[str, Any]:
    """Write ``pd-run.yaml`` for ``node_traces`` under ``output_dir``, run it once, and summarise the outcome.

    Never raises: a failed split or report comes back as ``status="failed"`` with the error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pd_report.log"
    config_path = output_dir / PD_REPORT_CONFIG_NAME
    result: dict[str, Any] = {"status": "failed", "output_dir": str(output_dir), "config": str(config_path)}
    try:
        jobs = _report_jobs(
            node_traces,
            trace_root=trace_root,
            output_dir=output_dir,
            platform=platform,
            tracelens_root=tracelens_root,
            log_path=log_path,
        )
        if not jobs:
            raise RuntimeError("no PD node traces to report")
        config_path.write_text(
            yaml.safe_dump(pd_report_config(jobs, output_dir=output_dir, analysis_mode=analysis_mode), sort_keys=False),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            "-m",
            "TraceLens.Reporting.generate_perf_report_from_config",
            "--config",
            str(config_path),
        ]
        rc = run_command(cmd, cwd=tracelens_root, log_path=log_path, timeout_s=_REPORT_TIMEOUT_S)
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
    """Run the combined report for a PD profile's ``role_node_traces`` and record it as ``pd_combined_report``.

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
        trace_root=Path(trace_dir),
        output_dir=Path(trace_dir).parent / "tracelens" / stamp,
        platform=(platform or os.environ.get("GPU_TYPE", "")).strip().upper(),
        analysis_mode=tracelens_analysis_mode(framework),
        tracelens_root=paths.tracelens_root(),
    )
    profile_result["pd_combined_report"] = summary
    if summary.get("status") == "ok":
        log.info("pd combined report: %d job(s) -> %s", len(summary["jobs"]), summary["aggregated_csvs"])
    else:
        log.warning("pd combined report failed: %s (config %s)", summary.get("error"), summary.get("config"))
    return summary
