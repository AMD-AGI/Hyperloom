# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a run's ``run.yaml`` into ``analysis.json`` through TraceLens, with no LLM.

This module only sequences TraceLens calls and writes the envelope. The task
grouping, per-row metrics and platform peaks are TraceLens code; the output is
read by the same readers as the agent route's ``analysis.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.io import atomic_write_json
from TraceLens.Agent.Analysis.category_analyses.analysis_utils import build_operation_metrics, perf_report_csv_dir
from TraceLens.Agent.Analysis.post_processing import TASK_SOURCE_COLUMNS, compute_tasks
from TraceLens.Agent.Analysis.utils.config_utils import SCOPE_STANDALONE, check_config
from TraceLens.Agent.Analysis.utils.deterministic_fallback import check_graph_replay_coverage
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    ops_summary_by_enhanced_category,
    perf_report_metadata,
)
from TraceLens.Reporting.generate_perf_report_from_config import load_run_config, run_config

from ._trace_gates import WorkloadShares
from .analysis_route import AnalysisRoute

_TITLE = "Perf report analysis"
_ANALYSIS_JSON = "analysis.json"
_UNIFIED_CSV = "unified_perf_summary.csv"


@dataclass(frozen=True)
class PerfReportAnalysis:
    """The written ``analysis.json``, the shares the gates read, and the quality warnings."""

    report_path: Path
    shares: WorkloadShares
    warnings: list[str]


def _top_operations(csv_dir: Path) -> list[dict]:
    """The ``top_operations`` rows: TraceLens's category rollup, in its order."""
    return [
        {
            "rank": rank,
            "category": row["op category"],
            "time_ms": row["total_direct_kernel_time_ms"],
            "pct_of_compute": row["Percentage (%)"],
            "ops": row["Count"],
            "potential_improvement_md": None,
        }
        for rank, row in enumerate(ops_summary_by_enhanced_category(str(csv_dir)), start=1)
    ]


def analyze_perf_report(run_config_path: Path) -> PerfReportAnalysis:
    """Run the TraceLens perf report for ``run_config_path`` and write ``analysis.json`` beside it.

    ``check_config`` adds the agent report flags (such as ``include_call_stack``) that
    ``write_run_config`` leaves out of ``run.yaml``, so both routes get the same report columns.
    """
    config = check_config(load_run_config(str(run_config_path)), SCOPE_STANDALONE).run_config
    run_config(config)

    output_dir = Path(config["output_dir"])
    csv_dir = Path(perf_report_csv_dir(str(output_dir)))
    unified_csv = csv_dir / _UNIFIED_CSV
    metadata = perf_report_metadata(str(csv_dir), config["jobs"][0]["platform"])
    ops = build_operation_metrics(
        pd.read_csv(unified_csv), metadata, {"extra_fields": list(TASK_SOURCE_COLUMNS)}, SCOPE_STANDALONE
    )

    coverage = check_graph_replay_coverage(unified_csv)
    warnings = [coverage.reason] if coverage.graph_under_recorded else []
    warnings += [
        f"High kernel-time variance in {op['name']} (cov {op['cov']})" for op in ops if op.get("high_variance")
    ]

    utilization = metadata["gpu_utilization"]
    shares = WorkloadShares(
        idle_pct=utilization["idle_time_percent"],
        compute_pct=utilization["computation_time_percent"],
        exposed_communication_pct=utilization["exposed_comm_time_percent"],
    )
    report_info = {
        "mode": AnalysisRoute.NON_LLM.value,
        "comparison_scope": SCOPE_STANDALONE,
        "trace_quality": "poor" if warnings else "ok",
    }
    if warnings:
        report_info["warnings"] = warnings
    report = {
        "report_info": report_info,
        "title": _TITLE,
        "executive_summary": {
            "narrative": "",
            "metrics": {
                "total_time_ms": utilization["total_time_ms"],
                "compute_pct": shares.compute_pct,
                "idle_pct": shares.idle_pct,
                "exposed_communication_pct": shares.exposed_communication_pct,
            },
        },
        "top_operations": _top_operations(csv_dir),
        "compute_optimizations": compute_tasks(ops),
        "fusion_optimizations": [],
    }
    report_path = output_dir / _ANALYSIS_JSON
    atomic_write_json(report_path, report)
    return PerfReportAnalysis(report_path=report_path, shares=shares, warnings=warnings)
