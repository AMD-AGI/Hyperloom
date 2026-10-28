###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the perf-report mapper (``_perf_report_analysis``).

The warnings test stubs ``run_config`` to write synthetic report CSVs; the report
contract test runs the real TraceLens perf report on a tiny trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY

import pandas as pd
import pytest
import yaml

from hyperloom.inference_optimizer.roofline_snapshot import extract_workload_summary
from hyperloom.orchestrator.trace_analysis import _perf_report_analysis as pra
from hyperloom.orchestrator.trace_analysis._analysis_json import load_report_tasks
from hyperloom.orchestrator.trace_analysis._trace_gates import WorkloadShares
from TraceLens.Agent.Analysis.post_processing import AnalysisReport, ComputeTask, ReportInfo

_OP_ROW = {
    "name": "op_a",
    "count": 20,
    "Kernel Time (µs)_sum": 20000.0,
    "Kernel Time (µs)_mean": 1000.0,
    "Kernel Time (µs)_std": 100.0,
    "Input Dims": "[[2, 3]]",
    "Input type": "['c10::BFloat16']",
    "call_stack_full": "['mod.py', 'kernel_a']",
}
_GPU_TIMELINE = [
    ("total_time", 100.0, 100.0),
    ("computation_time", 80.0, 80.0),
    ("exposed_comm_time", 5.0, 5.0),
    ("idle_time", 10.0, 10.0),
]
_TRACE_EVENTS = (
    [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": name,
            "ts": ts,
            "dur": 100,
            "pid": 1,
            "tid": 1,
            "args": {"External id": ext},
        }
        for name, ts, ext in [("op_a", 1000, 1), ("op_b", 1100, 2)]
    ]
    + [
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "hipLaunchKernel",
            "ts": ts,
            "dur": 20,
            "pid": 1,
            "tid": 1,
            "args": {"correlation": corr, "External id": ext},
        }
        for ts, corr, ext in [(1010, 5, 1), (1110, 6, 2)]
    ]
    + [
        {
            "ph": "X",
            "cat": "kernel",
            "name": name,
            "ts": ts,
            "dur": 300,
            "pid": 0,
            "tid": 7,
            "args": {"correlation": corr},
        }
        for name, ts, corr in [("kernel_a", 1200, 5), ("kernel_b", 1700, 6)]
    ]
)


def _write_run_yaml(tmp_path: Path, trace: dict) -> Path:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    config = {
        "analysis_mode": "default",
        "output_dir": str(tmp_path),
        "jobs": [{"trace_path": str(trace_path), "platform": "MI300X"}],
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def make_run_yaml(tmp_path, monkeypatch):
    """Return a factory for a ``run.yaml`` whose perf report is a stub writing the given rows."""

    def make(rows: list[dict]) -> Path:
        def stub_report(config):
            csv_dir = Path(config["output_dir"]) / "perf_report_csvs"
            csv_dir.mkdir()
            pd.DataFrame(rows).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
            pd.DataFrame(_GPU_TIMELINE, columns=["type", "time ms", "percent"]).to_csv(
                csv_dir / "gpu_timeline.csv", index=False
            )

        monkeypatch.setattr(pra, "run_config", stub_report)
        return _write_run_yaml(tmp_path, {})

    return make


@pytest.mark.parametrize(
    ("overrides", "warnings"),
    [
        ({"name": "hipGraphLaunch->kernel_a"}, [ANY]),
        ({"Kernel Time (µs)_std": 1500.0}, ["High kernel-time variance in op_a (cov 1.5)"]),
        ({}, []),
    ],
    ids=["graph_replay", "high_variance", "clean"],
)
def test_warnings_set_trace_quality(make_run_yaml, overrides, warnings):
    analysis = pra.analyze_perf_report(make_run_yaml([{**_OP_ROW, **overrides}]))

    report_info = json.loads(analysis.report_path.read_text(encoding="utf-8"))["report_info"]
    assert analysis.warnings == warnings
    assert report_info.get("warnings", []) == warnings
    assert report_info["trace_quality"] == ("poor" if warnings else "ok")


def test_top_operations_follow_category_rollup(make_run_yaml):
    rows = [
        {**_OP_ROW, "op category": "gemm", "operation_count": 20},
        {**_OP_ROW, "name": "op_b", "op category": "norm", "operation_count": 4, "Kernel Time (µs)_sum": 5000.0},
    ]

    report = json.loads(pra.analyze_perf_report(make_run_yaml(rows)).report_path.read_text(encoding="utf-8"))

    assert report["top_operations"] == [
        {
            "rank": 1,
            "category": "gemm",
            "time_ms": 20.0,
            "pct_of_compute": 80.0,
            "ops": 20,
            "potential_improvement_md": None,
        },
        {
            "rank": 2,
            "category": "norm",
            "time_ms": 5.0,
            "pct_of_compute": 20.0,
            "ops": 4,
            "potential_improvement_md": None,
        },
    ]


def test_written_report_matches_schema_and_reader_inputs(tmp_path):
    analysis = pra.analyze_perf_report(_write_run_yaml(tmp_path, {"traceEvents": _TRACE_EVENTS}))

    report = json.loads(analysis.report_path.read_text(encoding="utf-8"))
    assert set(report) == AnalysisReport.__required_keys__ | {"executive_summary", "top_operations"}
    assert set(report["report_info"]) == ReportInfo.__required_keys__
    assert set(report["compute_optimizations"][0]) == ComputeTask.__required_keys__
    assert analysis.shares == WorkloadShares(idle_pct=25.0, compute_pct=75.0, exposed_communication_pct=0.0)
    summary = extract_workload_summary(analysis.report_path)
    assert (summary["compute_pct"], summary["idle_pct"], summary["comm_pct"]) == (75.0, 25.0, 0.0)
    candidates = {c["name"]: c for c in load_report_tasks(analysis.report_path)}
    assert candidates["op_a"]["device_kernel_names"] == ["kernel_a"]
