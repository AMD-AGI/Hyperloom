# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for conc_sweep report collection."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors import collect_conc_sweep_summary


def _write_result(root: Path, task: str, variant: str, tput: float) -> None:
    out = root / "runs" / "conc_sweep" / task / variant / "benchmark" / "inferencex_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"output_throughput": tput, "request_throughput": tput / 1000}), encoding="utf-8")


def test_collect_conc_sweep_summary_recovers_newer_successful_task(tmp_path: Path):
    session = tmp_path / "session"
    (session / "reports").mkdir(parents=True)
    (session / "reports" / "conc_sweep_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "summary": {"successful_pairs": 0, "failed_pairs": 8, "best_conc": None},
                "comparison": [],
                "workspace": str(session / "runs" / "conc_sweep" / "old_failed_task"),
            }
        ),
        encoding="utf-8",
    )

    _write_result(session, "new_success_task", "baseline_conc1", 100.0)
    _write_result(session, "new_success_task", "optimized_conc1", 150.0)
    _write_result(session, "new_success_task", "baseline_conc4", 200.0)
    _write_result(session, "new_success_task", "optimized_conc4", 260.0)

    warnings: list[str] = []
    summary = collect_conc_sweep_summary(session, warnings)

    assert summary["status"] == "succeeded"
    assert summary["summary"]["successful_pairs"] == 2
    assert summary["summary"]["best_conc"] == 1
    assert summary["source"] == "recovered_from_runs"
    assert "original_report_path" in summary
    assert any("recovered" in w for w in warnings)


def test_collect_conc_sweep_summary_recovers_when_report_absent(tmp_path: Path):
    session = tmp_path / "session"
    (session / "reports").mkdir(parents=True)
    _write_result(session, "task_a", "baseline_conc1", 100.0)
    _write_result(session, "task_a", "optimized_conc1", 130.0)
    warnings: list[str] = []
    summary = collect_conc_sweep_summary(session, warnings)
    assert summary["status"] == "succeeded"
    assert summary["summary"]["successful_pairs"] == 1
    assert summary["source"] == "recovered_from_runs"
    assert any("absent" in w for w in warnings)

def test_collect_conc_sweep_summary_absent_and_no_runs_returns_empty(tmp_path: Path):
    session = tmp_path / "session"
    (session / "reports").mkdir(parents=True)
    warnings: list[str] = []
    assert collect_conc_sweep_summary(session, warnings) == {}


def test_collect_conc_sweep_summary_keeps_report_when_not_beaten(tmp_path: Path):
    session = tmp_path / "session"
    (session / "reports").mkdir(parents=True)
    (session / "reports" / "conc_sweep_summary.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "summary": {"successful_pairs": 2, "failed_pairs": 0},
                "comparison": [],
            }
        ),
        encoding="utf-8",
    )
    # Runs hold only one successful pair -> must NOT override the 2-pair report.
    _write_result(session, "task_a", "baseline_conc1", 100.0)
    _write_result(session, "task_a", "optimized_conc1", 130.0)
    warnings: list[str] = []
    summary = collect_conc_sweep_summary(session, warnings)
    assert summary["summary"]["successful_pairs"] == 2
    assert summary.get("source") != "recovered_from_runs"
