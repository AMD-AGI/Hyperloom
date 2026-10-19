# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A TraceLens report marked degraded gives no op shares, and its tool summary says so."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.toolbox.tools import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import ANALYSIS_JSON, DEGRADED_NOTICE, read_op_shares, with_op_shares

FALLBACK_WARNING = "graph-under-recorded (graph-replay fraction 100%)"
REQUEST = ToolRequest(tool="tracelens", branch="b1", params={"commit_id": "c1", "profile_request_id": "p1"})


@pytest.fixture(autouse=True)
def _table_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single flow's reader needs the TraceLens package (GPU image only); this one reads the same table shape."""

    def load_report_tasks(path: Path) -> list[dict[str, Any]]:
        tasks = json.loads(Path(path).read_text())["compute_optimizations"]
        return [{"name": t["operation"], "gpu_pct": m["pct_e2e"]} for t in tasks for m in t["members"]]

    module = types.ModuleType("hyperloom.orchestrator.trace_analysis.analysis_json")
    module.load_report_tasks = load_report_tasks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _member(kernel: str, pct: float) -> dict[str, Any]:
    return {"kernel_name": [kernel], "pct_e2e": pct, "time_ms": 1.0, "count": 1}


def _write(directory: Path, info: dict[str, Any]) -> ToolOutput:
    task = {"operation": "gemm", "members": [_member("k_gemm", 61.5)], "priority": 1}
    (directory / ANALYSIS_JSON).write_text(json.dumps({"report_info": info, "compute_optimizations": [task]}))
    (directory / "analysis.md").write_text("report")
    return ToolOutput(result={"analysis_md_path": str(directory / "analysis.md")}, summary="analysed")


def test_a_normal_report_keeps_its_op_shares(tmp_path: Path) -> None:
    output = with_op_shares(_write(tmp_path, {"mode": "agentic", "trace_quality": "good"}), REQUEST)
    assert dict(output.op_shares) == {"gemm": 61.5}
    assert output.commit_id == "c1" and output.summary == "analysed"


@pytest.mark.parametrize(
    "info",
    [
        {"mode": "deterministic-fallback", "trace_quality": "poor", "warnings": [FALLBACK_WARNING]},
        {"mode": "deterministic-fallback", "warnings": [FALLBACK_WARNING]},
        {"mode": "agentic", "trace_quality": "poor", "warnings": [FALLBACK_WARNING]},
    ],
)
def test_a_degraded_report_gives_no_shares_and_a_warning_in_the_summary(tmp_path: Path, info: dict[str, Any]) -> None:
    written = _write(tmp_path, info)
    assert read_op_shares(written.result) == {}
    output = with_op_shares(written, REQUEST)
    assert not output.op_shares and not output.commit_id
    assert DEGRADED_NOTICE in output.summary and FALLBACK_WARNING in output.summary


def test_a_report_without_report_info_is_not_degraded(tmp_path: Path) -> None:
    written = _write(tmp_path, {})
    assert dict(with_op_shares(written, REQUEST).op_shares) == {"gemm": 61.5}
