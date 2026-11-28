# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A TraceLens report marked degraded keeps the report's kernel shares, and its tool summary says they are kernel
shares; a kernel table the report found hollow gives none."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from hyperloom_multibranch.toolbox.tools import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import (
    ANALYSIS_JSON,
    DEGRADED_NOTICE,
    KERNEL_SHARES_NOTICE,
    with_op_shares,
)

FALLBACK_WARNING = "graph-under-recorded (graph-replay fraction 100%)"
REQUEST = ToolRequest(tool="tracelens", branch="b1", params={"commit_id": "c1", "profile_request_id": "p1"})


def test_a_degraded_report_keeps_the_report_routes_kernel_shares_and_says_they_are_kernel_shares(
    tmp_path: Path,
) -> None:
    info = {"mode": "deterministic-fallback", "trace_quality": "poor", "warnings": [FALLBACK_WARNING]}
    (tmp_path / ANALYSIS_JSON).write_text(json.dumps({"report_info": info}))
    (tmp_path / "analysis.md").write_text("report")
    table = tmp_path / "kernel_shares.csv"
    table.write_text("name,Percentage (%)\nCijk_gemm,61.5\npaged_attention,20.0\n", encoding="utf-8")
    result = {"analysis_md_path": str(tmp_path / "analysis.md"), "kernel_shares_path": str(table)}
    routed = ToolOutput(result=result, summary="analysed")

    output = with_op_shares(routed, REQUEST)

    assert output.op_shares == {"Cijk_gemm": 61.5, "paged_attention": 20.0} and output.commit_id == "c1"
    assert KERNEL_SHARES_NOTICE in output.summary and FALLBACK_WARNING in output.summary
    assert DEGRADED_NOTICE not in output.summary
    hollow = replace(routed, result={**routed.result, "window_degraded_reason": "kernels fill 0.6% of the span"})
    withheld = with_op_shares(hollow, REQUEST)
    assert not withheld.op_shares and "kernels fill 0.6% of the span" in withheld.summary
