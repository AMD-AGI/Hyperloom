# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The op shares are a complete per-op (or per-kernel) GPU time table, not the few optimisation tasks of TraceLens' report.

``data/tracelens_ops_summary.csv`` is a real ``ops_summary.csv`` of a TraceLens perf report (Qwen3-8B on SGLang, TP2,
one steady-state window). The report route names a kernel table of its own, which is read instead of it.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.contracts.notebook.records import MAX_OP_SHARES
from hyperloom_multibranch.toolbox.tools import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import _copy_back
from hyperloom_multibranch.toolbox.tools.op_shares import (
    ANALYSIS_JSON,
    KERNEL_SHARES_KEY,
    OPS_SUMMARY,
    read_op_shares,
    with_op_shares,
)

FIXTURE = Path(__file__).parent / "data" / "tracelens_ops_summary.csv"


def _report(directory: Path, table: Path = FIXTURE, info: dict[str, Any] | None = None) -> dict[str, str]:
    (directory / OPS_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(table, directory / OPS_SUMMARY)
    (directory / ANALYSIS_JSON).write_text(json.dumps({"report_info": info or {"mode": "agentic"}}))
    (directory / "analysis.md").write_text("report")
    return {"analysis_md_path": str(directory / "analysis.md")}


def test_the_full_table_sums_to_the_whole_gpu_time_with_gemm_and_attention_present(tmp_path: Path) -> None:
    shares = read_op_shares(_report(tmp_path))
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.01)
    assert shares["aten::mm"] == pytest.approx(25.32, abs=0.01)
    assert shares["aiter::mha_batch_prefill"] == pytest.approx(2.40, abs=0.01)
    assert max(shares, key=shares.__getitem__) == "aten::mm"
    assert shares["aten::copy_"] == pytest.approx(19.82, abs=0.01)


def test_only_the_largest_ops_are_kept(tmp_path: Path) -> None:
    result = _report(tmp_path)
    top = read_op_shares(result, limit=3)
    assert list(top) == ["aten::mm", "aten::copy_", "aten::mul"]


def test_a_table_longer_than_the_cap_is_trimmed_to_the_cap(tmp_path: Path) -> None:
    table = tmp_path / "long.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "Percentage (%)"])
        writer.writerows((f"op{i:03d}", 1.0 + i / 1000) for i in range(MAX_OP_SHARES + 10))
    shares = read_op_shares(_report(tmp_path, table))
    assert len(shares) == MAX_OP_SHARES
    assert "op000" not in shares and f"op{MAX_OP_SHARES + 9:03d}" in shares


def test_an_op_listed_twice_has_its_rows_summed_and_unreadable_rows_are_skipped(tmp_path: Path) -> None:
    table = tmp_path / "dup.csv"
    table.write_text("name,Percentage (%)\naten::mm,10\naten::mm,5.5\naten::bad,n/a\n,3\n")
    assert read_op_shares(_report(tmp_path, table)) == {"aten::mm": 15.5}


def test_a_report_without_the_table_gives_no_shares(tmp_path: Path) -> None:
    result = _report(tmp_path)
    (tmp_path / OPS_SUMMARY).unlink()
    assert read_op_shares(result) == {}


def test_the_table_is_copied_back_from_the_scratch_directory_with_the_report(tmp_path: Path) -> None:
    scratch, workdir = tmp_path / "scratch", tmp_path / "work"
    scratch.mkdir()
    result = _report(scratch)
    out = _copy_back(result, scratch, workdir)
    landed = Path(out["analysis_md_path"])
    assert landed.is_relative_to(workdir)
    assert (landed.parent / OPS_SUMMARY).is_file() and (landed.parent / ANALYSIS_JSON).is_file()
    assert sum(read_op_shares(out).values()) == pytest.approx(100.0, abs=0.01)


def _kernel_table(directory: Path) -> Path:
    table = directory / "kernel_shares.csv"
    table.write_text(
        "name,Percentage (%),total_time_ms,count\nkernel_gemm_0,60.5,3.0,160\nreshape_and_cache_flash,39.5,2.0,40\n"
    )
    return table


def test_a_result_naming_a_kernel_table_has_its_shares_read_from_it_not_from_the_ops_summary_beside_the_report(
    tmp_path: Path,
) -> None:
    result = {**_report(tmp_path), KERNEL_SHARES_KEY: str(_kernel_table(tmp_path))}
    assert read_op_shares(result) == {"kernel_gemm_0": 60.5, "reshape_and_cache_flash": 39.5}


@pytest.mark.parametrize("named", ["", None, "nowhere/kernel_shares.csv"])
def test_a_result_whose_kernel_table_is_empty_or_missing_gives_no_shares_even_beside_an_ops_summary(
    tmp_path: Path, named: str | None
) -> None:
    result = {**_report(tmp_path), KERNEL_SHARES_KEY: named}
    assert (tmp_path / OPS_SUMMARY).is_file() and read_op_shares(result) == {}


def test_the_kernel_table_is_copied_back_from_the_scratch_directory_and_read_there(tmp_path: Path) -> None:
    scratch, workdir = tmp_path / "scratch", tmp_path / "work"
    scratch.mkdir()
    out = _copy_back({**_report(scratch), KERNEL_SHARES_KEY: str(_kernel_table(scratch))}, scratch, workdir)
    assert Path(out[KERNEL_SHARES_KEY]).is_relative_to(workdir)
    assert read_op_shares(out) == {"kernel_gemm_0": 60.5, "reshape_and_cache_flash": 39.5}


def test_the_summary_says_how_many_misreported_events_the_shares_leave_out(tmp_path: Path) -> None:
    result = {KERNEL_SHARES_KEY: str(_kernel_table(tmp_path)), "excluded_events": 6, "excluded_claimed_ms": 54.84}
    request = ToolRequest("tracelens", "b1", {"commit_id": "c1"})
    noted = with_op_shares(ToolOutput(result, "trace analysed"), request)
    assert (
        noted.summary
        == "trace analysed | 6 GPU events the profiler misreported (claiming 54.84 ms) are left out of the shares"
    )
    quiet = with_op_shares(ToolOutput({**result, "excluded_events": 0}, "trace analysed"), request)
    assert quiet.summary == "trace analysed" and quiet.op_shares
