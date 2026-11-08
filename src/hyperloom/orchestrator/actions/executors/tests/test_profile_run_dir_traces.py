# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The profile executor and the capture classifier read a run_dir's three trace directories apart."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import profile as pf
from hyperloom.orchestrator.trace_analysis import is_capture_fragment
from inference_suite.contracts import RunLayout
from inference_suite.contracts.layout import CAPTURE_TRACE_DIRS


def _trace(path: Path, events: list[dict], pad: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": events, "pad": "x" * pad}, fh)
    return path


def _run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout(tmp_path / "round-000")
    layout.root.mkdir()
    layout.report.write_text("{}", encoding="utf-8")
    return layout


@pytest.mark.parametrize("capture_dir", CAPTURE_TRACE_DIRS)
def test_a_run_dir_capture_sidecar_is_never_a_serving_step(tmp_path: Path, capture_dir: str) -> None:
    layout = _run_dir(tmp_path)
    step = layout.profile_torch_trace / "rank0.pt.trace.json.gz"
    sidecar = layout.profile_dir / capture_dir / "rank0.pt.trace.json.gz"

    for root in (layout.root, layout.profile_dir, None):
        assert not is_capture_fragment(step, root)
        assert is_capture_fragment(sidecar, root)


def test_the_executor_takes_the_step_trace_and_finds_the_capture_beside_it(tmp_path: Path) -> None:
    layout = _run_dir(tmp_path)
    step = _trace(layout.profile_torch_trace / "rank0.pt.trace.json.gz", [{"cat": "cpu_op", "name": "aten::mm"}])
    capture = layout.profile_dir / CAPTURE_TRACE_DIRS[1]
    _trace(capture / "rank0.pt.trace.json.gz", [{"cat": "cpu_op", "name": "aten::mm"}], pad=100_000)
    result: dict = {}

    pf._record_workspace_traces(result, str(layout.root), "sglang", None)

    assert result["trace_dir"] == str(layout.profile_torch_trace)
    assert result["trace_files"] == [str(step)]
    assert result["profile_trace_selection_reason"] == "trace_dir_preferred"
    present = next(
        row for row in result["trace_health"]["checks"] if row["check_id"] == pf.CHECK_CAPTURE_TRACES_PRESENT
    )
    assert present["status"] == "passed" and present["detail"]["capture_dir"] == str(capture)


def test_a_run_dir_with_only_capture_sidecars_falls_back_to_them(tmp_path: Path) -> None:
    layout = _run_dir(tmp_path)
    sidecar = _trace(layout.profile_dir / CAPTURE_TRACE_DIRS[0] / "bs_1_rank0.json.gz", [])
    layout.profile_torch_trace.mkdir(parents=True)
    result: dict = {}

    pf._record_workspace_traces(result, str(layout.root), "sglang", None)

    assert result["profile_trace_selection_reason"] == "capture_only_fallback"
    assert result["trace_files"] == [str(sidecar)]


def test_a_run_dir_without_a_trace_fails_naming_its_trace_dirs(tmp_path: Path) -> None:
    layout = _run_dir(tmp_path)
    result: dict = {}

    pf._record_workspace_traces(result, str(layout.root), "sglang", None)

    assert result["error_class"] == "no_trace_files"
    assert all(str(path) in result["error"] for path in layout.profile_trace_dirs())
