# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the run_dir layout: the file names readers locate, and the ``server_ready_at`` stamp round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_suite.contracts import ReadyStamp, RunDirError, RunLayout, format_ready_stamp, parse_ready_stamp
from inference_suite.contracts.layout import GPU_METRICS, HOST_DERIVED


def test_ready_stamp_round_trips_through_its_file(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path)
    assert layout.read_ready_stamp() is None
    stamp = ReadyStamp(ready_unix=1762400000.125, boot_sec=312.5)
    layout.write_ready_stamp(stamp)
    assert layout.ready_stamp == tmp_path / "server_ready_at"
    assert layout.ready_stamp.read_text(encoding="utf-8") == "1762400000.125 312.500\n"
    assert layout.read_ready_stamp() == stamp
    assert sorted(path.name for path in tmp_path.iterdir()) == ["server_ready_at"]
    layout.clear_ready_stamp()
    assert layout.read_ready_stamp() is None


def test_a_reused_server_stamps_no_boot() -> None:
    assert parse_ready_stamp(format_ready_stamp(ReadyStamp(1762400000.0, 0))) == ReadyStamp(1762400000.0, 0.0)


@pytest.mark.parametrize("text", ["", "1762400000.0", "1762400000.0 nan", "0 5", "x 5", "1 2 3", "1762400000.0 -1"])
def test_a_text_that_is_not_a_whole_stamp_reads_as_none(text: str) -> None:
    assert parse_ready_stamp(text) is None


def test_the_stamp_reads_what_the_current_launcher_writes(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors._subprocess_kill import stamp_server_ready

    stamp_server_ready(str(tmp_path / "server.log"), 42.0)
    stamp = RunLayout(tmp_path).read_ready_stamp()
    assert stamp is not None and stamp.boot_sec == 42.0


def test_run_dir_names(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path)
    relative = {
        name: getattr(layout, name).relative_to(tmp_path).as_posix()
        for name in ("config", "launch", "report", "server_log", "phase_marks", "gpu_power", "gpu_samples")
    }
    assert relative == {
        "config": "config.yaml",
        "launch": "launch.json",
        "report": "report.json",
        "server_log": "server.log",
        "phase_marks": "phase_marks.json",
        "gpu_power": "gpu_power.json",
        "gpu_samples": "gpu_samples.csv",
    }
    assert layout.aiperf_request_export.relative_to(tmp_path).as_posix() == (
        "client/aiperf_artifacts/profile_export.jsonl"
    )
    assert [path.relative_to(tmp_path).as_posix() for path in layout.profile_trace_dirs()] == [
        "profile/torch_trace",
        "profile/capture_traces",
        "profile/graph_capture_profile",
    ]
    assert layout.profile_torch_trace == layout.profile_trace_dirs()[0]
    assert HOST_DERIVED == (GPU_METRICS,)


def test_claim_creates_an_absent_run_dir_and_accepts_an_empty_one(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "a" / "run")
    layout.claim()
    assert layout.root.is_dir()
    layout.claim()


def test_claim_refuses_a_run_dir_that_holds_anything(tmp_path: Path) -> None:
    (tmp_path / "leftover").write_text("", encoding="utf-8")
    with pytest.raises(RunDirError, match="not empty"):
        RunLayout(tmp_path).claim()
