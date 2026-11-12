# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the traces of a profile test are collected to, and what counts as a trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.profile_traces import (
    collect_traces,
    has_step_traces,
    has_traces,
    is_trace_file,
    round_has_traces,
)
from inference_suite.contracts import RunLayout


def _file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("name", "expected"),
    [("rank0.pt.trace.json.gz", True), ("t.json", True), ("server.log", False), ("trace.txt", False)],
)
def test_only_json_traces_are_traces(name: str, expected: bool) -> None:
    assert is_trace_file(name) is expected


def test_a_directory_without_a_trace_file_has_no_traces(tmp_path: Path) -> None:
    _file(tmp_path / "torch_trace" / "notes.txt")
    assert not has_traces(tmp_path / "torch_trace") and not has_traces(tmp_path / "missing")


@pytest.mark.parametrize(
    ("files", "steps"),
    [
        # What a NEXTN head's profile held: the boot's graph captures, no rank trace.
        (("torch_trace/graph_capture_profile/DecodeCudaGraphRunner_bs_10_rank0.json.gz",), False),
        (("capture_traces/bs_1_rank0.json",), False),
        (("torch_trace/cuda_graph_capture_bs_8.json",), False),
        (
            (
                "torch_trace/graph_capture_profile/DecodeCudaGraphRunner_bs_10_rank0.json.gz",
                "torch_trace/1700000000.0-TP-0.trace.json.gz",
            ),
            True,
        ),
    ],
)
def test_only_a_trace_that_is_not_a_graph_capture_sidecar_records_serving_steps(
    tmp_path: Path, files: tuple[str, ...], steps: bool
) -> None:
    for name in files:
        _file(tmp_path / "profile" / name)
    assert has_traces(tmp_path / "profile")
    assert has_step_traces(tmp_path / "profile") is steps


def test_trace_directories_are_moved_and_nothing_else_is_taken(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "round" / "suite_run_0")
    _file(layout.profile_torch_trace / "r0" / "a.pt.trace.json.gz", b"a")
    _file(layout.server_log, b"log")
    _file(layout.root / "torch_trace" / "outside.pt.trace.json.gz", b"not in profile/")

    moved = collect_traces(layout.root, tmp_path / "out" / "profile")

    assert moved == 1
    assert (tmp_path / "out" / "profile" / "torch_trace" / "r0" / "a.pt.trace.json.gz").read_bytes() == b"a"
    assert not layout.profile_torch_trace.exists() and layout.server_log.exists()
    assert (layout.root / "torch_trace" / "outside.pt.trace.json.gz").exists(), "only the run_dir's profile/ is read"


def test_a_directory_with_no_trace_file_stays_and_nothing_is_created(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "ws")
    _file(layout.profile_torch_trace / "notes.txt")

    assert collect_traces(layout.root, tmp_path / "out" / "profile") == 0
    assert (layout.profile_torch_trace / "notes.txt").exists() and not (tmp_path / "out").exists()
    assert not round_has_traces(layout.root)


def test_a_stale_directory_of_an_earlier_attempt_is_replaced_not_merged(tmp_path: Path) -> None:
    destination = tmp_path / "out" / "profile"
    _file(destination / "torch_trace" / "old.pt.trace.json.gz")
    layout = RunLayout(tmp_path / "ws")
    _file(layout.profile_torch_trace / "new.pt.trace.json.gz")

    collect_traces(layout.root, destination)

    assert sorted(p.name for p in (destination / "torch_trace").iterdir()) == ["new.pt.trace.json.gz"]


def test_a_run_dirs_trace_directories_are_moved_apart_not_flattened(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "round-000")
    _file(layout.profile_torch_trace / "rank0.pt.trace.json.gz", b"step")
    _file(layout.profile_dir / "capture_traces" / "bs_1_rank0.json", b"capture")
    _file(layout.profile_dir / "graph_capture_profile" / "DecodeCudaGraphRunner_bs_8_rank0.json.gz", b"graph")
    destination = tmp_path / "out" / "profile"
    assert round_has_traces(layout.root)

    assert collect_traces(layout.root, destination) == 3

    kept = {p.relative_to(destination).as_posix(): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert kept == {
        "torch_trace/rank0.pt.trace.json.gz": b"step",
        "capture_traces/bs_1_rank0.json": b"capture",
        "graph_capture_profile/DecodeCudaGraphRunner_bs_8_rank0.json.gz": b"graph",
    }
    assert has_step_traces(destination)
