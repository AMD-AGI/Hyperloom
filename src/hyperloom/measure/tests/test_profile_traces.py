# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the traces of a profile test are collected to, and what counts as a trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.profile_traces import collect_traces, has_step_traces, has_traces, is_trace_file


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
                "torch_trace/1792918450.3630219-TP-0.trace.json.gz",
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


def test_trace_directories_are_moved_with_their_names_and_nothing_is_left_behind(tmp_path: Path) -> None:
    workspace = tmp_path / "round" / "ws"
    _file(workspace / "torch_trace" / "r0" / "a.pt.trace.json.gz", b"a")
    _file(workspace.parent / "capture_traces" / "g.json", b"g")
    _file(workspace / "server.log", b"log")

    moved = collect_traces(workspace, tmp_path / "out" / "profile")

    assert moved == 2
    assert (tmp_path / "out" / "profile" / "torch_trace" / "r0" / "a.pt.trace.json.gz").read_bytes() == b"a"
    assert (tmp_path / "out" / "profile" / "capture_traces" / "g.json").read_bytes() == b"g"
    assert not (workspace / "torch_trace").exists() and (workspace / "server.log").exists()


def test_a_directory_with_no_trace_file_stays_and_nothing_is_created(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _file(workspace / "torch_trace" / "notes.txt")

    assert collect_traces(workspace, tmp_path / "out" / "profile") == 0
    assert (workspace / "torch_trace" / "notes.txt").exists() and not (tmp_path / "out").exists()


def test_a_stale_directory_of_an_earlier_attempt_is_replaced_not_merged(tmp_path: Path) -> None:
    destination = tmp_path / "out" / "profile"
    _file(destination / "torch_trace" / "old.pt.trace.json.gz")
    workspace = tmp_path / "ws"
    _file(workspace / "torch_trace" / "new.pt.trace.json.gz")

    collect_traces(workspace, destination)

    assert sorted(p.name for p in (destination / "torch_trace").iterdir()) == ["new.pt.trace.json.gz"]
