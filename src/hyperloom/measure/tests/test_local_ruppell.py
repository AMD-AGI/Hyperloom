# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ruppell in the local backend's profile test, against a stubbed ``run_grid``: the server is traced when ruppell is
ready, a traced attempt that fails is profiled again without it, and nothing of ruppell's changes the return code."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from hyperloom.measure import local, ruppell
from hyperloom.measure.tests.test_local_backend import (  # noqa: F401 (fixtures)
    _call,
    _Grid,
    _profile_point,
    _run,
    grid,
    magpie_backend,
)
from hyperloom.orchestrator.actions.executors import VariantResult

TRACED = {"RUPPELL_SGLANG": "1", "RUPPELL_DETAIL": "5/1000", "RUPPELL_TOOL_LIB": "/w/build/libruppell_tool.so"}


class _Tracer(ruppell.Tracer):
    """A tracer whose points carry :data:`TRACED` and whose report says what it was asked."""

    def __init__(self, work: Path, report: str = "ruppell: ok, 64 detail steps, check ok") -> None:
        super().__init__(work, work / "source", work / "build")
        work.mkdir(parents=True, exist_ok=True)
        self.answer = report
        self.reports: list[tuple[Path, int]] = []
        self.closed = False

    def traced_points(self, points, environ):
        traced = []
        for index, point in enumerate(points):
            config = yaml.safe_load(Path(point).read_text())
            config["benchmark"]["envs"] = {**config["benchmark"]["envs"], **TRACED}
            target = self.work / f"point_{index}.yaml"
            target.write_text(yaml.safe_dump(config))
            traced.append(target)
        return tuple(traced)

    def report(self, output_dir: Path, ranks: int) -> str:
        self.reports.append((Path(output_dir), ranks))
        return self.answer

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def tracer(tmp_path, monkeypatch) -> _Tracer:
    made = _Tracer(tmp_path / "ruppell-work")
    monkeypatch.setattr(ruppell, "prepare", lambda points, environ: made)
    return made


def _traced(config: dict[str, Any]) -> bool:
    return "RUPPELL_TOOL_LIB" in config["benchmark"]["envs"]


def _traced_result(grid: _Grid, *, trace: bool = True, **fields: Any) -> VariantResult:
    result = grid.result("test", **fields)
    if trace:
        (Path(result.workspace) / "torch_trace").mkdir(parents=True, exist_ok=True)
        (Path(result.workspace) / "torch_trace" / "rank0.pt.trace.json.gz").write_bytes(b"trace")
    return result


def _answers(grid: _Grid, monkeypatch, *results: Callable[[], VariantResult]) -> None:
    """``run_grid`` answering each call with the next of ``results``, made when the call runs, as its round would."""
    queue = list(results)
    original = _Grid.__call__

    async def call(self, **kwargs):
        if queue:
            self.answers["test"] = queue.pop(0)()
        return await original(self, **kwargs)

    monkeypatch.setattr(_Grid, "__call__", call)


def test_a_profile_with_ruppell_ready_traces_its_server_and_notes_the_report(grid, tracer, tmp_path, monkeypatch):
    _answers(grid, monkeypatch, lambda: _traced_result(grid))
    point = _profile_point(tmp_path)

    outcome = _run(_call(tmp_path, point, kind="profile"))

    (config,) = grid.configs
    assert _traced(config) and {k: config["benchmark"]["envs"][k] for k in TRACED} == TRACED
    assert not _traced(yaml.safe_load(point.read_text())), "the point itself is shared and stays as it was"
    assert tracer.reports == [(tmp_path / "out", 1)], "after the traces were gathered, for the point's TP"
    assert (tmp_path / "out" / "profile" / "torch_trace" / "rank0.pt.trace.json.gz").is_file()
    assert (outcome.returncode, outcome.notes) == (0, ("ruppell: ok, 64 detail steps, check ok",))
    assert tracer.closed


def test_a_report_that_failed_leaves_the_return_code_alone(grid, tmp_path, monkeypatch):
    made = _Tracer(tmp_path / "w", report="ruppell: report failed (the server left no records)")
    monkeypatch.setattr(ruppell, "prepare", lambda points, environ: made)
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert (outcome.returncode, outcome.failure_class) == (0, None)
    assert outcome.notes == ("ruppell: report failed (the server left no records)",)


def test_a_traced_attempt_that_failed_is_profiled_again_without_ruppell(grid, tracer, tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    log.write_text("Fatal Python error: Segmentation fault\n")
    failed = _traced_result(
        grid, trace=False, status="failed", returncode=1, error="bench rc=1\nmore", server_log_path=str(log)
    )
    first_workspace = Path(failed.workspace)
    (first_workspace / "server.log").write_text("the traced attempt")
    _answers(grid, monkeypatch, lambda: failed, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [True, False]
    assert outcome.notes == ("ruppell: attempt failed (bench rc=1), profiled again without it",)
    assert (outcome.returncode, [r.returncode for r in outcome.rounds]) == (0, [0])
    assert tracer.reports == [], "no report of an attempt that is not the one measured"
    kept = tmp_path / "out" / "ruppell" / "attempt_server.log"
    assert kept.read_text() == "Fatal Python error: Segmentation fault\n"
    assert not (first_workspace / "server.log").exists(), "the traced attempt's round is not in the output"


def test_a_traced_attempt_that_left_no_trace_is_profiled_again(grid, tracer, tmp_path, monkeypatch):
    _answers(grid, monkeypatch, lambda: _traced_result(grid, trace=False), lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [True, False]
    assert outcome.notes == (
        "ruppell: attempt failed (the profiled round left no trace file), profiled again without it",
    )
    assert outcome.returncode == 0 and "profile_no_trace" not in outcome.stderr


def test_a_traced_attempt_that_timed_out_is_not_run_again(grid, tracer, tmp_path, monkeypatch):
    _answers(
        grid, monkeypatch, lambda: _traced_result(grid, trace=False, status="failed", error_class="magpie_timeout")
    )

    with pytest.raises(local.MeasurementTimeout):
        _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [True]
    assert tracer.closed


def test_a_profile_ruppell_cannot_trace_runs_as_before_with_a_note(grid, tmp_path, monkeypatch):
    monkeypatch.setattr(ruppell, "prepare", lambda points, environ: "ruppell: skipped (not staged: no /runner-cache/x)")
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [False]
    assert (outcome.returncode, outcome.notes) == (0, ("ruppell: skipped (not staged: no /runner-cache/x)",))


def test_points_ruppell_cannot_be_given_run_untraced_with_a_note(grid, tracer, tmp_path, monkeypatch):
    def refuse(points, environ):
        raise ruppell.RuppellError("python -m ruppell run exited 1: no build")

    monkeypatch.setattr(tracer, "traced_points", refuse)
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [False]
    assert outcome.notes == ("ruppell: skipped (python -m ruppell run exited 1: no build)",)
    assert tracer.reports == [] and tracer.closed


def test_a_test_that_is_not_a_profile_never_builds_ruppell(grid, tmp_path, monkeypatch):
    def refuse(points, environ):
        raise AssertionError("prepared for a test that is not a profile")

    monkeypatch.setattr(ruppell, "prepare", refuse)

    outcome = _run(_call(tmp_path))

    assert (outcome.returncode, outcome.notes) == (0, ())
