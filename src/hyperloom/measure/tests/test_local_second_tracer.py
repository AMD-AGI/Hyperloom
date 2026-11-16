# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A second tracer in the local backend's profile test, against a stubbed ``run_grid``: the server is traced when a
tracer is ready, a traced attempt that fails is profiled again without it, and nothing of the tracer's changes the
return code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.measure import local, second_tracer
from hyperloom.measure.second_tracer import SecondTracer, TracerError
from hyperloom.measure.tests.test_local_backend import (  # noqa: F401 (fixtures)
    _call,
    _Grid,
    _profile_point,
    _run,
    grid,
)
from hyperloom.orchestrator.actions.executors import VariantResult
from inference_suite.contracts import RunLayout

NAME = "stand-in"
TRACED = {"STAND_IN_TRACER": "1", "STAND_IN_TOOL_LIB": "/w/build/libstand_in_tool.so"}


@dataclass
class _Tracer:
    """A stand-in tracer whose points carry :data:`TRACED` and whose report says what it was asked."""

    work: Path
    answer: tuple[str, ...] = (f"{NAME}: ok, 64 detail steps, check ok",)
    refuses: str = ""
    reports: list[tuple[Path, int]] = field(default_factory=list)
    closed: bool = False

    def traced_points(self, points, environ):
        if self.refuses:
            raise TracerError(self.refuses)
        self.work.mkdir(parents=True, exist_ok=True)
        traced = []
        for index, point in enumerate(points):
            config = yaml.safe_load(Path(point).read_text())
            config["benchmark"]["envs"] = {**config["benchmark"]["envs"], **TRACED}
            target = self.work / f"point_{index}.yaml"
            target.write_text(yaml.safe_dump(config))
            traced.append(target)
        return tuple(traced)

    def report(self, output_dir: Path, ranks: int) -> tuple[str, ...]:
        self.reports.append((Path(output_dir), ranks))
        return self.answer

    def close(self) -> None:
        self.closed = True


def _installed(monkeypatch, made: _Tracer) -> _Tracer:
    """Make ``made`` the tracer a profile test is given."""
    ready = SecondTracer(NAME, made.traced_points, made.report, made.close)
    monkeypatch.setattr(second_tracer, "prepare", lambda points, environ: ready)
    return made


@pytest.fixture
def tracer(tmp_path, monkeypatch) -> _Tracer:
    return _installed(monkeypatch, _Tracer(tmp_path / "tracer-work"))


def _traced(config: dict[str, Any]) -> bool:
    return "STAND_IN_TOOL_LIB" in config["benchmark"]["envs"]


def _traced_result(grid: _Grid, *, trace: bool = True, **fields: Any) -> VariantResult:
    result = grid.result("test", **fields)
    if trace:
        traces = RunLayout(Path(result.workspace)).profile_torch_trace
        traces.mkdir(parents=True, exist_ok=True)
        (traces / "rank0.pt.trace.json.gz").write_bytes(b"trace")
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


def test_a_profile_with_a_tracer_ready_traces_its_server_and_notes_the_report(grid, tracer, tmp_path, monkeypatch):
    _answers(grid, monkeypatch, lambda: _traced_result(grid))
    point = _profile_point(tmp_path)

    outcome = _run(_call(tmp_path, point, kind="profile"))

    (config,) = grid.configs
    assert _traced(config) and {k: config["benchmark"]["envs"][k] for k in TRACED} == TRACED
    assert not _traced(yaml.safe_load(point.read_text())), "the point itself is shared and stays as it was"
    assert tracer.reports == [(tmp_path / "out", 1)], "after the traces were gathered, for the point's TP"
    assert (tmp_path / "out" / "profile" / "torch_trace" / "rank0.pt.trace.json.gz").is_file()
    assert (outcome.returncode, outcome.notes) == (0, (f"{NAME}: ok, 64 detail steps, check ok",))
    assert tracer.closed


def test_a_report_that_failed_leaves_the_return_code_alone_and_every_note_of_it_is_carried(grid, tmp_path, monkeypatch):
    said = (f"{NAME}: report failed (the server left no records)", f"{NAME}: traced with packet capture off")
    _installed(monkeypatch, _Tracer(tmp_path / "w", answer=said))
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert (outcome.returncode, outcome.failure_class) == (0, None)
    assert outcome.notes == said


def test_a_traced_attempt_that_failed_is_profiled_again_without_the_tracer(grid, tracer, tmp_path, monkeypatch):
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
    assert outcome.notes == (f"{NAME}: attempt failed (bench rc=1), profiled again without it",)
    assert (outcome.returncode, [r.returncode for r in outcome.rounds]) == (0, [0])
    assert tracer.reports == [], "no report of an attempt that is not the one measured"
    kept = tmp_path / "out" / second_tracer.OUTPUT_DIR / second_tracer.ATTEMPT_LOG_FILE
    assert kept.read_text() == "Fatal Python error: Segmentation fault\n"
    assert not (first_workspace / "server.log").exists(), "the traced attempt's round is not in the output"


def test_a_traced_attempt_that_left_no_trace_is_profiled_again(grid, tracer, tmp_path, monkeypatch):
    _answers(grid, monkeypatch, lambda: _traced_result(grid, trace=False), lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [True, False]
    assert outcome.notes == (
        f"{NAME}: attempt failed (the profiled round left no trace file), profiled again without it",
    )
    assert outcome.returncode == 0 and "profile_no_trace" not in outcome.stderr


def test_a_traced_attempt_that_timed_out_is_not_run_again(grid, tracer, tmp_path, monkeypatch):
    _answers(grid, monkeypatch, lambda: _traced_result(grid, trace=False, status="failed", error_class="round_timeout"))

    with pytest.raises(local.MeasurementTimeout):
        _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [True]
    assert tracer.closed


def test_a_profile_a_tracer_cannot_trace_runs_as_before_with_its_note(grid, tmp_path, monkeypatch):
    why = f"{NAME}: skipped (not staged: no /runner-cache/x)"
    monkeypatch.setattr(second_tracer, "prepare", lambda points, environ: why)
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [False]
    assert (outcome.returncode, outcome.notes) == (0, (why,))


def test_a_profile_with_no_tracer_installed_runs_untraced_and_carries_no_note(grid, tmp_path, monkeypatch):
    monkeypatch.setattr(second_tracer.metadata, "entry_points", lambda *, group: [])
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [False]
    assert (outcome.returncode, outcome.notes) == (0, ())
    assert not (tmp_path / "out" / second_tracer.OUTPUT_DIR).exists()


def test_points_a_tracer_cannot_be_given_run_untraced_with_a_note(grid, tracer, tmp_path, monkeypatch):
    tracer.refuses = "the tool's probe exited 1: no build"
    _answers(grid, monkeypatch, lambda: _traced_result(grid))

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert [_traced(config) for config in grid.configs] == [False]
    assert outcome.notes == (f"{NAME}: skipped (the tool's probe exited 1: no build)",)
    assert tracer.reports == [] and tracer.closed


def test_a_test_that_is_not_a_profile_never_prepares_a_tracer(grid, tmp_path, monkeypatch):
    def refuse(points, environ):
        raise AssertionError("prepared for a test that is not a profile")

    monkeypatch.setattr(second_tracer, "prepare", refuse)

    outcome = _run(_call(tmp_path))

    assert (outcome.returncode, outcome.notes) == (0, ())
