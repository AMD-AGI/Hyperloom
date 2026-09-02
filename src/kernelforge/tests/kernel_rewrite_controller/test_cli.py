# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from kernelforge import cli
from kernelforge.kernel_rewrite_controller import ControllerRunError, run_controller
from kernelforge.kernel_rewrite_controller import controller
from kernelforge.kernel_rewrite_controller.scheduler import ScheduleResult
from kernelforge.kernel_rewrite_controller.contracts import (
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)


def _handoff(tmp_path: Path, *, omit: str = "") -> Path:
    root = tmp_path / "handoff"
    root.mkdir()
    for filename in (WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME):
        if filename != omit:
            (root / filename).write_text(f"# {filename}\n", encoding="utf-8")
    return root


def _invoke(tmp_path: Path, handoff: Path, *, budget: str = "10", output_name: str = "output"):
    return CliRunner().invoke(
        cli.main,
        [
            "kernel-rewrite-controller",
            "--handoff-dir",
            str(handoff),
            "--budget-minutes",
            budget,
            "--output-dir",
            str(tmp_path / output_name),
        ],
    )


def test_cli_declares_the_three_controller_arguments() -> None:
    command = cli.main.get_command(None, "kernel-rewrite-controller")

    assert command is not None
    assert {parameter.name for parameter in command.params} == {
        "handoff_dir",
        "budget_minutes",
        "output_dir",
    }


def test_cli_initializes_a_fresh_no_opportunity_result(tmp_path: Path) -> None:
    before = time.time()
    result = _invoke(tmp_path, _handoff(tmp_path))
    after = time.time()

    assert result.exit_code == 0, result.output
    assert "kernel-rewrite-controller: no_opportunity" in result.output
    output = tmp_path / "output"
    state = json.loads((output / "controller" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "no_opportunity"
    assert state["budget_minutes"] == 10.0
    assert before + 600 <= state["deadline_unix"] <= after + 600
    assert state["task_count"] == 0
    assert state["patch_count"] == 0
    assert (output / "controller" / "tasks").is_dir()
    assert (output / "controller" / "workspaces").is_dir()
    assert (output / "result" / "patches").is_dir()
    assert "**Status:** `no_opportunity`" in (output / "result" / "summary.md").read_text(encoding="utf-8")


def test_handoff_may_live_inside_the_cycle_output_directory(tmp_path: Path) -> None:
    cycle = tmp_path / "cycle-2"
    cycle.mkdir()
    handoff = cycle / "handoff"
    handoff.mkdir()
    for filename in (WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME):
        (handoff / filename).write_text(f"# {filename}\n", encoding="utf-8")

    state = run_controller(
        handoff_dir=handoff,
        budget_minutes=10,
        output_dir=cycle,
    )

    assert state.status == "no_opportunity"
    assert (cycle / "controller" / "state.json").is_file()
    assert (cycle / "result" / "summary.md").is_file()


def test_controller_summary_counts_prepared_task_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        controller,
        "dispatch_prepared_tasks",
        lambda *_args, **_kwargs: ScheduleResult(
            task_count=2,
            results=(
                SimpleNamespace(status="succeeded"),
                SimpleNamespace(status="failed"),
            ),
        ),
    )
    monkeypatch.setattr(
        controller,
        "published_operator_dirs",
        lambda _layout: (tmp_path / "output" / "result" / "patches" / "operator",),
    )

    state = run_controller(
        handoff_dir=_handoff(tmp_path),
        budget_minutes=10,
        output_dir=tmp_path / "output",
    )

    assert state.status == "completed"
    assert state.task_count == 2
    assert state.patch_count == 1


def test_invalid_handoff_is_persisted_as_failed(tmp_path: Path) -> None:
    result = _invoke(
        tmp_path,
        _handoff(tmp_path, omit=TRACE_EVIDENCE_FILENAME),
    )

    assert result.exit_code != 0
    assert "handoff validation failed" in result.output
    state = json.loads((tmp_path / "output" / "controller" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert TRACE_EVIDENCE_FILENAME in state["reason"]


def test_existing_controller_output_is_not_resumed(tmp_path: Path) -> None:
    handoff = _handoff(tmp_path)
    first = _invoke(tmp_path, handoff)
    second = _invoke(tmp_path, handoff)

    assert first.exit_code == 0
    assert second.exit_code != 0
    assert "already initialized and cannot be resumed" in second.output


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_budget_without_initializing_output(
    tmp_path: Path,
    budget: str,
) -> None:
    result = _invoke(tmp_path, _handoff(tmp_path), budget=budget)

    assert result.exit_code != 0
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf")])
def test_controller_api_rejects_invalid_budget(tmp_path: Path, budget: object) -> None:
    with pytest.raises(ControllerRunError, match="budget_minutes"):
        run_controller(
            handoff_dir=_handoff(tmp_path),
            budget_minutes=budget,  # type: ignore[arg-type]
            output_dir=tmp_path / "output",
        )
