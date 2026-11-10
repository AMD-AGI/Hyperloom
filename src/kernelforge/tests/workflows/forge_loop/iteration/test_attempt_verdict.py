# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measuring an attempt and ruling on it: repeated runs, assembly acceptance, agent errors, integrity violations and the label each verdict carries."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _lesson_result,
    _make_loop,
    _measurement_loop,
    _no_change_agent,
    _unused_supervisor,
)


@pytest.mark.asyncio
async def test_iteration_keeps_a_gain_that_repeats_across_all_three_runs(
    monkeypatch,
):
    candidate_ms = 1.0 / 1.005
    loop, benchmark_calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": candidate_ms,
            "case_times": {"small": candidate_ms, "large": candidate_ms},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {
                        "small": candidate_ms,
                        "large": candidate_ms,
                    },
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "three measurements",
        },
    )

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert result.bench_detail["mean_case_speedup"] == pytest.approx(1.005)
    assert result.bench_detail["measurement_mean_case_speedups"] == pytest.approx([1.005, 1.005, 1.005])
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0]["measurements"] == 3


def _faster_bench(candidate_ms=1.0 / 1.05):
    return {
        "success": True,
        "median_ms": candidate_ms,
        "case_times": {"small": candidate_ms, "large": candidate_ms},
        "unscored_cases": [],
        "measurement_count": 3,
        "measurements": [
            {
                "success": True,
                "case_times": {"small": candidate_ms, "large": candidate_ms},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
        "message": "three measurements",
    }


def _git_workspace(path: Path) -> Path:
    """The loop commits every iteration and diffs against HEAD, so a workspace is a repo that has one."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path


def _canonical_workspace(tmp_path, command: str) -> Path:
    # A task config shaped the way the arena's evaluator reads one. Only the
    # assembly backend asks the engine to run it; for every other backend its
    # presence must not move a verdict the driver has already reached.
    tmp_path.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "compile_command": [f"{sys.executable} -c 'pass'"],
                "correctness_command": [f"{sys.executable} -c {command!r}"],
            }
        )
    )
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("ships_a_config", [True, False])
async def test_a_non_assembly_keep_rests_on_the_driver_not_on_the_task_config(
    tmp_path, monkeypatch, capsys, ships_a_config
):
    """The driver judged this candidate in Step 4 and has measured it ever since.

    Re-running that verdict here would ask the same question of the same command
    through a configuration file whose shape the engine has no business knowing,
    and a consumer that declares evaluation differently could never earn a keep.
    Shipping a task config -- even one whose suite would reject -- must therefore
    change nothing, and no canonical verdict may reach the operator's log.
    """
    if ships_a_config:
        _canonical_workspace(tmp_path, "raise AssertionError('the engine must not run this')")
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=_git_workspace(tmp_path))

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert result.validation_passed is True
    assert "[canonical]" not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "unstable", "valid"])
async def test_assembly_keep_requires_fresh_source_relative_numerics(tmp_path, monkeypatch, kind):
    from kernelforge.tests.evaluation.correctness._numerical_contract import _task, evidence

    if kind != "missing":
        _task(tmp_path, evidence(candidate_repeat_db=27 if kind == "unstable" else 45))
    loop, _ = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=_git_workspace(tmp_path))
    loop.ic.kernel_backend = "assembly"
    loop.ic.pristine_baseline_wall_ms = 1.0

    result = await loop.run_one_iteration(1)

    assert result.kept is (kind == "valid")
    assert result.validation_passed is (kind == "valid")
    if kind != "missing":
        assert result.bench_detail["numerical_validation"]["contract_sha256"]
    if kind == "unstable":
        assert result.validation_outcome == "numerical_correctness_failure"
        assert "repeat_errors" in result.validation_summary


@pytest.mark.asyncio
async def test_snr_pass_with_failing_assembly_acceptance_is_reverted(tmp_path, monkeypatch, capsys):
    """The mla-decode run: 33.4 dB cleared forge's gate, 0.02468 broke the task's.

    The SNR probe passes, the candidate is 5% faster, and the suite the assembly
    backend declares rejects it. That candidate must not be kept, and the
    tolerance it broke -- not the dB figure -- has to reach the agent.
    """
    from kernelforge.tests.evaluation.correctness._numerical_contract import _task, evidence

    _task(tmp_path, evidence())
    tmp_path.joinpath("driver.py").write_text("raise AssertionError('normalized max err 0.02468 too high')\n")
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=_git_workspace(tmp_path))
    loop.ic.kernel_backend = "assembly"
    loop.ic.pristine_baseline_wall_ms = 1.0

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert result.validation_passed is False
    assert result.validation_outcome == "canonical_correctness_failure"
    assert "0.02468" in result.error_output
    assert "[canonical] FAIL" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_assembly_acceptance_is_skipped_for_a_candidate_that_is_not_faster(tmp_path, monkeypatch):
    """The suite is the expensive check; a slower candidate is reverted anyway."""
    workspace = _git_workspace(_canonical_workspace(tmp_path, "raise AssertionError('this must never run')"))
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(candidate_ms=2.0), workspace_dir=workspace)
    loop.ic.kernel_backend = "assembly"
    loop.ic.pristine_baseline_wall_ms = 1.0

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert result.validation_passed is True
    assert result.error_output == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("kernel_backend,expected_keep", [("", True), ("assembly", False)])
async def test_iteration_keeps_winning_mean_despite_one_regressed_case(monkeypatch, kernel_backend, expected_keep):
    loop, _benchmark_calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": 1.0,
            "case_times": {"small": 0.5, "large": 1.5},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {"small": 0.5, "large": 1.5},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "three measurements",
        },
    )

    loop.ic.kernel_backend = kernel_backend
    loop.ic.pristine_baseline_wall_ms = 1.0
    result = await loop.run_one_iteration(1)

    assert result.kept is expected_keep
    assert result.bench_detail["mean_case_speedup"] == pytest.approx((2.0 + 2.0 / 3.0) / 2.0)


def test_agent_error_is_reduced_and_persisted(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def failing_agent(*_args, **_kwargs):
        raise RuntimeError("agent exploded")

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=failing_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert state.iteration == 1
    assert state.stall.no_improvement_iters == 1
    assert loop.monitor.no_improve_streak == 1
    assert decisions == ["AGENT_ERROR"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — AGENT_ERROR" in history
    assert loop.archive.load_index() == []
    assert loop.tracker.get(loop.experiment.experiment_id).iterations == []


def test_agent_error_with_diff_is_handed_to_outer_validation(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    validated = []

    async def editing_then_failing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "change before SDK failure"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        raise RuntimeError("SDK stream failed after edit")

    async def reject_candidate(iteration, plan=""):
        validated.append((iteration, plan))
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary="canonical correctness failed",
        )

    monkeypatch.setattr(loop, "run_one_iteration", reject_candidate)
    asyncio.run(
        loop.run(runner_module.LoopAgents(agent_fn=editing_then_failing_agent, supervisor_fn=_unused_supervisor))
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert validated == [(1, "change before SDK failure")]
    assert decisions == ["REVERT_VALIDATION"]


def test_integrity_violation_restores_and_skips_canonical_validation(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    protected = workspace / "nested" / "tests" / "oracle.bin"
    protected.parent.mkdir(parents=True)
    protected.write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add protected oracle"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    validation_calls: list[int] = []

    async def unsafe_agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 99\n")
        protected.write_text("gamed\n")
        session_sink["plan"] = "unsafe candidate"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["integrity_violation"] = True
        session_sink["integrity_verdict"] = "violation"
        session_sink["integrity_reason"] = "modified nested/tests/oracle.bin"
        session_sink["integrity_restore"] = lambda: protected.write_text("original\n")
        return "unsafe candidate"

    async def unexpected_validation(*_args, **_kwargs):
        validation_calls.append(1)
        raise AssertionError("canonical validation must be skipped")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=unsafe_agent, supervisor_fn=_unused_supervisor)))

    assert validation_calls == []
    assert kernel.read_text() == "def kernel():\n    return 1\n"
    assert protected.read_text() == "original\n"
    assert loop.results[0].integrity_violation is True
    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert decisions == ["REVERT_INTEGRITY"]


def test_an_api_outage_is_not_recorded_as_an_agent_decision(tmp_path, monkeypatch):
    """An outage and a deliberate no-op leave the same empty diff.

    Only the end reason separates them, and recording the outage as NO_CHANGES
    tells the next Session that the agent looked and chose to change nothing.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def api_failed_agent(_kernel_path, _history, session_sink):
        session_sink["end_reason"] = "api_error"
        return "agent session ended with an API error"

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=api_failed_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert decisions == ["API_ERROR"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — API_ERROR" in history


def test_no_change_attempt_is_reduced_and_persisted(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert state.iteration == 1
    assert state.stall.no_improvement_iters == 1
    assert loop.monitor.no_improve_streak == 1
    assert decisions == ["NO_CHANGES"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — NO_CHANGES" in history
    lesson = (workspace / "forge_experiments" / "lessons" / "iter_001.md").read_text()
    assert lesson.strip().startswith("SCOPE: measured on ")
    assert "OUTCOME: NO_CHANGES" in lesson
    assert loop.archive.load_index() == []
    assert loop.tracker.get(loop.experiment.experiment_id).iterations == []


def test_decision_label_matches_every_outcome():
    from kernelforge.workflows.forge_loop.iteration.iteration_types import _decision_label

    assert _decision_label(_lesson_result(crashed=True)) == "CRASH"
    assert (
        _decision_label(_lesson_result(validation_passed=False, validation_summary="BUILD FAILED: boom"))
        == "BUILD_FAILED"
    )
    assert (
        _decision_label(_lesson_result(validation_passed=False, validation_summary="stage 3 failed"))
        == "REVERT_VALIDATION"
    )
    assert (
        _decision_label(
            _lesson_result(
                validation_passed=False,
                validation_summary="full suite timed out",
                validation_outcome="timeout",
            )
        )
        == "REVERT_VALIDATION_TIMEOUT"
    )
    assert (
        _decision_label(
            _lesson_result(
                validation_passed=False,
                validation_summary="driver crashed",
                validation_outcome="driver_error",
            )
        )
        == "REVERT_VALIDATION_ERROR"
    )
    assert _decision_label(_lesson_result(kept=True)) == "KEEP"
    assert _decision_label(_lesson_result(kept=False)) == "REVERT_PERF"
