# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every shape a scripted round has to be able to say."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hyperloom.orchestrator.rehearsal import (
    COMPLETED,
    CRASHED,
    EMPTY,
    HUNG,
    LaunchScenario,
    ScenarioError,
    ScriptedSpecialist,
    SpecialistStep,
    VirtualClock,
)
from hyperloom.orchestrator.specialists.subprocess_ import SpecialistSpawn, SpecialistSubprocessDispatcher


def _spawn(task_id: str, workspace: Path) -> SpecialistSpawn:
    """The dispatch the runner would hand the dispatcher; the script reads only the task id and workspace."""
    return SpecialistSpawn(
        task_id=task_id,
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="",
        user_prompt="",
    )


def test_a_scenario_that_names_neither_spelling_is_rejected():
    with pytest.raises(ScenarioError):
        LaunchScenario.from_dict({"name": "empty"})


@pytest.mark.asyncio
async def test_a_completed_specialist_writes_the_done_file_the_runner_reads(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=COMPLETED, done={"status": "patched"}),))

    result = await scripted.run(_spawn("spec-1", tmp_path / "ws"))

    assert result.done_payload == {"status": "patched"}
    assert (tmp_path / "ws" / "specialist_done.json").exists()
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_an_empty_completion_is_a_done_file_with_nothing_in_it(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=EMPTY),))

    result = await scripted.run(_spawn("spec-1", tmp_path / "ws"))

    assert result.done_payload == {}
    assert (tmp_path / "ws" / "specialist_done.json").exists()


@pytest.mark.asyncio
async def test_a_crash_leaves_no_done_file_for_the_runner_to_find(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=CRASHED),))

    result = await scripted.run(_spawn("spec-1", tmp_path / "ws"))

    assert result.done_payload is None
    assert not (tmp_path / "ws" / "specialist_done.json").exists()
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_a_hang_reports_a_stale_heartbeat_and_its_log_went_silent(tmp_path):
    from hyperloom.orchestrator.specialists.reap_watch import newest_activity, specialist_activity_paths

    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=HUNG),))

    result = await scripted.run(_spawn("spec-1", tmp_path / "ws"))

    assert result.stale_heartbeat
    assert result.exit_code is None
    assert newest_activity(specialist_activity_paths(tmp_path / "ws")) == 0.0


@pytest.mark.asyncio
async def test_a_retry_under_a_new_task_id_is_visible_as_a_retry(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=CRASHED), SpecialistStep(outcome=COMPLETED)))

    await scripted.run(_spawn("spec-1", tmp_path / "spec-1"))
    await scripted.run(_spawn("spec-2", tmp_path / "spec-2"))

    assert scripted.task_ids == ["spec-1", "spec-2"]


def test_the_virtual_clock_moves_only_when_something_charges_it():
    clock = VirtualClock()
    start_monotonic = clock.monotonic()
    start_wall = clock.wall()

    clock.advance(120.0)
    clock.advance(-5.0)

    # Both readings move by the same amount, so a watchdog arming on monotonic
    # time and an artifact stamped with wall time agree about the same step.
    assert clock.monotonic() - start_monotonic == pytest.approx(120.0)
    assert clock.wall() - start_wall == pytest.approx(120.0)


def test_the_scripted_specialist_takes_the_dispatch_the_real_dispatcher_takes():
    real = inspect.signature(SpecialistSubprocessDispatcher.run)
    scripted = inspect.signature(ScriptedSpecialist.run)

    assert list(scripted.parameters) == list(real.parameters)
    assert scripted.parameters["request"].annotation == real.parameters["request"].annotation
