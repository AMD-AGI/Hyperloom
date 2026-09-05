# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every shape a scripted round has to be able to say.

A scenario is only worth writing if the outcomes it can express cover the ones
a real round produces. These pin that coverage: each case is a shape that used
to require a GPU to reach, checked here through the same classifier and the
same result types production uses.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.bringup import observe_bringup
from hyperloom.orchestrator.rehearsal import (
    COMPLETED,
    CRASHED,
    DIED_SILENTLY,
    EMPTY,
    HANG,
    HUNG,
    READY,
    STAGE_FAILED,
    LaunchAttempt,
    LaunchScenario,
    ScenarioError,
    ScenarioExhausted,
    ScriptedLaunchBackend,
    ScriptedSpecialist,
    SpecialistStep,
    VirtualClock,
    installed_clock,
)


def _backend(*attempts: LaunchAttempt) -> ScriptedLaunchBackend:
    """Build a backend over ``attempts`` on a fresh clock."""
    return ScriptedLaunchBackend(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock())


def _play(backend: ScriptedLaunchBackend, slot, **kwargs) -> subprocess.CompletedProcess:
    """Run one launch against ``backend``, writing into ``slot``."""
    return backend.run(["magpie", "bench.yaml"], server_log_path=str(slot / "server.log"), **kwargs)


def test_a_clean_boot_serves_and_leaves_a_log_that_classifies_as_booted(tmp_path):
    backend = _backend(LaunchAttempt(outcome=READY, artifacts={"bench/benchmark_report.json": {"success": True}}))

    done = _play(backend, tmp_path)

    assert done.returncode == 0
    observed = observe_bringup(server_log=(tmp_path / "server.log").read_text(encoding="utf-8"))
    assert observed.observation.booted
    assert (tmp_path / "bench" / "benchmark_report.json").exists()


@pytest.mark.parametrize(
    "stage",
    [LadderStage.ARGV_PARSE, LadderStage.CONFIG_VALIDATE, LadderStage.WEIGHTS_LOADING, LadderStage.ENGINE_INIT],
)
def test_a_boot_that_fails_at_a_named_stage_is_classified_back_to_that_stage(tmp_path, stage):
    """The scenario names the wall; the classifier has to find the same one."""
    backend = _backend(LaunchAttempt(outcome=STAGE_FAILED, stage=stage.name))

    done = _play(backend, tmp_path)

    assert done.returncode != 0
    observed = observe_bringup(server_log=(tmp_path / "server.log").read_text(encoding="utf-8"))
    assert observed.observation.stage_failed == stage


def test_a_hang_is_reaped_on_the_hard_timeout_and_charges_it_to_the_clock(tmp_path):
    backend = _backend(LaunchAttempt(outcome=HANG, duration_sec=10.0))

    with pytest.raises(subprocess.TimeoutExpired):
        _play(backend, tmp_path, timeout=1800)

    assert backend.clock.elapsed == pytest.approx(1800.0)
    assert backend.calls[-1].returncode == -1


def test_a_process_that_dies_without_a_log_leaves_nothing_to_classify(tmp_path):
    backend = _backend(LaunchAttempt(outcome=DIED_SILENTLY, returncode=139))

    done = _play(backend, tmp_path)

    assert done.returncode == 139
    assert not (tmp_path / "server.log").exists()


def test_a_blocker_stack_peels_one_per_attempt_and_then_boots(tmp_path):
    scenario = LaunchScenario.from_dict({"blockers": ["argv_parse", "import", "engine_init"]})
    backend = ScriptedLaunchBackend(scenario=scenario, clock=VirtualClock())

    reached = []
    for _ in scenario.attempts:
        _play(backend, tmp_path)
        observed = observe_bringup(server_log=(tmp_path / "server.log").read_text(encoding="utf-8"))
        reached.append(observed.observation.stage_failed)

    assert reached == [LadderStage.ARGV_PARSE, LadderStage.IMPORT, LadderStage.ENGINE_INIT, None]
    assert backend.exhausted


def test_one_launch_past_the_end_of_the_scenario_is_an_error_not_a_repeat(tmp_path):
    backend = _backend(LaunchAttempt(outcome=READY))
    _play(backend, tmp_path)

    with pytest.raises(ScenarioExhausted):
        _play(backend, tmp_path)


def test_a_session_deadline_the_attempt_runs_past_stops_the_round(tmp_path):
    from hyperloom.orchestrator.actions.executors._subprocess_kill import SESSION_TIME_EXHAUSTED_RETURNCODE

    backend = _backend(LaunchAttempt(outcome=READY, duration_sec=600.0))

    with installed_clock(backend.clock):
        done = _play(backend, tmp_path, session_deadline_sec=backend.clock.deadline_in(60.0))

    assert done.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert backend.clock.elapsed == pytest.approx(60.0)


def test_a_deadline_from_a_clock_this_backend_does_not_read_is_refused(tmp_path):
    """A deadline is an instant on one clock, and this backend must be that clock.

    Left unchecked the comparison is silently wrong rather than absent: a real
    ``time.monotonic()`` deadline sits far above the virtual origin on a
    long-uptime host and far below it on a freshly booted one, so the budget
    branch never fires or always does.
    """
    backend = _backend(LaunchAttempt(outcome=READY))

    with pytest.raises(ScenarioError):
        _play(backend, tmp_path, session_deadline_sec=time.monotonic() + 60.0)

    assert backend.served == 0


def test_a_hang_that_outlives_the_budget_stops_for_the_budget_in_both_backends(tmp_path):
    """The double and production have to name the same cause when both gates trip.

    Production tests the session budget at the top of every poll iteration,
    before the hard-timeout gate, so a hang whose budget ran out reports the
    budget rather than a reap. A double that answered the other way would make
    a scenario about a hang eating the next attempt's budget -- the case the
    scenario format exists to express -- assert the wrong stop reason.
    """
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        SESSION_TIME_EXHAUSTED_RETURNCODE,
        run_with_session_kill,
    )

    hard_timeout = 0.4
    produced = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=hard_timeout,
        session_deadline_sec=time.monotonic() + hard_timeout,
    )

    backend = _backend(LaunchAttempt(outcome=HANG, duration_sec=30.0))
    with installed_clock(backend.clock):
        scripted = _play(
            backend,
            tmp_path,
            timeout=hard_timeout,
            session_deadline_sec=backend.clock.deadline_in(hard_timeout),
        )

    assert produced.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert scripted.returncode == produced.returncode


def test_a_scenario_that_names_neither_spelling_is_rejected():
    with pytest.raises(ScenarioError):
        LaunchScenario.from_dict({"name": "empty"})


def test_a_scenario_file_round_trips(tmp_path):
    import json

    path = tmp_path / "round.json"
    path.write_text(json.dumps({"name": "f", "blockers": ["import"]}), encoding="utf-8")

    scenario = LaunchScenario.from_file(path)

    assert scenario.name == "f"
    assert [a.outcome for a in scenario.attempts] == [STAGE_FAILED, READY]


@pytest.mark.asyncio
async def test_a_completed_specialist_writes_the_done_file_the_runner_reads(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=COMPLETED, done={"status": "patched"}),))

    result = await scripted.run(task_id="spec-1", workspace=tmp_path / "ws")

    assert result.done_payload == {"status": "patched"}
    assert (tmp_path / "ws" / "specialist_done.json").exists()
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_an_empty_completion_is_a_done_file_with_nothing_in_it(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=EMPTY),))

    result = await scripted.run(task_id="spec-1", workspace=tmp_path / "ws")

    assert result.done_payload == {}
    assert (tmp_path / "ws" / "specialist_done.json").exists()


@pytest.mark.asyncio
async def test_a_crash_leaves_no_done_file_for_the_runner_to_find(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=CRASHED),))

    result = await scripted.run(task_id="spec-1", workspace=tmp_path / "ws")

    assert result.done_payload is None
    assert not (tmp_path / "ws" / "specialist_done.json").exists()
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_a_hang_reports_a_stale_heartbeat_and_never_wrote_one(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=HUNG),))

    result = await scripted.run(task_id="spec-1", workspace=tmp_path / "ws")

    assert result.stale_heartbeat
    assert result.exit_code is None
    assert not (tmp_path / "ws" / "heartbeat.json").exists()


@pytest.mark.asyncio
async def test_a_retry_under_a_new_task_id_is_visible_as_a_retry(tmp_path):
    scripted = ScriptedSpecialist(steps=(SpecialistStep(outcome=CRASHED), SpecialistStep(outcome=COMPLETED)))

    await scripted.run(task_id="spec-1", workspace=tmp_path / "spec-1")
    await scripted.run(task_id="spec-2", workspace=tmp_path / "spec-2")

    assert scripted.task_ids == ["spec-1", "spec-2"]
    assert scripted.retried


def test_the_virtual_clock_moves_only_when_something_charges_it():
    clock = VirtualClock()
    start_monotonic = clock.monotonic()
    start_wall = clock.wall()

    clock.advance(120.0)
    clock.mark("boot")
    clock.advance(-5.0)

    # Both readings move by the same amount, so a watchdog arming on monotonic
    # time and an artifact stamped with wall time agree about the same step.
    assert clock.monotonic() - start_monotonic == pytest.approx(120.0)
    assert clock.wall() - start_wall == pytest.approx(120.0)
    assert clock.marks == ((120.0, "boot"),)
