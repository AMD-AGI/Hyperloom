# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A whole bring-up round, played in this process on a clock nobody waits for.

The round under test is the one that runs when a baseline will not boot: an
attempt fails, the failure is observed and recorded, the lane opens a repair
task against the registry, the gate re-validates it before dispatch, the
specialist answers, and the next attempt gets further. Every one of those steps
had its own test and none of them had a test of the sequence, because the
sequence needed a GPU.

Here the sequence is a scenario: three attempts peeling two serial blockers,
answered by scripted backends, charged to a virtual clock. What is asserted is
what a later reader of the session has to be able to find -- one observation
artifact per attempt, under the path the packaging reads, each naming a stage
further up the ladder than the last.
"""

from __future__ import annotations

import time
import types
from pathlib import Path

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from hyperloom.orchestrator.actions.executors import baseline as bl
from hyperloom.orchestrator.bringup import load_boot_observation
from hyperloom.orchestrator.bringup.reconcile import Reconciler
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.enablement.build import EnablementBuild
from hyperloom.orchestrator.enablement.lane import EnablementLane
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.rehearsal import (
    COMPLETED,
    CRASHED,
    STAGE_FAILED,
    LaunchScenario,
    ScriptedLaunchBackend,
    ScriptedSpecialist,
    SpecialistStep,
    boot_log_for,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.state.round_store import (
    FAILED,
    SETTLED,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry

#: The round: two blockers peeled one per attempt, then a boot that serves.
#: Written as a stack rather than three logs because what is under test is that
#: the round keeps going, not the wording of any one failure.
_SCENARIO = {
    "name": "two-blockers-then-boot",
    "blockers": [
        {"stage": "argv_parse", "duration_sec": 45.0},
        {"stage": "config_validate", "duration_sec": 180.0},
    ],
    "clean_after": True,
}


def _ctx(task_id: str) -> RunnerContext:
    from hyperloom.orchestrator.state.task_registry import Task

    return RunnerContext(
        task=Task(
            task_id=task_id,
            kind="baseline",
            state="running",
            params={},
            idempotency_key=task_id,
            requires_lanes=(),
        ),
        lease=None,
        extra={},
    )


@pytest.fixture
def round_slot(tmp_path, monkeypatch):
    """A session root, a round workspace, and a baseline with no salvage pass.

    The harvest/workspace helpers are stubbed because they scan for artifacts a
    real Magpie leaves and a scripted attempt does not; what this test reads is
    the boot observation, which the executor writes itself.
    """
    session = tmp_path / "session"
    session.mkdir()
    slot = tmp_path / "round"
    slot.mkdir()
    monkeypatch.setattr(bl, "harvest_leaked_artifacts", lambda *a, **k: [])
    monkeypatch.setattr(bl, "snapshot_workspaces", lambda *a, **k: set())
    monkeypatch.setattr(bl, "select_run_workspace", lambda *a, **k: None)
    return session, slot


@pytest.fixture
def registry(tmp_path):
    """A real ``TaskRegistry``, round store and lock manager over one database.

    One database on purpose: the acquire and the holder's task row commit in a
    single transaction, so a test that gave them separate connections would not
    be exercising the thing that makes them atomic.
    """
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield TaskRegistry(db), RoundStore(db), ResourceLockManager(SqliteLeaseBackend(db))
    db.close()


async def _bringup_attempt(session: Path, slot: Path, *, task_id: str) -> dict:
    """Run one bring-up attempt through the real baseline executor.

    Args:
        session: The session root.
        slot: The round workspace the attempt runs in.
        task_id: Task id the attempt runs under.

    Returns:
        dict: The round result, carrying the boot observation's path.
    """
    executor = bl.BaselineExecutor(session_dir=session, magpie_python="/usr/bin/python3")
    config = session / "bench.yaml"
    config.write_text("benchmark:\n  framework: vllm\n  model: scripted/Model\n", encoding="utf-8")
    return await executor._run_single_benchmark(
        config_path=config,
        output_dir=slot,
        timeout_sec=1800,
        override_result_dir=None,
        resolved_model="scripted/Model",
        materialized_config_path=config,
        inferencex_path="",
        effective_extra_server_args="",
        params={"framework": "vllm"},
        ctx=_ctx(task_id),
    )


def _lane(session: Path, tasks: TaskRegistry, rounds: RoundStore, launch_log: str, *, attempts: int):
    """Build the collaborator surface the enablement lane runs against.

    Only the coordinator attributes the lane actually reads are supplied; the
    lane's own methods are the real ones, so admission, the idempotency key and
    the registry row are all production behaviour.

    Args:
        session: The session root.
        tasks: The real task registry rows are created in.
        rounds: The real round store the lane acquires from.
        launch_log: The failure the round is repairing.
        attempts: How many enablement attempts have already been made.

    Returns:
        A shim carrying the real ``EnablementLane`` methods.
    """
    state = types.SimpleNamespace(
        framework="vllm",
        model_name="scripted/Model",
        model_path="",
        gpu_type="mi300x",
        enablement_mode="all",
        enablement=EnablementRound(attempts=attempts, launch_log=launch_log),
        baseline_tput=0.0,
        baseline_failure_streak=1,
        tick=attempts,
        stop_reason="",
        save=lambda *a, **k: None,
    )
    state.set_stop_reason = lambda value, **_kw: setattr(state, "stop_reason", str(value or ""))

    async def _noop(*_a, **_k):
        return None

    shim = types.SimpleNamespace(
        shared_state=state,
        tasks=tasks,
        rounds=rounds,
        session_dir=str(session),
        _run_deadline=None,
        _warm_specialist_params=_noop,
        _record_observation=_noop,
        action_registry=ACTION_CATALOGUE,
        state=types.SimpleNamespace(pending_proposals={}),
        _read_enablement_source_context=lambda _sig: "",
        _derive_checkpoint_weight_facts=lambda _log: "",
        _framework_gpu_params=lambda: {},
        _framework_authoring_lanes_ttl=lambda _params, *, base_ttl_sec: (["research_lane"], base_ttl_sec),
        _time_budget_denial_for_action=lambda _action: None,
    )
    for owner, name in (
        (Coordinator, "_build_enablement_specialist_params"),
        (Coordinator, "_discover_enablement_candidate_refs"),
        (Coordinator, "_registry_lanes_ttl"),
        (Coordinator, "_maybe_record_enablement_human_review"),
        (Coordinator, "_maybe_rearm_enablement"),
        (EnablementLane, "_maybe_enqueue_enablement_specialist"),
        (EnablementLane, "_refused_argv_is_terminal"),
        (EnablementLane, "_environment_fault_is_terminal"),
        (EnablementLane, "_environment_verdict"),
        (EnablementLane, "_enablement_in_flight"),
        (EnablementLane, "_round_has_live_work"),
        (EnablementLane, "_open_authoring_round"),
        (EnablementLane, "_renew_enablement_round"),
        (EnablementLane, "_settle_enablement_round"),
        (EnablementLane, "_charge_round_observation"),
        (EnablementBuild, "_maybe_enqueue_specialist_requested_build"),
        (EnablementBuild, "_maybe_escalate_to_targeted_build"),
    ):
        setattr(shim, name, types.MethodType(getattr(owner, name), shim))
    return shim


def _advanced_by(seconds: float):
    """A ``time.time`` reading ``seconds`` later than the one in force now.

    The round's lifecycle is minutes wide and nothing in this test waits, so
    the ticks are separated by moving the reading rather than the wall.

    Args:
        seconds: How much later the next reading is.

    Returns:
        Callable[[], float]: The replacement reading.
    """
    later = time.time() + float(seconds)
    return lambda: later


@pytest.mark.asyncio
async def test_a_bringup_round_peels_two_blockers_and_lands_one_observation_per_attempt(
    round_slot,
    registry,
    launch_backend,
    virtual_clock,
    monkeypatch,
):
    """Three attempts, three artifacts, a ladder that only ever goes up."""
    import hyperloom.agents.framework.sources as sources

    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])
    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, slot = round_slot
    tasks, rounds, _locks = registry
    clock = virtual_clock
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO), clock=clock)
    launch_backend(launches)
    # The first dispatch dies without a done-file and the round retries it; the
    # retry is a fresh row, so the specialist sees a task id no earlier dispatch
    # used. That, not a counter, is what tells a retry from a resumption.
    specialists = ScriptedSpecialist(
        steps=(
            SpecialistStep(name="crash", outcome=CRASHED, duration_sec=60.0),
            SpecialistStep(name="repair", outcome=COMPLETED, done={"status": "patched"}, duration_sec=240.0),
        ),
        clock=clock,
    )

    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=session,
        shared_state=None,
        strict_paths=False,
    )

    observations: list = []
    dispatched: list[str] = []

    for tick in range(len(launches.scenario.attempts)):
        result = await _bringup_attempt(session, slot, task_id=f"baseline-{tick}")
        loaded = load_boot_observation(result.get("boot_observation_path"))
        assert loaded.observation is not None, loaded.degraded
        observations.append(loaded.observation)
        if loaded.observation.booted:
            break

        # The excerpt is what the round recorded off the server's own log, and
        # the only evidence a wrapper that printed nothing leaves behind.
        excerpt = loaded.observation.excerpt
        # What the Coordinator does first on every tick. The previous round's
        # specialist crashed without settling anything, and the lane will not
        # open a second round while the first is still held: the repair pass is
        # what ends it, and it runs before anything is admitted.
        await Reconciler(
            rounds=rounds,
            tasks=tasks,
            locks=ResourceLockManager(SqliteLeaseBackend(rounds.db)),
            shared_state=None,
            terminal_holder_cap_sec=0.0,
        ).run(time.time())
        # A confirmed kill keeps excluding until the reap grace passes: the
        # kernel has not torn the address space down when the signal returns,
        # and the cards outlive the process that held them by longer still. The
        # next tick is therefore on the far side of that grace.
        monkeypatch.setattr(time, "time", _advanced_by(1.0))
        lane = _lane(session, tasks, rounds, excerpt.text if excerpt is not None else "", attempts=tick)
        task_id = await lane._maybe_enqueue_enablement_specialist()
        assert task_id, "the lane opened no repair task for a boot that failed"
        row = await tasks.get(task_id)
        # The gate re-reads the persisted row before anything dispatches it,
        # which is the check a forged database row has to get past.
        gate.validate_dispatched_task(row.kind, row.params)
        dispatched.append(task_id)

        workspace = session / "runs" / "specialist" / task_id
        outcome = await specialists.run(task_id=task_id, workspace=workspace)
        await tasks.transition(task_id, "running")
        await tasks.transition(task_id, "succeeded" if outcome.done_payload else "failed")

    # Every attempt left exactly one artifact, at the path the packaging reads.
    artifacts = sorted((reports_dir(session) / "bringup").glob("*.json"))
    assert len(artifacts) == len(launches.scenario.attempts)
    assert launches.served == len(launches.scenario.attempts)

    # And the round got further each time, ending on a server that answered.
    stages = [o.stage_failed or LadderStage.ACCURACY_OK for o in observations]
    assert stages == sorted(stages), stages
    assert stages[0] == LadderStage.ARGV_PARSE
    assert stages[1] == LadderStage.CONFIG_VALIDATE
    assert observations[-1].booted

    # Two repair dispatches under two different ids: the lane minted a fresh
    # idempotency key for the retry rather than reopening the crashed task,
    # which is the only difference between a retry and a resumption a consumer
    # outside the lane can see.
    assert len(dispatched) == 2
    assert dispatched[0] != dispatched[1], dispatched

    # Every declared step was charged to the one clock both doubles share, and
    # none of it was spent waiting.
    assert clock.elapsed == pytest.approx(45.0 + 180.0 + 30.0 + 60.0 + 240.0)


#: A round that never gets anywhere: the same wall, attempt after attempt, and
#: no clean boot at the end. Written as explicit attempts rather than a blocker
#: stack because the point is that nothing is peeled.
_UNCLEARED = {
    "name": "a-wall-that-never-moves",
    "attempts": [
        {"name": f"attempt-{index}", "outcome": STAGE_FAILED, "stage": "engine_init", "duration_sec": 240.0}
        for index in range(4)
    ],
}

#: The same wall, alternating between one the launcher rejected on its own
#: arguments and one the engine hit after doing real work. Each class has its
#: own terminal, and neither sequence on its own reaches three of a kind.
_ALTERNATING = {
    "name": "argument-then-engine-then-argument",
    "attempts": [
        {"name": "bad-arg", "outcome": STAGE_FAILED, "stage": "argv_parse", "duration_sec": 5.0},
        {"name": "engine", "outcome": STAGE_FAILED, "stage": "engine_init", "duration_sec": 240.0},
        {"name": "bad-arg-again", "outcome": STAGE_FAILED, "stage": "argv_parse", "duration_sec": 5.0},
    ],
}


def _baseline_task(task_id: str):
    """Return the registry row a baseline attempt settles under."""
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(
        task_id=task_id,
        kind="baseline",
        state="running",
        params={"config_path": "baseline.yaml"},
        idempotency_key=task_id,
    )


@pytest.fixture
def coordinator(session_dir):
    """A real Coordinator over a temp session, with silent LLM backends."""
    from hyperloom.orchestrator.roles.base import Backend
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, ScriptedPlan

    backends: dict[str, Backend] = {
        name: MockBackend(ScriptedPlan(turns=[]), name=name) for name in ("orchestration", "critic", "robustness")
    }
    return Coordinator(session_dir, backends=backends)


async def _settle_failures(coordinator, session, slot, scenario_len: int) -> list[dict]:
    """Play every attempt and hand each failure to the writeback path.

    Args:
        coordinator: The Coordinator whose state the failures are charged to.
        session: The session root the executor writes observations under.
        slot: The round workspace each attempt runs in.
        scenario_len: How many attempts the scenario declares.

    Returns:
        list[dict]: The round result of every attempt that was played, in
        order; short of ``scenario_len`` when a stop reason was set first.
    """
    played: list[dict] = []
    for index in range(scenario_len):
        result = await _bringup_attempt(session, slot, task_id=f"baseline-{index}")
        played.append(result)
        await coordinator._handle_unpromotable_result(_baseline_task(f"baseline-{index}"), result)
        if coordinator.shared_state.stop_reason:
            break
    return played


@pytest.mark.asyncio
async def test_a_baseline_that_keeps_failing_reaches_the_prelude_terminal(
    coordinator,
    round_slot,
    launch_backend,
    virtual_clock,
):
    """A round with an open bring-up does not buy the baseline extra retries.

    The round is deliberately made to look maximally alive -- patches stacked,
    a specialist in flight, attempts on the board -- because that shape used to
    stand the terminal down, and a session in it dispatched baselines until the
    wall-clock ran out.
    """
    from hyperloom.orchestrator.phases import machine_state

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_UNCLEARED), clock=virtual_clock)
    launch_backend(launches)

    state = coordinator.shared_state
    state.enablement_mode = "all"
    state.enablement.attempts = 2
    state.enablement.kept_patches = ["srt/attention.py"]

    played = await _settle_failures(coordinator, session, slot, len(launches.scenario.attempts))

    assert state.baseline_failure_streak == 3
    assert state.stop_reason == "baseline_failed"
    assert machine_state.exit_terminal_prelude(state) == (
        "prelude_baseline_failed",
        {"baseline_failure_streak": 3},
    )
    # The fourth attempt was never played: the session stopped on the third.
    assert len(played) == 3
    assert launches.served == 3


@pytest.mark.asyncio
async def test_the_enablement_attempt_cap_stops_a_round_that_keeps_asking(
    round_slot,
    registry,
    monkeypatch,
):
    """The allowance is spent by counting, so a progressing round still ends."""
    import hyperloom.agents.framework.sources as sources

    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node
    from hyperloom.orchestrator.loop.coordinator import _ENABLEMENT_MAX_ATTEMPTS

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])
    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, _slot = round_slot
    tasks, rounds, _locks = registry
    launch_log = boot_log_for(LadderStage.ENGINE_INIT)

    # One under the cap still dispatches, so the cap is what stops the next one
    # rather than something else about a round this deep.
    lane = _lane(session, tasks, rounds, launch_log, attempts=_ENABLEMENT_MAX_ATTEMPTS - 1)
    assert await lane._maybe_enqueue_enablement_specialist()
    assert not lane.shared_state.stop_reason
    # That round has to end before the next one can ask for the machine, and a
    # revert is how a round that repaired nothing ends.
    await lane._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert await rounds.held() is None

    lane = _lane(session, tasks, rounds, launch_log, attempts=_ENABLEMENT_MAX_ATTEMPTS)
    assert await lane._maybe_enqueue_enablement_specialist() == ""
    assert lane.shared_state.stop_reason == "enablement_attempts_exhausted"
    assert int(lane.shared_state.enablement.attempts) == _ENABLEMENT_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_a_round_settles_when_the_caller_has_no_reason_to_give(
    round_slot,
    registry,
    monkeypatch,
):
    """Most settles carry a reason and some carry none; both end the round."""
    import hyperloom.agents.framework.sources as sources

    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])
    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, _slot = round_slot
    tasks, rounds, _locks = registry
    lane = _lane(session, tasks, rounds, boot_log_for(LadderStage.ENGINE_INIT), attempts=0)
    assert await lane._maybe_enqueue_enablement_specialist()
    held = await rounds.held()
    assert held is not None

    await lane._settle_enablement_round(FAILED, reason="")

    assert await rounds.held() is None
    row = await rounds.get(held.round_id)
    assert (row.state, row.outcome) == (SETTLED, FAILED)


@pytest.mark.asyncio
async def test_the_arg_error_terminal_survives_an_alternating_failure_sequence(
    coordinator,
    round_slot,
    launch_backend,
    virtual_clock,
):
    """Two rejected-argument launches end the run even when one boot separates them.

    The engine failure between them is not evidence that the arguments changed,
    so it must not clear what the first rejection recorded.
    """
    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_ALTERNATING), clock=virtual_clock)
    launch_backend(launches)

    played = await _settle_failures(coordinator, session, slot, len(launches.scenario.attempts))

    assert [r.get("error_class") for r in played] == [
        "fast_exit_arg_error",
        "subprocess_nonzero",
        "fast_exit_arg_error",
    ]
    state = coordinator.shared_state
    assert state.baseline_arg_error_streak == 2
    assert state.baseline_failure_streak == 1
    assert state.stop_reason == "baseline_arg_error"
