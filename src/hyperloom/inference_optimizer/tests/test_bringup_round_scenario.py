# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A whole bring-up round, played in this process on a clock nobody waits for."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.enablement.build import EnablementBuild
from hyperloom.orchestrator.enablement.lane import EnablementLane
from hyperloom.orchestrator.enablement.params import EnablementParams
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.phases.machine_state import PHASE_ENABLEMENT
from hyperloom.orchestrator.rehearsal import (
    boot_log_for,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.state.round_store import (
    FAILED,
    SETTLED,
    RoundRequest,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry

#: The round: two blockers peeled one per attempt, then a boot that serves.
#: Written as a stack rather than three logs because what is under test is that
#: the round keeps going, not the wording of any one failure.


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
    """A session root and a round workspace."""
    session = tmp_path / "session"
    session.mkdir()
    slot = tmp_path / "round"
    slot.mkdir()
    return session, slot


@pytest.fixture
def registry(tmp_path):
    """A real ``TaskRegistry``, round store and lock manager over one database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield TaskRegistry(db), RoundStore(db), SqliteLeaseBackend(db)
    db.close()


def _lane(session: Path, tasks: TaskRegistry, rounds: RoundStore, launch_log: str):
    """Build the collaborator surface the enablement lane runs against.

    Args:
        session: The session root.
        tasks: The real task registry rows are created in.
        rounds: The real round store the lane acquires from.
        launch_log: The failure the round is repairing.

    Returns:
        A shim carrying the real ``EnablementLane`` methods.
    """
    state = types.SimpleNamespace(
        framework="vllm",
        model_name="scripted/Model",
        model_path="",
        gpu_type="mi300x",
        tp=1,
        max_model_len=0,
        enablement_mode="all",
        enablement=EnablementRound(launch_log=launch_log),
        baseline_tput=0.0,
        baseline_failure_streak=1,
        phase=PHASE_ENABLEMENT,
        macro_cycle=0,
        tick=0,
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
        run_deadline=None,
        warm_specialist_params=_noop,
        bus=types.SimpleNamespace(record_observation=_noop),
        state=types.SimpleNamespace(pending_proposals={}),
        _read_enablement_source_context=lambda _sig: "",
        _derive_checkpoint_weight_facts=lambda _log: "",
        framework_authoring_lanes_ttl=lambda _params, *, base_ttl_sec: (["research_lane"], base_ttl_sec),
        time_budget_denial_for_action=lambda _action: None,
        _enqueue_on_validated_board=_noop,
        # Attributes exposed by CoordinatorCollaborator properties; set directly
        # on the shim since property descriptors don't apply to SimpleNamespace.
        knowledge_plane=None,
        recipe_kb=None,
    )
    for owner, name in (
        (EnablementParams, "build_enablement_specialist_params"),
        (EnablementParams, "_discover_enablement_candidate_refs"),
        (EnablementLane, "_maybe_record_enablement_human_review"),
        (EnablementLane, "maybe_rearm_enablement"),
        (EnablementLane, "_maybe_enqueue_enablement_specialist"),
        (EnablementLane, "enablement_admitted"),
        (EnablementLane, "_check_argv_terminal"),
        (EnablementLane, "_check_environment_terminal"),
        (EnablementLane, "_environment_verdict"),
        (EnablementLane, "enablement_in_flight"),
        (EnablementLane, "_round_has_live_work"),
        (EnablementLane, "_open_authoring_round"),
        (EnablementLane, "_renew_enablement_round"),
        (EnablementLane, "settle_enablement_round"),
        (EnablementLane, "close_lane_event"),
        (EnablementBuild, "maybe_enqueue_specialist_requested_build"),
        (EnablementBuild, "maybe_escalate_to_targeted_build"),
    ):
        setattr(shim, name, types.MethodType(getattr(owner, name), shim))
    # Collaborator access goes through self._coord, so route it back to the shim.
    shim._coord = types.SimpleNamespace(
        run_deadline=None,
        enablement_params=shim,
        enablement_build=shim,
        specialist_dispatch=shim,
        gpu_lanes=shim,
    )
    return shim


#: A round that never gets anywhere: the same wall, attempt after attempt, and
#: no clean boot at the end. Written as explicit attempts rather than a blocker
#: stack because the point is that nothing is peeled.

#: The same wall, alternating between one the launcher rejected on its own
#: arguments and one the engine hit after doing real work. Each class has its
#: own terminal, and neither sequence on its own reaches three of a kind.


async def _seed_failed_rounds(rounds: RoundStore, n: int) -> None:
    """Settle N FAILED rounds into the ledger as precondition for cap tests."""

    for i in range(n):
        rid = f"seed-failed-{i:03d}"
        holder = f"holder-{i:03d}"
        now = float(i)
        await rounds.open(rid, RoundRequest(now_unix=now, request_id=rid), holder_task_id=holder, lease_sec=3600.0)
        await rounds.settle(
            await rounds.get(rid), RoundRequest(now_unix=now + 1.0, request_id=f"settle-{rid}"), outcome=FAILED
        )


@pytest.mark.asyncio
async def test_the_enablement_attempt_cap_stops_a_round_that_keeps_asking(
    round_slot,
    registry,
    monkeypatch,
):
    """Consecutive rounds that repair nothing exhaust the cap and stop the run."""
    import hyperloom.agents.framework.sources as sources

    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node
    from hyperloom.orchestrator.phases.machine_state import ENABLEMENT_MAX_ATTEMPTS as _ENABLEMENT_MAX_ATTEMPTS

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])
    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, _slot = round_slot
    tasks, rounds, _locks = registry
    launch_log = boot_log_for(LadderStage.ENGINE_INIT)

    # One under the cap: the lane still dispatches.
    await _seed_failed_rounds(rounds, _ENABLEMENT_MAX_ATTEMPTS - 1)
    lane = _lane(session, tasks, rounds, launch_log)
    assert await lane._maybe_enqueue_enablement_specialist()
    assert not lane.shared_state.stop_reason
    # That round has to end before the next one can ask for the machine, and a
    # revert is how a round that repaired nothing ends.
    await lane.maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert await rounds.held() is None

    # Now at the cap: the lane must stop.
    assert await rounds.consecutive_stalled() == _ENABLEMENT_MAX_ATTEMPTS
    assert await lane._maybe_enqueue_enablement_specialist() == ""
    assert lane.shared_state.stop_reason == "enablement_attempts_exhausted"


@pytest.mark.asyncio
async def test_an_advancing_round_does_not_exhaust_the_cap(
    round_slot,
    registry,
    monkeypatch,
):
    """An advancing round resets the streak so a bring-up outlives the cap."""
    import hyperloom.agents.framework.sources as sources

    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node
    from hyperloom.orchestrator.phases.machine_state import ENABLEMENT_MAX_ATTEMPTS as _ENABLEMENT_MAX_ATTEMPTS

    monkeypatch.setattr(sources, "enumerate_candidates", lambda _request: [])
    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, _slot = round_slot
    tasks, rounds, _locks = registry
    launch_log = boot_log_for(LadderStage.ENGINE_INIT)

    # Seed cap - 1 failed rounds so the next dispatch brings us to cap - 1 stalled.
    await _seed_failed_rounds(rounds, _ENABLEMENT_MAX_ATTEMPTS - 1)
    lane = _lane(session, tasks, rounds, launch_log)
    assert await lane._maybe_enqueue_enablement_specialist()
    assert not lane.shared_state.stop_reason

    # An advancing result settles the open round as ADVANCED, resetting the streak.
    await lane.maybe_rearm_enablement(
        {"enablement": True, "status": "advanced", "patches_applied": ["vllm/platforms/rocm.py"]}
    )
    assert await rounds.held() is None
    assert await rounds.consecutive_stalled() == 0

    # The advance cleared the failure backstop; a new baseline failure lets the
    # lane dispatch again, and the run is still live.
    lane.shared_state.baseline_failure_streak = 1
    assert await lane._maybe_enqueue_enablement_specialist()
    assert not lane.shared_state.stop_reason


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
    lane = _lane(session, tasks, rounds, boot_log_for(LadderStage.ENGINE_INIT))
    assert await lane._maybe_enqueue_enablement_specialist()
    held = await rounds.held()
    assert held is not None

    await lane.settle_enablement_round(FAILED, reason="")

    assert await rounds.held() is None
    row = await rounds.get(held.round_id)
    assert (row.state, row.outcome) == (SETTLED, FAILED)
