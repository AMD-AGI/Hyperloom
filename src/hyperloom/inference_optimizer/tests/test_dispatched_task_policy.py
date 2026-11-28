# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline against an open bring-up round, where an agent delegates one and where a queued one is dispatched."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session import paths
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.round_store import BOOTED, RoundStore
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec

from .conftest import silent_backends

_ORIGIN = {"phase": "EXPLORE", "macro_cycle": 0, "tick": 0}
_ROUND_RULE = "enablement_round_in_flight"


def _runner(tmp_path: Path, monkeypatch) -> SubAgentRunner:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    sub = SubAgentRunner(
        SqliteLeaseBackend(db),
        TaskRegistry(db, dispatch_origin_provider=lambda: _ORIGIN),
        RoundStore(db),
        session_dir=sd,
    )
    sub.shared_state = SharedState.load_or_init(sd)
    return sub


def _record_runs(sub: SubAgentRunner, kind: str) -> dict:
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor(kind, _stub)
    return executed


async def _open_round(rounds, *, holder_task_id: str, round_id: str = "round-7") -> None:
    opened = await rounds.open(round_id, holder_task_id=holder_task_id, now_unix=1.0, lease_sec=600.0)
    assert opened.ok


@pytest.fixture
async def coord(session_dir):
    c = Coordinator(session_dir, backends=silent_backends())
    c.shared_state.phase = "PRELUDE"
    yield c
    await c.stop()


def _baseline_delegate() -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "baseline", "params": {}})


async def _denied_rules(c: Coordinator) -> list[str]:
    observed = await c.bus.tail(topic="observation", n=50)
    return [m.payload["rule"] for m in observed if m.payload.get("kind") == "policy_denied"]


# -- At dispatch ------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_baseline_queued_before_the_round_opened_is_cancelled_at_dispatch_and_never_runs(tmp_path, monkeypatch):
    """The round holds no lane a baseline needs, so the dispatch is where a baseline already queued is stopped."""
    sub = _runner(tmp_path, monkeypatch)
    executed = _record_runs(sub, "baseline")
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="second-bring-up"),
    )
    await _open_round(sub.rounds, holder_task_id="spec-abc")

    res = await sub.run_task(task)

    assert res.state == "cancelled"
    assert res.error_class == "policy_enablement_round_in_flight"
    assert "spec-abc" in (res.error or "")
    assert executed["ran"] is False
    updated = await sub.tasks.get(task.task_id)
    assert updated.state == "cancelled"
    terminal = updated.history[-1]["evidence"]
    assert terminal["reason"] == "policy_denied"
    assert terminal["rule"] == _ROUND_RULE
    assert terminal["outcome"]["error_class"] == "policy_enablement_round_in_flight"


@pytest.mark.asyncio
async def test_a_baseline_runs_once_the_round_that_refused_it_has_settled(tmp_path, monkeypatch):
    """The store is read at the dispatch, not carried from an earlier pass."""
    sub = _runner(tmp_path, monkeypatch)
    executed = _record_runs(sub, "baseline")
    await _open_round(sub.rounds, holder_task_id="spec-abc")
    refused, _ = await sub.tasks.create_or_return_existing(TaskSpec(kind="baseline", params={}, idempotency_key="a"))
    assert (await sub.run_task(refused)).state == "cancelled"

    settled = await sub.rounds.settle(await sub.rounds.get("round-7"), outcome=BOOTED, now_unix=2.0)
    assert settled.ok
    admitted, _ = await sub.tasks.create_or_return_existing(TaskSpec(kind="baseline", params={}, idempotency_key="b"))
    res = await sub.run_task(admitted)

    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatch_denied_baseline_is_not_charged_to_baseline_failures(coord):
    """A baseline refused at dispatch measured nothing, so repeated refusals cannot stop the run as baseline_failed."""
    await _open_round(coord.rounds, holder_task_id="spec-abc")
    for i in range(3):
        task, _ = await coord.tasks.create_or_return_existing(
            TaskSpec(kind="baseline", params={}, idempotency_key=f"denied-baseline-{i}")
        )
        result = await coord.dispatcher.run_task_registered(task)
        await coord.dispatcher.reap_dispatched_task(task, result)
        assert result.state == "cancelled"
        assert result.error_class == "policy_enablement_round_in_flight"
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    state = coord.shared_state
    assert state.baseline_failure_streak == 0
    assert state.baseline_total_failures == 0
    assert state.stop_reason == ""
    assert not state.last_action_failures


@pytest.mark.asyncio
async def test_a_refused_baseline_gives_back_the_lanes_it_took(coord):
    """The refusal comes after the lanes are taken, so it is also what releases them."""
    await _open_round(coord.rounds, holder_task_id="spec-abc")
    lanes, ttl = coord.dispatcher.registry_lanes_ttl("baseline")
    assert lanes
    task, _ = await coord.tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="held-lanes", requires_lanes=lanes, lease_ttl_sec=ttl)
    )

    result = await coord.dispatcher.run_task_registered(task)

    assert result.state == "cancelled"
    assert coord.locks.lanes_by_task_sync() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        pytest.param("roofline", {"reason": "prelude_bootstrap"}, id="internal_roofline"),
        pytest.param(
            "recover",
            {"reason": "gpu_memory_leaked", "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]}},
            id="recover",
        ),
        pytest.param("framework_agent", {"candidate": {"repo": "x/y", "pr_number": 1}}, id="internal_framework_agent"),
        pytest.param("integrate_patch", {"specialist_task_id": "spec-ok", "apply_only": True}, id="integrate_patch"),
        pytest.param("specialist", {"enablement": True, "domain": "enablement_specialist"}, id="the_rounds_specialist"),
    ],
)
async def test_every_other_kind_runs_while_a_round_is_open(tmp_path, monkeypatch, kind, params):
    """The round keeps a second bring-up off the machine and nothing else: its own legs are dispatched under it."""
    sub = _runner(tmp_path, monkeypatch)
    executed = _record_runs(sub, kind)
    await _open_round(sub.rounds, holder_task_id="someone-else")
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(kind=kind, params=params, idempotency_key=f"queued-{kind}"),
    )

    res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatched_revalidation_holding_its_own_round_runs(tmp_path, monkeypatch):
    """The revalidation baseline executes rather than being cancelled at dispatch.

    A revalidation only exists once a KEEP measured throughput, and a denial
    there cancels the row instead of deferring it. What admits it is the round
    it opened: the store names this task the holder, and the holder is not
    refused by its own acquire.
    """
    sub = _runner(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.baseline_tput = 1000.0
    executed = _record_runs(sub, "baseline")
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(
            kind="baseline",
            params={"reason": "enablement_eval_revalidation"},
            idempotency_key="enablement-revalidation",
        ),
    )
    await _open_round(sub.rounds, holder_task_id=task.task_id, round_id="revalidation-1")

    result = await sub.run_task(task)

    assert result.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_killed_running_task_keeps_its_result(tmp_path, monkeypatch):
    """A task cancelled mid-flight still returns its executor result."""
    sub = _runner(tmp_path, monkeypatch)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(
            kind="report",
            params={},
            idempotency_key="k-kill-midflight",
        ),
    )

    async def _stub(ctx):
        # Simulate a kill landing while the executor is still running.
        await sub.tasks.transition(ctx.task.task_id, "cancelled", evidence={"reason": "killed"})
        return {"produced": "work"}

    sub.register_executor("report", _stub)

    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result == {"produced": "work"}
    updated = await sub.tasks.get(task.task_id)
    assert updated.state == "cancelled"


# -- At the intent ----------------------------------------------------------
@pytest.mark.asyncio
async def test_a_delegated_baseline_is_denied_while_a_round_holds_the_machine(coord):
    """A second bring-up fights the first for the same cards and ports."""
    await _open_round(coord.rounds, holder_task_id="spec-abc")

    await coord.router.handle_intent("orchestration", _baseline_delegate())

    assert await _denied_rules(coord) == [_ROUND_RULE]
    assert await coord.tasks.by_state("queued") == []
    (denial,) = [
        m.payload for m in await coord.bus.tail(topic="observation", n=50) if m.payload.get("kind") == "policy_denied"
    ]
    assert "spec-abc" in denial["reason"]


@pytest.mark.asyncio
async def test_the_round_is_what_a_baseline_is_told_in_a_phase_that_offers_none(coord):
    """ENABLEMENT offers no baseline, and the round is the refusal that holds wherever one is offered."""
    coord.shared_state.phase = "ENABLEMENT"
    await _open_round(coord.rounds, holder_task_id="spec-abc")

    await coord.router.handle_intent("orchestration", _baseline_delegate())

    assert await _denied_rules(coord) == [_ROUND_RULE]


@pytest.mark.asyncio
async def test_a_round_opened_after_the_ticks_repair_pass_denies_in_the_same_tick(coord):
    """The pass at the top of the tick is not where the rule gets its answer: the round store is."""
    await coord.reconciler.run(time.time())
    await _open_round(coord.rounds, holder_task_id="spec-abc")

    await coord.router.handle_intent("orchestration", _baseline_delegate())

    assert await _denied_rules(coord) == [_ROUND_RULE]
    assert await coord.tasks.by_state("queued") == []


@pytest.mark.asyncio
async def test_a_round_settled_after_the_ticks_repair_pass_denies_nothing_in_the_same_tick(coord):
    """The other half of the same read: a baseline is not refused for a round that ended during the turn."""
    await _open_round(coord.rounds, holder_task_id="spec-abc")
    await coord.reconciler.run(time.time())
    settled = await coord.rounds.settle(await coord.rounds.get("round-7"), outcome=BOOTED, now_unix=2.0)
    assert settled.ok

    await coord.router.handle_intent("orchestration", _baseline_delegate())

    assert await _denied_rules(coord) == []
    assert [task.kind for task in await coord.tasks.by_state("queued")] == ["baseline"]


@pytest.mark.asyncio
async def test_an_established_anchor_alone_does_not_refuse_a_baseline(coord):
    """Re-measuring a reference the run already has is a workflow opinion.

    The rule is about the cards and ports a live round is using, not about how
    many baselines a session may take, so a session that already has its figure
    can still take another one while nothing holds the machine.
    """
    coord.shared_state.baseline_tput = 1000.0

    await coord.router.handle_intent("orchestration", _baseline_delegate())

    assert _ROUND_RULE not in await _denied_rules(coord)
    assert [task.kind for task in await coord.tasks.by_state("queued")] == ["baseline"]
