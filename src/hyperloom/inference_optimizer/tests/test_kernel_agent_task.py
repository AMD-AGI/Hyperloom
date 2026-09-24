# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The KERNEL_AGENT phase's work runs as one lane-holding ``kernel_agent`` task."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import CancelledError as FuturesCancelledError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    ACTION_CATALOGUE,
    COORDINATOR_INTERNAL_ACTIONS,
)
from hyperloom.orchestrator.actions.cancel_channel import cancel_scope_listener
from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_SWEEP, SharedState
from hyperloom.orchestrator.state.task_registry import Task

_KERNEL_AGENT_LANES = ("server_lifecycle", "workspace_mutation", "benchmark_lane")


@pytest.fixture
def coord(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
    }
    c = Coordinator(sd, backends=backends)

    async def _noop(*_args, **_kwargs):
        return None

    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.specialist_dispatch._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]
    c.shared_state.kernel_enabled = True
    yield c


def _arm_kernel_phase(st):
    """Park the session mid-KERNEL with plenty of phase and session time left."""
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=20)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=20)).timestamp()
    st.start_ts = (now - timedelta(minutes=30)).isoformat()
    st.max_minutes = 96 * 60


def _spend_the_phase_budget(st):
    spent = datetime.now(timezone.utc) - timedelta(hours=30)
    st.phase_started_ts = spent.isoformat()
    st.phase_started_unix = spent.timestamp()


async def _create_kernel_agent(c, *, key: str = "kernel_agent_c0"):
    task, _ = await c.tasks.create_or_return_existing(
        kind="kernel_agent",
        params={"from_phase": ps.PHASE_FRAMEWORK_AGENT},
        idempotency_key=key,
        requires_lanes=list(_KERNEL_AGENT_LANES),
        lease_ttl_sec=3600,
    )
    return task


def _blocking_executor(release: asyncio.Event, started: asyncio.Event | None = None):
    async def _run(_ctx: Any) -> dict[str, Any]:
        if started is not None:
            started.set()
        await release.wait()
        return {"status": "ok", "route": "test"}

    return _run


def _skip_phase_entry_effects(c, monkeypatch) -> None:
    async def _noop(**_kwargs):
        return None

    monkeypatch.setattr(c.phase_machine, "_on_phase_entered", _noop)


async def _settle(c, task_id: str) -> None:
    entry = c.dispatcher._inflight_actions.get(task_id)
    if entry is not None:
        await asyncio.wait_for(entry.atask, timeout=5.0)


def test_kernel_agent_is_catalogued_as_a_coordinator_internal_lane_holder():
    meta = ACTION_CATALOGUE["kernel_agent"]
    assert meta.requires_lanes == _KERNEL_AGENT_LANES
    assert "kernel_agent" in COORDINATOR_INTERNAL_ACTIONS
    assert "kernel_agent" in ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_KERNEL_AGENT]
    # Allowed in the phase so it survives the transition sweep, never proposable by a model.
    assert "kernel_agent" not in ps.allowed_actions_for(ps.PHASE_KERNEL_AGENT)


@pytest.mark.asyncio
async def test_entering_kernel_enqueues_one_lane_holding_task_and_returns(coord, monkeypatch):
    """The entry hook only enqueues; the phase's work never runs on the tick that entered it."""
    from hyperloom.orchestrator.phases import machine as phase_machine_mod

    c = coord
    st = c.shared_state
    _arm_kernel_phase(st)
    st.phase = ps.PHASE_FRAMEWORK_AGENT
    ran: list[Any] = []

    async def _must_not_run(ctx):
        ran.append(ctx)
        return {}

    c.sub.register_executor("kernel_agent", _must_not_run)
    monkeypatch.setattr(
        phase_machine_mod._phase_state,
        "compute_next_phase",
        lambda *_args, **_kwargs: (ps.PHASE_KERNEL_AGENT, "test_enter_kernel", {"source": "test"}),
    )

    await asyncio.wait_for(c._advance_phase_if_needed(), timeout=2.0)

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert ran == []
    queued = [t for t in await c.tasks.queued() if t.kind == "kernel_agent"]
    assert len(queued) == 1
    task = queued[0]
    assert tuple(task.requires_lanes) == _KERNEL_AGENT_LANES
    assert task.params["from_phase"] == ps.PHASE_FRAMEWORK_AGENT
    remaining = ps.phase_budget_remaining_seconds(st, budget_pct=c._phase_budget_pct)
    assert task.lease_ttl_sec == pytest.approx(remaining, abs=5.0)


@pytest.mark.asyncio
async def test_resumed_entry_reuses_a_live_task_and_replaces_a_settled_one(coord):
    c = coord
    st = c.shared_state
    _arm_kernel_phase(st)

    await c._on_enter_kernel(from_phase=ps.PHASE_FRAMEWORK_AGENT)
    first = [t for t in await c.tasks.queued() if t.kind == "kernel_agent"]
    await c._on_enter_kernel(from_phase="resume")
    assert [t.task_id for t in await c.tasks.queued() if t.kind == "kernel_agent"] == [first[0].task_id]

    await c.tasks.transition(first[0].task_id, "running")
    await c.tasks.transition(first[0].task_id, "failed", evidence={"reason": "dead_holder"})
    await c._on_enter_kernel(from_phase="resume")

    requeued = [t for t in await c.tasks.queued() if t.kind == "kernel_agent"]
    assert len(requeued) == 1
    assert requeued[0].task_id != first[0].task_id
    assert requeued[0].params["from_phase"] == "resume"


@pytest.mark.asyncio
async def test_the_pump_returns_while_the_kernel_agent_task_runs(coord):
    """A pump that joined the phase's whole pipeline would freeze every tick until it returned."""
    c = coord
    _arm_kernel_phase(c.shared_state)
    release, started = asyncio.Event(), asyncio.Event()
    c.sub.register_executor("kernel_agent", _blocking_executor(release, started))
    task = await _create_kernel_agent(c)

    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=2.0)
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert (await c.tasks.get(task.task_id)).state == "running"
    release.set()
    await _settle(c, task.task_id)
    assert (await c.tasks.get(task.task_id)).state == "succeeded"


@pytest.mark.parametrize(
    ("label", "arm", "reason"),
    [
        (
            "controller finished",
            lambda st: (
                setattr(st, "kernel_optimizer", "forge"),
                setattr(
                    st,
                    "kernel_rewrite_controller_result",
                    {"status": sorted(ps.CONTROLLER_TERMINAL_STATUSES)[0], "macro_cycle": 0},
                ),
            ),
            "kernel_controller_done",
        ),
        (
            "skip_to_sweep hint",
            lambda st: st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP),
            "kernel_no_more_leverage",
        ),
        (
            "idle streak",
            lambda st: (
                setattr(st, "kernel_idle_ticks", ps.KERNEL_IDLE_MAX_TICKS),
                setattr(st, "kernel_idle_since_unix", time.time() - ps.KERNEL_IDLE_MIN_SECONDS - 1.0),
            ),
            "kernel_no_more_leverage",
        ),
    ],
)
def test_kernel_agent_in_flight_blocks_every_leverage_exit(label, arm, reason):
    st = SharedState(session_id="s")
    _arm_kernel_phase(st)
    arm(st)

    exit_now = ps.exit_normal_kernel(st)
    assert exit_now is not None and exit_now[0] == reason, label
    assert ps.exit_normal_kernel(st, kernel_work_in_flight=True) is None, label

    inputs = ps.workflow_predicate_inputs(st, kernel_work_in_flight=True)
    assert inputs["pending_work"]["kernel_agent_in_flight"] is True
    assert ps.compute_next_phase(st, kernel_work_in_flight=True) is None, label
    assert ps.replay_next_phase(inputs) is None, label


def test_kernel_agent_in_flight_never_blocks_a_budget_exit():
    st = SharedState(session_id="s")
    _arm_kernel_phase(st)
    _spend_the_phase_budget(st)
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)

    exit_now = ps.exit_normal_kernel(st, kernel_work_in_flight=True)
    transition = ps.compute_next_phase(st, kernel_work_in_flight=True)

    assert exit_now is not None
    assert exit_now[0] in {"kernel_phase_budget_exhausted", "kernel_budget_cap"}
    assert transition is not None
    assert transition[1] == exit_now[0]
    assert ps.replay_next_phase(transition[2]["predicate_inputs"]) == transition


@pytest.mark.asyncio
async def test_kernel_holds_while_its_task_is_in_flight_and_leaves_once_it_settles(coord, monkeypatch):
    """The idle guard must not hand the GPUs to SWEEP while the kernel_agent task still holds them."""
    c = coord
    _skip_phase_entry_effects(c, monkeypatch)
    st = c.shared_state
    _arm_kernel_phase(st)
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
    task = await _create_kernel_agent(c)
    await c.tasks.transition(task.task_id, "running")

    for _ in range(ps.KERNEL_IDLE_MAX_TICKS * 5):
        await c._advance_phase_if_needed()
        st.kernel_idle_since_unix = datetime.now(timezone.utc).timestamp() - ps.KERNEL_IDLE_MIN_SECONDS * 10

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert st.kernel_idle_ticks == 0

    await c.tasks.transition(task.task_id, "succeeded", evidence={"result_keys": []})
    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP
    assert st.phase_history[-1]["reason"] == "kernel_no_more_leverage"


def _listening_executor(started: asyncio.Event):
    async def _run(_ctx: Any) -> dict[str, Any]:
        with cancel_scope_listener() as scope:
            started.set()
            while not scope.cancelled:
                await asyncio.sleep(0.01)
            raise FuturesCancelledError(scope.reason)

    return _run


@pytest.mark.asyncio
async def test_a_spent_phase_budget_stops_the_kernel_agent_and_leaves_kernel(coord, monkeypatch):
    """The transition barrier stops the running pipeline and frees its lanes before SWEEP starts."""
    c = coord
    _skip_phase_entry_effects(c, monkeypatch)
    st = c.shared_state
    _arm_kernel_phase(st)
    started = asyncio.Event()
    c.sub.register_executor("kernel_agent", _listening_executor(started))
    task = await _create_kernel_agent(c)
    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=2.0)
    await asyncio.wait_for(started.wait(), timeout=2.0)
    _spend_the_phase_budget(st)

    await asyncio.wait_for(c._advance_phase_if_needed(), timeout=30.0)

    assert st.phase == ps.PHASE_SWEEP
    assert st.phase_history[-1]["reason"] in {"kernel_phase_budget_exhausted", "kernel_budget_cap"}
    assert (await c.tasks.get(task.task_id)).state == "cancelled"
    assert not any((await c.locks.lane_holders()).values())


@pytest.mark.asyncio
async def test_a_running_kernel_agent_keeps_roofline_queued_until_it_returns(coord):
    """benchmark_lane conflicts with profile_lane, so no analysis shares the GPUs with the phase's pipeline."""
    c = coord
    _arm_kernel_phase(c.shared_state)
    release, started = asyncio.Event(), asyncio.Event()
    c.sub.register_executor("kernel_agent", _blocking_executor(release, started))
    rooflines: list[str] = []

    async def _roofline(ctx):
        rooflines.append(ctx.task.task_id)
        return {"status": "ok"}

    c.sub.register_executor("roofline", _roofline)
    agent = await _create_kernel_agent(c)
    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=2.0)
    await asyncio.wait_for(started.wait(), timeout=2.0)

    lanes, ttl = c._registry_lanes_ttl("roofline")
    roofline, _ = await c.tasks.create_or_return_existing(
        kind="roofline",
        params={"source": "coordinator_internal", "reason": "test"},
        idempotency_key="roofline-under-kernel-agent",
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
    )
    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=2.0)

    assert rooflines == []
    assert (await c.tasks.get(roofline.task_id)).state == "queued"

    release.set()
    await _settle(c, agent.task_id)
    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=5.0)

    assert rooflines == [roofline.task_id]
    assert (await c.tasks.get(roofline.task_id)).state == "succeeded"


@pytest.mark.asyncio
async def test_a_spent_session_cancels_the_running_kernel_agent(coord):
    c = coord
    st = c.shared_state
    _arm_kernel_phase(st)
    started = asyncio.Event()
    c.sub.register_executor("kernel_agent", _listening_executor(started))
    task = await _create_kernel_agent(c)
    await asyncio.wait_for(c.dispatcher._pump_dispatcher_once(), timeout=2.0)
    await asyncio.wait_for(started.wait(), timeout=2.0)

    st.max_minutes = 60
    st.elapsed_minutes = lambda **_kw: 60.0  # type: ignore[method-assign]
    assert st.session_budget_usable_sec() == 0.0
    await asyncio.wait_for(c.dispatcher._cancel_inflight_that_outlived_the_session(), timeout=30.0)
    await _settle(c, task.task_id)

    assert (await c.tasks.get(task.task_id)).state == "cancelled"
    assert not any((await c.locks.lane_holders()).values())


def test_the_time_budget_gate_admits_kernel_agent_while_one_baseline_round_fits(coord):
    c = coord
    st = c.shared_state
    st.baseline_runtime_sec = 600.0
    st.max_minutes = 120
    # 120-minute session: 120 s closing reserve, so 100 min spent leaves 18 usable minutes.
    st.elapsed_minutes = lambda **_kw: 100.0  # type: ignore[method-assign]
    assert c._time_budget_denial_for_action("kernel_agent") is None

    # 112 min spent leaves 6 usable minutes: not even the 10-minute baseline round fits.
    st.elapsed_minutes = lambda **_kw: 112.0  # type: ignore[method-assign]
    denied = c._time_budget_denial_for_action("kernel_agent")
    assert denied is not None and denied.rule == "time_budget"


def _covered_roofline() -> Task:
    return Task(
        task_id="covered-roofline",
        kind="roofline",
        state="running",
        params={"source": "coordinator_internal", "reason": "kernel_entry"},
        idempotency_key="internal-analysis-kernel_entry",
    )


@pytest.mark.asyncio
async def test_a_covered_step_runs_inside_the_lanes_its_caller_holds(coord):
    """A reprofile under the kernel_agent lease needs no profile_lane and leaves no row for the pump."""
    c = coord
    agent = await _create_kernel_agent(c)
    lease = await c.locks.try_acquire_many(
        list(_KERNEL_AGENT_LANES),
        holder_id=agent.task_id,
        task_id=agent.task_id,
        action="kernel_agent",
        ttl_sec=60,
    )
    assert lease is not None
    seen: list[Any] = []

    async def _roofline(ctx):
        seen.append(ctx)
        return {"status": "ok"}

    c.sub.register_executor("roofline", _roofline)

    result = await c.sub.execute_covered(_covered_roofline())

    assert result == {"status": "ok"}
    assert seen[0].lease is None
    assert seen[0].extra["shared_state"] is c.shared_state
    assert [t.task_id for t in await c.tasks.queued()] == [agent.task_id]
    assert (await c.locks.lane_holders()).get("benchmark_lane") == 1


@pytest.mark.asyncio
async def test_a_covered_step_that_raises_propagates_to_its_caller(coord):
    c = coord

    async def _crash(_ctx):
        raise RuntimeError("profile crashed")

    c.sub.register_executor("roofline", _crash)

    with pytest.raises(RuntimeError, match="profile crashed"):
        await c.sub.execute_covered(_covered_roofline())


@pytest.mark.asyncio
async def test_phase_transition_waits_until_no_task_is_running(coord, monkeypatch):
    c = coord
    _skip_phase_entry_effects(c, monkeypatch)
    st = c.shared_state
    _arm_kernel_phase(st)
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
    task = await _create_kernel_agent(c)
    await c.tasks.transition(task.task_id, "running")

    await c._advance_phase_if_needed()
    assert st.phase == ps.PHASE_KERNEL_AGENT

    await c.tasks.transition(task.task_id, "succeeded", evidence={"result_keys": []})
    await c._advance_phase_if_needed()
    assert st.phase == ps.PHASE_SWEEP
    assert await c.tasks.running() == []


@pytest.mark.asyncio
async def test_phase_transition_drops_queued_work_the_next_phase_does_not_allow(coord, monkeypatch):
    c = coord
    _skip_phase_entry_effects(c, monkeypatch)
    st = c.shared_state
    _arm_kernel_phase(st)
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
    roofline = await c.tasks.create(kind="roofline", params={}, idempotency_key="left-behind-roofline")

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP
    assert (await c.tasks.get(roofline.task_id)).state == "cancelled"
    assert await c.tasks.queued() == []


def test_specialist_is_allowed_in_enablement_but_not_in_kernel():
    assert "specialist" in ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_ENABLEMENT]
    assert "specialist" not in ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_KERNEL_AGENT]
