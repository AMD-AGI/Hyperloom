# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the pump returning on the first completion while dispatched work keeps running."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Gated:
    """Executor that runs until its gate opens."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()

    async def __call__(self, ctx) -> dict:
        self.calls.append(ctx.task.task_id)
        self.entered.set()
        await self.gate.wait()
        return {"runner_status": "succeeded"}


class _Instant(_Gated):
    """Executor that finishes immediately."""

    def __init__(self) -> None:
        super().__init__()
        self.gate.set()


@pytest.fixture
def coord(tmp_path: Path):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="decouple-test")
    state.max_minutes = 30
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )


async def _enqueue(coord, kind: str, key: str, lanes: list[str] | None = None, params: dict | None = None):
    await coord.tasks.create_or_return_existing(
        kind=kind, params=params or {}, idempotency_key=key, requires_lanes=lanes or [], lease_ttl_sec=600
    )


def _in_flight_kinds(coord) -> list[str]:
    return [entry.kind for entry in coord.dispatcher._inflight_actions.values()]


async def _settle(coord) -> None:
    await asyncio.wait_for(coord.dispatcher.wait_for_running_work(timeout=5), timeout=5)


async def test_pump_returns_while_its_only_task_keeps_running(coord):
    slow = _Gated()
    coord.sub.register_executor("slow_action", slow)
    await _enqueue(coord, "slow_action", "slow")

    started = time.monotonic()
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)

    assert time.monotonic() - started < 2
    assert _in_flight_kinds(coord) == ["slow_action"]
    slow.gate.set()
    await _settle(coord)
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    assert slow.calls and _in_flight_kinds(coord) == []


async def test_a_pump_that_books_a_completion_spawns_the_task_it_unblocked(coord):
    first, second = _Instant(), _Instant()
    coord.sub.register_executor("first_action", first)
    coord.sub.register_executor("second_action", second)
    await _enqueue(coord, "first_action", "first", ["benchmark_lane"])
    await _enqueue(coord, "second_action", "second", ["benchmark_lane"])
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await _settle(coord)
    assert first.calls and not second.calls

    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await _settle(coord)

    assert second.calls


async def test_a_running_task_is_not_dispatched_again_by_a_later_pump(coord):
    slow = _Gated()
    coord.sub.register_executor("slow_action", slow)
    await _enqueue(coord, "slow_action", "slow")
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)

    slow.gate.set()
    await _settle(coord)
    assert len(slow.calls) == 1


async def test_gpu_specialist_stays_exclusive_with_a_running_explore_across_pumps(coord):
    explore, specialist = _Gated(), _Instant()
    coord.sub.register_executor("explore", explore)
    coord.sub.register_executor("specialist", specialist)
    await _enqueue(coord, "explore", "explore", ["benchmark_lane"])
    await _enqueue(
        coord,
        "specialist",
        "gpu-spec",
        ["gpu_research_lane"],
        {"domain": "serving_specialist", "gap_canonical_id": "gap.test"},
    )

    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await asyncio.wait_for(explore.entered.wait(), timeout=5)
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    assert not specialist.calls

    explore.gate.set()
    await _settle(coord)
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await _settle(coord)
    assert specialist.calls


async def test_frozen_admission_spawns_nothing(coord):
    fast = _Instant()
    coord.sub.register_executor("fast_action", fast)
    await _enqueue(coord, "fast_action", "fast")

    coord.dispatcher.admission_frozen = True
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await _settle(coord)
    assert not fast.calls

    coord.dispatcher.admission_frozen = False
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await _settle(coord)
    assert fast.calls


async def test_spawning_a_specialist_resets_its_domain_stale_counter(coord):
    from hyperloom.orchestrator.specialists.domains import get_domain

    specialist = _Gated()
    coord.sub.register_executor("specialist", specialist)
    anchor = get_domain("serving_specialist").kb_anchor
    for _ in range(5):
        coord.shared_state.bump_domain_round_counters()
    await _enqueue(
        coord,
        "specialist",
        "cpu-spec",
        ["research_lane"],
        {"domain": "serving_specialist", "gap_canonical_id": "gap.test"},
    )

    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    assert coord.shared_state.rounds_since_last_specialist[anchor] == 0

    specialist.gate.set()
    await _settle(coord)
    assert specialist.calls


async def test_waiting_for_running_work_returns_when_an_action_finishes(coord):
    slow = _Gated()
    coord.sub.register_executor("slow_action", slow)
    await _enqueue(coord, "slow_action", "slow")
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)

    started = time.monotonic()
    await coord.dispatcher.wait_for_running_work(timeout=0.2)
    assert _in_flight_kinds(coord) == ["slow_action"]
    assert time.monotonic() - started >= 0.2

    asyncio.get_running_loop().call_later(0.1, slow.gate.set)
    await asyncio.wait_for(coord.dispatcher.wait_for_running_work(timeout=60), timeout=5)
    assert slow.calls


async def test_an_unbooked_completion_holds_the_phase_transition(coord):
    coord.dispatcher._completion_queue.append((SimpleNamespace(task_id="t", kind="specialist", params={}), object()))
    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.baseline_tput = 1.0

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == "PRELUDE"
    assert coord.dispatcher.admission_frozen


async def test_the_closing_transition_releases_a_held_barrier(coord):
    async def _close_sequence(*, reason: str) -> bool:
        return True

    coord.phase_close.ensure_close_sequence = _close_sequence
    coord.dispatcher.admission_frozen = True
    coord.shared_state.closing_phase = True
    coord.shared_state.phase = "CLOSE"

    await coord.phase_machine.advance_phase_if_needed()

    assert not coord.dispatcher.admission_frozen


async def test_the_closing_grace_waits_on_a_report_that_outlived_the_close_sequence(coord):
    report = _Gated()
    coord.sub.register_executor("report", report)
    real_enter = coord.phase_close.enter_closing_phase
    closing_tick: list[int] = []

    async def _enter(*, grace_sec: float):
        closing_tick.append(int(coord.shared_state.tick))
        asyncio.get_running_loop().call_later(0.5, report.gate.set)
        return await real_enter(grace_sec=grace_sec)

    async def _close_sequence(*, reason: str) -> bool:
        coord.shared_state.close_sequence_done = True
        return True

    coord.phase_close.enter_closing_phase = _enter
    coord.phase_close.ensure_close_sequence = _close_sequence

    await asyncio.wait_for(coord.run(max_minutes=0.001, closing_grace_sec=30.0, tick_interval_sec=0.0), timeout=20)

    assert report.calls
    assert int(coord.shared_state.tick) - closing_tick[0] < 20
