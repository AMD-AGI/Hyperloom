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

    async def __call__(self, ctx) -> dict:
        self.calls.append(ctx.task.task_id)
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
    coord = Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )
    coord._dispatcher_poll_sec = 0.05
    return coord


async def _enqueue(coord, kind: str, key: str, lanes: list[str] | None = None, params: dict | None = None):
    await coord.tasks.create_or_return_existing(
        kind=kind, params=params or {}, idempotency_key=key, requires_lanes=lanes or [], lease_ttl_sec=600
    )


def _in_flight_kinds(coord) -> list[str]:
    return [entry.kind for entry in coord._inflight_actions.values()]


async def test_pump_returns_after_first_completion_while_slow_work_keeps_running(coord):
    fast, slow = _Instant(), _Gated()
    coord.sub.register_executor("fast_action", fast)
    coord.sub.register_executor("slow_action", slow)
    await _enqueue(coord, "fast_action", "fast")
    await _enqueue(coord, "slow_action", "slow")

    started = time.monotonic()
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)

    assert time.monotonic() - started < 2
    assert fast.calls and slow.calls
    assert _in_flight_kinds(coord) == ["slow_action"]
    slow.gate.set()
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)
    assert _in_flight_kinds(coord) == []


async def test_a_running_task_is_not_dispatched_again_by_a_later_pump(coord):
    fast, slow = _Instant(), _Gated()
    coord.sub.register_executor("fast_action", fast)
    coord.sub.register_executor("slow_action", slow)
    await _enqueue(coord, "fast_action", "fast")
    await _enqueue(coord, "slow_action", "slow")
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)

    asyncio.get_running_loop().call_later(0.2, slow.gate.set)
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)

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

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    await asyncio.sleep(0.3)
    assert explore.calls and not specialist.calls

    explore.gate.set()
    await asyncio.wait_for(pump, timeout=5)
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)
    assert specialist.calls


async def test_frozen_admission_spawns_nothing(coord):
    fast = _Instant()
    coord.sub.register_executor("fast_action", fast)
    await _enqueue(coord, "fast_action", "fast")

    coord.admission_frozen = True
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)
    assert not fast.calls

    coord.admission_frozen = False
    await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=5)
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

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    await asyncio.sleep(0.3)
    assert specialist.calls
    assert coord.shared_state.rounds_since_last_specialist[anchor] == 0

    specialist.gate.set()
    await asyncio.wait_for(pump, timeout=5)


async def test_an_unbooked_completion_holds_the_phase_transition(coord):
    coord._completion_queue.append((SimpleNamespace(task_id="t", kind="specialist", params={}), object()))
    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.baseline_tput = 1.0

    await coord._advance_phase_if_needed()

    assert coord.shared_state.phase == "PRELUDE"
    assert coord.admission_frozen
