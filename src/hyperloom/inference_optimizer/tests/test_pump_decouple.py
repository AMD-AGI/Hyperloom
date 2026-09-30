# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the decoupled pump: returns on first completion, deduplification and barrier freeze."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_coord(tmp_path: Path):
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="decouple-test")
    state.max_minutes = 30
    state.save(tmp_path)

    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {name: MockBackend(idle) for name in ("orchestration", "critic")}
    return Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


class _SlowExecutor:
    """Executor that holds for ``hold_sec`` before succeeding."""

    def __init__(self, hold_sec: float = 0.2):
        self.hold_sec = hold_sec
        self.calls: list[str] = []

    async def __call__(self, ctx) -> dict:
        self.calls.append(ctx.task.task_id)
        await asyncio.sleep(self.hold_sec)
        return {
            "runner_status": "succeeded",
            "task_id": ctx.task.task_id,
        }


class _FastExecutor:
    """Executor that returns immediately."""

    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, ctx) -> dict:
        self.calls.append(ctx.task.task_id)
        return {
            "runner_status": "succeeded",
            "task_id": ctx.task.task_id,
        }


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pump_returns_after_first_completion_not_after_all(tmp_path: Path):
    """A fast task and a slow task: the pump returns once the fast one is booked,
    while the slow task keeps running in the background."""
    coord = _build_coord(tmp_path)
    fast = _FastExecutor()
    slow = _SlowExecutor(hold_sec=5.0)
    coord.sub.register_executor("fast_action", fast)
    coord.sub.register_executor("slow_action", slow)

    async def _run():
        from hyperloom.orchestrator.phases.machine_state import PHASE_ALLOWED_ACTIONS

        coord.shared_state.phase = "ENABLEMENT"

        await coord.tasks.create_or_return_existing(
            kind="fast_action",
            params={},
            idempotency_key="fast-t1",
            requires_lanes=[],
            lease_ttl_sec=60,
        )
        await coord.tasks.create_or_return_existing(
            kind="slow_action",
            params={},
            idempotency_key="slow-t1",
            requires_lanes=[],
            lease_ttl_sec=60,
        )

        t_start = time.monotonic()
        await coord._pump_dispatcher_once()
        elapsed = time.monotonic() - t_start

        # Pump must return well before the slow task finishes (< 1 s).
        assert elapsed < 2.0, f"pump took {elapsed:.2f}s; slow task still running"
        # Fast task was called.
        assert fast.calls, "fast executor never ran"
        # Slow task was dispatched (inflight by kind).
        assert any(v.kind == "slow_action" for v in coord._inflight_actions.values())

    asyncio.run(_run())


def test_no_task_dispatched_twice_across_pump_calls(tmp_path: Path):
    """The same queued task is not dispatched a second time on the next pump call
    if it is still in _inflight_actions."""
    coord = _build_coord(tmp_path)
    slow = _SlowExecutor(hold_sec=5.0)
    coord.sub.register_executor("slow_action", slow)

    async def _run():
        coord.shared_state.phase = "ENABLEMENT"

        await coord.tasks.create_or_return_existing(
            kind="slow_action",
            params={},
            idempotency_key="slow-dedup",
            requires_lanes=[],
            lease_ttl_sec=60,
        )

        await coord._pump_dispatcher_once()
        call_count_after_first_pump = len(slow.calls)

        await coord._pump_dispatcher_once()
        call_count_after_second_pump = len(slow.calls)

        assert call_count_after_first_pump == 1, "should have been dispatched once"
        assert call_count_after_second_pump == 1, "must not be dispatched again while inflight"

    asyncio.run(_run())


def test_gpu_specialist_stays_exclusive_with_explore_across_pumps(tmp_path: Path):
    """A GPU specialist (holding gpu_research_lane) must not be dispatched while
    an explore (holding benchmark_lane, which conflicts with gpu_research_lane) is
    already running, even across multiple pump calls."""
    coord = _build_coord(tmp_path)
    slow_explore = _SlowExecutor(hold_sec=5.0)
    gpu_spec = _FastExecutor()
    coord.sub.register_executor("explore", slow_explore)
    coord.sub.register_executor("specialist", gpu_spec)

    async def _run():
        coord.shared_state.phase = "FRAMEWORK_AGENT"
        coord.shared_state.gpu_specialist_capacity = 4
        coord.shared_state.research_lane_capacity = 4

        await coord.tasks.create_or_return_existing(
            kind="explore",
            params={"source": "test"},
            idempotency_key="explore-1",
            requires_lanes=["benchmark_lane"],
            lease_ttl_sec=3600,
        )
        await coord.tasks.create_or_return_existing(
            kind="specialist",
            params={"domain": "serving_specialist", "needs_gpu": True, "gpu_count": 1},
            idempotency_key="gpu-spec-1",
            requires_lanes=["gpu_research_lane"],
            lease_ttl_sec=3600,
        )

        # First pump: explore starts, GPU specialist cannot start (lane conflict).
        await coord._pump_dispatcher_once()
        assert "explore-1" in coord._inflight_actions or slow_explore.calls

        # Second pump: GPU specialist still must not start.
        await coord._pump_dispatcher_once()
        assert not gpu_spec.calls, "GPU specialist must not run while explore holds benchmark_lane"

    asyncio.run(_run())


def test_barrier_admission_freeze_prevents_new_spawns(tmp_path: Path):
    """While _admit_frozen is True, the pump does not spawn new tasks."""
    coord = _build_coord(tmp_path)
    fast = _FastExecutor()
    coord.sub.register_executor("fast_action", fast)

    async def _run():
        coord.shared_state.phase = "ENABLEMENT"
        coord._admit_frozen = True

        await coord.tasks.create_or_return_existing(
            kind="fast_action",
            params={},
            idempotency_key="frozen-t1",
            requires_lanes=[],
            lease_ttl_sec=60,
        )

        await coord._pump_dispatcher_once()
        assert not fast.calls, "pump must not spawn while _admit_frozen is True"

        coord._admit_frozen = False
        await coord._pump_dispatcher_once()
        assert fast.calls, "pump must spawn after _admit_frozen is cleared"

    asyncio.run(_run())


def test_unbooked_completion_counts_as_running_in_barrier(tmp_path: Path):
    """The barrier logic counts pending completions as running, preventing transition."""
    coord = _build_coord(tmp_path)

    async def _run():
        from types import SimpleNamespace

        # Plant a pending completion.
        fake_task = SimpleNamespace(task_id="fake-t1", kind="specialist", params={})
        coord._completion_queue.append((fake_task, object(), None))

        # Set up a state where PRELUDE would normally transition to ENABLEMENT.
        coord.shared_state.phase = "PRELUDE"
        coord.shared_state.baseline_tput = 1.0
        coord.shared_state.tp = 0
        coord.shared_state.gpu_specialist_capacity = 0

        prior_phase = coord.shared_state.phase
        await coord._advance_phase_if_needed()
        # Phase must not have advanced while there is a pending completion.
        assert coord.shared_state.phase == prior_phase, "phase must not advance while completion queue is non-empty"
        # _admit_frozen should be set (barrier entered but held due to pending bookkeeping).
        assert coord._admit_frozen, "_admit_frozen must be set during barrier hold"

    asyncio.run(_run())
