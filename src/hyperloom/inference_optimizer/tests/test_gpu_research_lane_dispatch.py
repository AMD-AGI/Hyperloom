# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""A ``needs_gpu`` specialist delegate acquires gpu_research_lane and gets a
budget-sourced lease TTL (so the lane never expires mid-run and lets serving
grab the cards). CPU specialists are unchanged (research_lane only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.bus.gpu_pool import GPU_LEASE_TTL_GRACE
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _build_coord(tmp_path: Path, *, gpu_capacity: int) -> Coordinator:
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="ws2-lane")
    state.gpu_specialist_capacity = gpu_capacity
    state.save(tmp_path)

    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        name: MockBackend(idle)
        for name in ("orchestration", "kernel_agent", "critic", "robustness")
    }
    return Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
        knowledge_plane=None,
    )


def _delegate(params: dict) -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "specialist", "params": params},
    )


async def _queued_specialist(coord: Coordinator):
    tasks = await coord.tasks.queued()
    spec = [t for t in tasks if t.kind == "specialist"]
    assert spec, "no specialist task was queued"
    return spec[-1]


@pytest.mark.asyncio
async def test_needs_gpu_specialist_acquires_gpu_research_lane(tmp_path):
    coord = _build_coord(tmp_path, gpu_capacity=8)
    coord.shared_state.macro_cycle = 0
    await coord._handle_delegate(
        "orchestration",
        _delegate(
            {
                "scope": "freeform",
                "task_description": "probe a kernel on GPU",
                "needs_gpu": True,
                "gpu_count": 2,
            }
        ),
    )
    task = await _queued_specialist(coord)
    assert "gpu_research_lane" in task.requires_lanes
    # research_lane is kept for LLM-concurrency accounting.
    assert "research_lane" in task.requires_lanes
    # TTL re-sourced to the GPU wall budget × (1 + grace).
    budget = coord._specialist_wall_budget_sec(needs_gpu=True)
    assert task.lease_ttl_sec == max(
        1800, int(budget * (1.0 + GPU_LEASE_TTL_GRACE))
    )
    # Iron law: lane TTL ≥ the agent's wall-budget kill.
    assert task.lease_ttl_sec >= int(budget)


@pytest.mark.asyncio
async def test_cpu_specialist_has_no_gpu_research_lane(tmp_path):
    coord = _build_coord(tmp_path, gpu_capacity=8)
    await coord._handle_delegate(
        "orchestration",
        _delegate(
            {
                "scope": "freeform",
                "task_description": "read the scheduler",
            }
        ),
    )
    task = await _queued_specialist(coord)
    assert "gpu_research_lane" not in task.requires_lanes
    assert "research_lane" in task.requires_lanes
    assert task.lease_ttl_sec == 1800  # registry default, unchanged
