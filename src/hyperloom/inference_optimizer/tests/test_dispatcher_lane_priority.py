# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT measures a reviewed patch before grids queued ahead of it on the same lanes; ENABLEMENT keeps FIFO."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ._dispatch_helpers import pump_until_settled
from hyperloom.orchestrator.state.task_registry import TaskSpec

_SERVING_LANES = ["server_lifecycle", "benchmark_lane"]


def _build_coord(tmp_path: Path, phase: str):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="lane-priority")
    state.phase = phase
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )


class _Recorder:
    def __init__(self, order: list[str]):
        self.order = order

    async def __call__(self, ctx) -> dict:
        self.order.append(ctx.task.kind)
        await asyncio.sleep(0.01)
        return {"status": "ok"}


async def _queue_approved_patch(coord) -> None:
    coord.shared_state.record_specialist_patch_verdict("spec-1", "approve")
    await coord.tasks.create_or_return_existing(
        TaskSpec(
            kind="integrate_patch",
            params={"specialist_task_id": "spec-1", "patches": ["p.patch"]},
            idempotency_key="integrate-spec-1",
            requires_lanes=["workspace_mutation", *_SERVING_LANES],
        ),
    )


async def test_framework_agent_dispatches_approved_patch_before_earlier_queued_explore(tmp_path):
    coord = _build_coord(tmp_path, "FRAMEWORK_AGENT")
    order: list[str] = []
    recorder = _Recorder(order)
    coord.sub.register_executor("explore", recorder)
    coord.sub.register_executor("integrate_patch", recorder)
    for i in range(2):
        await coord.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params={"grid": [{"name": f"v{i}", "extra_server_args": "--stream-interval 4"}]},
                idempotency_key=f"explore-{i}",
                requires_lanes=list(_SERVING_LANES),
            ),
        )
    await _queue_approved_patch(coord)

    await pump_until_settled(coord.dispatcher)

    assert order == ["integrate_patch", "explore", "explore"]


async def test_enablement_keeps_fifo_so_a_patch_waits_for_the_queued_revalidation(tmp_path):
    coord = _build_coord(tmp_path, "ENABLEMENT")
    order: list[str] = []
    recorder = _Recorder(order)
    coord.sub.register_executor("baseline", recorder)
    coord.sub.register_executor("integrate_patch", recorder)
    await coord.tasks.create_or_return_existing(
        TaskSpec(
            kind="baseline",
            params={},
            idempotency_key="enablement_revalidation:gen0",
            requires_lanes=list(_SERVING_LANES),
        ),
    )
    await _queue_approved_patch(coord)

    await pump_until_settled(coord.dispatcher)

    assert order == ["baseline", "integrate_patch"]
