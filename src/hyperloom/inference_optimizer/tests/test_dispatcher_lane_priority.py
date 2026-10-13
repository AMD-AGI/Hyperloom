# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reviewed patch is measured before configuration grids that queued ahead of it on the same lanes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ._dispatch_helpers import pump_until_settled

_SERVING_LANES = ["server_lifecycle", "benchmark_lane"]


def _build_coord(tmp_path: Path):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="lane-priority")
    state.phase = "FRAMEWORK_AGENT"
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        knowledge_plane=None,
    )


class _Recorder:
    """Executor stub that records the order its kinds were dispatched in."""

    def __init__(self, order: list[str]):
        self.order = order

    async def __call__(self, ctx) -> dict:
        self.order.append(ctx.task.kind)
        await asyncio.sleep(0.01)
        return {"status": "ok"}


async def test_approved_patch_is_dispatched_before_earlier_queued_explore(tmp_path):
    coord = _build_coord(tmp_path)
    order: list[str] = []
    recorder = _Recorder(order)
    coord.sub.register_executor("explore", recorder)
    coord.sub.register_executor("integrate_patch", recorder)
    coord.shared_state.record_specialist_patch_verdict("spec-1", "approve")

    for i in range(2):
        await coord.tasks.create_or_return_existing(
            kind="explore",
            params={"grid": [{"name": f"v{i}", "extra_server_args": "--stream-interval 4"}]},
            idempotency_key=f"explore-{i}",
            requires_lanes=list(_SERVING_LANES),
        )
    await coord.tasks.create_or_return_existing(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-1", "patches": ["p.patch"]},
        idempotency_key="integrate-spec-1",
        requires_lanes=["workspace_mutation", *_SERVING_LANES],
    )

    await pump_until_settled(coord.dispatcher)

    assert order == ["integrate_patch", "explore", "explore"]
