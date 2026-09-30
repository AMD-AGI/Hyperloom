# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long dispatcher joins do not require a supervisor progress stamp."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


async def _build_coord(tmp_path: Path):
    """Build a minimal Coordinator rooted at ``tmp_path``."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="pump-tick-heartbeat")
    state.save(tmp_path)
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    return Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


@pytest.mark.asyncio
async def test_pump_joins_long_work_without_a_supervisor_stamp(tmp_path, monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()
    coord = await _build_coord(tmp_path)
    coord._dispatcher_poll_sec = 0.02
    assert not hasattr(coord.reconciler, "stamp_progress")
    reaped = AsyncMock(wraps=coord._reap_dispatched_task)
    monkeypatch.setattr(coord, "_reap_dispatched_task", reaped)
    monkeypatch.setattr(coord, "_is_promotable_result", lambda *_args: True)
    monkeypatch.setattr(coord, "_promote_to_shared_state", AsyncMock())
    monkeypatch.setattr(coord, "_fact_write_hook", AsyncMock())
    calls = []

    async def execute(ctx):
        calls.append(ctx.task.task_id)
        entered.set()
        await release.wait()
        return {"status": "ok"}

    coord.sub.register_executor("profile", execute)
    task = await coord.tasks.create(kind="profile", params={}, idempotency_key="long-profile")
    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not pump.done()
        assert (await coord.tasks.get(task.task_id)).state == "running"
        reaped.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(pump, timeout=5)
    try:
        assert calls == [task.task_id]
        reaped.assert_awaited_once()
        assert (await coord.tasks.get(task.task_id)).state == "succeeded"
        events = await coord.db.fetchall("SELECT payload FROM events WHERE topic='delegated_result'")
        assert len(events) == 1
        assert not coord._executions
    finally:
        await coord.stop()
