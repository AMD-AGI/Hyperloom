# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-running dispatched work does not require a supervisor progress stamp."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ._dispatch_helpers import pump_until_settled


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
        knowledge_plane=None,
    )


@pytest.mark.asyncio
async def test_long_work_is_booked_without_a_supervisor_stamp(tmp_path, monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()
    coord = await _build_coord(tmp_path)
    assert not hasattr(coord.reconciler, "stamp_progress")
    reaped = AsyncMock(wraps=coord.dispatcher.reap_dispatched_task)
    monkeypatch.setattr(coord.dispatcher, "reap_dispatched_task", reaped)
    monkeypatch.setattr(coord.writeback, "is_promotable_result", lambda *_args: True)
    monkeypatch.setattr(coord.writeback, "promote_to_shared_state", AsyncMock())
    monkeypatch.setattr(coord.recipe_journal, "fact_write_hook", AsyncMock())
    calls = []

    async def execute(ctx):
        calls.append(ctx.task.task_id)
        entered.set()
        await release.wait()
        return {"status": "ok"}

    coord.sub.register_executor("profile", execute)
    task = await coord.tasks.create(kind="profile", params={}, idempotency_key="long-profile")
    try:
        await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), 5)
        await asyncio.wait_for(entered.wait(), 5)
        assert (await coord.tasks.get(task.task_id)).state == "running"
        reaped.assert_not_awaited()
    finally:
        release.set()
        await pump_until_settled(coord.dispatcher)
    try:
        assert calls == [task.task_id]
        reaped.assert_awaited_once()
        assert (await coord.tasks.get(task.task_id)).state == "succeeded"
        events = await coord.db.fetchall("SELECT payload FROM events WHERE topic='delegated_result'")
        assert len(events) == 1
        assert not coord.dispatcher._executions
    finally:
        await coord.stop()
