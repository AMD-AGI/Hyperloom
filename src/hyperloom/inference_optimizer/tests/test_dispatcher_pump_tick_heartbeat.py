# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long dispatcher joins do not require a supervisor progress stamp."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

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


def _pump_joins_one_task(monkeypatch, *, release: asyncio.Event) -> dict[str, int]:
    """Strip the pump down to one dispatched task that ends only when released."""
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    counts = {"spawned": 0, "reaped": 0}

    async def _held() -> SimpleNamespace:
        await release.wait()
        return SimpleNamespace(ok=True)

    async def _spawn_fitting_queued(self, *_args, **_kwargs):
        if counts["spawned"]:
            return []
        counts["spawned"] += 1
        task = SimpleNamespace(task_id="long-baseline", kind="baseline")
        return [(task, asyncio.create_task(_held()), None)]

    async def _reap(self, *_args, **_kwargs) -> None:
        counts["reaped"] += 1

    async def _noop(self, *_args, **_kwargs) -> None:
        return None

    async def _not_shutting_down(self, *_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(DispatcherCollaborator, "_spawn_fitting_queued", _spawn_fitting_queued)
    monkeypatch.setattr(DispatcherCollaborator, "_reap_dispatched_task", _reap)
    monkeypatch.setattr(DispatcherCollaborator, "_reclaim_stale_dispatch_state", _noop)
    monkeypatch.setattr(DispatcherCollaborator, "cancel_inflight_actions", _noop)
    monkeypatch.setattr(
        DispatcherCollaborator,
        "_cancel_inflight_that_outlived_the_session",
        _not_shutting_down,
    )
    monkeypatch.setattr(DispatcherCollaborator, "_dispatch_paused_for_phase_budget", lambda self: False)
    return counts


@pytest.mark.asyncio
async def test_pump_joins_long_work_without_a_supervisor_stamp(tmp_path, monkeypatch):
    release = asyncio.Event()
    counts = _pump_joins_one_task(monkeypatch, release=release)
    coord = await _build_coord(tmp_path)
    coord._dispatcher_poll_sec = 0.02
    assert not hasattr(coord.reconciler, "stamp_progress")

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.sleep(0.1)
        assert not pump.done()
    finally:
        release.set()
        await asyncio.wait_for(pump, timeout=5)
    assert counts == {"spawned": 1, "reaped": 1}
