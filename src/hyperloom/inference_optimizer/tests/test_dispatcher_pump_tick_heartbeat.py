# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pump must keep the supervisor's tick stamp fresh while it joins a task.

``_pump_dispatcher_once`` synchronously joins every dispatched kind outside
``_NOT_JOINED_KINDS``, so a baseline that spends an hour loading a model runs
inside one tick body. The stamp the supervisor watches is written at the main
loop's tick boundary, which that body has not reached yet: with no stamp from
the join wait, a coordinator doing exactly what it was asked to do looks
identical to a wedged one and gets killed.
"""

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
        "robustness": MockBackend(idle_plan),
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

    monkeypatch.setattr(
        DispatcherCollaborator, "_spawn_fitting_queued", _spawn_fitting_queued
    )
    monkeypatch.setattr(DispatcherCollaborator, "_reap_dispatched_task", _reap)
    monkeypatch.setattr(DispatcherCollaborator, "_reclaim_stale_dispatch_state", _noop)
    monkeypatch.setattr(DispatcherCollaborator, "cancel_inflight_actions", _noop)
    monkeypatch.setattr(
        DispatcherCollaborator,
        "_cancel_inflight_that_outlived_the_session",
        _not_shutting_down,
    )
    monkeypatch.setattr(
        DispatcherCollaborator, "_dispatch_paused_for_phase_budget", lambda self: False
    )
    return counts


@pytest.mark.asyncio
async def test_pump_stamps_tick_while_joining_a_long_task(tmp_path, monkeypatch):
    """A stamp lands, and keeps advancing, for as long as the join wait runs."""
    from hyperloom.orchestrator.supervisor import store as supervisor_store

    release = asyncio.Event()
    counts = _pump_joins_one_task(monkeypatch, release=release)
    coord = await _build_coord(tmp_path)
    coord._dispatcher_poll_sec = 0.02

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.sleep(0.2)
        first = supervisor_store.read_tick(tmp_path)
        await asyncio.sleep(0.2)
        second = supervisor_store.read_tick(tmp_path)
    finally:
        release.set()
        await asyncio.wait_for(pump, timeout=5)

    assert counts == {"spawned": 1, "reaped": 1}
    assert first is not None, "pump joined a long task without ever stamping the tick"
    assert second is not None
    assert second.stamped_unix > first.stamped_unix, (
        "tick stamp went stale while the pump was still polling its join"
    )


@pytest.mark.asyncio
async def test_pump_stamp_carries_the_current_tick(tmp_path, monkeypatch):
    """The stamp reports the tick in flight, not a placeholder."""
    from hyperloom.orchestrator.supervisor import store as supervisor_store

    release = asyncio.Event()
    _pump_joins_one_task(monkeypatch, release=release)
    coord = await _build_coord(tmp_path)
    coord._dispatcher_poll_sec = 0.02
    coord.shared_state.tick = 7

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.sleep(0.2)
        stamp = supervisor_store.read_tick(tmp_path)
    finally:
        release.set()
        await asyncio.wait_for(pump, timeout=5)

    assert stamp is not None
    assert stamp.tick == 7


@pytest.mark.asyncio
async def test_pump_survives_an_unwritable_stamp(tmp_path, monkeypatch):
    """Liveness reporting is best-effort: a failing stamp cannot break dispatch."""
    from hyperloom.orchestrator.loop import dispatcher as dispatcher_mod

    release = asyncio.Event()
    counts = _pump_joins_one_task(monkeypatch, release=release)
    coord = await _build_coord(tmp_path)
    coord._dispatcher_poll_sec = 0.02

    def _explode(*_args, **_kwargs):
        raise OSError("read-only session dir")

    monkeypatch.setattr(dispatcher_mod.supervisor_store, "stamp_tick", _explode)

    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.sleep(0.2)
    finally:
        release.set()
        await asyncio.wait_for(pump, timeout=5)

    assert counts == {"spawned": 1, "reaped": 1}
