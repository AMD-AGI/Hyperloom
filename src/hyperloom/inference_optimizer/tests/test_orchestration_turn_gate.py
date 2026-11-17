# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration takes a turn when something it reads has changed, or once the heartbeat elapses, in every phase."""

from __future__ import annotations

import time

import pytest

from hyperloom.inference_optimizer.protocol.intent import DEFAULT_IDLE_BODY
from hyperloom.orchestrator.bus.message_bus import LANES_BUSY_SOURCE, Message
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.task_registry import TaskSpec
from .conftest import silent_backends

_HOLD_PHASES = (ps.PHASE_PRELUDE, ps.PHASE_FRAMEWORK_AGENT, ps.PHASE_KERNEL_AGENT, ps.PHASE_SWEEP)
_ONE_TURN_THEN_HOLD = ["orchestration", "critic", "critic", "critic"]


@pytest.fixture
async def ticking(session_dir, monkeypatch):
    """A Coordinator with silent backends whose ticks run only the reactor turns, and the turns they ran."""
    c = Coordinator(session_dir, backends=silent_backends())
    turns: list[str] = []
    reactor_pass = c.reactor_pass

    async def _noop(*_args, **_kwargs):
        return None

    async def _turn(name, **kwargs):
        turns.append(name)
        return await reactor_pass(name, **kwargs)

    monkeypatch.setattr(c, "reactor_pass", _turn)
    monkeypatch.setattr(c.reconciler, "run", _noop)
    monkeypatch.setattr(c.phase_machine, "advance_phase_if_needed", _noop)
    monkeypatch.setattr(c.phase_machine, "pump_current_phase", _noop)
    monkeypatch.setattr(c.dispatcher, "pump_dispatcher_once", _noop)
    monkeypatch.setattr(c.enablement_lane, "pump_enablement_safely", _noop)
    c._last_maintenance_ts = time.monotonic()
    c.shared_state.phase = ps.PHASE_PRELUDE
    try:
        yield c, turns
    finally:
        await c.stop()


@pytest.mark.parametrize("phase", _HOLD_PHASES)
async def test_a_tick_with_nothing_new_runs_no_orchestration_turn(ticking, phase):
    c, turns = ticking
    c.shared_state.phase = phase

    await c.tick(3)

    assert turns == _ONE_TURN_THEN_HOLD


async def _enter_sweep(c: Coordinator) -> None:
    c.shared_state.phase = ps.PHASE_SWEEP


async def _queue_a_task(c: Coordinator) -> None:
    await c.tasks.create_or_return_existing(TaskSpec(kind="roofline", params={}, idempotency_key="gate-roofline"))


async def _cancel_the_queued_task(c: Coordinator) -> None:
    (task,) = await c.tasks.queued()
    assert await c.tasks.cancel_if_queued(task.task_id, kind=task.kind, reason="test")


async def _answer_a_request(c: Coordinator) -> None:
    payload = {"kind": "integrate_done", "status": "ok", "result": {"status": "ok"}, "source": "integrate"}
    await c.bus.append_and_seq(Message.new("kernel_agent", "orchestration", "response", payload))


@pytest.mark.parametrize("change", [_enter_sweep, _queue_a_task, _cancel_the_queued_task, _answer_a_request])
async def test_a_change_admits_one_orchestration_turn(ticking, change):
    c, turns = ticking
    await c.tasks.create_or_return_existing(TaskSpec(kind="roofline", params={}, idempotency_key="gate-standing"))
    await c.tick(1)

    await change(c)
    await c.tick(2)

    assert turns == ["orchestration", "critic", "orchestration", "critic", "critic"]


async def _defer_on_busy_lanes(c: Coordinator) -> None:
    payload = {
        "kind": "integrate_done",
        "status": "deferred",
        "result": {"status": "deferred", "reason": "lanes_busy", "lanes": ["benchmark_lane"]},
        "source": LANES_BUSY_SOURCE,
    }
    await c.bus.append_and_seq(Message.new("kernel_agent", "orchestration", "response", payload))


async def _critic_idles(c: Coordinator) -> None:
    await c.bus.append_and_seq(Message.new("critic", "*", "observation", {"body_md": DEFAULT_IDLE_BODY}))


@pytest.mark.parametrize("message", [_defer_on_busy_lanes, _critic_idles])
async def test_a_message_that_asks_nothing_admits_no_orchestration_turn(ticking, message):
    c, turns = ticking
    await c.tick(1)

    await message(c)
    await c.tick(2)

    assert turns == _ONE_TURN_THEN_HOLD


@pytest.mark.parametrize("phase", _HOLD_PHASES)
async def test_the_heartbeat_admits_an_orchestration_turn_when_nothing_changed(ticking, phase):
    c, _turns = ticking
    c.shared_state.phase = phase
    await c.tick(1)
    now = time.time()

    assert not await c.conversation.admit_turn("orchestration", now + 10.0)
    assert await c.conversation.admit_turn("orchestration", now + 901.0)


async def test_the_turn_a_message_wakes_shows_it_and_no_critic_idle_reply(ticking):
    c, _turns = ticking
    await c.tick(1)
    for _ in range(400):
        await _critic_idles(c)
    await _answer_a_request(c)

    inbox = (await c.conversation.compose_prompt("orchestration")).turn.split("=== Inbox for orchestration", 1)[1]

    assert "integrate_done" in inbox
    assert DEFAULT_IDLE_BODY not in inbox
    assert "held for the next turn" not in inbox
