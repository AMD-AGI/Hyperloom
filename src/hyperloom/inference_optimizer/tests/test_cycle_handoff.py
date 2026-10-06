# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SWEEP handoff turn and the macro-cycle loopback that consumes its directive."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.collaborator import OrchestrationPrompt
from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.roles import MockBackend, MockTurn, ScriptedPlan

from .conftest import make_coordinator

_DIRECTIVE = "attack the KV cache, deprioritise scheduler knobs"


def _sweep_coordinator(session_dir, *, replies: list[MockTurn]):
    orchestration = MockBackend(ScriptedPlan(turns=replies), name="orchestration")
    coord = make_coordinator(session_dir)
    coord.backends["orchestration"] = orchestration
    st = coord.shared_state
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_SWEEP
    st.start_ts = (now - timedelta(hours=1)).isoformat()
    st.max_minutes = 96 * 60
    st.macro_cycle = 0
    st.cumulative_gain_validated = 7.0
    st.gain_at_cycle_start = 0.0
    st.last_conc_sweep = {"status": "running"}
    return coord, orchestration


@pytest.mark.asyncio
async def test_sweep_asks_for_the_directive_once_per_cycle(session_dir):
    coord, orchestration = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])

    await coord.phase_sweep.pump()
    await coord.phase_sweep.pump()

    assert len(orchestration.calls) == 1
    assert "MACRO-CYCLE HANDOFF" in orchestration.calls[0]["prompt"]
    assert coord.shared_state.orchestration_memory == {
        "next_cycle_directive": _DIRECTIVE,
        "for_cycle": 0,
        "parse_error": "",
    }


@pytest.mark.asyncio
async def test_a_reply_without_intents_is_not_an_error(session_dir):
    coord, _ = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])

    await coord.phase_sweep.pump()

    observations = await coord.bus.tail(n=50, topic="observation")
    assert not any((o.payload or {}).get("kind") == "no_intent_emitted" for o in observations)


@pytest.mark.asyncio
async def test_the_handoff_turn_is_billed_like_any_reactor_turn(session_dir, monkeypatch):
    coord, _ = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])
    billed: list[str] = []
    monkeypatch.setattr(coord, "_trace_reactor_llm_call", lambda agent, result, **_: billed.append(agent))

    await coord.phase_sweep.pump()

    assert billed == ["orchestration"]


@pytest.mark.asyncio
async def test_a_sweep_that_settles_on_entry_hands_off_before_the_machine_leaves(session_dir):
    from hyperloom.orchestrator.phases.machine import Transition

    coord, orchestration = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])
    st = coord.shared_state
    st.last_conc_sweep = {}
    st.conc_sweep_enabled = False

    await coord.phase_sweep.on_enter_sweep(
        Transition(
            from_phase=ps.PHASE_FRAMEWORK_AGENT, to_phase=ps.PHASE_SWEEP, reason="test", evidence={}, loopback=False
        )
    )

    assert st.last_conc_sweep["status"] == "skipped"
    assert len(orchestration.calls) == 1
    assert st.orchestration_memory["for_cycle"] == 0
    assert st.orchestration_memory["next_cycle_directive"] == _DIRECTIVE


@pytest.mark.asyncio
async def test_a_failed_transition_save_still_enters_the_next_cycle_and_then_raises(session_dir, monkeypatch):
    from unittest.mock import AsyncMock

    coord, _ = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])
    st = coord.shared_state
    st.last_conc_sweep = {"status": "succeeded"}
    entered = AsyncMock()
    restarted = AsyncMock()
    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", entered)
    monkeypatch.setattr(coord.phase_macro_cycle, "run_cycle_soft_restart", restarted)
    real_save = st.save
    failed: list[bool] = []

    def _save(session_dir):
        if st.phase == ps.PHASE_FRAMEWORK_AGENT and not failed:
            failed.append(True)
            raise OSError("session dir unavailable")
        real_save(session_dir)

    monkeypatch.setattr(st, "save", _save)

    with pytest.raises(OSError, match="session dir unavailable"):
        await coord.phase_machine.advance_phase_if_needed()

    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert entered.await_args.kwargs["to_phase"] == ps.PHASE_FRAMEWORK_AGENT
    restarted.assert_awaited_once()
    events = await coord.bus.tail(n=20, topic="event")
    assert any((e.payload or {}).get("kind") == "phase_transition" for e in events)


@pytest.mark.asyncio
async def test_a_failed_handoff_turn_records_the_error_and_leaves_no_directive(session_dir, monkeypatch):
    from hyperloom.orchestrator.roles.base import LLMCallFailed

    coord, orchestration = _sweep_coordinator(session_dir, replies=[])

    async def _timed_out(**_kwargs):
        raise LLMCallFailed("Claude backend timed out: turn exceeded its 1500s wall-clock bound")

    monkeypatch.setattr(orchestration, "run", _timed_out)

    await coord.phase_sweep.pump()

    assert coord.shared_state.orchestration_memory["for_cycle"] == 0
    assert coord.shared_state.orchestration_memory["next_cycle_directive"] == ""
    observations = await coord.bus.tail(n=50, topic="observation")
    assert any((o.payload or {}).get("kind") == "backend_error" for o in observations)


@pytest.mark.asyncio
async def test_no_handoff_is_requested_when_no_further_cycle_is_feasible(session_dir):
    coord, orchestration = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])
    coord.shared_state.macro_cycle = ps.DEFAULT_MAX_MACRO_CYCLES - 1

    await coord.phase_sweep.pump()

    assert orchestration.calls == []
    assert coord.shared_state.orchestration_memory == {}


@pytest.mark.asyncio
async def test_the_next_cycle_opens_on_the_directive_and_enters_as_one_sequence(session_dir):
    coord, _ = _sweep_coordinator(session_dir, replies=[MockTurn(raw_text=_DIRECTIVE)])
    st = coord.shared_state
    await coord.phase_sweep.pump()
    st.last_conc_sweep = {"status": "succeeded"}
    directives: list[str] = []
    coord.orch_prompt = OrchestrationPrompt(
        overrides={"orchestration": "ORIGINAL"},
        is_user_supplied=False,
        rebuild=lambda **kw: directives.append(kw["cycle_directive"]) or "REBUILT",
    )
    order: list[str] = []
    open_cycle = st.open_macro_cycle

    def _open_cycle(**kwargs):
        order.append(f"open_cycle phase={st.phase} cycle={st.macro_cycle}")
        return open_cycle(**kwargs)

    async def _soft_restart(**_kwargs):
        order.append("soft_restart")

    st.open_macro_cycle = _open_cycle  # type: ignore[method-assign]
    coord.phase_macro_cycle.run_cycle_soft_restart = _soft_restart  # type: ignore[method-assign]

    await coord.phase_machine.advance_phase_if_needed()

    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1
    assert order == [f"open_cycle phase={ps.PHASE_FRAMEWORK_AGENT} cycle=0", "soft_restart"]
    assert directives[-1] == _DIRECTIVE
    assert st.phase_history[-1]["cycle"] == 1
