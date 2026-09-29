# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The idle gate holds orchestration turns that would see nothing new, and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.idle_gate import IdleTickGate, normalize_prompt, prompt_digest
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

# Two consecutive KERNEL-phase ticks from a real session (Qwen3.5-122B, tick 357 -> 358),
# trimmed: only the clock, the tick counter, message ids and the critic's ack differ.
_TICK_A = """SESSION_DIR=/s
=== Phase ===
phase     : KERNEL_AGENT
budget    : pct=0.47 elapsed_sec=8625 cumulative_sec=18451 remaining_sec=12520
reloop    : cycle_reloop_feasible=true threshold_sec=10800 session_remaining_sec=15669 (projected)
allowed   : integrate
=== Mission progress ===
gain      : validated=77.59%
time      : elapsed=1178.8min remaining=261.2min budget=1440min
=== Time budget ===
elapsed=1178.8min  remaining=261.2min  budget=1440min  closing_phase=False
=== Shared session state ===
tick=357  target_gap_pct=222.41
pending_keep_kernels=['llm_input_residual_rmsnorm']
=== Inbox for orchestration (newest last) ===
  seq=1032 msg_id=a890b2dc781f46bbbe9e7126f9a3795e from=kernel_agent topic=response payload={'in_reply_to': 'cfa3970ee6b548a38826ece1fda71c1f', 'kind': 'integrate_done', 'status': 'deferred', 'result': {'status': 'deferred', 'reason': 'lanes_busy'}}
  seq=1033 msg_id=bb2052ae864c476191148a04a1b1ca20 from=critic topic=observation payload={'topic': 'observation', 'body_md': 'ok (critic)'}
"""
_TICK_B = (
    _TICK_A.replace(
        "elapsed_sec=8625 cumulative_sec=18451 remaining_sec=12520",
        "elapsed_sec=8663 cumulative_sec=18489 remaining_sec=12481",
    )
    .replace("session_remaining_sec=15669", "session_remaining_sec=15631")
    .replace("elapsed=1178.8min remaining=261.2min", "elapsed=1179.5min remaining=260.5min")
    .replace("elapsed=1178.8min  remaining=261.2min", "elapsed=1179.5min  remaining=260.5min")
    .replace("tick=357", "tick=358")
    .replace("seq=1032 msg_id=a890b2dc781f46bbbe9e7126f9a3795e", "seq=1035 msg_id=d17cf336ee6c4764868cc30a777a6d9e")
    .replace("'in_reply_to': 'cfa3970ee6b548a38826ece1fda71c1f'", "'in_reply_to': 'a64457922db941fa9d7816c51f2c1c03'")
    .replace("seq=1033 msg_id=bb2052ae864c476191148a04a1b1ca20", "seq=1036 msg_id=a1f5fce20ab148208a62ef8052bc4edb")
)


def test_consecutive_idle_ticks_normalize_to_the_same_prompt():
    assert _TICK_A != _TICK_B
    assert prompt_digest(_TICK_A) == prompt_digest(_TICK_B)
    norm = normalize_prompt(_TICK_A)
    assert "ok (critic)" not in norm
    assert "elapsed_sec" not in norm and "tick=357" not in norm
    assert "pending_keep_kernels=['llm_input_residual_rmsnorm']" in norm
    assert "'status': 'deferred'" in norm


@pytest.mark.parametrize(
    "change",
    [
        ("pending_keep_kernels=['llm_input_residual_rmsnorm']", "pending_keep_kernels=(none)"),
        ("'reason': 'lanes_busy'", "'reason': 'integrated'"),
        ("allowed   : integrate", "allowed   : integrate, report"),
        ("validated=77.59%", "validated=81.02%"),
    ],
)
def test_a_real_change_changes_the_digest(change):
    old, new = change
    assert prompt_digest(_TICK_A) != prompt_digest(_TICK_A.replace(old, new))


def test_ages_and_empty_inbox_spellings_do_not_reopen_the_gate():
    # A held tick does not refresh agent_last_active, so its age grows on every held tick.
    a = _TICK_A + "agent_last_active=critic=0s ago, orchestration=0s ago\n"
    b = _TICK_B + "agent_last_active=critic=0s ago, orchestration=75s ago\n"
    assert prompt_digest(a) == prompt_digest(b)
    empty = "=== Inbox for orchestration ===\n(no new messages)\n"
    acks_only = (
        "=== Inbox for orchestration (newest last) ===\n"
        "  seq=7 msg_id=0123456789abcdef from=critic topic=observation "
        "payload={'topic': 'observation', 'body_md': 'ok (critic)'}\n"
    )
    assert prompt_digest(empty) == prompt_digest(acks_only)


def test_gate_skips_only_unchanged_prompts_inside_the_heartbeat():
    gate = IdleTickGate(enabled=True, heartbeat_sec=300)
    assert not gate.should_skip(_TICK_A, now=0.0)  # nothing recorded yet
    gate.record_turn(_TICK_A, now=0.0)
    assert gate.should_skip(_TICK_B, now=45.0)
    assert gate.should_skip(_TICK_B, now=90.0)
    assert gate.skipped == 2 and gate.streak == 2
    changed = _TICK_B.replace("pending_keep_kernels=['llm_input_residual_rmsnorm']", "pending_keep_kernels=(none)")
    assert not gate.should_skip(changed, now=100.0)
    assert not gate.should_skip(_TICK_B, now=300.0)  # heartbeat due
    gate.reset()
    assert not gate.should_skip(_TICK_B, now=301.0)


def test_disabled_gate_never_skips(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "0")
    gate = IdleTickGate()
    gate.record_turn(_TICK_A, now=0.0)
    assert not gate.enabled
    assert not gate.should_skip(_TICK_A, now=1.0)


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


async def _orchestration_calls(session_dir: Path, ticks: int) -> tuple[int, int]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    orchestration = MockBackend(silent, name="o")
    c = Coordinator(session_dir, backends={"orchestration": orchestration, "critic": MockBackend(silent, name="c")})
    try:
        await c.tick(ticks)
        return len(orchestration.calls), int(c.shared_state.orchestration_idle_skips)
    finally:
        await c.stop()


async def test_coordinator_holds_idle_orchestration_turns(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "1")
    calls, skips = await _orchestration_calls(session_dir, ticks=12)
    assert (calls, skips) == (1, 11)


async def test_coordinator_calls_every_tick_with_the_gate_off(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "0")
    calls, skips = await _orchestration_calls(session_dir, ticks=6)
    assert (calls, skips) == (6, 0)
