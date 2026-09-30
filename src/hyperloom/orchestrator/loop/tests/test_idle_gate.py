# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""IdleTickGate: skip an orchestration turn whose prompt holds nothing new, never past the heartbeat."""

from __future__ import annotations

from hyperloom.orchestrator.loop.idle_gate import IdleTickGate, normalize_prompt, prompt_digest

_TICK_606 = """=== Time budget ===
budget    : pct=0.65 elapsed_sec=1809 cumulative_sec=31644 remaining_sec=18984
reloop    : cycle_reloop_feasible=true threshold_sec=10800 session_remaining_sec=44255 (projected)
time      : elapsed=642.4min remaining=737.6min budget=1380min
elapsed=642.4min  remaining=737.6min  budget=1380min  closing_phase=False
current_action=tick 605: still blocked, no change since tick 603.
tick=606  target_gap_pct=500.00
=== Inbox for orchestration (newest last) ===
  seq=2050 msg_id=3d2e620b4795434dbf545e0477a8f25b from=coordinator topic=observation kind='backend_error' payload={'agent': 'critic'}
"""

_TICK_607 = """=== Time budget ===
budget    : pct=0.65 elapsed_sec=1851 cumulative_sec=31687 remaining_sec=18941
reloop    : cycle_reloop_feasible=true threshold_sec=10800 session_remaining_sec=44213 (projected)
time      : elapsed=643.1min remaining=736.9min budget=1380min
elapsed=643.1min  remaining=736.9min  budget=1380min  closing_phase=False
current_action=tick 606: still blocked. The Critic is still failing with the same auth error.
tick=607  target_gap_pct=500.00
=== Inbox for orchestration (newest last) ===
  seq=2050 msg_id=3d2e620b4795434dbf545e0477a8f25b from=coordinator topic=observation kind='backend_error' payload={'agent': 'critic'}
  seq=2052 msg_id=6c5e4350adab46de9eb4dc01e93928c9 from=coordinator topic=observation kind='backend_error' payload={'agent': 'critic'}
"""


def test_consecutive_stuck_ticks_normalize_to_the_same_prompt():
    """Recorded from a session that re-sent this prompt every minute for hours."""
    assert prompt_digest(_TICK_606) == prompt_digest(_TICK_607)


def test_flags_on_clock_lines_still_count():
    closing = _TICK_607.replace("closing_phase=False", "closing_phase=True")
    assert prompt_digest(closing) != prompt_digest(_TICK_607)


def test_a_new_inbox_message_counts():
    fresh = _TICK_607 + "  seq=2053 msg_id=aa11bb22cc33dd44ee55ff6600112233 from=coordinator kind='delegated_result'\n"
    assert prompt_digest(fresh) != prompt_digest(_TICK_607)


def test_normalize_drops_only_volatile_parts():
    out = normalize_prompt(_TICK_607)
    assert "current_action" not in out
    assert "closing_phase=False" in out
    assert "tick=N" in out
    assert out.count("kind='backend_error'") == 1


def test_gate_skips_unchanged_prompts_until_the_heartbeat():
    gate = IdleTickGate(enabled=True, heartbeat_sec=900.0)
    assert not gate.should_skip(_TICK_606, now=0.0)
    gate.record_sent(_TICK_606, now=0.0)
    assert gate.should_skip(_TICK_607, now=60.0)
    assert gate.should_skip(_TICK_607, now=899.0)
    assert not gate.should_skip(_TICK_607, now=900.0)
    assert gate.skipped == 2


def test_gate_lets_a_changed_prompt_through():
    gate = IdleTickGate(enabled=True, heartbeat_sec=900.0)
    gate.record_sent(_TICK_606, now=0.0)
    assert not gate.should_skip(_TICK_607.replace("backend_error", "delegated_result"), now=60.0)


def test_disabled_gate_never_skips():
    gate = IdleTickGate(enabled=False)
    gate.record_sent(_TICK_606, now=0.0)
    assert not gate.should_skip(_TICK_606, now=1.0)


def test_from_env():
    assert IdleTickGate.from_env({}).enabled
    assert not IdleTickGate.from_env({"HYPERLOOM_IDLE_TICK_GATE": "0"}).enabled
    assert IdleTickGate.from_env({"HYPERLOOM_IDLE_TICK_HEARTBEAT_SEC": "120"}).heartbeat_sec == 120.0
    assert IdleTickGate.from_env({"HYPERLOOM_IDLE_TICK_HEARTBEAT_SEC": "x"}).heartbeat_sec == 900.0
