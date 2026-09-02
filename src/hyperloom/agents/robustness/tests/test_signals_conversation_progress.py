# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :func:`evaluate_conversation_progress_signals`."""

from __future__ import annotations

from hyperloom.agents.robustness.role.prompt_inputs import (
    ConversationProgress,
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals.conversation_progress import (
    ConversationProgressConfig,
    evaluate_conversation_progress_signals,
)


def _ctx(
    *,
    ticks: int = 0,
    threshold: int = 12,
    severity: str = "ok",
    last_tick: int = 0,
    closing_phase: bool = False,
    stop_reason: str = "",
    no_progress: bool = False,
) -> ReactorContext:
    snap = SharedStateSnapshot(closing_phase=closing_phase, stop_reason=stop_reason)
    cp = (
        None
        if no_progress
        else ConversationProgress(
            ticks_without_progress=ticks,
            threshold=threshold,
            severity=severity,
            last_progress_tick=last_tick,
        )
    )
    return ReactorContext(shared_state=snap, conversation_progress=cp)


def test_fires_on_high_severity():
    ctx = _ctx(ticks=15, threshold=12, severity="high", last_tick=3)
    syms = evaluate_conversation_progress_signals(ctx)
    assert len(syms) == 1
    s = syms[0]
    assert s.name == "conversation_no_progress"
    assert s.severity.value == "high"
    assert "15" in s.summary
    assert "threshold=12" in s.summary


def test_silent_on_ok_severity():
    ctx = _ctx(ticks=3, threshold=12, severity="ok")
    assert evaluate_conversation_progress_signals(ctx) == []


def test_silent_when_no_progress_block():
    ctx = _ctx(no_progress=True)
    assert evaluate_conversation_progress_signals(ctx) == []


def test_silent_when_closing_phase():
    ctx = _ctx(ticks=20, severity="high", closing_phase=True)
    assert evaluate_conversation_progress_signals(ctx) == []


def test_silent_when_stop_reason_set():
    ctx = _ctx(ticks=20, severity="high", stop_reason="sweep_done")
    assert evaluate_conversation_progress_signals(ctx) == []


def test_disabled_config():
    ctx = _ctx(ticks=20, severity="high")
    assert evaluate_conversation_progress_signals(ctx, config=ConversationProgressConfig(enabled=False)) == []


def test_evidence_fields_populated():
    ctx = _ctx(ticks=15, threshold=12, severity="high", last_tick=7)
    syms = evaluate_conversation_progress_signals(ctx)
    ev = syms[0].evidence
    assert ev["ticks_without_progress"] == 15
    assert ev["threshold"] == 12
    assert ev["last_progress_tick"] == 7
