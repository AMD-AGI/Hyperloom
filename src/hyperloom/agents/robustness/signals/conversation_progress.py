# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conversation no-progress circuit-breaker signal.

Fires ``conversation_no_progress`` (HIGH) when the Coordinator reports a
stalled orchestration conversation. Alert-only: the ladder attaches no
recovery intent, so a stall never terminates the run on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext
from .symptom import Symptom, SymptomSeverity


@dataclass
class ConversationProgressConfig:
    """Tunables for :func:`evaluate_conversation_progress_signals`.

    Attributes:
        enabled (bool): Set to ``False`` to disable this signal entirely.
    """

    enabled: bool = True


def evaluate_conversation_progress_signals(
    ctx: ReactorContext,
    *,
    config: ConversationProgressConfig | None = None,
) -> list[Symptom]:
    """Emit a HIGH symptom when the Coordinator reports a stalled conversation.

    Args:
        ctx (ReactorContext): Per-tick input carrying a
            ``conversation_progress`` record parsed from the Coordinator prompt.
        config (ConversationProgressConfig | None): Tunables; defaults to
            :class:`ConversationProgressConfig` when ``None``.

    Returns:
        list[Symptom]: At most one HIGH symptom per tick.
    """
    cfg = config or ConversationProgressConfig()
    if not cfg.enabled:
        return []
    snap = ctx.shared_state
    if snap.closing_phase or snap.stop_reason:
        return []
    cp = ctx.conversation_progress
    if cp is None:
        return []
    if cp.severity != "high":
        return []
    return [
        Symptom(
            name="conversation_no_progress",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"orchestration stalled for {cp.ticks_without_progress} ticks "
                f"(threshold={cp.threshold}); last progress at tick {cp.last_progress_tick}"
            ),
            evidence={
                "ticks_without_progress": cp.ticks_without_progress,
                "threshold": cp.threshold,
                "last_progress_tick": cp.last_progress_tick,
            },
            source="shared_state",
            suggestion="operator intervention; wind down via report if the budget is tight",
        )
    ]
