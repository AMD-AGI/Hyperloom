# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conversation no-progress circuit-breaker signal.

Fires a HIGH ``conversation_no_progress`` symptom when the Coordinator
reports that the orchestration conversation has been stalled for at least
``threshold`` ticks without any measurable advancement (new KEEP / stack
growth / validated-gain bump / phase advance).

The reactor is the external safety net here.  It never auto-terminates the
run; it raises an alert so the operator can intervene.
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
                "severity": cp.severity,
                "last_progress_tick": cp.last_progress_tick,
            },
            source="coordinator_prompt",
            suggestion=(
                "Operator intervention recommended. If the wall-clock budget is "
                "nearly exhausted, consider triggering a report wind-down."
            ),
        )
    ]
