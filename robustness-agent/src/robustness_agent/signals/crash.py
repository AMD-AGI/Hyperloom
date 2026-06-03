"""Crash-count signal.

 lets ``crash_count >= crash_emergency_threshold``
(default 25) terminate the run with ``stop_reason=emergency``. We fire
escalating severity well before that:

* >= 2  consecutive crashes -> medium symptom suggesting recover delegate
* >= 5  consecutive crashes -> high symptom suggesting strategy change
* >= 10 consecutive crashes -> high symptom flagged ``emergency``
"""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class CrashConfig:
    """Crash-count thresholds for :func:`evaluate_crash_signals`.

    Attributes:
        medium_threshold (int): Consecutive crashes at which a MEDIUM
            ``crash_count_rising`` symptom fires.
        high_threshold (int): Consecutive crashes at which a HIGH
            ``crash_count_high`` symptom fires.
        emergency_threshold (int): Consecutive crashes at which a HIGH
            ``crash_count_emergency`` symptom fires.
    """

    medium_threshold: int = 2
    high_threshold: int = 5
    emergency_threshold: int = 10


def evaluate_crash_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: CrashConfig | None = None,
) -> list[Symptom]:
    """Emit an escalating crash-count symptom from the shared-state counter.

    Picks MEDIUM/HIGH severity and a matching symptom name based on how many
    consecutive crashes the session has recorded.

    Args:
        ctx (ReactorContext): Reactor context providing the shared-state crash
            count and current action.
        data (SourceData): Collected source data (unused but kept for a uniform
            rule signature).
        config (CrashConfig | None): Tunables; defaults to :class:`CrashConfig`
            when ``None``.

    Returns:
        list[Symptom]: A one-element list with the crash symptom once the medium
            threshold is reached, otherwise an empty list.
    """
    cfg = config or CrashConfig()
    crash_count = ctx.shared_state.crash_count
    if crash_count < cfg.medium_threshold:
        return []

    severity: SymptomSeverity
    suggestion: str
    name: str
    if crash_count >= cfg.emergency_threshold:
        severity = SymptomSeverity.HIGH
        name = "crash_count_emergency"
        suggestion = "escalate_strategy_change with stop / emergency hint"
    elif crash_count >= cfg.high_threshold:
        severity = SymptomSeverity.HIGH
        name = "crash_count_high"
        suggestion = "escalate_strategy_change suggesting baseline revert"
    else:
        severity = SymptomSeverity.MEDIUM
        name = "crash_count_rising"
        suggestion = "delegate(recover) or escalate strategy change"

    return [
        Symptom(
            name=name,
            severity=severity,
            summary=(
                f"crash_count={crash_count} (>= {cfg.medium_threshold}); "
                f"current_action={ctx.shared_state.current_action or '(idle)'}"
            ),
            evidence={
                "crash_count": crash_count,
                "current_action": ctx.shared_state.current_action,
            },
            subject={"agent": "session"},
            source="shared_state",
            suggestion=suggestion,
        )
    ]


__all__ = ["CrashConfig", "evaluate_crash_signals"]
