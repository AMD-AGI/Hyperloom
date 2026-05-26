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
    medium_threshold: int = 2
    high_threshold: int = 5
    emergency_threshold: int = 10


def evaluate_crash_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: CrashConfig | None = None,
) -> list[Symptom]:
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
