# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Symptom data shape produced by signal rules and consumed by ActionLadder."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SymptomSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass
class Symptom:
    """One rule firing, ready for the ActionLadder.

    Attributes
    ----------
    name:
        Stable identifier the ActionLadder dispatches on.
    severity:
        :class:`SymptomSeverity`; gates whether soft / hard actions are reachable.
    summary:
        One-line summary surfaced in alerts and findings.
    evidence:
        Structured ``detail`` payload; keep it small (persisted verbatim).
    subject:
        Identifying tuple for de-dup and downstream targetting.
    source:
        Data source that produced the signal (``"server"`` / ``"local"``).
    suggestion:
        Optional hint for the ``escalate_strategy_change`` next_action_hint.
    """

    name: str
    severity: SymptomSeverity
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    subject: dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    suggestion: str = ""

    def dedup_key(self) -> tuple[str, ...]:
        """Stable identity used by the classifier to drop duplicates."""
        if not self.subject:
            return (self.name,)
        return (self.name, *sorted(f"{k}={v}" for k, v in self.subject.items()))


__all__ = ["Symptom", "SymptomSeverity"]
