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
        Stable identifier the ActionLadder dispatches on
        (e.g. ``"agent_stall"``, ``"repeated_failure"``).
    severity:
        :class:`SymptomSeverity`. Maps to the alert severity emitted
        downstream and gates whether soft / hard actions are reachable.
    summary:
        Human-readable one-line summary surfaced in alerts and findings.
    evidence:
        Structured payload the alert ``detail`` field carries through;
        keep it small (Coordinator persists it verbatim).
    subject:
        Identifying tuple for de-duplication and downstream targetting
        (e.g. ``{"agent": "kernel"}`` / ``{"pod": "brain-0", "namespace": "..."}``).
    source:
        Label of the data source that produced the signal
        (``"server"`` / ``"local"``).
    suggestion:
        Optional hint the ActionLadder uses when it builds the
        ``next_action_hint`` for an ``escalate_strategy_change`` intent.
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
