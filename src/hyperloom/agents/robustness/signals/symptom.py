# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Symptom data shape produced by signal rules and consumed by ActionLadder."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SymptomSeverity(str, Enum):
    """Severity level of a :class:`Symptom`."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Ordinal rank used for severity comparison and sorting."""
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass
class Symptom:
    """One rule firing, ready for the ActionLadder."""

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
