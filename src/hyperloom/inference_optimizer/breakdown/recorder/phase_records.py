# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The records a phase event is written from, each normalized into the row it persists as."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import (
    text_or_none as _text_or_none,
    to_float as _float_or_none,
    to_int as _int_or_none,
)

from .event_fields import (
    as_dict as _as_dict,
    clip as _clip,
    now_iso_seconds as _now,
)

#: Transition reasons and marker evidence are bounded prose, not logs.
MAX_REASON_CHARS = 500

__all__ = [
    "MAX_REASON_CHARS",
    "ActionDispatch",
    "ActionSettlement",
    "PhaseCycle",
    "PhaseEntry",
    "PhaseExit",
    "PhaseMarker",
    "PhaseProposal",
    "SpecialistRound",
]


@dataclass(frozen=True)
class PhaseCycle:
    """The phase and macro cycle a phase event is scoped to."""

    phase: str
    macro_cycle: int

    def label(self) -> str:
        """The phase name as rows spell it."""
        return str(self.phase or "").strip().upper()


@dataclass(frozen=True)
class PhaseEntry:
    """Entry into a phase.

    ``sequence`` is the transition's ``phase_history`` position and keys the
    segment row; ``entered_unix`` measures the segment when it closes.
    """

    at: PhaseCycle
    sequence: int
    from_phase: str = ""
    reason: str = ""
    evidence: Mapping[str, Any] | None = None
    entered_at: str = ""
    entered_unix: float | None = None

    def row(self, entered: str) -> dict[str, Any]:
        """The segment row this entry persists as, entered at ``entered``."""
        return {
            "sequence": int(self.sequence or 0),
            "from_phase": str(self.from_phase or "").strip().upper(),
            "entered_at": entered,
            "entered_unix": _float_or_none(self.entered_unix),
            "entered_reason": _clip(self.reason, MAX_REASON_CHARS),
            "entered_evidence": _as_dict(self.evidence),
        }


@dataclass(frozen=True)
class PhaseExit:
    """Exit from a phase."""

    at: PhaseCycle
    to_phase: str = ""
    reason: str = ""
    evidence: Mapping[str, Any] | None = None
    exited_at: str = ""
    exited_unix: float | None = None

    def settle(self, exited: str, duration: float | None) -> dict[str, Any]:
        """The fields that settle the open segment, exited at ``exited``."""
        return {
            "to_phase": str(self.to_phase or "").strip().upper(),
            "exited_at": exited,
            "exited_unix": _float_or_none(self.exited_unix),
            "exit_reason": _clip(self.reason, MAX_REASON_CHARS),
            "exit_evidence": _as_dict(self.evidence),
            "duration_sec": duration,
        }


@dataclass(frozen=True)
class PhaseMarker:
    """A non-transition marker raised in a phase; ``sequence`` is its ``phase_history`` position."""

    at: PhaseCycle
    sequence: int
    reason: str = ""
    evidence: Mapping[str, Any] | None = None
    ts: str = ""

    def row(self) -> dict[str, Any]:
        """The marker row this marker persists as."""
        return {
            "sequence": int(self.sequence or 0),
            "reason": _clip(self.reason, MAX_REASON_CHARS),
            "evidence": _as_dict(self.evidence),
            "ts": str(self.ts or "") or _now(),
        }


@dataclass(frozen=True)
class ActionDispatch:
    """An action dispatched against its authoring phase."""

    at: PhaseCycle
    action: str
    task_id: str
    tick: int = 0
    dispatch_class: str | None = None
    allowed: bool | None = True
    denial_rule: str | None = None
    dispatched_at: str = ""
    dispatched_unix: float | None = None

    def row(self) -> dict[str, Any]:
        """The action row this dispatch persists as."""
        row: dict[str, Any] = {
            "action": str(self.action or ""),
            "task_id": str(self.task_id),
            "phase": self.at.label(),
            "macro_cycle": int(self.at.macro_cycle or 0),
            "tick": int(self.tick or 0),
            "dispatched_at": str(self.dispatched_at or "") or _now(),
            "dispatched_unix": _float_or_none(self.dispatched_unix),
        }
        if self.dispatch_class is not None:
            row.update(
                dispatch_class=str(self.dispatch_class),
                allowed=bool(self.allowed),
                denial_rule=_text_or_none(self.denial_rule),
            )
        return row


@dataclass(frozen=True)
class ActionSettlement:
    """The verdict a dispatched action's row settles with."""

    task_id: str
    status: str = ""
    decision: str = ""
    error_class: Any = None
    workspace: Any = None
    settled_at: str = ""
    settled_unix: float | None = None

    def row(self) -> dict[str, Any]:
        """The action row fields this settlement persists as."""
        return {
            "task_id": str(self.task_id),
            "status": str(self.status or ""),
            "decision": str(self.decision or ""),
            "error_class": _text_or_none(self.error_class),
            "workspace": _text_or_none(self.workspace),
            "settled_at": str(self.settled_at or "") or _now(),
            "settled_unix": _float_or_none(self.settled_unix),
        }


@dataclass(frozen=True)
class SpecialistRound:
    """What a specialist round produced; a field left as ``None`` was not reported.

    ``proposal_set`` is deliberately absent: for the FRAMEWORK arm each proposal
    owns a row on the framework event, and for the scouts the product is findings.
    """

    task_id: str
    round_id: str = ""
    domain: Any = None
    gap_canonical_id: Any = None
    summary: Any = None
    reason: Any = None
    source: Any = None
    tags: Any = None
    new_findings: Any = None
    residual_questions: Any = None
    notes: Any = None
    proposals_total: Any = None
    empty: Any = None
    confidence: Any = None
    ensemble_scores: Mapping[str, Any] | None = None

    def row(self) -> dict[str, Any]:
        """The action row fields this round persists as."""
        key = str(self.task_id or "")
        row: dict[str, Any] = {"task_id": key}
        if str(self.round_id or "") and str(self.round_id) != key:
            row["round_id"] = str(self.round_id)
        texts = {
            "domain": self.domain,
            "gap_canonical_id": self.gap_canonical_id,
            "summary": self.summary,
            "reason": self.reason,
            "source": self.source,
        }
        for name, text in texts.items():
            if text is not None:
                row[name] = _clip(str(text or ""), MAX_REASON_CHARS)
        lists = {
            "tags": self.tags,
            "new_findings": self.new_findings,
            "residual_questions": self.residual_questions,
            "notes": self.notes,
        }
        for name, items in lists.items():
            if items is not None:
                row[name] = [str(item) for item in (items or []) if str(item or "")]
        if self.proposals_total is not None:
            row["proposals_total"] = _int_or_none(self.proposals_total)
        if self.empty is not None:
            row["empty"] = bool(self.empty)
        if self.confidence is not None:
            row["confidence"] = _float_or_none(self.confidence)
        if self.ensemble_scores:
            row["ensemble_scores"] = _as_dict(self.ensemble_scores)
        return row


@dataclass(frozen=True)
class PhaseProposal:
    """A proposal as the phase raised it, written when it is minted rather than when acted on."""

    at: PhaseCycle
    proposal_msg_id: str
    action: str
    from_agent: str = ""
    tick: int = 0
    predicted_gain_pct: Any = None
    candidate_id: Any = None
    variant_name: Any = None
    proposed_at: str = ""

    def row(self) -> dict[str, Any]:
        """The proposal row this proposal persists as."""
        return {
            "proposal_msg_id": str(self.proposal_msg_id),
            "action": str(self.action or ""),
            "from_agent": str(self.from_agent or ""),
            "phase": self.at.label(),
            "macro_cycle": int(self.at.macro_cycle or 0),
            "tick": int(self.tick or 0),
            "predicted_gain_pct": _float_or_none(self.predicted_gain_pct),
            "candidate_id": _text_or_none(self.candidate_id),
            "variant_name": _text_or_none(self.variant_name),
            "proposed_at": str(self.proposed_at or "") or _now(),
        }
