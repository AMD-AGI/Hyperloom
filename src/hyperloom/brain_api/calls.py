# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two calls a brain answers: propose candidates, and review candidates someone proposed.

Both return only decisions. Measuring a candidate, adopting it and reverting it stay in the body, so every answer
a brain gives can be scored offline by a dataset that knows what the candidates are worth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from hyperloom.brain_api.state import Candidate, Gap
from hyperloom.brain_api.wire import WireError


class CallSite(str, enum.Enum):
    """Where the body calls the brain."""

    PROPOSE = "propose"
    REVIEW = "review"


@dataclass(frozen=True)
class ProposeAsk:
    """What the body asks for: candidates in a domain, aimed at a gap, within a budget.

    ``domain`` is a specialist domain key, or empty for a freeform ask described by ``task``. ``patches`` says
    whether candidates may carry source patches; when it is false only config changes are accepted.
    """

    domain: str = ""
    gap: Gap | None = None
    task: str = ""
    max_candidates: int = 6
    patches: bool = False
    wall_budget_sec: float = 0.0
    max_turns: int = 0

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise WireError("ProposeAsk.max_candidates must be at least 1")
        if not (self.domain or self.task):
            raise WireError("ProposeAsk needs a domain or a task")


@dataclass(frozen=True)
class Proposal:
    """The candidates a brain proposes, best first, and what it learned on the way."""

    candidates: tuple[Candidate, ...] = ()
    summary: str = ""
    confidence: float = 0.0
    findings: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "questions", tuple(self.questions))
        if not 0.0 <= self.confidence <= 1.0:
            raise WireError("Proposal.confidence must be within [0, 1]")


@dataclass(frozen=True)
class ReviewAsk:
    """Candidates to review before any of them is measured, and the rules the review must hold to."""

    candidates: tuple[Candidate, ...]
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.candidates:
            raise WireError("ReviewAsk needs at least one candidate")
        _unique_names(self.candidates, "ReviewAsk")


class Decision(str, enum.Enum):
    """A review's call on one candidate: measure it, drop it, or hold it back for now."""

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True)
class Verdict:
    """The review of one candidate, by name."""

    candidate: str
    decision: Decision
    predicted_gain_pct: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Review:
    """One verdict per reviewed candidate."""

    verdicts: tuple[Verdict, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdicts", tuple(self.verdicts))


def _unique_names(candidates: tuple[Candidate, ...], where: str) -> None:
    names = [c.name for c in candidates]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise WireError(f"{where}: duplicate candidate names {dupes}")


def check_proposal(ask: ProposeAsk, proposal: Proposal) -> Proposal:
    """Return ``proposal`` if it answers ``ask``: within the candidate cap, unique names, patches only if allowed."""
    if len(proposal.candidates) > ask.max_candidates:
        raise WireError(f"Proposal has {len(proposal.candidates)} candidates, the ask allows {ask.max_candidates}")
    _unique_names(proposal.candidates, "Proposal")
    if not ask.patches:
        patched = [c.name for c in proposal.candidates if c.patches]
        if patched:
            raise WireError(f"Proposal carries patches the ask does not allow: {patched}")
    return proposal


def check_review(ask: ReviewAsk, review: Review) -> Review:
    """Return ``review`` if it holds exactly one verdict for every candidate of ``ask``."""
    asked = [c.name for c in ask.candidates]
    given = [v.candidate for v in review.verdicts]
    if sorted(given) != sorted(asked):
        raise WireError(f"Review covers {sorted(given)}, the ask holds {sorted(asked)}")
    return review
