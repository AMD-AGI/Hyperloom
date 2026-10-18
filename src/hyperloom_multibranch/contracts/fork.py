# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Requests to the scheduler for a new branch, to carry a correctness fix that costs its branch performance or to
repair the defect that blocked a gain, and the scheduler's answer.

On ``ACCEPTED`` the answer names the new branch; for a fork the gate cuts it and lands the fix on it before the
scheduler starts the branch's specialist. The requesting specialist stays on its own branch either way. ``PENDING``
waits for a free branch slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict


@dataclass(frozen=True)
class ForkRequest:
    """Asks for a new branch, cut at ``parent_commit``, to carry the fix ``from_branch`` leaves out."""

    from_branch: str
    parent_commit: str
    reason: str


@dataclass(frozen=True)
class RepairRequest:
    """Asks for a new branch, cut at the blocked candidate's parent commit, to repair what blocked its gain.

    ``intent.branch`` is the requesting branch and ``intent.parent_commit`` the candidate's parent, where the repair
    branch is cut; ``judgement.request_ids`` starts with the candidate's first test. The new specialist starts from
    ``candidate_delta`` and the ``diagnosis`` (root cause and repair plan) the requesting specialist wrote; nothing lands
    until a repaired candidate passes the gate. The scheduler and the notebook open one repair per blocked intent.
    """

    intent: Intent
    candidate_delta: RecipeDelta
    judgement: Judgement
    diagnosis: str

    def __post_init__(self) -> None:
        if self.judgement.verdict is not Verdict.BLOCKED:
            raise ValueError(f"only a blocked gain is repaired, not a {self.judgement.verdict.value} one")


class AdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ForkAdmission:
    """The answer to a fork or a repair request; ``branch`` names the new branch, exactly when it is accepted."""

    status: AdmissionStatus
    reason: str = ""
    branch: str = ""

    def __post_init__(self) -> None:
        if (self.status is AdmissionStatus.ACCEPTED) != bool(self.branch):
            raise ValueError("an accepted request, and only an accepted one, names its new branch")


class ForkAdmissionService(Protocol):
    """The scheduler admits a new branch against the branch cap, or queues it."""

    async def admit(self, request: ForkRequest) -> ForkAdmission: ...

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission: ...


__all__ = ["AdmissionStatus", "ForkAdmission", "ForkAdmissionService", "ForkRequest", "RepairRequest"]
