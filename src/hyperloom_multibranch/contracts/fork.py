# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Requests to the scheduler for a new branch, to carry a correctness fix that costs its branch performance or to
repair the defect that blocked a gain, and the scheduler's answer.

On ``ACCEPTED`` the answer names the new branch. ``PENDING`` waits for a free branch slot and names the branch id
reserved for it. The requesting specialist stays on its own branch either way. The scheduler cuts a repair branch
itself; a fork's branch it has the gate land (:class:`ReservedBranches`) before it starts the branch's specialist, and
a fork or repair it voids it hands back to the gate to record.
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
    """The answer to a fork or a repair request; ``branch`` names the new or reserved branch, exactly when it is not
    rejected."""

    status: AdmissionStatus
    reason: str = ""
    branch: str = ""

    def __post_init__(self) -> None:
        if bool(self.branch) != (self.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.PENDING}):
            raise ValueError("an accepted or pending request, and only such a one, names its new branch")


class ForkAdmissionService(Protocol):
    """The scheduler admits a new branch against the branch cap, or queues it."""

    async def admit(self, request: ForkRequest) -> ForkAdmission: ...

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission: ...


class ReservedBranches(Protocol):
    """The rule gate's side of the branches the scheduler reserved for its forks and repairs, named by the branch.

    ``land_split`` cuts a fork's branch at the candidate's parent commit, commits the fix on it and records the
    candidate as ``SPLIT_COMMITTED`` and the branch's ``SPLIT``; true once it has, false for a branch no fork waits
    on. ``settle_voided`` records a voided fork's or repair's candidate as only recorded, which leaves its intent free
    for another repair; a branch already landed or never reserved is logged and left alone.
    """

    async def land_split(self, branch: str) -> bool: ...

    async def settle_voided(self, branch: str, reason: str) -> None: ...


__all__ = [
    "AdmissionStatus",
    "ForkAdmission",
    "ForkAdmissionService",
    "ForkRequest",
    "RepairRequest",
    "ReservedBranches",
]
