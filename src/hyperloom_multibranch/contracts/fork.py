# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Requests to the scheduler for a new branch, to carry a correctness fix that costs its branch performance or to
repair the defect that blocked a gain, and the scheduler's answer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict


@dataclass(frozen=True)
class ForkRequest:
    """Asks for a new branch, cut at ``parent_commit``, that carries the fix the requesting branch leaves out.

    ``repro_case`` locates the case that fails on ``parent_commit`` and passes with the fix; descendants of the fork
    keep running it.
    """

    parent_commit: str
    fix_delta: RecipeDelta
    repro_case: str
    reason: str


@dataclass(frozen=True)
class RepairRequest:
    """Asks for a new branch, cut at the blocked candidate's parent commit, to repair what blocked its gain.

    The new specialist starts from ``candidate_delta`` and the ``diagnosis`` (root cause and repair plan) the requesting
    specialist wrote; nothing lands until a repaired candidate passes the gate. The scheduler and the notebook open one
    repair per blocked intent.
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
class Admission:
    """``PENDING`` waits for a free branch slot and is the first thing the next refill starts."""

    status: AdmissionStatus
    reason: str = ""


class BranchRequests(Protocol):
    """The scheduler admits a new branch against the branch cap, or queues it."""

    async def request_fork(self, request: ForkRequest) -> Admission: ...

    async def request_repair(self, request: RepairRequest) -> Admission: ...


__all__ = ["Admission", "AdmissionStatus", "BranchRequests", "ForkRequest", "RepairRequest"]
