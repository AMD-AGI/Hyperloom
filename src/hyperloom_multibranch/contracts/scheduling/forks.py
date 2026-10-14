# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fork admission: what the scheduler answers when the rule gate asks for a new branch.

The gate awaits :meth:`ForkAdmissionService.admit`. On ``ACCEPTED`` it cuts the branch the answer names, commits the
fix on it, records the ``SPLIT`` lineage event, and awaits :meth:`ForkAdmissionService.fork_committed`; the scheduler
then starts a specialist there. The original specialist stays on its own branch either way. ``PENDING`` (admitted
later, when a slot frees up) is part of the vocabulary; how a pending fork is handed back to the gate is settled with
the queue of pending forks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hyperloom_multibranch.contracts.scheduling.peers import ForkRequest


class AdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass(frozen=True)
class ForkAdmission:
    """The answer to a fork request; ``branch`` is the new branch's id, given exactly when it is accepted."""

    status: AdmissionStatus
    reason: str
    branch: str = ""

    def __post_init__(self) -> None:
        if (self.status is AdmissionStatus.ACCEPTED) != bool(self.branch):
            raise ValueError("an accepted fork, and only one, names its new branch")


class ForkAdmissionService(Protocol):
    async def admit(self, request: ForkRequest) -> ForkAdmission: ...

    async def fork_committed(self, branch: str, fix_commit: str) -> None:
        """The gate committed the fix on the admitted ``branch``; its specialist starts from ``fix_commit``."""


__all__ = ["AdmissionStatus", "ForkAdmission", "ForkAdmissionService"]
