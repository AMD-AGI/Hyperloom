# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The quota and priority policy the scheduler hands the toolbox (multi-branch design sections 4.1 item 5 and 5.2).

The toolbox queues tests by :class:`PriorityClass`, highest first in :data:`PRIORITY_ORDER`, keeps each branch's
running tests at or under its K, and keeps all running tests at or under ``max_parallel_tests``. Branch K may add up
to more than the global cap; the queue applies back-pressure. A request names its class by putting
``PriorityClass.value`` in ``TestRequest.priority``.

Confirmation retests, reproduction checks and session measurements do not take a slot of their branch's K: a branch
whose K is full of exploration tests must still be able to confirm a positive result. They do count toward
``max_parallel_tests``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Names the priority order, the K exemption and the meaning of the fields below; recorded in the manifest fingerprint.
QUOTA_POLICY_VERSION = "mb1-quota-1"


class PriorityClass(str, Enum):
    """What a queued test is for.

    ``SESSION_MEASUREMENT`` is the scheduler's own: the PRELUDE baseline tests and the FINAL validation, which run
    while no specialist branch does.
    """

    SESSION_MEASUREMENT = "session_measurement"
    CONFIRMATION_RETEST = "confirmation_retest"
    REPRO_CHECK = "repro_check"
    INTEGRATION_MERGE = "integration_merge"
    PROFILE_REFRESH = "profile_refresh"
    EXPLORATION = "exploration"


#: Highest priority first.
PRIORITY_ORDER: tuple[PriorityClass, ...] = tuple(PriorityClass)

#: Classes that never take a slot of their branch's K.
OUTSIDE_BRANCH_K: frozenset[PriorityClass] = frozenset(
    {PriorityClass.SESSION_MEASUREMENT, PriorityClass.CONFIRMATION_RETEST, PriorityClass.REPRO_CHECK}
)


@dataclass(frozen=True)
class BranchQuota:
    """How many K-bound tests of ``branch`` may run at once; a branch without quota is stopped, not at K 0."""

    branch: str
    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"the K of {self.branch} must be at least 1, got {self.k}")


@dataclass(frozen=True)
class QuotaPolicy:
    """Every running branch's K, the integration branch's included, and the global cap on running tests."""

    max_parallel_tests: int
    branches: tuple[BranchQuota, ...]
    version: str = QUOTA_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.max_parallel_tests < 1:
            raise ValueError(f"max_parallel_tests must be at least 1, got {self.max_parallel_tests}")
        names = [quota.branch for quota in self.branches]
        if len(names) != len(set(names)):
            raise ValueError(f"a branch has more than one quota: {sorted(names)}")

    def k_of(self, branch: str) -> int | None:
        """The K of ``branch``; ``None`` for a branch that has no quota and may run no K-bound test."""

        return next((quota.k for quota in self.branches if quota.branch == branch), None)

    @staticmethod
    def uses_branch_k(priority: PriorityClass) -> bool:
        return priority not in OUTSIDE_BRANCH_K


__all__ = [
    "OUTSIDE_BRANCH_K",
    "PRIORITY_ORDER",
    "QUOTA_POLICY_VERSION",
    "BranchQuota",
    "PriorityClass",
    "QuotaPolicy",
]
