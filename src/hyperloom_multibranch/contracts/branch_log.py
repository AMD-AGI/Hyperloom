# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's long-term memory: the specialist writes and reads it, the blackboard stores it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BranchLogType(str, Enum):
    HYPOTHESIS = "hypothesis"
    ATTEMPT = "attempt"
    RESULT = "result"
    PLAN = "plan"


class HypothesisStatus(str, Enum):
    OPEN = "open"
    FALSIFIED = "falsified"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class BranchLogEntry:
    """One entry; ``supersedes`` names the entry it updates, and ``request_ids`` the tests it is about."""

    branch: str
    kind: BranchLogType
    content: str
    request_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    hypothesis_status: HypothesisStatus | None = None


__all__ = ["BranchLogEntry", "BranchLogType", "HypothesisStatus"]
