# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which kernel version each forge-loop iteration starts from."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class SearchPolicy(str, Enum):
    """The rule that decides which measured candidates later iterations build on."""

    SEQUENTIAL = "sequential"
    SEQANY = "seqany"


DEFAULT_SEARCH_POLICY = SearchPolicy.SEQUENTIAL

# Implementer lanes a sequential round runs when the caller does not choose.
SEQUENTIAL_DEFAULT_LANES = 3

DECISION_KEEP = "KEEP"
DECISION_ACCEPT = "ACCEPT"
# Decisions whose candidate is committed on the campaign branch.
COMMITTED_DECISIONS = frozenset({DECISION_KEEP, DECISION_ACCEPT})


def parse_search_policy(value: str) -> SearchPolicy:
    """Resolve a configured policy name, refusing anything the loop cannot run."""
    try:
        return SearchPolicy(str(value).strip().lower())
    except ValueError:
        known = ", ".join(policy.value for policy in SearchPolicy)
        raise ValueError(f"unsupported search policy {value!r}; expected one of {known}") from None


def resolve_lanes(policy: SearchPolicy, requested: int | None) -> int:
    """How many Implementer lanes a round runs under ``policy``; ``None`` asks for the policy's default.

    ``seqany`` is a single chain: every valid lane candidate would be accepted and applied over the others without a
    speed check, so it runs exactly one lane.
    """
    if requested is not None and requested < 1:
        raise ValueError(f"lanes must be at least 1, got {requested}")
    if policy is SearchPolicy.SEQANY:
        if requested not in (None, 1):
            raise ValueError(f"search policy 'seqany' runs a single lane; got {requested} lanes")
        return 1
    return SEQUENTIAL_DEFAULT_LANES if requested is None else requested


@dataclass(frozen=True)
class SearchCandidate:
    """One iteration's candidate: the version it was made from and what became of it."""

    iteration: int
    parent_iteration: int
    parent_commit: str
    decision: str
    # Set for a KEEP or an ACCEPT, the decisions that commit the candidate.
    commit_hash: str = ""
    # Set whenever the candidate was scored.
    mean_case_speedup: float | None = None
    # Per-case times of a committed candidate, which a later iteration may start from.
    case_times: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration <= 0:
            raise ValueError(f"candidate iteration must be a positive integer: {self.iteration!r}")
        if (
            isinstance(self.parent_iteration, bool)
            or not isinstance(self.parent_iteration, int)
            or not 0 <= self.parent_iteration < self.iteration
        ):
            raise ValueError(f"candidate {self.iteration} parent iteration must precede it: {self.parent_iteration!r}")
        if not str(self.parent_commit or "").strip():
            raise ValueError(f"candidate {self.iteration} has no parent commit")
        if not str(self.decision or "").strip():
            raise ValueError(f"candidate {self.iteration} has no decision")
        committed = self.decision in COMMITTED_DECISIONS
        if committed != bool(str(self.commit_hash or "").strip()):
            raise ValueError(
                f"candidate {self.iteration} decision {self.decision} "
                f"{'requires' if committed else 'must not carry'} a commit"
            )
        if self.mean_case_speedup is not None and (
            isinstance(self.mean_case_speedup, bool)
            or not math.isfinite(float(self.mean_case_speedup))
            or float(self.mean_case_speedup) <= 0
        ):
            raise ValueError(f"candidate {self.iteration} score must be a positive finite number")

    @property
    def committed(self) -> bool:
        return self.decision in COMMITTED_DECISIONS


@dataclass(frozen=True)
class StartingVersion:
    """The committed kernel version an iteration's candidate is made from."""

    iteration: int
    commit_hash: str
    mean_case_speedup: float | None
    case_times: dict[str, float]


def accepts(policy: SearchPolicy, *, valid: bool, improves_best: bool) -> bool:
    """Whether a measured candidate becomes the next iteration's starting version."""
    if policy is SearchPolicy.SEQUENTIAL:
        return valid and improves_best
    if policy is SearchPolicy.SEQANY:
        return valid
    raise ValueError(f"unsupported search policy: {policy!r}")


def select_starting_version(
    policy: SearchPolicy,
    *,
    candidates: Sequence[SearchCandidate],
    best: StartingVersion,
) -> StartingVersion:
    """The version the next iteration starts from, derived from the candidate records."""
    if policy is SearchPolicy.SEQUENTIAL:
        return best
    if policy is SearchPolicy.SEQANY:
        for record in reversed(candidates):
            if record.committed:
                return StartingVersion(
                    iteration=record.iteration,
                    commit_hash=record.commit_hash,
                    mean_case_speedup=record.mean_case_speedup,
                    case_times=dict(record.case_times),
                )
        return best
    raise ValueError(f"unsupported search policy: {policy!r}")
