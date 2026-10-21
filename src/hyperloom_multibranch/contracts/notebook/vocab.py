# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closed vocabularies of the experiment notebook, and which role writes which part of it."""

from __future__ import annotations

from enum import Enum

#: The branch the integrator owns; every other branch is a specialist's.
INTEGRATION_BRANCH = "integration"


class Role(str, Enum):
    """Who a notebook handle acts as. Each state has one writer role; reads are open to all."""

    TOOLBOX = "toolbox"
    GATE = "gate"
    INTEGRATOR = "integrator"
    SCHEDULER = "scheduler"
    SPECIALIST_RUNTIME = "specialist_runtime"
    OBSERVER = "observer"


class ClaimStatus(str, Enum):
    """A claim is held until its test result arrives, its holder lets it go, or nobody renews it in time."""

    HELD = "held"
    COMPLETED = "completed"
    RELEASED = "released"
    EXPIRED = "expired"


class DedupStatus(str, Enum):
    """What the notebook knows about an intent on a state (a parent commit); ``testing_elsewhere``: another branch holds
    a claim on it, or has a test of it in flight, on another state, or has a test of the same change in flight on this
    one."""

    FREE = "free"
    IN_PROGRESS = "in_progress"
    TESTED = "tested"
    SUSPECTED = "suspected"
    TESTING_ELSEWHERE = "testing_elsewhere"


class Conclusion(str, Enum):
    """What came of a tested candidate: the rule gate's action on a specialist branch, or the integrator's merge.

    ``DIAGNOSIS_PENDING`` is a blocked gain waiting for its specialist to send it for repair or decline it.
    ``UNVERIFIED``: a result the decision rested on did not re-verify against its signed bundle, so nothing was landed
    or requested; the notebook counts the test as not measured.
    """

    COMMITTED = "committed"
    COMMITTED_WITH_REPRO = "committed_with_repro"
    SPLIT_REQUESTED = "split_requested"
    SPLIT_PENDING = "split_pending"
    SPLIT_COMMITTED = "split_committed"
    DIAGNOSIS_PENDING = "diagnosis_pending"
    REPAIR_REQUESTED = "repair_requested"
    RECORDED_ONLY = "recorded_only"
    ACCURACY_FAILED = "accuracy_failed"
    UNVERIFIED = "unverified"
    MERGED = "merged"
    NOT_STACKABLE = "not_stackable"


#: Conclusions that put a commit on a specialist branch.
COMMITTING = frozenset({Conclusion.COMMITTED, Conclusion.COMMITTED_WITH_REPRO, Conclusion.SPLIT_COMMITTED})
#: Conclusions only the integrator reaches, on the integration branch.
INTEGRATION_CONCLUSIONS = frozenset({Conclusion.MERGED, Conclusion.NOT_STACKABLE})


class LineageKind(str, Enum):
    """Events that shape the branch tree; commits and merges come from judgements and merge attempts instead."""

    ROOT = "root"
    OPEN = "open"
    SPLIT = "split"
    STOP = "stop"


#: The one role that records each lineage event.
LINEAGE_WRITERS: dict[LineageKind, Role] = {
    LineageKind.ROOT: Role.SCHEDULER,
    LineageKind.OPEN: Role.SCHEDULER,
    LineageKind.SPLIT: Role.GATE,
    LineageKind.STOP: Role.SCHEDULER,
}


class CommitOrigin(str, Enum):
    """How a commit came to exist."""

    ROOT = "root"
    COMMIT = "commit"
    SPLIT = "split"
    MERGE = "merge"


class MergeOutcome(str, Enum):
    """Where a merge attempt stands: tested and merged or not, untested because it conflicted or because its change is
    already on the integration head, or still testing."""

    MERGED = "merged"
    NOT_STACKABLE = "not_stackable"
    CONFLICT_SKIPPED = "conflict_skipped"
    ALREADY_ON_HEAD = "already_on_head"
    PENDING = "pending"


class InboxKind(str, Enum):
    """What an inbox item points at; a ``brief`` item points at the branch's head and asks for its brief again, and a
    ``peer_outcome`` item at another branch's test that refused the branch's claim, once that test ended."""

    RESULT = "result"
    JUDGEMENT = "judgement"
    TOOL_RESULT = "tool_result"
    LINEAGE = "lineage"
    BRIEF = "brief"
    PEER_OUTCOME = "peer_outcome"


class UsageSource(str, Enum):
    """Whose LLM spent the money: a branch's specialist, or the scheduler's judge, resolver, the rule gate's fix
    critic, or the LLM that writes frontier scorers at CLOSE."""

    SPECIALIST = "specialist"
    JUDGE = "judge"
    RESOLVER = "resolver"
    CRITIC = "critic"
    SCORER = "scorer"


class QuarantineAction(str, Enum):
    QUARANTINE = "quarantine"
    LIFT = "lift"


class LearningKind(str, Enum):
    """What a specialist's learning is: a way to fail, how the system behaves here, how a tool or setting behaves, what
    a hypothesis came to, or something else."""

    PITFALL = "pitfall"
    INSIGHT = "insight"
    TOOLING = "tooling"
    HYPOTHESIS = "hypothesis"
    OTHER = "other"


class FrontierBasis(str, Enum):
    """What a frontier score rests on, so its reader knows how far to trust it.

    ``trend`` is the branch's own recent gain; ``prior`` what similar archived branches still gained; ``observing`` the
    floor of a branch whose kernel-layer direction may still pay off; ``new`` the floor of a branch too young to have a
    trend; ``pending`` the gain of a measured candidate waiting for its confirm; ``headroom`` the roofline bound that
    capped any of them; ``amdahl`` the most that speeding up the head's largest op could gain, which capped any of them;
    ``custom`` the estimate of a scorer other than the built-in formula, whose components say what it read.
    """

    TREND = "trend"
    PRIOR = "prior"
    OBSERVING = "observing"
    NEW = "new"
    PENDING = "pending"
    HEADROOM = "headroom"
    AMDAHL = "amdahl"
    CUSTOM = "custom"


class UnfinishedReason(str, Enum):
    """Why a result of the session is worth finishing: a commit of a stopped branch that never reached the integration
    head, an explore that kept accuracy and fell just short of its bar, a blocked candidate whose repair never
    finished, or an explore withdrawn before its result."""

    UNMERGED = "unmerged"
    NEAR_BAR = "near_bar"
    REPAIR_UNFINISHED = "repair_unfinished"
    WITHDRAWN = "withdrawn"


__all__ = [
    "COMMITTING",
    "INTEGRATION_BRANCH",
    "INTEGRATION_CONCLUSIONS",
    "LINEAGE_WRITERS",
    "ClaimStatus",
    "CommitOrigin",
    "Conclusion",
    "DedupStatus",
    "FrontierBasis",
    "InboxKind",
    "LearningKind",
    "LineageKind",
    "MergeOutcome",
    "QuarantineAction",
    "Role",
    "UnfinishedReason",
    "UsageSource",
]
