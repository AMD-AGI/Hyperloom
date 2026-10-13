# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The objects the notebook defines: claims, test records, lineage, and what its queries return."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from hyperloom_multibranch.notebook.peers import (
    BranchLogEntry,
    Intent,
    IntentLayer,
    Judgement,
    TestRequest,
    TestResult,
    Verdict,
)
from hyperloom_multibranch.notebook.vocab import (
    ClaimStatus,
    CommitOrigin,
    Conclusion,
    DedupStatus,
    InboxKind,
    LineageKind,
    MergeOutcome,
    QuarantineAction,
    Role,
)

CLAIM_TOKEN_LENGTH = 24


@dataclass(frozen=True)
class Writer:
    """The identity a notebook handle writes as; a specialist runtime writes for its own ``branch``."""

    role: Role
    name: str
    branch: str = ""


@dataclass(frozen=True)
class ClaimKey:
    """The same intent on the same state: ``parent_commit`` is the state."""

    parent_commit: str
    intent_fingerprint: str

    @property
    def token(self) -> str:
        """A file-name-safe name of the key."""

        joined = f"{self.parent_commit}\x00{self.intent_fingerprint}"
        return hashlib.sha256(joined.encode()).hexdigest()[:CLAIM_TOKEN_LENGTH]


@dataclass(frozen=True)
class Claim:
    """A lease on a key, held by a branch so a restarted specialist keeps it; times are Unix seconds."""

    key: ClaimKey
    holder: str
    intent: Intent
    action_fingerprint: str
    ttl_seconds: float
    acquired_at: float
    heartbeat_at: float
    expires_at: float
    status: ClaimStatus
    request_id: str | None = None
    generation: int = 0


@dataclass(frozen=True)
class TestRecord:
    """Everything known about one test (design section 5.2): request, result, judgement, conclusion and commit."""

    __test__ = False

    request: TestRequest
    result: TestResult | None = None
    judgement: Judgement | None = None
    conclusion: Conclusion | None = None
    commit_id: str | None = None

    @property
    def request_id(self) -> str:
        return self.request.request_id


@dataclass(frozen=True)
class DedupAnswer:
    """Whether an intent on a state is free; ``records`` are this state's, ``elsewhere`` only for reference.

    ``TESTED``: the same action was measured on this state. ``IN_PROGRESS``: ``claim`` holds the key, or another claim
    on this state carries the same action. ``SUSPECTED``: the same intent was measured here through another action.
    """

    key: ClaimKey
    action_fingerprint: str
    status: DedupStatus
    claim: Claim | None = None
    records: tuple[TestRecord, ...] = ()
    elsewhere: tuple[TestRecord, ...] = ()


@dataclass(frozen=True)
class MergeAttempt:
    """The integrator merging source commits onto the integration head; without a test it was skipped untested."""

    source_commits: tuple[str, ...]
    integration_head: str
    request_id: str | None = None
    skipped_reason: str = ""


@dataclass(frozen=True)
class LineageEvent:
    """A branch starting (``from_commit``), splitting off ``from_branch`` with fix ``commit_id``, or stopping.

    ``ROOT`` records the baseline commit ``commit_id`` the integration branch starts from.
    """

    kind: LineageKind
    branch: str
    commit_id: str = ""
    from_commit: str = ""
    from_branch: str = ""
    reason: str = ""


@dataclass(frozen=True)
class BaselineRecord:
    """The PRELUDE baseline the rule gate measured, per metric; every recorded value is anchored to it."""

    root_commit: str
    values: Mapping[str, float]
    dispersion_pct: Mapping[str, float]
    threshold_pct: Mapping[str, float]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegressionCase:
    """A correctness fix's reproduction case; it applies to every descendant of ``fix_commit``."""

    case_ref: str
    fix_commit: str
    request_id: str


@dataclass(frozen=True)
class QuarantineEvent:
    """A node, or some of its GPUs, taken out of or returned to service by the toolbox."""

    node: str
    action: QuarantineAction
    reason: str
    gpu_ids: tuple[int, ...] = ()
    request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResultPointer:
    """Where an asynchronous tool's result is; the result itself never enters the notebook."""

    branch: str
    job_id: str
    location: str
    summary: str


@dataclass(frozen=True)
class InboxItem:
    """A pointer pushed to a branch: a request id, a tool job id, or a lineage entry id."""

    kind: InboxKind
    branch: str
    pointer: str
    summary: str
    at: float


@dataclass(frozen=True)
class InboxPage:
    """New items, and the cursor to poll from next; the cursor is opaque."""

    items: tuple[InboxItem, ...]
    cursor: str


@dataclass(frozen=True)
class BranchLogRecord:
    entry_id: str
    at: float
    entry: BranchLogEntry


@dataclass(frozen=True)
class FrontierScore:
    """Expected gain over the PRELUDE baseline from one more unit of budget, and the parts it is made of."""

    expected_gain_pct: float
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchProgress:
    """One branch at a glance; ``frontier`` is filled from MB2 on."""

    branch: str
    running: bool
    start_commit: str
    head_commit: str
    tests: int
    commits: int
    gpu_seconds: float
    head_value: Mapping[str, float]
    frontier: FrontierScore | None = None


@dataclass(frozen=True)
class CommitNode:
    """A commit of the experiment repository; a merge also names the source commits it brought in."""

    commit_id: str
    parent_commit: str
    branch: str
    origin: CommitOrigin
    request_id: str | None = None
    merged_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class BranchNode:
    branch: str
    origin: LineageKind
    start_commit: str
    head_commit: str
    from_branch: str = ""
    stop_reason: str | None = None


@dataclass(frozen=True)
class LineageView:
    """The branch tree: every commit and every branch."""

    commits: tuple[CommitNode, ...]
    branches: tuple[BranchNode, ...]

    def commit(self, commit_id: str) -> CommitNode | None:
        return next((node for node in self.commits if node.commit_id == commit_id), None)

    def ancestry(self, commit_id: str) -> tuple[str, ...]:
        """``commit_id`` and every commit it builds on, through parents and the sources merges brought in."""

        by_id = {node.commit_id: node for node in self.commits}
        seen: list[str] = []
        pending = [commit_id]
        while pending:
            current = pending.pop()
            node = by_id.get(current)
            if current in seen or node is None:
                continue
            seen.append(current)
            pending.extend([*node.merged_from, node.parent_commit])
        return tuple(seen)


@dataclass(frozen=True)
class MergeState:
    """The latest merge attempt a commit took part in."""

    outcome: MergeOutcome
    integration_head: str
    request_id: str | None = None
    merged_commit: str | None = None


@dataclass(frozen=True)
class CommittedCommit:
    """A specialist commit as the integrator sees it; ``gain_pct`` is its own gain over its parent commit."""

    commit_id: str
    branch: str
    parent_commit: str
    request_id: str
    gain_pct: float
    fixes_correctness: bool
    foundation_fix: bool
    merged_into: str | None = None
    last_merge: MergeState | None = None


@dataclass(frozen=True)
class RecordedValue:
    """A commit's recorded value per metric, and the tests it comes from (the baseline tests for the root)."""

    commit_id: str
    values: Mapping[str, float]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class TriedDirection:
    """One test on a state a new specialist starts from or builds on."""

    commit_id: str
    request_id: str
    target: str
    mechanism: str
    layer: IntentLayer | None
    verdict: Verdict | None
    conclusion: Conclusion | None
    gain_pct: float | None


@dataclass(frozen=True)
class StartupBrief:
    """What a fill specialist reads at start; ``leads`` and ``lessons`` come from cross-session recall (MB2)."""

    start_commit: str
    tried: tuple[TriedDirection, ...]
    leads: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    entry_id: str
    kind: str
    branch: str
    summary: str


__all__ = [
    "CLAIM_TOKEN_LENGTH",
    "BaselineRecord",
    "BranchLogRecord",
    "BranchNode",
    "BranchProgress",
    "Claim",
    "ClaimKey",
    "CommitNode",
    "CommittedCommit",
    "DedupAnswer",
    "FrontierScore",
    "InboxItem",
    "InboxPage",
    "LineageEvent",
    "LineageView",
    "MergeAttempt",
    "MergeState",
    "QuarantineEvent",
    "RecordedValue",
    "RegressionCase",
    "SearchHit",
    "StartupBrief",
    "TestRecord",
    "ToolResultPointer",
    "TriedDirection",
    "Writer",
]
