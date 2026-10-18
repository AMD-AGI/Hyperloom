# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What this session's own record says before the next decision.

* Failure tiers: a failed test of a branch's concern is ``same_state`` when it was tried on the branch head itself,
  ``ancestor`` when on a commit the head builds on, ``elsewhere`` otherwise. A later commit of the same intent on the
  same state redeems it.
* Direction occupancy: per direction (target, mechanism, layer), the states it was tried on and how, the running
  branches whose head carries one of its commits, and the branches holding a live claim on it.
* Hypothesis ledger: the latest version of every hypothesis a specialist logged, following ``supersedes``.

Whether two branches converge is the scheduler's call and is not analysed here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.contracts.notebook.peers import BranchLogType, HypothesisStatus, Intent, IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import COMMITTING, ClaimStatus, Conclusion
from hyperloom_multibranch.notebook.knowledge.lessons import LessonKind, lesson_of

_REDEEMING = COMMITTING | {Conclusion.MERGED}
_FAILING = frozenset({LessonKind.FAILED, LessonKind.PITFALL})


class FailureTier(str, Enum):
    SAME_STATE = "same_state"
    ANCESTOR = "ancestor"
    ELSEWHERE = "elsewhere"


_TIER_ORDER = list(FailureTier)


@dataclass(frozen=True)
class FailedTest:
    tier: FailureTier
    request_id: str
    branch: str
    parent_commit: str
    direction: str
    outcome: str
    gain_pct: float | None

    def render(self) -> str:
        return f"- [{self.tier.value}] {self.direction} on {self.parent_commit}: {self.outcome} ({self.request_id})"


def failures_at(snapshot: NotebookSnapshot, head: str) -> tuple[FailedTest, ...]:
    """Every failed test no commit redeemed, tiered by how close the state it was tried on is to ``head``."""

    ancestry = set(snapshot.lineage.ancestry(head)) if head else set()
    redeemed = {
        (record.request.parent_commit, record.request.intent_fingerprint)
        for record in snapshot.records
        if record.conclusion in _REDEEMING
    }
    failed = []
    for record in snapshot.records:
        lesson = lesson_of(record, snapshot)
        key = (record.request.parent_commit, record.request.intent_fingerprint)
        if lesson is None or lesson.kind not in _FAILING or key in redeemed:
            continue
        if lesson.parent_commit == head:
            tier = FailureTier.SAME_STATE
        else:
            tier = FailureTier.ANCESTOR if lesson.parent_commit in ancestry else FailureTier.ELSEWHERE
        failed.append(
            FailedTest(
                tier,
                record.request_id,
                record.request.branch,
                lesson.parent_commit,
                lesson.direction,
                lesson.outcome,
                lesson.gain_pct,
            )
        )
    return tuple(sorted(failed, key=lambda item: _TIER_ORDER.index(item.tier)))


def repeated_failures(failures: Sequence[FailedTest], at_least: int = 2) -> tuple[tuple[str, int], ...]:
    """The directions that failed at least ``at_least`` times, most often first."""

    counts = Counter(item.direction for item in failures)
    return tuple((direction, count) for direction, count in counts.most_common() if count >= at_least)


@dataclass(frozen=True)
class DirectionCoverage:
    """How the session has covered one direction."""

    target: str
    mechanism: str
    layer: IntentLayer
    tried_on: tuple[str, ...]
    outcomes: tuple[str, ...]
    committed_on: tuple[str, ...]
    claimed_by: tuple[str, ...]
    best_gain_pct: float | None

    @property
    def name(self) -> str:
        return f"{self.target} / {self.mechanism}"

    def render(self) -> str:
        best = "n/a" if self.best_gain_pct is None else f"{self.best_gain_pct:+.2f}%"
        held = f"; claimed by {', '.join(self.claimed_by)}" if self.claimed_by else ""
        carried = f"; on the heads of {', '.join(self.committed_on)}" if self.committed_on else ""
        return f"- {self.name} ({self.layer.value}): tried on {len(self.tried_on)} states, best {best}{carried}{held}"


def _coverage(
    intent: Intent, records: Sequence[TestRecord], snapshot: NotebookSnapshot, heads: dict[str, set[str]]
) -> DirectionCoverage:
    outcomes = tuple(
        lesson.text for lesson in (lesson_of(record, snapshot) for record in records) if lesson is not None
    )
    commits = {record.commit_id for record in records if record.commit_id}
    gains = [record.judgement.gain_pct for record in records if record.judgement is not None]
    return DirectionCoverage(
        target=intent.target,
        mechanism=intent.mechanism,
        layer=intent.layer,
        tried_on=tuple(dict.fromkeys(record.request.parent_commit for record in records)),
        outcomes=outcomes,
        committed_on=tuple(sorted(branch for branch, ancestry in heads.items() if commits & ancestry)),
        claimed_by=tuple(
            sorted(
                {
                    claim.intent.branch
                    for claim in snapshot.claims
                    if claim.status is ClaimStatus.HELD and claim.intent.fingerprint == intent.fingerprint
                }
            )
        ),
        best_gain_pct=max(gains) if gains else None,
    )


def direction_occupancy(snapshot: NotebookSnapshot) -> tuple[DirectionCoverage, ...]:
    """Every direction an intent named, in the order it was first claimed."""

    running = {row.branch for row in snapshot.progress if row.running}
    heads = {
        node.branch: set(snapshot.lineage.ancestry(node.head_commit))
        for node in snapshot.lineage.branches
        if node.branch in running
    }
    return tuple(
        _coverage(
            intent,
            [record for record in snapshot.records if record.request.intent_fingerprint == fingerprint],
            snapshot,
            heads,
        )
        for fingerprint, intent in snapshot.intents.items()
    )


@dataclass(frozen=True)
class HypothesisRow:
    """The latest version of one hypothesis; ``first_entry`` is the entry that posted it."""

    first_entry: str
    latest_entry: str
    branch: str
    statement: str
    status: HypothesisStatus
    evidence: tuple[str, ...]
    updated_at: float

    def render(self) -> str:
        cited = f" [{', '.join(self.evidence)}]" if self.evidence else ""
        return f"- {self.status.value}: {self.statement} ({self.branch}){cited}"


def hypothesis_ledger(snapshot: NotebookSnapshot) -> tuple[HypothesisRow, ...]:
    """Every hypothesis in the order it was first posted, each at its latest version."""

    rows: dict[str, HypothesisRow] = {}
    first_of: dict[str, str] = {}
    for branch, log in snapshot.branch_logs.items():
        for record in log:
            entry = record.entry
            if entry.kind is not BranchLogType.HYPOTHESIS:
                continue
            first = first_of.get(entry.supersedes or "", record.entry_id)
            first_of[record.entry_id] = first
            rows[first] = HypothesisRow(
                first_entry=first,
                latest_entry=record.entry_id,
                branch=branch,
                statement=entry.content,
                status=entry.hypothesis_status or HypothesisStatus.OPEN,
                evidence=entry.request_ids,
                updated_at=record.at,
            )
    return tuple(rows.values())


__all__ = [
    "DirectionCoverage",
    "FailedTest",
    "FailureTier",
    "HypothesisRow",
    "direction_occupancy",
    "failures_at",
    "hypothesis_ledger",
    "repeated_failures",
]
