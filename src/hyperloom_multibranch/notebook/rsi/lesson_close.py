# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's lesson steps (RSI): the session's learnings into the lesson pool, and what its branches did after a pooled
lesson was shown to them.

:func:`publish_lessons` writes every learning of the session that no later one superseded: a new lesson, or, when it
supersedes a past session's lesson, directly or through an earlier learning of its branch that did, a revision of that
lesson that replaces its heads. A revision of a lesson that is no longer live begins a lesson of its own. A learning
that ``retires`` a live lesson retires it, even when a later learning of its branch replaced its text: a retirement is
a step the branch took, not a statement. A learning that says what a live lesson says, once case and spacing are set
aside, adds nothing. Every id comes from the session and the notebook entry, so a CLOSE run again writes nothing
twice.

:func:`session_standings` reads the session's retrieval log beside its notebook. For each pooled lesson a branch was
shown, from the first showing on, it sets down what the branch's tests in the lesson's direction did
(:class:`~hyperloom_multibranch.notebook.rsi.lessons.LessonStanding`); a lesson that names no direction counts as
shown only. A branch that wrote a live lesson again without having been shown it rediscovered it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.records import (
    Learning,
    LearningRecord,
    NotebookSnapshot,
    SessionProfile,
    is_lesson_id,
)
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.knowledge.learnings import measures
from hyperloom_multibranch.notebook.learning_search import current_learnings, names, outcome_of
from hyperloom_multibranch.notebook.rsi.learning_eval import PAST, BranchTest, branch_tests, first_shown, rediscoveries
from hyperloom_multibranch.notebook.rsi.lessons import (
    Lesson,
    LessonEvidence,
    LessonPool,
    LessonRetirement,
    LessonRevision,
    LessonStanding,
    SessionStandings,
    lesson_id_for,
    revision_id_for,
    specialist_author,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions

NEW, REVISED, RETIRED, DUPLICATE, FAILED = "new", "revised", "retired", "duplicate", "failed"
_TAKEN_UP = frozenset({LearningKind.INSIGHT, LearningKind.HYPOTHESIS})
_HEEDED, _IGNORED, _CONTRADICTED, _PAID, _SHOWN, _REDISCOVERED = (
    "heeded",
    "ignored",
    "contradicted",
    "paid",
    "shown",
    "rediscovered",
)


@dataclass(frozen=True)
class LessonPublication:
    """What :func:`publish_lessons` did, by outcome: ``new`` lessons, ``revised`` and ``retired`` ones, ``duplicate``
    learnings a live lesson already says, and writes that ``failed``, each failure's reason in ``reasons``."""

    counts: Mapping[str, int] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def written(self) -> int:
        return sum(self.counts.get(outcome, 0) for outcome in (NEW, REVISED, RETIRED))


def _profile(snapshot: NotebookSnapshot) -> SessionProfile:
    if snapshot.profile is None:
        raise NotRecorded("PRELUDE has not written the session profile yet")
    return snapshot.profile


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def _revised_lesson(record: LearningRecord, by_id: Mapping[str, LearningRecord]) -> str | None:
    """The past session's lesson ``record`` revises: the one it supersedes, or the one an earlier learning of its
    branch that it supersedes revised."""

    seen: set[str] = set()
    reference = record.learning.supersedes
    while reference is not None and not is_lesson_id(reference) and reference not in seen:
        seen.add(reference)
        earlier = by_id.get(reference)
        reference = earlier.learning.supersedes if earlier is not None else None
    return reference if is_lesson_id(reference) else None


class _Publication:
    """One session's learnings on their way into ``pool``, against the pool as it stood when CLOSE began."""

    def __init__(self, snapshot: NotebookSnapshot, pool: LessonPool) -> None:
        profile = _profile(snapshot)
        self._pool = pool
        self._session = profile.session_id
        self._conditions = task_conditions(profile)
        self._records = {test.request_id: test for test in snapshot.records}
        self._by_id = {record.entry_id: record for record in snapshot.learnings}
        self._lessons = pool.lessons()
        self._said = {_normal(head.content) for head in self._live_heads()}
        self.counts: Counter[str] = Counter()
        self.reasons: list[str] = []

    def _live_heads(self) -> list[LessonRevision]:
        return [head for lesson in self._lessons.values() if lesson.retired is None for head in lesson.heads]

    def _live(self, lesson_id: str | None) -> Lesson | None:
        lesson = self._lessons.get(lesson_id) if lesson_id is not None else None
        return lesson if lesson is not None and lesson.retired is None else None

    def _evidence(self, learning: Learning) -> tuple[LessonEvidence, ...]:
        cited = (self._records[request_id] for request_id in learning.request_ids if request_id in self._records)
        return tuple(LessonEvidence(self._session, test.request_id, outcome_of(test), measures(test)) for test in cited)

    def _write(self, outcome: str, write: Callable[[], bool]) -> None:
        try:
            if write():
                self.counts[outcome] += 1
        except (OSError, ValueError) as exc:
            self.counts[FAILED] += 1
            self.reasons.append(f"{type(exc).__name__}: {exc}")

    def retire(self, record: LearningRecord) -> None:
        lesson = self._live(record.learning.retires)
        if lesson is None:
            return
        author = specialist_author(self._session, record.learning.branch)
        retirement = LessonRetirement(
            lesson.lesson_id, author, record.at, record.learning.content, self._evidence(record.learning)
        )
        self._write(RETIRED, lambda: self._pool.retire(retirement))

    def publish(self, record: LearningRecord) -> None:
        learning = record.learning
        text = _normal(learning.content)
        if text in self._said:
            self.counts[DUPLICATE] += 1
            return
        lesson = self._live(_revised_lesson(record, self._by_id))
        revision = LessonRevision(
            revision_id_for(self._session, record.entry_id),
            lesson.lesson_id if lesson is not None else lesson_id_for(self._session, record.entry_id),
            learning.kind,
            learning.content,
            self._conditions,
            specialist_author(self._session, learning.branch),
            record.at,
            session_id=self._session,
            target=learning.target,
            mechanism=learning.mechanism,
            layer=learning.layer,
            supersedes=tuple(head.revision_id for head in lesson.heads) if lesson is not None else (),
            evidence=self._evidence(learning),
        )
        self._write(REVISED if lesson is not None else NEW, lambda: self._pool.write(revision))
        self._said.add(text)


def publish_lessons(snapshot: NotebookSnapshot, pool: LessonPool) -> LessonPublication:
    """CLOSE: the session's learnings into ``pool``; raises :class:`NotRecorded` before PRELUDE wrote the profile."""

    publication = _Publication(snapshot, pool)
    for record in snapshot.learnings:
        if record.learning.retires is not None:
            publication.retire(record)
    for record in current_learnings(snapshot):
        if record.learning.retires is None:
            publication.publish(record)
    return LessonPublication(dict(publication.counts), tuple(publication.reasons))


def _pitfall_outcome(probe: Learning, later: Sequence[BranchTest]) -> str:
    tried = [test for test in later if names(probe, test.intent)]
    if not tried:
        return _HEEDED
    if any(test.committed for test in tried):
        return _CONTRADICTED
    return _IGNORED if any(test.ended for test in tried) else ""


def _outcome(head: LessonRevision, later: Sequence[BranchTest]) -> str:
    """What a branch's tests after it was shown ``head`` say of the lesson; empty when they say nothing."""

    if not (head.target or head.mechanism):
        return ""
    probe = Learning("", head.kind, head.content, head.target, head.mechanism)
    if head.kind is LearningKind.PITFALL:
        return _pitfall_outcome(probe, later)
    paid = head.kind in _TAKEN_UP and bool(later) and names(probe, later[0].intent) and later[0].committed
    return _PAID if paid else ""


def session_standings(
    snapshot: NotebookSnapshot, pool: LessonPool, log: Sequence[Mapping[str, Any]]
) -> SessionStandings:
    """What the session's branches did after each pooled lesson ``log`` says they were shown, and which live lessons
    they wrote again unshown; raises :class:`NotRecorded` before PRELUDE wrote the profile."""

    profile = _profile(snapshot)
    lessons = pool.lessons()
    latest = {
        lesson_id: max(lesson.heads, key=lambda head: head.written_at)
        for lesson_id, lesson in lessons.items()
        if lesson.heads
    }
    live = {lesson_id: head for lesson_id, head in latest.items() if lessons[lesson_id].retired is None}
    shown = {pair: at for pair, at in first_shown(log, PAST).items() if pair[1] in latest}
    tests = branch_tests(snapshot)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (branch, lesson_id), at in shown.items():
        counts[lesson_id][_SHOWN] += 1
        outcome = _outcome(latest[lesson_id], [test for test in tests.get(branch, ()) if test.at > at])
        if outcome:
            counts[lesson_id][outcome] += 1
    texts = {lesson_id: (head.session_id, head.content) for lesson_id, head in live.items()}
    for _branch, lesson_id in rediscoveries(snapshot, texts, shown):
        counts[lesson_id][_REDISCOVERED] += 1
    standings = {lesson_id: LessonStanding(**found) for lesson_id, found in counts.items()}
    return SessionStandings(profile.session_id, snapshot.taken_at, standings)


__all__ = [
    "DUPLICATE",
    "FAILED",
    "NEW",
    "RETIRED",
    "REVISED",
    "LessonPublication",
    "publish_lessons",
    "session_standings",
]
