# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's lesson steps (RSI): the session's learnings into the lesson pool, and what its branches did after a pooled
lesson was shown to them.

:func:`publish_lessons` writes every learning of the session that no later one superseded: a new lesson, or, when it
supersedes a past session's lesson, directly or through an earlier learning of its branch that did, a revision of that
lesson that replaces its heads. A revision of a lesson that is no longer live begins a lesson of its own. A learning
that ``retires`` a live lesson retires it, even when a later learning of its branch replaced its text: a retirement is
a step the branch took, not a statement. A learning that says what a live lesson or an earlier learning of the session
says, once case and spacing are set aside or :data:`NEAR_DUPLICATE` of their words are shared, adds nothing. A session
starts at most :data:`MAX_NEW_LESSONS` lessons, those that cite tests first; a revision of a live lesson is never
held back. Every id comes from the session and the blackboard entry, so a CLOSE run again writes nothing twice.

:func:`session_standings` reads the session's retrieval log beside its blackboard. For each pooled lesson a branch was
shown, from the first showing on, it sets down what the branch's explore tests of the lesson's direction family did
(:class:`~hyperloom_multibranch.blackboard.rsi.lessons.LessonStanding`): borne out when they came out as the lesson says,
none committing after a pitfall, one after an insight or a hypothesis, contradicted when they came out the other way.
A branch that tested none of the family, or a lesson that names no family or says nothing of how a direction comes
out, counts as shown only. A branch that wrote a live lesson again without having been shown it rediscovered it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyperloom_multibranch.contracts.blackboard import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import (
    Learning,
    LearningRecord,
    BlackboardSnapshot,
    SessionProfile,
    TestRecord,
    is_lesson_id,
)
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.blackboard.families import note_family
from hyperloom_multibranch.blackboard.knowledge.learnings import measures
from hyperloom_multibranch.blackboard.learning_search import current_learnings, outcome_of, word_overlap, words
from hyperloom_multibranch.blackboard.rsi.learning_eval import PAST, first_shown, rediscoveries
from hyperloom_multibranch.blackboard.rsi.lessons import (
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
from hyperloom_multibranch.blackboard.rsi.menu import explored, record_family
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions

NEW, REVISED, RETIRED, DUPLICATE, CAPPED, FAILED = "new", "revised", "retired", "duplicate", "capped", "failed"
#: The share of its words a learning shares with a live lesson or an earlier learning that makes it say the same.
NEAR_DUPLICATE = 0.8
#: How many lessons a session starts at most.
MAX_NEW_LESSONS = 12
#: The kinds of learning that say a direction pays; a pitfall says it fails.
_SAYS_IT_PAYS = frozenset({LearningKind.INSIGHT, LearningKind.HYPOTHESIS})
_BORNE_OUT, _CONTRADICTED, _SHOWN, _REDISCOVERED = "borne_out", "contradicted", "shown", "rediscovered"


@dataclass(frozen=True)
class LessonPublication:
    """What :func:`publish_lessons` did, by outcome: ``new`` lessons, ``revised`` and ``retired`` ones, ``duplicate``
    learnings a live lesson or an earlier learning already says, ``capped`` ones past :data:`MAX_NEW_LESSONS`, and
    writes that ``failed``, each failure's reason in ``reasons``."""

    counts: Mapping[str, int] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def written(self) -> int:
        return sum(self.counts.get(outcome, 0) for outcome in (NEW, REVISED, RETIRED))


def _profile(snapshot: BlackboardSnapshot) -> SessionProfile:
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

    def __init__(self, snapshot: BlackboardSnapshot, pool: LessonPool) -> None:
        profile = _profile(snapshot)
        self._pool = pool
        self._session = profile.session_id
        self._conditions = task_conditions(profile)
        self._records = {test.request_id: test for test in snapshot.records}
        self._by_id = {record.entry_id: record for record in snapshot.learnings}
        self._lessons = pool.lessons()
        self._said = {_normal(head.content) for head in self._live_heads()}
        self._worded = [words(head.content) for head in self._live_heads()]
        self.counts: Counter[str] = Counter()
        self.reasons: list[str] = []

    def _says_again(self, text: str) -> bool:
        mine = words(text)
        return _normal(text) in self._said or any(
            word_overlap(mine, theirs) >= NEAR_DUPLICATE for theirs in self._worded
        )

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
        if self._says_again(learning.content):
            self.counts[DUPLICATE] += 1
            return
        lesson = self._live(_revised_lesson(record, self._by_id))
        if lesson is None and self.counts[NEW] >= MAX_NEW_LESSONS:
            self.counts[CAPPED] += 1
            return
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
        self._said.add(_normal(learning.content))
        self._worded.append(words(learning.content))


def publish_lessons(snapshot: BlackboardSnapshot, pool: LessonPool) -> LessonPublication:
    """CLOSE: the session's learnings into ``pool``; raises :class:`NotRecorded` before PRELUDE wrote the profile."""

    publication = _Publication(snapshot, pool)
    for record in snapshot.learnings:
        if record.learning.retires is not None:
            publication.retire(record)
    cited_first = sorted(current_learnings(snapshot), key=lambda record: (not record.learning.request_ids, record.at))
    for record in cited_first:
        if record.learning.retires is None:
            publication.publish(record)
    return LessonPublication(dict(publication.counts), tuple(publication.reasons))


def _outcome(head: LessonRevision, later: Sequence[TestRecord], snapshot: BlackboardSnapshot) -> str:
    """What a branch's explore tests ``later`` than its showing of ``head`` say of the lesson, by those of its family;
    empty when they say nothing."""

    family = note_family(head.target, head.mechanism, head.layer, head.content)
    if family is None or (head.kind is not LearningKind.PITFALL and head.kind not in _SAYS_IT_PAYS):
        return ""
    tried = [record for record in later if (found := record_family(snapshot, record)) and family.covers(found)]
    if not tried:
        return ""
    committed = any(record.commit_id is not None for record in tried)
    return _CONTRADICTED if committed == (head.kind is LearningKind.PITFALL) else _BORNE_OUT


def session_standings(
    snapshot: BlackboardSnapshot, pool: LessonPool, log: Sequence[Mapping[str, Any]]
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
    tests: dict[str, list[TestRecord]] = defaultdict(list)
    for record in snapshot.records:
        if explored(record):
            tests[record.request.branch].append(record)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (branch, lesson_id), at in shown.items():
        counts[lesson_id][_SHOWN] += 1
        later = [record for record in tests.get(branch, ()) if record.requested_at > at]
        outcome = _outcome(latest[lesson_id], later, snapshot)
        if outcome:
            counts[lesson_id][outcome] += 1
    texts = {lesson_id: (head.session_id, head.content) for lesson_id, head in live.items()}
    for _branch, lesson_id in rediscoveries(snapshot, texts, shown):
        counts[lesson_id][_REDISCOVERED] += 1
    standings = {lesson_id: LessonStanding(**found) for lesson_id, found in counts.items()}
    return SessionStandings(profile.session_id, snapshot.taken_at, standings)


__all__ = [
    "CAPPED",
    "DUPLICATE",
    "FAILED",
    "MAX_NEW_LESSONS",
    "NEAR_DUPLICATE",
    "NEW",
    "RETIRED",
    "REVISED",
    "LessonPublication",
    "publish_lessons",
    "session_standings",
]
