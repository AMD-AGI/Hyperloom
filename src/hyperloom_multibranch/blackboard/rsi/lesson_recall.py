# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' lessons, from the RSI lesson pool, for a branch's next decision (RSI).

:class:`LessonRecall` is the session's one learnings finder with a lesson pool. This session's learnings and the
pool's live lessons that bear on a query come in one list (:meth:`LessonRecall.notes`): the family asked about first,
the same family before the same target (:func:`~hyperloom_multibranch.blackboard.learning_search.kinship`), then by the
words they share, a word of the direction counting double, a lesson's weighed by how similar the session it was
learnt in is to this one (none below :data:`~hyperloom_multibranch.blackboard.rsi.similarity.MIN_SIMILARITY`) and by its
standing (:func:`~hyperloom_multibranch.blackboard.rsi.lessons.standing_factor`); among equals this session's comes
first, then the newer. A lesson's family is that of the direction it names, else of its words. A lesson this session
revised or retired is no longer shown; the learning that did it is, among this session's. Each lesson shows its id,
which a learning of this session may revise or retire it by, the conditions it differs in, whether a test measured it
and how often sessions bore it out. A granted claim, the learnings tool and a brief list
:data:`~hyperloom_multibranch.blackboard.learning_search.NOTES` of them; a menu line of a brief, its family's best note
(:meth:`LessonRecall.family_note`). The pool is read once, when first asked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from hyperloom_kb import ReadStatus
from hyperloom_multibranch.contracts.blackboard.records import LearningRecord, BlackboardSnapshot, is_lesson_id
from hyperloom_multibranch.blackboard.families import Family, note_family
from hyperloom_multibranch.blackboard.knowledge.learning_recall import (
    LearningRecall,
    Notes,
    PastLearning,
    PastLearnings,
)
from hyperloom_multibranch.blackboard.knowledge.recall import ExperienceReader
from hyperloom_multibranch.blackboard.learning_search import (
    NOTES,
    SAME_FAMILY,
    Match,
    Retrieval,
    RetrievalLog,
    RetrievalSource,
    kinship,
    note_lines,
    note_score,
    query_family,
    render_session_learning,
    session_scored,
    words,
)
from hyperloom_multibranch.blackboard.rsi.lessons import (
    Lesson,
    LessonPool,
    LessonRevision,
    standing_factor,
    standing_of,
)
from hyperloom_multibranch.blackboard.rsi.similarity import (
    MIN_SIMILARITY,
    TaskConditions,
    differing_conditions,
    task_conditions,
    task_similarity,
)

#: A family's notes a menu line of a brief shows.
FAMILY_NOTES = 1


@dataclass(frozen=True)
class _Loaded:
    lessons: dict[str, Lesson]
    heads: tuple[LessonRevision, ...]
    factors: dict[str, float]
    confirmations: dict[str, int]
    revision: str = ""


def edited_lessons(snapshot: BlackboardSnapshot) -> frozenset[str]:
    """The past sessions' lessons a learning of this session revised or retired."""

    named = (record.learning.retires or record.learning.supersedes for record in snapshot.learnings)
    return frozenset(lesson for lesson in named if is_lesson_id(lesson))


def _item(head: LessonRevision, task: TaskConditions, confirmations: int) -> PastLearning:
    source = head.session_id or head.author
    differing = differing_conditions(task, head.conditions)
    return PastLearning(head.lesson_id, head.content, head.measured, source, differing, confirmations=confirmations)


def _ranked(
    here: list[tuple[Match, LearningRecord]], past: list[tuple[Match, PastLearning]], limit: int
) -> list[LearningRecord | PastLearning]:
    """``here`` and ``past`` in one list, best first, this session's first among equals."""

    merged: list[tuple[Match, int, LearningRecord | PastLearning]] = [(match, 0, item) for match, item in here]
    merged += [(match, 1, item) for match, item in past]
    merged.sort(key=lambda entry: (-entry[0].kin, -entry[0].score, entry[1], -entry[0].at))
    return [item for _, _, item in merged[:limit]]


def _notes(chosen: list[LearningRecord | PastLearning], snapshot: BlackboardSnapshot) -> Notes:
    lines = tuple(
        render_session_learning(item, snapshot) if isinstance(item, LearningRecord) else item.render()
        for item in chosen
    )
    here = tuple(item for item in chosen if isinstance(item, LearningRecord))
    past = tuple(item for item in chosen if isinstance(item, PastLearning))
    return Notes(lines, here, past, tuple(isinstance(item, LearningRecord) for item in chosen))


class LessonRecall(LearningRecall):
    """One session's retrieval over its learnings and the lesson pool ``pool``, shared by every branch's tools and
    briefs; ``reader``, ``identities`` and ``log`` as for :class:`LearningRecall`."""

    def __init__(
        self, reader: ExperienceReader | None = None, *, pool: LessonPool | None = None, log: RetrievalLog | None = None
    ) -> None:
        super().__init__(reader, log=log)
        self._pool = pool
        self._loaded: _Loaded | None = None

    def _load(self) -> _Loaded:
        with self._lock:
            if self._loaded is None:
                lessons = self._pool.lessons() if self._pool is not None else {}
                standings = self._pool.standings() if self._pool is not None else ()
                heads = tuple(head for lesson in lessons.values() if lesson.retired is None for head in lesson.heads)
                found = {key: standing_of(standings, key) for key in lessons}
                factors = {key: standing_factor(standing) for key, standing in found.items()}
                confirmations = {key: standing.confirmations for key, standing in found.items()}
                revision = self._pool.digest() if self._pool is not None else ""
                self._loaded = _Loaded(lessons, heads, factors, confirmations, revision)
            return self._loaded

    def past_revision(self) -> str:
        """The pool's digest as this recall read it, the one read a session works on; empty without a pool."""

        return self._load().revision

    def knows(self, lesson_id: str) -> bool:
        """Whether ``lesson_id`` names a live lesson of the pool."""

        lesson = self._load().lessons.get(lesson_id)
        return lesson is not None and lesson.retired is None

    def _scored(
        self, snapshot: BlackboardSnapshot, query: str, family: Family | None
    ) -> list[tuple[Match, PastLearning]]:
        """The pool's live lessons that bear on ``query`` or are about ``family``, each with its match; none before
        the session's profile."""

        if snapshot.profile is None or self._pool is None:
            return []
        loaded = self._load()
        task = task_conditions(snapshot.profile)
        hidden = edited_lessons(snapshot)
        wanted = words(query)
        found = []
        for head in loaded.heads:
            if head.lesson_id in hidden or head.session_id == snapshot.profile.session_id:
                continue
            similarity = task_similarity(task, head.conditions)
            if similarity < MIN_SIMILARITY:
                continue
            kin = kinship(family, note_family(head.target, head.mechanism, head.layer, head.content))
            overlap = note_score(wanted, head.content, head.target, head.mechanism)
            if kin or overlap:
                weight = similarity * loaded.factors.get(head.lesson_id, 1.0)
                item = _item(head, task, loaded.confirmations.get(head.lesson_id, 0))
                found.append((Match(kin, overlap * weight, head.written_at), item))
        return found

    def pooled(
        self, snapshot: BlackboardSnapshot, query: str, *, limit: int, family: Family | None = None
    ) -> tuple[PastLearning, ...]:
        """The pool's live lessons that bear on ``query``, about ``family`` first, the best ``limit`` first."""

        return tuple(_ranked([], self._scored(snapshot, query, family), limit))

    def past_for(self, snapshot: BlackboardSnapshot, query: str, *, limit: int) -> PastLearnings | None:
        """The pool's lessons for ``query``, after the Experience KB's when one is configured; ``None`` before the
        session's profile, or with neither."""

        read = super().past_for(snapshot, query, limit=limit)
        if snapshot.profile is None or self._pool is None:
            return read
        kept = read.items if read is not None and read.available else ()
        items = (*kept, *self.pooled(snapshot, query, limit=limit))[:limit]
        warnings = read.warnings if read is not None else ()
        return PastLearnings(ReadStatus.COMPLETED.value, read.read_id if read is not None else "", items, warnings)

    def notes(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, family: Family | None = None) -> Notes:
        """This session's learnings and the pool's lessons that bear on ``query``, in one list of ``limit``."""

        chosen = _ranked(session_scored(snapshot, query, family), self._scored(snapshot, query, family), limit)
        return _notes(chosen, snapshot)

    def family_note(self, snapshot: BlackboardSnapshot, family: Family) -> Notes:
        """The best of this session's learnings and the pool's lessons about ``family`` itself, for its menu line."""

        same = [(match, record) for match, record in session_scored(snapshot, "", family) if match.kin == SAME_FAMILY]
        pooled = [(match, item) for match, item in self._scored(snapshot, "", family) if match.kin == SAME_FAMILY]
        return _notes(_ranked(same, pooled, FAMILY_NOTES), snapshot)

    def _logged(self, snapshot: BlackboardSnapshot, retrieval: Retrieval, found: Notes) -> list[str]:
        self.record(snapshot, retrieval, found.here, found.logged())
        return note_lines(found.lines)

    async def about(
        self, snapshot: BlackboardSnapshot, query: str, *, branch: str, family: Family | None = None
    ) -> list[str]:
        """The :data:`~hyperloom_multibranch.blackboard.learning_search.NOTES` notes about a claimed direction, in one
        list, as one retrieval of the log."""

        found = await asyncio.to_thread(self.notes, snapshot, query, limit=NOTES, family=family)
        return self._logged(snapshot, Retrieval(branch, RetrievalSource.CLAIM, query), found)

    async def lines(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """The ``limit`` notes that bear on ``query``, the family it names first, in one list."""

        found = await asyncio.to_thread(self.notes, snapshot, query, limit=limit, family=query_family(query))
        return self._logged(snapshot, Retrieval(branch, RetrievalSource.TOOL, query), found)


__all__ = ["FAMILY_NOTES", "LessonRecall", "edited_lessons"]
