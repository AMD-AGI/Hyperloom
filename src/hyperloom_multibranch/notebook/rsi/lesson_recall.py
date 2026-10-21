# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' lessons, from the RSI lesson pool, for a branch's next decision (RSI).

:class:`LessonRecall` is the session's one learnings finder with a lesson pool: this session's learnings first, as
:class:`~hyperloom_multibranch.notebook.knowledge.learning_recall.LearningRecall` gives them, then the pool's live
lessons that bear on the query, after any the Experience KB gives when one is configured. A lesson bears on a query by
the words they share, a word of the direction it is about counting double, times how similar the session it was learnt
in is to this one (none below
:data:`~hyperloom_multibranch.notebook.rsi.similarity.MIN_SIMILARITY`), times its standing
(:func:`~hyperloom_multibranch.notebook.rsi.lessons.standing_factor`); among equals the newer comes first. A lesson this
session revised or retired is no longer shown here; the learning that did it is, among this session's. Each lesson
shows its id, which a learning of this session may revise or retire it by, the conditions it differs in and whether a
test measured it. A granted claim lists :data:`CLAIM_LESSONS` of them after this session's learnings about its
direction. The pool is read once, when first asked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from hyperloom_kb import ReadStatus
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, is_lesson_id
from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall, PastLearning, PastLearnings
from hyperloom_multibranch.notebook.knowledge.recall import ExperienceReader
from hyperloom_multibranch.notebook.learning_search import (
    CLAIM_LEARNINGS,
    Retrieval,
    RetrievalLog,
    RetrievalSource,
    claim_lines,
    session_matches,
    words,
)
from hyperloom_multibranch.notebook.rsi.lessons import (
    Lesson,
    LessonPool,
    LessonRevision,
    standing_factor,
    standing_of,
)
from hyperloom_multibranch.notebook.rsi.similarity import (
    MIN_SIMILARITY,
    TaskConditions,
    differing_conditions,
    task_conditions,
    task_similarity,
)

#: How many past sessions' lessons a granted claim lists.
CLAIM_LESSONS = 2
CLAIM_LESSONS_HEADING = "Past sessions' lessons about it:"


@dataclass(frozen=True)
class _Loaded:
    lessons: dict[str, Lesson]
    heads: tuple[LessonRevision, ...]
    factors: dict[str, float]


def edited_lessons(snapshot: NotebookSnapshot) -> frozenset[str]:
    """The past sessions' lessons a learning of this session revised or retired."""

    named = (record.learning.retires or record.learning.supersedes for record in snapshot.learnings)
    return frozenset(lesson for lesson in named if is_lesson_id(lesson))


def _item(head: LessonRevision, task: TaskConditions) -> PastLearning:
    source = head.session_id or head.author
    return PastLearning(
        head.lesson_id, head.content, head.measured, source, differing_conditions(task, head.conditions)
    )


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
                factors = {key: standing_factor(standing_of(standings, key)) for key in lessons}
                self._loaded = _Loaded(lessons, heads, factors)
            return self._loaded

    def knows(self, lesson_id: str) -> bool:
        """Whether ``lesson_id`` names a live lesson of the pool."""

        lesson = self._load().lessons.get(lesson_id)
        return lesson is not None and lesson.retired is None

    def pooled(self, snapshot: NotebookSnapshot, query: str, *, limit: int) -> tuple[PastLearning, ...]:
        """The pool's live lessons that bear on ``query`` for ``snapshot``'s session, the best ``limit`` first; none
        before its profile."""

        if snapshot.profile is None or self._pool is None:
            return ()
        loaded = self._load()
        task = task_conditions(snapshot.profile)
        hidden = edited_lessons(snapshot)
        wanted = words(query)
        scored = []
        for head in loaded.heads:
            if head.lesson_id in hidden or head.session_id == snapshot.profile.session_id:
                continue
            similarity = task_similarity(task, head.conditions)
            overlap = len(wanted & words(head.content)) + 2 * len(wanted & words(f"{head.target} {head.mechanism}"))
            if overlap and similarity >= MIN_SIMILARITY:
                scored.append((overlap * similarity * loaded.factors.get(head.lesson_id, 1.0), head.written_at, head))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return tuple(_item(head, task) for _, _, head in scored[:limit])

    def past_for(self, snapshot: NotebookSnapshot, query: str, *, limit: int) -> PastLearnings | None:
        """The pool's lessons for ``query``, after the Experience KB's when one is configured; ``None`` before the
        session's profile, or with neither."""

        read = super().past_for(snapshot, query, limit=limit)
        if snapshot.profile is None or self._pool is None:
            return read
        kept = read.items if read is not None and read.available else ()
        items = (*kept, *self.pooled(snapshot, query, limit=limit))[:limit]
        warnings = read.warnings if read is not None else ()
        return PastLearnings(ReadStatus.COMPLETED.value, read.read_id if read is not None else "", items, warnings)

    async def about(self, snapshot: NotebookSnapshot, query: str, *, branch: str) -> list[str]:
        """This session's learnings about a claimed direction, then :data:`CLAIM_LESSONS` past sessions' lessons, as
        one retrieval of the log."""

        here = session_matches(snapshot, query, limit=CLAIM_LEARNINGS)
        lessons = await asyncio.to_thread(self.pooled, snapshot, query, limit=CLAIM_LESSONS)
        past = PastLearnings(ReadStatus.COMPLETED.value, items=lessons) if lessons else None
        self.record(snapshot, Retrieval(branch, RetrievalSource.CLAIM, query), here, past)
        listed = [CLAIM_LESSONS_HEADING, *(item.render() for item in lessons)] if lessons else []
        return [*claim_lines(here, snapshot), *listed]


__all__ = ["CLAIM_LESSONS", "CLAIM_LESSONS_HEADING", "LessonRecall", "edited_lessons"]
