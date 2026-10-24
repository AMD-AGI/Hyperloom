# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the scheduler's LLM curates the RSI lesson pool, a bounded number of steps a session (RSI).

After the session's learnings are pooled and the lessons it showed have their standing (:mod:`rsi.lesson_close`),
:func:`curate_lessons` offers the LLM at most :data:`MAX_OFFERED` live lessons: those proposed for retirement
(:func:`~hyperloom_multibranch.notebook.rsi.lessons.retirement_proposals`), those two sessions revised at once, those
this session wrote, the lessons that say nearly what one of those says under alike conditions, and those this session
showed. It may take :data:`MAX_STEPS` steps, each one checked before it is written:

* ``merge`` lessons that say the same thing under alike conditions into one: a revision of the one kept that
  replaces its heads, and the others retired as merged into it;
* ``rewrite`` a lesson unclear, too long or out of date on its evidence: a revision that replaces its heads;
* ``retire`` a lesson proposed for retirement, or one a test of this session contradicted.

A text is a learning's, :data:`~hyperloom_multibranch.notebook.local.notebook.LEARNING_MIN_CHARS` to
:data:`~hyperloom_multibranch.notebook.local.notebook.LEARNING_MAX_CHARS` characters. Each written record names the
curator, the session and the model. The step never stops CLOSE: what it could not do it says, as a
:class:`~hyperloom_multibranch.notebook.rsi.session.CloseStep`; what the call cost is the scheduler's to record, as
``UsageSource.CURATOR``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.learning_search import word_overlap, words
from hyperloom_multibranch.notebook.local.notebook import LEARNING_MAX_CHARS, LEARNING_MIN_CHARS
from hyperloom_multibranch.notebook.rsi.archive import NO_ARCHIVE, SKIPPED, TreeArchive
from hyperloom_multibranch.notebook.rsi.authoring import Complete
from hyperloom_multibranch.notebook.rsi.learning_eval import REDISCOVERY_OVERLAP
from hyperloom_multibranch.notebook.rsi.lessons import (
    Lesson,
    LessonPool,
    LessonRetirement,
    LessonRevision,
    SessionStandings,
    curator_author,
    retirement_proposals,
    revision_id_for,
    standing_of,
)
from hyperloom_multibranch.notebook.rsi.session import FAILED, NO_PROFILE, STEP_ERRORS, WRITTEN, CloseStep
from hyperloom_multibranch.notebook.rsi.similarity import MIN_SIMILARITY, task_similarity

log = logging.getLogger(__name__)

MAX_OFFERED = 40
MAX_STEPS = 10
NO_LESSONS = "no lesson to curate"
MERGE, REWRITE, RETIRE = "merge", "rewrite", "retire"
_MERGED, _REWRITTEN, _RETIRED, _REFUSED = "merged", "rewritten", "retired", "refused"
_ACTION, _LESSON, _CONTENT, _WHY = "action", "lesson", "content", "why"

CURATION_SYSTEM_PROMPT = f"""You curate the lesson pool of a multi-branch LLM inference optimization system: what the
specialists of past sessions learnt, which the specialists of later sessions are shown before they decide. Keep the
pool short, true and specific. For the lessons you are given you may take at most {MAX_STEPS} steps:
- merge: lessons that say the same thing under alike conditions become one; give the one to keep, the ones merged into
  it, and the merged text;
- rewrite: a lesson unclear, too long or out of date on its evidence gets a better text;
- retire: a lesson proposed for retirement, or one this session's tests contradicted, is shown to no session again.
Never invent a measurement: every number and every outcome in a text you write is in a lesson you merge or rewrite.
A text says what was learnt in {LEARNING_MIN_CHARS} to {LEARNING_MAX_CHARS} characters. Take no step rather than a
doubtful one. Answer with one JSON object and nothing else:
{{"actions": [{{"action": "merge", "lesson": "lsn-...", "merge": ["lsn-..."], "content": "...", "why": "..."}},
{{"action": "rewrite", "lesson": "lsn-...", "content": "...", "why": "..."}},
{{"action": "retire", "lesson": "lsn-...", "why": "..."}}]}}"""


@dataclass(frozen=True)
class CurationStep:
    """One step the LLM asked for: ``action`` on ``lesson``, the lessons it merges into it, the new text, and why."""

    action: str
    lesson: str
    merge: tuple[str, ...] = ()
    content: str = ""
    why: str = ""


@dataclass(frozen=True)
class _Offer:
    """What the curator may touch: the live lessons offered, which of them retirement may take, and the standings."""

    lessons: Mapping[str, Lesson]
    retirable: frozenset[str]
    standings: tuple[SessionStandings, ...]


def _latest(lesson: Lesson) -> LessonRevision:
    return max(lesson.heads, key=lambda head: head.written_at)


def _alike(first: Lesson, second: Lesson) -> bool:
    return task_similarity(_latest(first).conditions, _latest(second).conditions) >= MIN_SIMILARITY


def _near(mine: Sequence[Lesson], live: Mapping[str, Lesson]) -> list[str]:
    """The live lessons that say nearly what one of ``mine`` says, under alike conditions."""

    said = {lesson.lesson_id: words(_latest(lesson).content) for lesson in live.values()}
    found = []
    for lesson in mine:
        for other in live.values():
            close = word_overlap(said[lesson.lesson_id], said[other.lesson_id]) >= REDISCOVERY_OVERLAP
            if other.lesson_id != lesson.lesson_id and close and _alike(lesson, other):
                found.append(other.lesson_id)
    return found


def _marked(session_id: str, standings: Sequence[SessionStandings]) -> tuple[list[str], set[str]]:
    """The lessons session ``session_id`` showed, and those of them it contradicted."""

    shown: list[str] = []
    contradicted: set[str] = set()
    for item in (one for one in standings if one.session_id == session_id):
        shown.extend(item.lessons)
        contradicted.update(key for key, mark in item.lessons.items() if mark.contradicted)
    return shown, contradicted


def offer(session_id: str, lessons: Mapping[str, Lesson], standings: Sequence[SessionStandings]) -> _Offer:
    """The live lessons the curator of session ``session_id`` is given, the most pressing first, :data:`MAX_OFFERED`
    at most, and the ones among them it may retire without a contradiction of this session."""

    live = {key: lesson for key, lesson in lessons.items() if lesson.retired is None and lesson.heads}
    proposed = [key for key in retirement_proposals(lessons, standings) if key in live]
    forked = [key for key, lesson in live.items() if len(lesson.heads) > 1]
    mine = [lesson for lesson in live.values() if any(head.session_id == session_id for head in lesson.heads)]
    shown, contradicted = _marked(session_id, standings)
    ordered = [*proposed, *forked, *(lesson.lesson_id for lesson in mine), *_near(mine, live), *shown]
    chosen = [key for key in dict.fromkeys(ordered) if key in live][:MAX_OFFERED]
    return _Offer({key: live[key] for key in chosen}, frozenset(proposed) | contradicted, tuple(standings))


def _lesson_lines(lesson: Lesson, found: _Offer) -> list[str]:
    head = _latest(lesson)
    standing = standing_of(found.standings, lesson.lesson_id)
    about = f" about {head.target} / {head.mechanism}" if head.target or head.mechanism else ""
    retirable = "; retirable" if lesson.lesson_id in found.retirable else ""
    record = (
        f"shown to {standing.shown} branches: heeded {standing.heeded}, ignored and failed {standing.ignored}, "
        f"contradicted {standing.contradicted}, paid {standing.paid}, written again unshown {standing.rediscovered}"
    )
    lines = [f"- {lesson.lesson_id} [{head.kind.value}{about}] {record}{retirable}"]
    for one in lesson.heads:
        tested = "; ".join(f"{item.request_id}: {item.outcome}" for item in one.evidence) or "no test cited"
        lines.append(f"  text ({one.author}): {one.content} (tests: {tested})")
    return lines


def curation_prompt(snapshot: NotebookSnapshot, found: _Offer) -> str:
    """What the curator reads: this session's conditions, then each offered lesson with its record and its texts."""

    profile = snapshot.profile
    conditions = f"{profile.model} on {profile.hardware}, {profile.framework}, tp {profile.tp}" if profile else ""
    body = [line for lesson in found.lessons.values() for line in _lesson_lines(lesson, found)]
    return "\n".join([f"This session: {conditions}.", "", "The lessons you may curate:", *body])


def _steps(answer: str) -> list[Any] | None:
    start, end = answer.find("{"), answer.rfind("}")
    try:
        document = json.loads(answer[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return None
    actions = document.get("actions") if isinstance(document, dict) else None
    return actions if isinstance(actions, list) else None


def _text(raw: object) -> str:
    return " ".join(str(raw or "").split())


def _refusal(step: CurationStep, found: _Offer, touched: Collection[str]) -> str:
    """Why ``step`` cannot be taken; empty when it can."""

    named = (step.lesson, *step.merge)
    if any(key not in found.lessons for key in named) or len(set(named)) != len(named):
        return f"{step.action} {step.lesson}: it names a lesson not offered, or one twice"
    if any(key in touched for key in named):
        return f"{step.action} {step.lesson}: a lesson takes one step"
    if step.action == RETIRE:
        return "" if step.lesson in found.retirable else f"retire {step.lesson}: neither proposed nor contradicted here"
    if not LEARNING_MIN_CHARS <= len(step.content) <= LEARNING_MAX_CHARS:
        return f"{step.action} {step.lesson}: a text of {len(step.content)} characters"
    if step.action == MERGE and not all(_alike(found.lessons[step.lesson], found.lessons[key]) for key in step.merge):
        return f"merge {step.lesson}: the lessons were learnt under conditions too unlike"
    return ""


def _step(item: object) -> CurationStep | None:
    """``item`` of the answer's actions as a step; ``None`` when it is none of :data:`MERGE`, :data:`REWRITE` and
    :data:`RETIRE`."""

    if not isinstance(item, dict) or item.get(_ACTION) not in (MERGE, REWRITE, RETIRE):
        return None
    merged = item.get(MERGE) if item[_ACTION] == MERGE else ()
    return CurationStep(
        str(item[_ACTION]),
        str(item.get(_LESSON) or ""),
        tuple(str(key) for key in merged) if isinstance(merged, list) else (),
        _text(item.get(_CONTENT)),
        _text(item.get(_WHY)),
    )


def read_steps(answer: str, found: _Offer) -> tuple[list[CurationStep], list[str]]:
    """The steps of ``answer`` the curator may take, :data:`MAX_STEPS` at most, and why it refused the others."""

    raw = _steps(answer)
    if raw is None:
        return [], ["the answer holds no JSON object with a list of actions"]
    taken: list[CurationStep] = []
    refused: list[str] = []
    for item in raw:
        step = _step(item)
        touched = {key for done in taken for key in (done.lesson, *done.merge)}
        if step is None:
            why = f"not a step: {str(item)[:80]}"
        else:
            why = _refusal(step, found, touched) or (f"more than {MAX_STEPS} steps" if len(taken) >= MAX_STEPS else "")
        if why or step is None:
            refused.append(why)
        else:
            taken.append(step)
    return taken, refused


@dataclass(frozen=True)
class _Writer:
    pool: LessonPool
    author: str
    session_id: str
    at: float

    def revise(self, lesson: Lesson, step: CurationStep, merged: Sequence[Lesson] = ()) -> bool:
        head = _latest(lesson)
        evidence = tuple(dict.fromkeys(item for one in (lesson, *merged) for h in one.heads for item in h.evidence))
        revision = LessonRevision(
            revision_id_for(self.session_id, "curator", lesson.lesson_id),
            lesson.lesson_id,
            head.kind,
            step.content,
            head.conditions,
            self.author,
            self.at,
            session_id=self.session_id,
            target=head.target,
            mechanism=head.mechanism,
            layer=head.layer,
            supersedes=tuple(one.revision_id for one in lesson.heads),
            evidence=evidence,
        )
        return self.pool.write(revision)

    def retire(self, lesson_id: str, reason: str) -> bool:
        return self.pool.retire(LessonRetirement(lesson_id, self.author, self.at, reason))


def _take(step: CurationStep, found: _Offer, writer: _Writer) -> str:
    """Write ``step``; the count it adds to, empty when everything it would write is there already."""

    lesson = found.lessons[step.lesson]
    if step.action == RETIRE:
        return _RETIRED if writer.retire(step.lesson, step.why or "retired by the curator") else ""
    if step.action == REWRITE:
        return _REWRITTEN if writer.revise(lesson, step) else ""
    merged = [found.lessons[key] for key in step.merge]
    wrote = writer.revise(lesson, step, merged)
    for other in merged:
        wrote = writer.retire(other.lesson_id, f"merged into {step.lesson}: {step.why}".rstrip(": ")) or wrote
    return _MERGED if wrote else ""


async def curate_lessons(
    snapshot: NotebookSnapshot,
    complete: Complete,
    *,
    model: str,
    archive: TreeArchive | None = None,
    pool: LessonPool | None = None,
) -> CloseStep:
    """CLOSE, after the lessons are published: ask the LLM, through ``complete``, to curate the lessons offered from
    ``pool``, by default the one beside ``archive``, by default the environment's, and write the steps it may take.
    ``model`` names the LLM in what it writes. A step that cannot be done says why and never stops CLOSE."""

    profile = snapshot.profile
    if profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    if pool is None:
        target = archive if archive is not None else TreeArchive.from_env()
        if target is None:
            return CloseStep(SKIPPED, NO_ARCHIVE)
        pool = LessonPool.beside(target)
    try:
        found = offer(profile.session_id, pool.lessons(), pool.standings())
    except STEP_ERRORS as exc:
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    if not found.lessons:
        return CloseStep(SKIPPED, NO_LESSONS)
    try:
        answer = await complete(CURATION_SYSTEM_PROMPT, curation_prompt(snapshot, found))
    except Exception as exc:
        log.warning("CLOSE could not ask the LLM to curate the lessons: %s", exc, exc_info=True)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    return _written(
        read_steps(answer, found),
        found,
        _Writer(pool, curator_author(profile.session_id, model), profile.session_id, snapshot.taken_at),
    )


def _written(read: tuple[list[CurationStep], list[str]], found: _Offer, writer: _Writer) -> CloseStep:
    """The steps ``read`` lets the curator take, written, and the step's outcome."""

    steps, refused = read
    counts = {_REFUSED: len(refused)}
    for step in steps:
        try:
            done = _take(step, found, writer)
        except STEP_ERRORS as exc:
            refused.append(f"{step.action} {step.lesson}: {type(exc).__name__}: {exc}")
            counts[_REFUSED] += 1
            continue
        if done:
            counts[done] = counts.get(done, 0) + 1
    written = sum(counts.get(key, 0) for key in (_MERGED, _REWRITTEN, _RETIRED))
    return CloseStep(WRITTEN if written else SKIPPED, "; ".join(refused), counts)


__all__ = [
    "CURATION_SYSTEM_PROMPT",
    "MAX_OFFERED",
    "MAX_STEPS",
    "MERGE",
    "NO_LESSONS",
    "RETIRE",
    "REWRITE",
    "CurationStep",
    "curate_lessons",
    "curation_prompt",
    "offer",
    "read_steps",
]
