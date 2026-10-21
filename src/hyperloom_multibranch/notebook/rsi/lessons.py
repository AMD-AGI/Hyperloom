# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lesson pool: specialists' learnings kept for the sessions after theirs, revised and retired, never edited (RSI).

A lesson is a learning a session left the sessions after it. ``<archive>/lessons/``, beside the archive's ``trees/``
and ``scorers/``, holds files each written once:

* ``revisions/<revision_id>.json``: a :class:`LessonRevision`, one version of a lesson. A revision ``supersedes`` the
  revisions of its lesson it replaces; those nothing supersedes are the lesson's heads, one as a rule, two when two
  sessions revised the lesson at once, until the curator merges them.
* ``retired/<lesson_id>.json``: a :class:`LessonRetirement`. A retired lesson is shown to no session again; its
  revisions stay, for the record.
* ``purged/<lesson_id>.json``: a :class:`LessonPurge`, a person's step for a text that must not exist: the lesson's
  revisions and retirement are deleted, and only its id and why remain.
* ``standings/<session_id>.json``: what one session's branches did after its lessons were shown to them
  (:class:`SessionStandings`); retrieval weighs a lesson by its record (:func:`standing_factor`), and a lesson the
  sessions keep contradicting is proposed for retirement (:func:`retirement_proposals`).

Three kinds of editor write here, each named in what it writes (:func:`specialist_author`, :func:`curator_author`,
:func:`human_author`): a session's specialists, whose learnings CLOSE publishes (:mod:`rsi.lesson_close`); the
scheduler's LLM, curating at CLOSE (:mod:`rsi.lesson_curation`); and a person, through
``python -m hyperloom_multibranch.notebook.rsi.lessons list|show|revise|retire|purge``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import LESSON_ID, is_lesson_id
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.local.notebook import LEARNING_MAX_CHARS, LEARNING_MIN_CHARS
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions

log = logging.getLogger(__name__)

REVISION_FORMAT = "multibranch.rsi.lesson-revision.v1"
RETIREMENT_FORMAT = "multibranch.rsi.lesson-retirement.v1"
PURGE_FORMAT = "multibranch.rsi.lesson-purge.v1"
STANDINGS_FORMAT = "multibranch.rsi.lesson-standings.v1"
LESSONS = "lessons"
REVISION_ID = re.compile(r"rev-[0-9a-f]{16}")
#: A lesson contradicted in this many sessions, and borne out in none, is proposed for retirement.
CONTRADICTING_SESSIONS = 2
_REVISIONS, _RETIRED, _PURGED, _STANDINGS = "revisions", "retired", "purged", "standings"
_JSON, _SUFFIX, _ENCODING, _FORMAT, _RECORD = "*.json", ".json", "utf-8", "format", "record"
_SESSION_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}")
#: Fixed, so the lessons a session leaves get the same ids however often its CLOSE runs.
_NAMESPACE = uuid.UUID("6f1d3a52-9c4e-4b8a-a7d2-0e5c9b1f3a64")
Kept = TypeVar("Kept")


def lesson_id_for(*parts: str) -> str:
    """The id of a lesson first written from ``parts``, such as the session and the notebook entry it came from."""

    return "lsn-" + uuid.uuid5(_NAMESPACE, "/".join(parts)).hex[:12]


def revision_id_for(*parts: str) -> str:
    """The id of a revision written from ``parts``."""

    return "rev-" + uuid.uuid5(_NAMESPACE, "/".join(("revision", *parts))).hex[:16]


def specialist_author(session_id: str, branch: str) -> str:
    return f"specialist:{session_id}/{branch}"


def curator_author(session_id: str, model: str) -> str:
    return f"curator:{session_id}/{model}"


def human_author(name: str) -> str:
    return f"human:{name}"


@dataclass(frozen=True)
class LessonEvidence:
    """A test behind a lesson: the session and the request that ran it, how it came out, and whether it measured."""

    session_id: str
    request_id: str
    outcome: str
    measured: bool = False


@dataclass(frozen=True)
class LessonRevision:
    """One version of lesson ``lesson_id``: what it says (``kind``, ``content``, and the direction it is about), the
    conditions of the session it was learnt in, the tests behind it, who wrote it and when, and the revisions of the
    same lesson it ``supersedes``. ``session_id`` is the session it came from, empty for a person's."""

    revision_id: str
    lesson_id: str
    kind: LearningKind
    content: str
    conditions: TaskConditions
    author: str
    written_at: float
    session_id: str = ""
    target: str = ""
    mechanism: str = ""
    layer: IntentLayer | None = None
    supersedes: tuple[str, ...] = ()
    evidence: tuple[LessonEvidence, ...] = ()

    @property
    def measured(self) -> bool:
        """Whether a test behind it measured: a judgement, or a candidate that failed."""

        return any(item.measured for item in self.evidence)


@dataclass(frozen=True)
class LessonRetirement:
    """Lesson ``lesson_id`` shown to no session again, by ``author`` at ``retired_at``, for ``reason``."""

    lesson_id: str
    author: str
    retired_at: float
    reason: str
    evidence: tuple[LessonEvidence, ...] = ()


@dataclass(frozen=True)
class LessonPurge:
    """Lesson ``lesson_id`` deleted by ``author`` at ``purged_at``, for ``reason``; nothing else of it remains."""

    lesson_id: str
    author: str
    purged_at: float
    reason: str


@dataclass(frozen=True)
class LessonStanding:
    """What the branches of a session did after a lesson was shown to them.

    ``shown`` counts the branches it was shown to. A pitfall ``heeded`` kept a branch off its direction; one
    ``ignored`` saw a branch test the direction anyway and fail as it said; one ``contradicted`` saw such a test
    commit. An insight or a hypothesis ``paid`` when the branch's next test, in its direction, committed.
    ``rediscovered`` counts the branches that wrote it again without having been shown it.
    """

    shown: int = 0
    heeded: int = 0
    ignored: int = 0
    contradicted: int = 0
    paid: int = 0
    rediscovered: int = 0

    @property
    def borne_out(self) -> int:
        """The outcomes that bore it out: a failure it foretold, a direction it pointed to that paid, a rediscovery."""

        return self.ignored + self.paid + self.rediscovered


@dataclass(frozen=True)
class SessionStandings:
    """The standing of each pooled lesson one session showed, by lesson id."""

    session_id: str
    written_at: float
    lessons: Mapping[str, LessonStanding] = field(default_factory=dict)


@dataclass(frozen=True)
class Lesson:
    """A lesson as the pool holds it: every revision, the earliest written first; its heads, the revisions nothing
    supersedes; and its retirement, when it has one."""

    lesson_id: str
    revisions: tuple[LessonRevision, ...]
    heads: tuple[LessonRevision, ...]
    retired: LessonRetirement | None = None


def _document(path: Path, kind: type[Kept], name: str) -> Kept | None:
    """The ``kind`` record ``path`` holds; ``None``, with a warning, when it cannot be read as one."""

    try:
        document = json.loads(path.read_text(encoding=_ENCODING))
        if document.get(_FORMAT) != name:
            raise ValueError(f"unsupported format {document.get(_FORMAT)!r}")
        return decode(kind, document.get(_RECORD))
    except (AttributeError, OSError, KeyError, TypeError, ValueError) as exc:
        log.warning("skipping unreadable lesson record %s: %s", path, exc)
        return None


def _documents(folder: Path, kind: type[Kept], name: str) -> list[Kept]:
    found = (_document(path, kind, name) for path in sorted(folder.glob(_JSON)))
    return [item for item in found if item is not None]


def _publish(path: Path, name: str, record: object) -> bool:
    return publish_once(path, encode_json({_FORMAT: name, _RECORD: encode(record)}))


def _check_lesson_id(lesson_id: str) -> None:
    if not is_lesson_id(lesson_id):
        raise ValueError(f"{lesson_id!r} is no lesson id: {LESSON_ID.pattern}")


def _check_revision(revision: LessonRevision) -> None:
    _check_lesson_id(revision.lesson_id)
    if not REVISION_ID.fullmatch(revision.revision_id):
        raise ValueError(f"{revision.revision_id!r} is no revision id: {REVISION_ID.pattern}")
    if not LEARNING_MIN_CHARS <= len(revision.content) <= LEARNING_MAX_CHARS:
        raise ValueError(
            f"a lesson says what was learnt in {LEARNING_MIN_CHARS} to {LEARNING_MAX_CHARS} characters, "
            f"not {len(revision.content)}"
        )


def _lesson(lesson_id: str, revisions: Sequence[LessonRevision], retired: LessonRetirement | None) -> Lesson:
    superseded = {item for revision in revisions for item in revision.supersedes}
    heads = tuple(revision for revision in revisions if revision.revision_id not in superseded)
    return Lesson(lesson_id, tuple(revisions), heads, retired)


class LessonPool:
    """The lesson pool: ``<root>/lessons``, beside an archive's trees under ``<root>/trees``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / LESSONS

    @classmethod
    def beside(cls, archive: TreeArchive) -> LessonPool:
        return cls(archive.root.parent)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LessonPool | None:
        """The pool a session uses unless it is given one: beside the archive
        :data:`~hyperloom_multibranch.notebook.rsi.archive.ARCHIVE_ENV` names, ``None`` when it names none."""

        archive = TreeArchive.from_env(env)
        return None if archive is None else cls.beside(archive)

    def revisions(self) -> tuple[LessonRevision, ...]:
        """Every readable revision, the earliest written first."""

        found = _documents(self.root / _REVISIONS, LessonRevision, REVISION_FORMAT)
        return tuple(sorted(found, key=lambda item: (item.written_at, item.revision_id)))

    def revision(self, revision_id: str) -> LessonRevision | None:
        path = self.root / _REVISIONS / f"{revision_id}{_SUFFIX}"
        return _document(path, LessonRevision, REVISION_FORMAT) if path.is_file() else None

    def retirements(self) -> tuple[LessonRetirement, ...]:
        found = _documents(self.root / _RETIRED, LessonRetirement, RETIREMENT_FORMAT)
        return tuple(sorted(found, key=lambda item: item.retired_at))

    def purges(self) -> tuple[LessonPurge, ...]:
        found = _documents(self.root / _PURGED, LessonPurge, PURGE_FORMAT)
        return tuple(sorted(found, key=lambda item: item.purged_at))

    def lessons(self) -> dict[str, Lesson]:
        """Every lesson not purged, by id, with its revisions, its heads and its retirement."""

        purged = {item.lesson_id for item in self.purges()}
        retired = {item.lesson_id: item for item in self.retirements()}
        grouped: dict[str, list[LessonRevision]] = defaultdict(list)
        for revision in self.revisions():
            if revision.lesson_id not in purged:
                grouped[revision.lesson_id].append(revision)
        return {lesson_id: _lesson(lesson_id, found, retired.get(lesson_id)) for lesson_id, found in grouped.items()}

    def live(self) -> tuple[LessonRevision, ...]:
        """The heads of every lesson neither retired nor purged, the earliest written first."""

        heads = [head for lesson in self.lessons().values() if lesson.retired is None for head in lesson.heads]
        return tuple(sorted(heads, key=lambda item: (item.written_at, item.revision_id)))

    def write(self, revision: LessonRevision) -> bool:
        """``revision``, written once; ``False`` when it is there already. Raises ``ValueError`` for a malformed id, a
        text the notebook would refuse as a learning, a purged lesson, or a superseded revision of another lesson."""

        _check_revision(revision)
        if (self.root / _PURGED / f"{revision.lesson_id}{_SUFFIX}").exists():
            raise ValueError(f"lesson {revision.lesson_id} was purged")
        for superseded in revision.supersedes:
            held = self.revision(superseded)
            if held is None or held.lesson_id != revision.lesson_id:
                raise ValueError(f"{superseded} is no revision of lesson {revision.lesson_id}")
        return _publish(self.root / _REVISIONS / f"{revision.revision_id}{_SUFFIX}", REVISION_FORMAT, revision)

    def retire(self, retirement: LessonRetirement) -> bool:
        """``retirement``, written once; ``False`` when the lesson was retired already."""

        _check_lesson_id(retirement.lesson_id)
        path = self.root / _RETIRED / f"{retirement.lesson_id}{_SUFFIX}"
        return _publish(path, RETIREMENT_FORMAT, retirement)

    def purge(self, purge: LessonPurge) -> int:
        """Record ``purge``, then delete its lesson's revisions and retirement; how many revisions it deleted. The
        record comes first, so a reader in between already finds the lesson gone."""

        _check_lesson_id(purge.lesson_id)
        _publish(self.root / _PURGED / f"{purge.lesson_id}{_SUFFIX}", PURGE_FORMAT, purge)
        doomed = [
            path
            for path in sorted((self.root / _REVISIONS).glob(_JSON))
            if (found := _document(path, LessonRevision, REVISION_FORMAT)) is not None
            and found.lesson_id == purge.lesson_id
        ]
        for path in doomed:
            path.unlink(missing_ok=True)
        (self.root / _RETIRED / f"{purge.lesson_id}{_SUFFIX}").unlink(missing_ok=True)
        return len(doomed)

    def record_standings(self, standings: SessionStandings) -> bool:
        """``standings``, written once a session; ``False`` when that session's are there already."""

        if not _SESSION_FILE.fullmatch(standings.session_id):
            raise ValueError(f"session id {standings.session_id!r} cannot name a standings file")
        path = self.root / _STANDINGS / f"{standings.session_id}{_SUFFIX}"
        return _publish(path, STANDINGS_FORMAT, standings)

    def standings(self) -> tuple[SessionStandings, ...]:
        """Every session's standings, the earliest written first."""

        found = _documents(self.root / _STANDINGS, SessionStandings, STANDINGS_FORMAT)
        return tuple(sorted(found, key=lambda item: item.written_at))


def standing_of(standings: Iterable[SessionStandings], lesson_id: str) -> LessonStanding:
    """``lesson_id``'s standing summed over the sessions of ``standings``."""

    found = [item.lessons[lesson_id] for item in standings if lesson_id in item.lessons]
    return LessonStanding(
        **{item.name: sum(getattr(one, item.name) for one in found) for item in fields(LessonStanding)}
    )


def standing_factor(standing: LessonStanding) -> float:
    """How a lesson's record weighs it in retrieval: one more than what bore it out, over one more than twice what
    contradicted it; 1 for a lesson no session has tried."""

    return (1 + standing.borne_out) / (1 + 2 * standing.contradicted)


def retirement_proposals(lessons: Mapping[str, Lesson], standings: Sequence[SessionStandings]) -> tuple[str, ...]:
    """The live lessons contradicted in :data:`CONTRADICTING_SESSIONS` sessions or more and borne out in none."""

    proposed = []
    for lesson_id, lesson in lessons.items():
        mine = [item.lessons[lesson_id] for item in standings if lesson_id in item.lessons]
        contradicting = sum(1 for item in mine if item.contradicted)
        if (
            lesson.retired is None
            and contradicting >= CONTRADICTING_SESSIONS
            and not any(item.borne_out for item in mine)
        ):
            proposed.append(lesson_id)
    return tuple(sorted(proposed))


__all__ = [
    "CONTRADICTING_SESSIONS",
    "LESSONS",
    "PURGE_FORMAT",
    "RETIREMENT_FORMAT",
    "REVISION_FORMAT",
    "REVISION_ID",
    "STANDINGS_FORMAT",
    "Lesson",
    "LessonEvidence",
    "LessonPool",
    "LessonPurge",
    "LessonRetirement",
    "LessonRevision",
    "LessonStanding",
    "SessionStandings",
    "curator_author",
    "human_author",
    "lesson_id_for",
    "retirement_proposals",
    "revision_id_for",
    "specialist_author",
    "standing_factor",
    "standing_of",
]
