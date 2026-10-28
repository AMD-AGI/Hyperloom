# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This session's learnings, matched to what a branch is about to decide; what the blackboard tools find on their own.

A learning is about a direction family (:mod:`blackboard.families`): that of the first test it cites, else of the
direction or the words it gives (:func:`learning_family`). A query matches a learning no later one superseded by its
family first, the same family before the same target (:func:`kinship`), then by the words they share, a word of the
direction the learning is about counting double; among equals the newer comes first. A word is lower-cased, and a
compound one, such as a server flag, also counts as each of its parts, so ``--cuda-graph-max-bs`` matches a query for
"cuda graph"; a number under four digits, the batch size or concurrency nearly every learning mentions, does not count.
A granted claim lists :data:`NOTES` notes about its direction, this session's and past sessions' in one list.
:class:`LearningFinder` is what the tools ask for learnings: :class:`SessionLearnings` answers from this session
alone, the knowledge package's ``LearningRecall`` adds past sessions' from the Experience KB, and RSI's
``LessonRecall`` past sessions' lessons from the lesson pool, which a learning of this session may revise or retire.

Given a :class:`RetrievalLog`, a finder appends one :data:`LEARNINGS_LOG_FORMAT` line per retrieval: when, to which
branch, through what (the learnings tool, a granted claim, a startup brief), for what query, and what came back. Set
beside what each branch did next, the log is how retrieval gets evaluated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from hyperloom.common.io import append_jsonl
from hyperloom_multibranch.contracts.blackboard.peers import Intent, Validity
from hyperloom_multibranch.contracts.blackboard.records import Learning, LearningRecord, BlackboardSnapshot, TestRecord
from hyperloom_multibranch.blackboard.codec import key_sorted
from hyperloom_multibranch.blackboard.families import Family, family_of, note_family

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9][a-z0-9._+-]*")
_PARTS = re.compile(r"[._:/+-]+")
#: A batch size or a concurrency every other learning mentions; a GEMM shape, four digits or more, still counts.
_SMALL_NUMBER = re.compile(r"\d{1,3}")
_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "is", "it", "no", "not", "of", "on", "or", "the", "this", "to", "with"}
)
SESSION_HEADING = "This session's learnings:"
#: How a granted claim heads the notes about its direction.
NOTES_HEADING = "Notes about it, the most relevant first:"
#: How many notes a granted claim and a brief list, this session's and past sessions' together.
NOTES = 5
#: The :func:`kinship` of two notes about one family.
SAME_FAMILY = 2
LEARNINGS_LOG_FORMAT = "multibranch.learnings-log.v1"
#: The retrieval log's usual name in the session's reports folder.
LEARNINGS_LOG = "learnings.jsonl"


class RetrievalSource(str, Enum):
    """How a branch came to be shown learnings: it asked, its claim was granted, or its startup brief listed them."""

    TOOL = "tool"
    CLAIM = "claim"
    BRIEF = "brief"


@dataclass(frozen=True)
class ShownLead:
    """A lead a startup brief showed: ``ref`` the Experience or the Recipe KB record it came from, and ``change`` the
    change it suggests, summarized as the Experience export summarizes a change (empty when it names none)."""

    ref: str
    change: str = ""


@dataclass(frozen=True)
class Retrieval:
    """One retrieval of learnings: for ``branch``, through ``source``, for ``query``; a brief also shows ``leads``."""

    branch: str
    source: RetrievalSource
    query: str
    leads: tuple[ShownLead, ...] = ()


@dataclass(frozen=True)
class RetrievalLog:
    """The session's log of the learnings shown to its branches; a failed append is logged and holds up nobody.

    A line names this session's learnings by entry id, past sessions' by Experience id, and the past read's status
    (empty when nothing read past sessions); a brief's line also lists the leads it showed, each with its change.
    """

    path: Path

    def record(
        self,
        snapshot: BlackboardSnapshot,
        retrieval: Retrieval,
        here: Sequence[LearningRecord],
        past: Sequence[str] = (),
        past_status: str = "",
    ) -> None:
        line: dict[str, object] = {
            "format": LEARNINGS_LOG_FORMAT,
            "at": snapshot.taken_at,
            "branch": retrieval.branch,
            "source": retrieval.source.value,
            "query": retrieval.query,
            "here": [record.entry_id for record in here],
            "past": list(past),
            "past_status": past_status,
        }
        if retrieval.leads:
            line["leads"] = [{"ref": lead.ref, "change": lead.change} for lead in retrieval.leads]
        try:
            append_jsonl(self.path, key_sorted(line), make_parents=True)
        except OSError as exc:
            log.warning("could not append the learnings retrieval to %s: %s", self.path, exc)


class LearningFinder(Protocol):
    def here(
        self, snapshot: BlackboardSnapshot, retrieval: Retrieval, *, limit: int, family: Family | None = None
    ) -> tuple[LearningRecord, ...]:
        """This session's learnings that bear on ``retrieval``'s query, about ``family`` first, best first, without
        waiting on any knowledge base."""

    async def lines(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """The learnings that bear on ``query`` for ``branch``, as headed lines; empty when none does."""

    async def about(
        self, snapshot: BlackboardSnapshot, query: str, *, branch: str, family: Family | None = None
    ) -> list[str]:
        """What ``branch``'s granted claim lists about its direction ``query`` of ``family``: :data:`NOTES` notes in
        one list under :data:`NOTES_HEADING`, logged as one retrieval; empty when nothing bears on it."""

    def knows(self, lesson_id: str) -> bool:
        """Whether ``lesson_id`` names a past session's lesson, live, that a learning of this session may revise or
        retire."""


def _same_words(left: str, right: str) -> bool:
    return " ".join(left.lower().split()) == " ".join(right.lower().split())


def names(learning: Learning, intent: Intent | None) -> bool:
    """Whether ``learning`` names ``intent``'s direction: its target and mechanism, those it gives, the same."""

    if intent is None or not (learning.target or learning.mechanism):
        return False
    same_target = not learning.target or _same_words(learning.target, intent.target)
    return same_target and (not learning.mechanism or _same_words(learning.mechanism, intent.mechanism))


def current_learnings(snapshot: BlackboardSnapshot) -> tuple[LearningRecord, ...]:
    """The learnings no later learning of the same branch superseded, oldest first."""

    superseded = {record.learning.supersedes for record in snapshot.learnings if record.learning.supersedes}
    return tuple(record for record in snapshot.learnings if record.entry_id not in superseded)


def outcome_of(record: TestRecord) -> str:
    """How a cited test came out, in a few words."""

    result, judged = record.result, record.judgement
    if record.withdrawn_at is not None:
        return "withdrawn"
    if result is None:
        return "in flight"
    if result.validity is Validity.MEASUREMENT_FAILED:
        return "not measured"
    if result.validity is Validity.CANDIDATE_FAILED:
        return f"the candidate failed ({result.failure_reason or 'no reason given'})"
    if judged is None:
        return "measured, not judged yet"
    conclusion = f", {record.conclusion.value}" if record.conclusion is not None else ""
    return f"{judged.verdict.value} {judged.gain_pct:+.2f}%{conclusion}"


def words(text: str) -> set[str]:
    """The words of ``text`` a match counts: lower-cased, two characters or more, with no sentence's full stop; common
    short words and numbers under four digits left out; a compound word counts whole and as each of its parts."""

    found = set()
    for word in _WORD.findall(text.lower()):
        whole = word.rstrip(".")
        found.add(whole)
        found.update(_PARTS.split(whole))
    return {word for word in found if len(word) > 1 and word not in _STOPWORDS and not _SMALL_NUMBER.fullmatch(word)}


def word_overlap(mine: set[str], theirs: set[str]) -> float:
    """The share of the smaller of two word sets the other holds too: 1 when one text's words are all the other's."""

    return len(mine & theirs) / max(1, min(len(mine), len(theirs)))


def kinship(wanted: Family | None, note: Family | None) -> int:
    """How close a note's family is to the one ``wanted``: :data:`SAME_FAMILY` when it covers it
    (:meth:`~hyperloom_multibranch.blackboard.families.Family.covers`), 1 for the same target, 0 otherwise or without
    either."""

    if wanted is None or note is None:
        return 0
    return SAME_FAMILY if note.covers(wanted) else int(note.target == wanted.target)


def query_family(query: str) -> Family | None:
    """The family a query's words name; ``None`` when they name no target."""

    return note_family("", query, None)


def family_of_learning(
    learning: Learning, records: Iterable[TestRecord], intents: Mapping[str, Intent]
) -> Family | None:
    """The family ``learning`` is about: that of the first test of ``records`` it cites whose intent ``intents``
    holds, else of the direction or the words it gives."""

    tests = {test.request_id: test for test in records if test.request_id in learning.request_ids}
    for request_id in learning.request_ids:
        test = tests.get(request_id)
        intent = intents.get(test.request.intent_fingerprint or "") if test is not None else None
        if test is not None and intent is not None:
            return family_of(intent.target, intent.mechanism, intent.layer, test.request.delta)
    return note_family(learning.target, learning.mechanism, learning.layer, learning.content)


def learning_family(record: LearningRecord, snapshot: BlackboardSnapshot) -> Family | None:
    """The family one of the session's learnings is about (:func:`family_of_learning`)."""

    return family_of_learning(record.learning, snapshot.records, snapshot.intents)


def note_score(wanted: set[str], content: str, target: str, mechanism: str) -> int:
    """The words ``wanted`` a note shares with its text, and twice those it shares with the direction it names."""

    return len(wanted & words(content)) + 2 * len(wanted & words(f"{target} {mechanism}"))


@dataclass(frozen=True)
class Match:
    """How a note bears on a query: its :func:`kinship` with the family asked about, its :func:`note_score`, and when
    it was written."""

    kin: int
    score: float
    at: float


def session_scored(
    snapshot: BlackboardSnapshot, query: str, family: Family | None = None
) -> list[tuple[Match, LearningRecord]]:
    """This session's current learnings that bear on ``query`` or are about ``family``, each with its match."""

    wanted = words(query)
    found = []
    for record in current_learnings(snapshot):
        learning = record.learning
        score = note_score(wanted, learning.content, learning.target, learning.mechanism)
        kin = kinship(family, learning_family(record, snapshot)) if family is not None else 0
        if kin or score:
            found.append((Match(kin, score, record.at), record))
    return found


def session_matches(
    snapshot: BlackboardSnapshot, query: str, *, limit: int, family: Family | None = None
) -> tuple[LearningRecord, ...]:
    """This session's current learnings that bear on ``query``, about ``family`` first, best first."""

    scored = sorted(
        session_scored(snapshot, query, family), key=lambda item: (-item[0].kin, -item[0].score, -item[0].at)
    )
    return tuple(record for _, record in scored[:limit])


def render_session_learning(record: LearningRecord, snapshot: BlackboardSnapshot) -> str:
    """One of this session's learnings as a line: who learnt it, what kind, about what, and the tests behind it."""

    learning = record.learning
    about = f" about {learning.target} / {learning.mechanism}" if learning.target or learning.mechanism else ""
    by_id = {test.request_id: test for test in snapshot.records}
    cited = "; ".join(
        f"{request_id}: {outcome_of(by_id[request_id])}" for request_id in learning.request_ids if request_id in by_id
    )
    evidence = f" ({cited})" if cited else ""
    return f"- [{learning.branch}, {learning.kind.value}{about}] {learning.content}{evidence} [{record.entry_id}]"


def fit_lines(lines: Sequence[str], max_chars: int) -> list[str]:
    """Whole lines, in order, within ``max_chars``; a line that does not fit is left out and the next ones tried."""

    kept, used = [], 0
    for line in lines:
        if used + len(line) + 1 <= max_chars:
            kept.append(line)
            used += len(line) + 1
    return kept


class SessionLearnings:
    """The finder without a knowledge base: this session's learnings only; ``log`` keeps what each retrieval showed."""

    def __init__(self, log: RetrievalLog | None = None) -> None:
        self.log = log

    def here(
        self, snapshot: BlackboardSnapshot, retrieval: Retrieval, *, limit: int, family: Family | None = None
    ) -> tuple[LearningRecord, ...]:
        """This session's matches, best first, logged as ``retrieval``."""

        found = session_matches(snapshot, retrieval.query, limit=limit, family=family)
        if self.log is not None:
            self.log.record(snapshot, retrieval, found)
        return found

    async def lines(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """This session's matches, the family the query names first, under their heading; empty when none matched."""

        found = self.here(
            snapshot, Retrieval(branch, RetrievalSource.TOOL, query), limit=limit, family=query_family(query)
        )
        rendered = [render_session_learning(record, snapshot) for record in found]
        return [SESSION_HEADING, *rendered] if rendered else []

    async def about(
        self, snapshot: BlackboardSnapshot, query: str, *, branch: str, family: Family | None = None
    ) -> list[str]:
        """This session's :data:`NOTES` best matches under :data:`NOTES_HEADING`; empty when none matched."""

        found = self.here(snapshot, Retrieval(branch, RetrievalSource.CLAIM, query), limit=NOTES, family=family)
        return note_lines([render_session_learning(record, snapshot) for record in found])

    def knows(self, lesson_id: str) -> bool:
        """``False``: this finder holds no past session's lesson."""

        return False


def note_lines(rendered: Sequence[str]) -> list[str]:
    """``rendered``, the notes about a claimed direction, under :data:`NOTES_HEADING`; empty for none."""

    return [NOTES_HEADING, *rendered] if rendered else []


__all__ = [
    "LEARNINGS_LOG",
    "LEARNINGS_LOG_FORMAT",
    "NOTES",
    "NOTES_HEADING",
    "SAME_FAMILY",
    "SESSION_HEADING",
    "LearningFinder",
    "Match",
    "Retrieval",
    "RetrievalLog",
    "RetrievalSource",
    "SessionLearnings",
    "ShownLead",
    "current_learnings",
    "family_of_learning",
    "fit_lines",
    "kinship",
    "learning_family",
    "names",
    "note_lines",
    "note_score",
    "outcome_of",
    "query_family",
    "render_session_learning",
    "session_matches",
    "session_scored",
    "word_overlap",
    "words",
]
