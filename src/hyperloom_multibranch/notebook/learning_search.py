# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This session's learnings, matched to what a branch is about to decide; what the notebook tools find on their own.

A query matches a learning no later one superseded by the words they share, a word of the direction the learning is
about counting double; among equals the newer comes first. :class:`LearningFinder` is what the tools ask for
learnings: :class:`SessionLearnings` answers from this session alone, and the knowledge package's
``LearningRecall`` adds past sessions' from the Experience KB.

Given a :class:`RetrievalLog`, a finder appends one :data:`LEARNINGS_LOG_FORMAT` line per retrieval: when, to which
branch, through what (the learnings tool, a granted claim, a startup brief), for what query, and what came back. Set
beside what each branch did next, the log is how retrieval gets evaluated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from hyperloom.common.io import append_jsonl
from hyperloom_multibranch.contracts.notebook.peers import Validity
from hyperloom_multibranch.contracts.notebook.records import LearningRecord, NotebookSnapshot, TestRecord

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9][a-z0-9._+-]*")
_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "is", "it", "no", "not", "of", "on", "or", "the", "this", "to", "with"}
)
SESSION_HEADING = "This session's learnings:"
LEARNINGS_LOG_FORMAT = "multibranch.learnings-log.v1"
#: The retrieval log's usual name in the session's reports folder.
LEARNINGS_LOG = "learnings.jsonl"


class RetrievalSource(str, Enum):
    """How a branch came to be shown learnings: it asked, its claim was granted, or its startup brief listed them."""

    TOOL = "tool"
    CLAIM = "claim"
    BRIEF = "brief"


@dataclass(frozen=True)
class RetrievalLog:
    """The session's log of the learnings shown to its branches; a failed append is logged and holds up nobody.

    A line names this session's learnings by entry id, past sessions' by Experience id, and the past read's status
    (empty when nothing read past sessions).
    """

    path: Path

    def record(
        self,
        snapshot: NotebookSnapshot,
        *,
        branch: str,
        source: RetrievalSource,
        query: str,
        here: Sequence[LearningRecord],
        past: Sequence[str] = (),
        past_status: str = "",
    ) -> None:
        line = {
            "format": LEARNINGS_LOG_FORMAT,
            "at": snapshot.taken_at,
            "branch": branch,
            "source": source.value,
            "query": query,
            "here": [record.entry_id for record in here],
            "past": list(past),
            "past_status": past_status,
        }
        try:
            append_jsonl(self.path, line, make_parents=True, sort_keys=True)
        except OSError as exc:
            log.warning("could not append the learnings retrieval to %s: %s", self.path, exc)


class LearningFinder(Protocol):
    def here(
        self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str, source: RetrievalSource
    ) -> tuple[LearningRecord, ...]:
        """This session's learnings that bear on ``query``, best first, without waiting on any knowledge base."""

    async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """The learnings that bear on ``query`` for ``branch``, as headed lines; empty when none does."""


def current_learnings(snapshot: NotebookSnapshot) -> tuple[LearningRecord, ...]:
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
    """The words of ``text`` a match counts: lower-cased, two characters or more, common short words left out."""

    return {word for word in _WORD.findall(text.lower()) if len(word) > 1 and word not in _STOPWORDS}


def session_matches(snapshot: NotebookSnapshot, query: str, *, limit: int) -> tuple[LearningRecord, ...]:
    """This session's current learnings that share a word with ``query``, best first."""

    wanted = words(query)
    scored = []
    for record in current_learnings(snapshot):
        learning = record.learning
        about = words(f"{learning.target} {learning.mechanism}")
        score = len(wanted & words(learning.content)) + 2 * len(wanted & about)
        if score:
            scored.append((score, record.at, record))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return tuple(record for _, _, record in scored[:limit])


def render_session_learning(record: LearningRecord, snapshot: NotebookSnapshot) -> str:
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
        self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str, source: RetrievalSource
    ) -> tuple[LearningRecord, ...]:
        """This session's matches, best first, logged as one retrieval by ``branch`` through ``source``."""

        found = session_matches(snapshot, query, limit=limit)
        if self.log is not None:
            self.log.record(snapshot, branch=branch, source=source, query=query, here=found)
        return found

    async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """This session's matches under their heading; empty when none matched."""

        found = self.here(snapshot, query, limit=limit, branch=branch, source=RetrievalSource.TOOL)
        rendered = [render_session_learning(record, snapshot) for record in found]
        return [SESSION_HEADING, *rendered] if rendered else []


__all__ = [
    "LEARNINGS_LOG",
    "LEARNINGS_LOG_FORMAT",
    "SESSION_HEADING",
    "LearningFinder",
    "RetrievalLog",
    "RetrievalSource",
    "SessionLearnings",
    "current_learnings",
    "fit_lines",
    "outcome_of",
    "render_session_learning",
    "session_matches",
    "words",
]
