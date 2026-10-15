# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This session's learnings, matched to what a branch is about to decide; what the notebook tools find on their own.

A query matches a learning no later one superseded by the words they share, a word of the direction the learning is
about counting double; among equals the newer comes first. :class:`LearningFinder` is what the tools ask for
learnings: :class:`SessionLearnings` answers from this session alone, and the knowledge package's
``LearningRecall`` adds past sessions' from the Experience KB.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from hyperloom_multibranch.contracts.notebook.peers import Validity
from hyperloom_multibranch.contracts.notebook.records import LearningRecord, NotebookSnapshot, TestRecord

_WORD = re.compile(r"[a-z0-9][a-z0-9._+-]*")
_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "is", "it", "no", "not", "of", "on", "or", "the", "this", "to", "with"}
)
SESSION_HEADING = "This session's learnings:"


class LearningFinder(Protocol):
    async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int) -> list[str]:
        """The learnings that bear on ``query``, as headed lines; empty when none does."""


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
    """The finder without a knowledge base: this session's learnings only."""

    async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int) -> list[str]:
        """This session's matches under their heading; empty when none matched."""

        found = [render_session_learning(record, snapshot) for record in session_matches(snapshot, query, limit=limit)]
        return [SESSION_HEADING, *found] if found else []


__all__ = [
    "SESSION_HEADING",
    "LearningFinder",
    "SessionLearnings",
    "current_learnings",
    "fit_lines",
    "outcome_of",
    "render_session_learning",
    "session_matches",
    "words",
]
