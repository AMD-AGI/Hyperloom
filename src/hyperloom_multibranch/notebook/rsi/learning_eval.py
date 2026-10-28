# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the learnings shown to a branch bore on what it did next: one session's retrieval, evaluated (MB2).

Read beside the session's retrieval log (``learnings.jsonl``, :class:`learning_search.RetrievalLog`) and its notebook,
each learning of this session shown to a branch counts once, at its first showing, when it names a direction (a target
or a mechanism):

* a pitfall is *avoided* when the branch requested no test in its direction afterwards;
* an insight or a hypothesis is *taken up* when the branch's next test is in its direction.

Then every test a branch requested after its first retrieval counts once: *in a shown direction* when a learning shown
to the branch before it named the test's direction, *elsewhere* otherwise; how many of each led to a commit sets the
learnings' directions beside the rest. Past sessions' learnings are logged by id only, an Experience's or a pooled
lesson's (:mod:`rsi.lessons`), so they are counted, not matched; the lessons apart too. Given the pool's live lessons,
a learning of this session that says one of another session again, its branch never shown it, is a *rediscovery*:
what the retrieval missed (:func:`rediscoveries`, which CLOSE's lesson standings use too).

A lead a startup brief showed counts once too, at its first showing to a branch: *taken up* when a later test of the
branch makes every change the lead names (each word of its change, as the Experience export summarizes a change), and
*committed* when such a test committed. A lead that names no change is counted, not matched. Every share comes with its
counts; what they mean for the retrieval and the briefs, vector search included, is MB2's call to make with them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_jsonl
from hyperloom_multibranch.contracts.notebook.peers import Intent, TestKind
from hyperloom_multibranch.contracts.notebook.records import Learning, NotebookSnapshot, Writer, is_lesson_id
from hyperloom_multibranch.contracts.notebook.vocab import COMMITTING, LearningKind, Role
from hyperloom_multibranch.notebook.knowledge.experience import change_document
from hyperloom_multibranch.notebook.learning_search import current_learnings, names, word_overlap, words
from hyperloom_multibranch.notebook.local.notebook import LocalNotebook
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool
from hyperloom_multibranch.notebook.rsi.report import share, table

STUDY_WRITER = "rsi-study"
#: A learning that shares this share of the smaller word set with a lesson says that lesson again.
REDISCOVERY_OVERLAP = 0.5
_TAKEN_UP = frozenset({LearningKind.INSIGHT, LearningKind.HYPOTHESIS})


@dataclass(frozen=True)
class LearningEvaluation:
    """What one session's retrievals showed, and what the branches did after them."""

    retrievals: int
    by_source: Mapping[str, int]
    pitfalls_shown: int
    pitfalls_avoided: int
    insights_shown: int
    insights_taken_up: int
    tests_in_shown_directions: int
    committed_in_shown_directions: int
    tests_elsewhere: int
    committed_elsewhere: int
    past_shown: int
    past_distinct: int
    leads_shown: int = 0
    leads_taken_up: int = 0
    leads_committed: int = 0
    lessons_shown: int = 0
    lessons_distinct: int = 0
    learnings_written: int = 0
    rediscovered: int = 0

    def render(self) -> str:
        """The evaluation as markdown: a table of the measures, each with its counts."""

        sources = ", ".join(f"{source} {count}" for source, count in sorted(self.by_source.items())) or "none"
        rows = [
            ("learnings that say a past lesson again, unshown", share(self.rediscovered, self.learnings_written)),
            ("pitfalls avoided", share(self.pitfalls_avoided, self.pitfalls_shown)),
            ("insights and hypotheses taken up", share(self.insights_taken_up, self.insights_shown)),
            (
                "tests committed, in a shown direction",
                share(self.committed_in_shown_directions, self.tests_in_shown_directions),
            ),
            ("tests committed, elsewhere", share(self.committed_elsewhere, self.tests_elsewhere)),
            ("brief leads taken up", share(self.leads_taken_up, self.leads_shown)),
            ("brief leads taken up and committed", share(self.leads_committed, self.leads_shown)),
        ]
        return "\n".join(
            [
                f"Learnings retrieval: {self.retrievals} retrievals ({sources}); past sessions' learnings shown "
                f"{self.past_shown} times, {self.past_distinct} distinct; of them pooled lessons {self.lessons_shown} "
                f"times, {self.lessons_distinct} distinct.",
                "",
                *table(("measure", "share"), rows),
            ]
        )


#: The retrieval log's field of this session's learnings shown, and of past sessions' (Experience or lesson ids).
HERE, PAST = "here", "past"


@dataclass(frozen=True)
class BranchTest:
    """One test a branch requested: when, in which direction, whether it committed, whether it ended with a result,
    and the words of its change."""

    at: float
    intent: Intent | None
    committed: bool
    change: frozenset[str] = frozenset()
    ended: bool = False


def branch_tests(snapshot: NotebookSnapshot) -> dict[str, list[BranchTest]]:
    """Each branch's tests of kind ``test``, the earliest requested first."""

    found: dict[str, list[BranchTest]] = defaultdict(list)
    for record in sorted(snapshot.records, key=lambda item: item.requested_at):
        if record.request.kind is TestKind.TEST:
            intent = snapshot.intents.get(record.request.intent_fingerprint)
            change = frozenset(str(change_document(record.request.delta)["summary"]).split())
            committed = record.conclusion in COMMITTING
            ended = record.result is not None
            found[record.request.branch].append(BranchTest(record.requested_at, intent, committed, change, ended))
    return found


def first_shown(log: Sequence[Mapping[str, Any]], ids: str = HERE) -> dict[tuple[str, str], float]:
    """When each branch was first shown each id of the log lines' ``ids`` field, by (branch, id)."""

    shown: dict[tuple[str, str], float] = {}
    for line in sorted(log, key=lambda item: float(item["at"])):
        for entry_id in line.get(ids, ()):
            shown.setdefault((str(line["branch"]), str(entry_id)), float(line["at"]))
    return shown


def _uptake(
    shown: Mapping[tuple[str, str], float], learnings: Mapping[str, Learning], tests: Mapping[str, list[BranchTest]]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for (branch, entry_id), at in shown.items():
        learning = learnings.get(entry_id)
        if learning is None or not (learning.target or learning.mechanism):
            continue
        later = [test for test in tests.get(branch, ()) if test.at > at]
        if learning.kind is LearningKind.PITFALL:
            counts["pitfalls"] += 1
            counts["avoided"] += not any(names(learning, test.intent) for test in later)
        elif learning.kind in _TAKEN_UP:
            counts["insights"] += 1
            counts["taken_up"] += bool(later) and names(learning, later[0].intent)
    return counts


def _outcomes(
    shown: Mapping[tuple[str, str], float], learnings: Mapping[str, Learning], tests: Mapping[str, list[BranchTest]]
) -> Counter[str]:
    by_branch: dict[str, list[tuple[float, Learning]]] = defaultdict(list)
    for (branch, entry_id), at in shown.items():
        if entry_id in learnings:
            by_branch[branch].append((at, learnings[entry_id]))
    counts: Counter[str] = Counter()
    for branch, seen in by_branch.items():
        first = min(at for at, _ in seen)
        for test in (item for item in tests.get(branch, ()) if item.at > first):
            bucket = "shown" if any(at < test.at and names(learning, test.intent) for at, learning in seen) else "else"
            counts[bucket] += 1
            counts[f"{bucket}_committed"] += test.committed
    return counts


def _leads(log: Sequence[Mapping[str, Any]], tests: Mapping[str, list[BranchTest]]) -> Counter[str]:
    first: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
    for line in sorted(log, key=lambda item: float(item["at"])):
        for lead in line.get("leads", ()):
            change = frozenset(str(lead.get("change") or "").split())
            first.setdefault((str(line["branch"]), str(lead["ref"])), (float(line["at"]), change))
    counts: Counter[str] = Counter()
    for (branch, _), (at, change) in first.items():
        counts["shown"] += 1
        taken = [test for test in tests.get(branch, ()) if change and test.at > at and change <= test.change]
        counts["taken_up"] += bool(taken)
        counts["committed"] += any(test.committed for test in taken)
    return counts


def rediscoveries(
    snapshot: NotebookSnapshot, lessons: Mapping[str, tuple[str, str]], shown: Collection[tuple[str, str]]
) -> set[tuple[str, str]]:
    """The (branch, lesson) pairs where a current learning of the session says again a lesson of another session,
    ``lessons`` giving each lesson's session and text, the branch never shown it (``shown`` holds what was)."""

    session_id = snapshot.profile.session_id if snapshot.profile is not None else ""
    found = set()
    for record in current_learnings(snapshot):
        mine = words(record.learning.content)
        for lesson_id, (origin, text) in lessons.items():
            pair = (record.learning.branch, lesson_id)
            if origin == session_id or pair in shown or pair in found:
                continue
            if word_overlap(mine, words(text)) >= REDISCOVERY_OVERLAP:
                found.add(pair)
    return found


def evaluate_learnings(
    snapshot: NotebookSnapshot, log: Sequence[Mapping[str, Any]], lessons: Mapping[str, tuple[str, str]] | None = None
) -> LearningEvaluation:
    """The retrieval of ``snapshot``'s session, read off its retrieval ``log`` (decoded lines); given the pool's live
    ``lessons`` (each one's session and text), what it missed of them too."""

    learnings = {record.entry_id: record.learning for record in snapshot.learnings}
    tests = branch_tests(snapshot)
    shown = first_shown(log)
    uptake, outcomes = _uptake(shown, learnings, tests), _outcomes(shown, learnings, tests)
    leads = _leads(log, tests)
    past = [str(item) for line in log for item in line.get(PAST, ())]
    pooled = [item for item in past if is_lesson_id(item)]
    missed = rediscoveries(snapshot, lessons, first_shown(log, PAST)) if lessons is not None else set()
    return LearningEvaluation(
        retrievals=len(log),
        by_source=dict(Counter(str(line["source"]) for line in log)),
        pitfalls_shown=uptake["pitfalls"],
        pitfalls_avoided=uptake["avoided"],
        insights_shown=uptake["insights"],
        insights_taken_up=uptake["taken_up"],
        tests_in_shown_directions=outcomes["shown"],
        committed_in_shown_directions=outcomes["shown_committed"],
        tests_elsewhere=outcomes["else"],
        committed_elsewhere=outcomes["else_committed"],
        past_shown=len(past),
        past_distinct=len(set(past)),
        leads_shown=leads["shown"],
        leads_taken_up=leads["taken_up"],
        leads_committed=leads["committed"],
        lessons_shown=len(pooled),
        lessons_distinct=len(set(pooled)),
        learnings_written=len(current_learnings(snapshot)),
        rediscovered=len(missed),
    )


async def read_session(notebook_root: Path, log_path: Path, archive: Path | None = None) -> LearningEvaluation:
    """The evaluation of the session whose notebook is ``notebook_root``, read as an observer, and its log; with
    ``archive``, against the live lessons of the pool beside it too."""

    snapshot = await LocalNotebook(notebook_root, Writer(Role.OBSERVER, STUDY_WRITER)).snapshot()
    log = read_jsonl(log_path)
    if archive is None:
        return evaluate_learnings(snapshot, log)
    live = LessonPool.beside(TreeArchive(archive)).live()
    return evaluate_learnings(snapshot, log, {head.lesson_id: (head.session_id, head.content) for head in live})


__all__ = [
    "HERE",
    "PAST",
    "REDISCOVERY_OVERLAP",
    "STUDY_WRITER",
    "BranchTest",
    "LearningEvaluation",
    "branch_tests",
    "evaluate_learnings",
    "first_shown",
    "names",
    "read_session",
    "rediscoveries",
]
