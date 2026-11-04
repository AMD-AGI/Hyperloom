# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the learnings shown to a branch bore on what it did next: one session's retrieval, evaluated (MB2).

Read beside the session's retrieval log (``learnings.jsonl``, :class:`learning_search.RetrievalLog`) and its blackboard,
each learning of this session shown to a branch counts once, at its first showing, when a direction family
(:mod:`blackboard.families`) can be read off it: that of a test it cites, else of the direction or the words it gives.
A test is in that direction when what it changed, or what its intent says, falls in the family's target and, unless
the family leaves it open, its mechanism, at either layer: two specialists seldom word one direction alike.

* a pitfall is *avoided* when the branch requested no test in its direction afterwards;
* an insight or a hypothesis is *taken up* when it requested one.

Then every test a branch requested after its first retrieval counts once: *in a shown direction* when it is in the
direction of a learning shown to the branch before it, *elsewhere* otherwise; how many of each led to a commit sets the
learnings' directions beside the rest. Past sessions' learnings are logged by id only, an Experience's or a pooled
lesson's (:mod:`rsi.lessons`). An Experience is counted, not matched. A pooled lesson, given the pool, is matched as
this session's learnings are, by the direction and kind its revision gives, and counts apart from them: whether the
branches avoided its pitfalls, took up its insights, and how the tests in its directions fared. Given the pool's live
lessons, a learning of this session that says one of another session again, its branch never shown it, is a
*rediscovery*: what the retrieval missed (:func:`rediscoveries`, which CLOSE's lesson standings use too).

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

from hyperloom_common.jsonio import read_jsonl
from hyperloom_multibranch.contracts.blackboard.peers import Intent, TestKind
from hyperloom_multibranch.contracts.blackboard.records import Learning, BlackboardSnapshot, Writer, is_lesson_id
from hyperloom_multibranch.contracts.blackboard.vocab import COMMITTING, LearningKind, Role
from hyperloom_multibranch.blackboard.families import Family, Mechanism, family_of
from hyperloom_multibranch.blackboard.knowledge.experience import change_document
from hyperloom_multibranch.blackboard.learning_search import current_learnings, family_of_learning, word_overlap, words
from hyperloom_multibranch.blackboard.local.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive
from hyperloom_multibranch.blackboard.rsi.lessons import LessonPool, LessonRevision
from hyperloom_multibranch.blackboard.rsi.report import share, table

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
    lesson_pitfalls_shown: int = 0
    lesson_pitfalls_avoided: int = 0
    lesson_insights_shown: int = 0
    lesson_insights_taken_up: int = 0
    tests_in_lesson_directions: int = 0
    committed_in_lesson_directions: int = 0
    tests_elsewhere_after_lessons: int = 0
    committed_elsewhere_after_lessons: int = 0

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
            ("pooled lessons: pitfalls avoided", share(self.lesson_pitfalls_avoided, self.lesson_pitfalls_shown)),
            (
                "pooled lessons: insights and hypotheses taken up",
                share(self.lesson_insights_taken_up, self.lesson_insights_shown),
            ),
            (
                "tests committed, in a pooled lesson's direction",
                share(self.committed_in_lesson_directions, self.tests_in_lesson_directions),
            ),
            (
                "tests committed, elsewhere after a pooled lesson",
                share(self.committed_elsewhere_after_lessons, self.tests_elsewhere_after_lessons),
            ),
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
    the words of its change, and its families: of what it changed, and of what its intent says."""

    at: float
    intent: Intent | None
    committed: bool
    change: frozenset[str] = frozenset()
    ended: bool = False
    families: frozenset[Family] = frozenset()


def branch_tests(snapshot: BlackboardSnapshot) -> dict[str, list[BranchTest]]:
    """Each branch's tests of kind ``test``, the earliest requested first."""

    found: dict[str, list[BranchTest]] = defaultdict(list)
    for record in sorted(snapshot.records, key=lambda item: item.requested_at):
        if record.request.kind is TestKind.TEST:
            intent = snapshot.intents.get(record.request.intent_fingerprint)
            change = frozenset(str(change_document(record.request.delta)["summary"]).split())
            committed = record.conclusion in COMMITTING
            ended = record.result is not None
            families: frozenset[Family] = frozenset()
            if intent is not None:
                said = family_of(intent.target, intent.mechanism, intent.layer)
                families = frozenset(
                    {family_of(intent.target, intent.mechanism, intent.layer, record.request.delta), said}
                )
            test = BranchTest(record.requested_at, intent, committed, change, ended, families)
            found[record.request.branch].append(test)
    return found


def in_direction(about: Family | None, test: BranchTest) -> bool:
    """Whether ``test`` is in the direction of a learning about ``about``: its target and, unless ``about`` leaves it
    open, its mechanism, at either layer, by what the test changed or what its intent says."""

    if about is None:
        return False
    open_mechanism = about.mechanism is Mechanism.OTHER
    return any(
        found.target == about.target and (open_mechanism or found.mechanism is about.mechanism)
        for found in test.families
    )


def first_shown(log: Sequence[Mapping[str, Any]], ids: str = HERE) -> dict[tuple[str, str], float]:
    """When each branch was first shown each id of the log lines' ``ids`` field, by (branch, id)."""

    shown: dict[tuple[str, str], float] = {}
    for line in sorted(log, key=lambda item: float(item["at"])):
        for entry_id in line.get(ids, ()):
            shown.setdefault((str(line["branch"]), str(entry_id)), float(line["at"]))
    return shown


def _uptake(
    shown: Mapping[tuple[str, str], float],
    learnings: Mapping[str, Learning],
    about: Mapping[str, Family | None],
    tests: Mapping[str, list[BranchTest]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for (branch, entry_id), at in shown.items():
        learning, family = learnings.get(entry_id), about.get(entry_id)
        if learning is None or family is None:
            continue
        followed = any(in_direction(family, test) for test in tests.get(branch, ()) if test.at > at)
        if learning.kind is LearningKind.PITFALL:
            counts["pitfalls"] += 1
            counts["avoided"] += not followed
        elif learning.kind in _TAKEN_UP:
            counts["insights"] += 1
            counts["taken_up"] += followed
    return counts


def _outcomes(
    shown: Mapping[tuple[str, str], float], about: Mapping[str, Family | None], tests: Mapping[str, list[BranchTest]]
) -> Counter[str]:
    by_branch: dict[str, list[tuple[float, Family | None]]] = defaultdict(list)
    for (branch, entry_id), at in shown.items():
        if entry_id in about:
            by_branch[branch].append((at, about[entry_id]))
    counts: Counter[str] = Counter()
    for branch, seen in by_branch.items():
        first = min(at for at, _ in seen)
        for test in (item for item in tests.get(branch, ()) if item.at > first):
            bucket = "shown" if any(at < test.at and in_direction(family, test) for at, family in seen) else "else"
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
    snapshot: BlackboardSnapshot, lessons: Mapping[str, tuple[str, str]], shown: Collection[tuple[str, str]]
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
    snapshot: BlackboardSnapshot,
    log: Sequence[Mapping[str, Any]],
    lessons: Mapping[str, tuple[str, str]] | None = None,
    pooled: Mapping[str, Learning] | None = None,
) -> LearningEvaluation:
    """The retrieval of ``snapshot``'s session, read off its retrieval ``log`` (decoded lines); given the pool's live
    ``lessons`` (each one's session and text), what it missed of them too; given the ``pooled`` lessons, by id, as the
    learnings their revisions give, what the branches did after them, as for this session's learnings."""

    learnings = {record.entry_id: record.learning for record in snapshot.learnings}
    about = {
        key: family_of_learning(learning, snapshot.records, snapshot.intents) for key, learning in learnings.items()
    }
    tests = branch_tests(snapshot)
    shown = first_shown(log)
    uptake, outcomes = _uptake(shown, learnings, about, tests), _outcomes(shown, about, tests)
    lessons_of = pooled or {}
    lesson_about = {lesson_id: family_of_learning(learning, (), {}) for lesson_id, learning in lessons_of.items()}
    lesson_shown = {key: at for key, at in first_shown(log, PAST).items() if key[1] in lessons_of}
    lesson_uptake, lesson_outcomes = (
        _uptake(lesson_shown, lessons_of, lesson_about, tests),
        _outcomes(lesson_shown, lesson_about, tests),
    )
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
        lesson_pitfalls_shown=lesson_uptake["pitfalls"],
        lesson_pitfalls_avoided=lesson_uptake["avoided"],
        lesson_insights_shown=lesson_uptake["insights"],
        lesson_insights_taken_up=lesson_uptake["taken_up"],
        tests_in_lesson_directions=lesson_outcomes["shown"],
        committed_in_lesson_directions=lesson_outcomes["shown_committed"],
        tests_elsewhere_after_lessons=lesson_outcomes["else"],
        committed_elsewhere_after_lessons=lesson_outcomes["else_committed"],
    )


def _as_learning(revision: LessonRevision) -> Learning:
    return Learning("", revision.kind, revision.content, revision.target, revision.mechanism, revision.layer)


async def read_session(blackboard_root: Path, log_path: Path, archive: Path | None = None) -> LearningEvaluation:
    """The evaluation of the session whose blackboard is ``blackboard_root``, read as an observer, and its log; with
    ``archive``, against the lessons of the pool beside it too: the live ones for what the session missed, and every
    one, retired ones as well, since a lesson shown then may have been retired since, for what it shown led to."""

    snapshot = await LocalBlackboard(blackboard_root, Writer(Role.OBSERVER, STUDY_WRITER)).snapshot()
    log = read_jsonl(log_path)
    if archive is None:
        return evaluate_learnings(snapshot, log)
    pool = LessonPool.beside(TreeArchive(archive))
    live = {head.lesson_id: (head.session_id, head.content) for head in pool.live()}
    every = {head.lesson_id: _as_learning(head) for lesson in pool.lessons().values() for head in lesson.heads}
    return evaluate_learnings(snapshot, log, live, every)


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
    "in_direction",
    "read_session",
    "rediscoveries",
]
