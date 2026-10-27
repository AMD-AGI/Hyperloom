# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How branches like this session's went on, for the scheduler's judge and the stop rule it applies.

The judge stops a branch on its stall: the judged tests it has run past its last commit. Stopping trades the
branch's next tests for a new branch's, so :func:`stall_odds` reads two curves off the archived search trees, one
observation for every judged test:

* ``h(j)``, how often a branch's ``j``-th judged test past its last commit committed;
* ``g(i)``, how often the ``i``-th judged test of a branch opened mid-session, after its session's opening batch
  (:data:`OPENING_SECONDS`), committed while it had not committed yet. A branch opened now, into the slot a stop
  frees, starts after the opening batch took the easy wins, so the opening batch would overstate it.

From them, ``p(k)``, the odds that a branch stalled ``k`` tests commits within its next :data:`HORIZON_TESTS`, is
``1 - (1 - h(k+1)) ... (1 - h(k+HORIZON_TESTS))``, and ``q``, the odds a branch opened mid-session commits within its
first :data:`HORIZON_TESTS`, the same product over ``g``. Counting each test once, rather than whole windows, keeps a branch
the judge stopped early as evidence for the tests it did run. The stop bar, :attr:`StallOdds.stall_tests`, is the
first ``k`` from :data:`MIN_STALL_TESTS` on whose ``p(k)`` falls below ``q``. Where too few branches ran that far to
read ``p(k)``, the bar is that ``k``, and :data:`DEFAULT_STALL_TESTS` at least: the archive cannot show keeping such a
branch is worse, so it is kept until probation shows more. The odds also give the median time from the opening of a
branch opened mid-session to its first commit, which tells the judge when a branch opened late has no time left to
commit.

A session of the same model type weighs 1 and any other :data:`OTHER_MODEL_WEIGHT`; sessions of another workload do
not count. The archive's weight at each test is capped at :data:`PRIOR_TESTS` observations, so the running session's
own branches, which count in full, move the odds as it goes. :func:`session_stall_odds` reads them for a running
session from its own snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions

#: How many judged tests past a stall, or into a new branch, the odds look for a commit in.
HORIZON_TESTS = 4
MIN_STALL_TESTS = 4
MAX_STALL_TESTS = 10
DEFAULT_STALL_TESTS = 6
#: The last test position the curves tell apart; later tests count at this one.
MAX_POSITION = MAX_STALL_TESTS + HORIZON_TESTS
#: How much a session of another model type weighs beside one of the same, which weighs 1.
OTHER_MODEL_WEIGHT = 0.5
#: The most observations the archive weighs at one test position, so the running session's own move it.
PRIOR_TESTS = 20.0
#: The fewest observations, weighted, a test position needs to be read.
MIN_SEEN = 5.0
#: How long after a session's first branch its opening batch still opens: the archived sessions opened theirs within
#: this, and every branch after it opened mid-session.
OPENING_SECONDS = 15 * 60.0
MAX_CHARS = 1500
MODEL_TYPE_LABEL = "model_type"
WORKLOAD_LABEL = "workload"
DEFAULT_WORKLOAD = "inferencex"
_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class Odds:
    """How often a test committed: ``hits`` of ``seen`` observations, both weighted."""

    hits: float = 0.0
    seen: float = 0.0

    @property
    def rate(self) -> float | None:
        """``hits`` over ``seen``; ``None`` below :data:`MIN_SEEN` observations."""

        return self.hits / self.seen if self.seen >= MIN_SEEN else None

    def plus(self, other: Odds) -> Odds:
        return Odds(self.hits + other.hits, self.seen + other.seen)

    def capped(self, cap: float) -> Odds:
        """These odds worth at most ``cap`` observations, the rate kept."""

        return self if self.seen <= cap else Odds(self.hits * cap / self.seen, cap)


def within(curve: Mapping[int, Odds], first: int, tests: int = HORIZON_TESTS) -> float | None:
    """The odds of a commit within ``tests`` tests from position ``first`` on ``curve``; ``None`` when one of those
    positions cannot be read."""

    missed = 1.0
    for position in range(first, first + tests):
        rate = curve.get(min(position, MAX_POSITION), Odds()).rate
        if rate is None:
            return None
        missed *= 1.0 - rate
    return 1.0 - missed


@dataclass(frozen=True)
class StallOdds:
    """The odds a session's stop rule rests on.

    ``hazard`` maps a test position past the last commit to ``h``'s odds, ``new_branch`` the test position of a branch
    opened mid-session to ``g``'s, and ``opening_branch`` that of a branch of the opening batch to the same odds,
    which only a replay reads. ``first_commit_minutes`` is the weighted median time from a mid-session branch's
    opening to its first commit, and ``stall_tests`` the stop bar; ``sessions`` and ``same_model`` say how many
    archived sessions counted, and how many of this model type, and ``session_tests`` how many judged tests of the
    running session.
    """

    sessions: int
    same_model: int
    hazard: Mapping[int, Odds] = field(default_factory=dict)
    new_branch: Mapping[int, Odds] = field(default_factory=dict)
    opening_branch: Mapping[int, Odds] = field(default_factory=dict)
    first_commit_minutes: float | None = None
    stall_tests: int = DEFAULT_STALL_TESTS
    session_tests: int = 0

    def after_stall(self, stall: int) -> float | None:
        """``p(stall)``: the odds of a commit within :data:`HORIZON_TESTS` more tests after ``stall`` without one."""

        return within(self.hazard, stall + 1)

    @property
    def new_branch_odds(self) -> float | None:
        """``q``: the odds of a new branch committing within its first :data:`HORIZON_TESTS` tests."""

        return within(self.new_branch, 1)

    def render(self, max_chars: int = MAX_CHARS) -> str:
        """The odds as prompt text of at most ``max_chars``; a line that does not fit is left out, never cut."""

        kept: list[str] = []
        used = 0
        for line in _lines(self):
            if used + len(line) + 1 <= max_chars:
                kept.append(line)
                used += len(line) + 1
        return "\n".join(kept)


def judged_tests(branch: TreeBranch) -> list[float]:
    """When the branch's tests were judged: their result times, or request times, in a tree archived before that."""

    times = branch.judged_times
    if times is None:
        times = branch.ended_times if branch.ended_times is not None else branch.test_times
    return sorted(times)


def _committing(branch: TreeBranch, judged: Sequence[float]) -> set[int]:
    """The positions in ``judged`` of the tests behind the branch's own commits that did not fail to merge: the last
    judged before each."""

    found = set()
    for commit in branch.commits:
        if commit.origin is not CommitOrigin.COMMIT or commit.merge_failed:
            continue
        before = [index for index, when in enumerate(judged) if when <= commit.t_s]
        if before:
            found.add(before[-1])
    return found


def hazard_observations(branch: TreeBranch) -> Iterator[tuple[int, bool]]:
    """Each judged test of ``branch``, as (its position past the branch's last commit that did not fail to merge, 1
    for the first, whether it made one)."""

    judged = judged_tests(branch)
    committing = _committing(branch, judged)
    position = 0
    for index in range(len(judged)):
        position += 1
        committed = index in committing
        yield position, committed
        if committed:
            position = 0


def new_branch_observations(branch: TreeBranch) -> Iterator[tuple[int, bool]]:
    """The judged tests of ``branch``, opened after the session's root, up to its first commit: (the test's position
    in the branch, whether it committed); none for the root's branch."""

    if branch.origin is LineageKind.ROOT:
        return
    for position, committed in hazard_observations(branch):
        yield position, committed
        if committed:
            return


def _first_commit_minutes(branch: TreeBranch) -> float | None:
    own = [commit.t_s for commit in branch.commits if commit.origin is CommitOrigin.COMMIT]
    return (min(own) - branch.opened_s) / _SECONDS_PER_MINUTE if own else None


def _added(curve: Mapping[int, Odds], position: int, odds: Odds) -> dict[int, Odds]:
    key = min(position, MAX_POSITION)
    return {**curve, key: curve.get(key, Odds()).plus(odds)}


def opened_with(tree: SearchTree) -> float:
    """The last moment a branch of ``tree`` opened with its opening batch: :data:`OPENING_SECONDS` past its first
    steered branch."""

    return min((branch.opened_s for branch in tree.branches if branch.steered), default=0.0) + OPENING_SECONDS


@dataclass
class _Tally:
    hazard: dict[int, Odds] = field(default_factory=dict)
    new_branch: dict[int, Odds] = field(default_factory=dict)
    opening_branch: dict[int, Odds] = field(default_factory=dict)
    first_commits: list[tuple[float, float]] = field(default_factory=list)

    def add(self, tree: SearchTree, weight: float) -> None:
        opening = opened_with(tree)
        for branch in tree.branches:
            if not branch.steered:
                continue
            for position, committed in hazard_observations(branch):
                self.hazard = _added(self.hazard, position, Odds(weight * committed, weight))
            mid_session = branch.opened_s > opening
            for position, committed in new_branch_observations(branch):
                seen = Odds(weight * committed, weight)
                if mid_session:
                    self.new_branch = _added(self.new_branch, position, seen)
                else:
                    self.opening_branch = _added(self.opening_branch, position, seen)
            minutes = _first_commit_minutes(branch)
            if mid_session and minutes is not None and branch.origin is not LineageKind.ROOT:
                self.first_commits.append((minutes, weight))

    def capped(self) -> _Tally:
        return _Tally(
            {position: odds.capped(PRIOR_TESTS) for position, odds in self.hazard.items()},
            {position: odds.capped(PRIOR_TESTS) for position, odds in self.new_branch.items()},
            {position: odds.capped(PRIOR_TESTS) for position, odds in self.opening_branch.items()},
            list(self.first_commits),
        )

    def plus(self, other: _Tally) -> _Tally:
        return _Tally(
            _sum(self.hazard, other.hazard),
            _sum(self.new_branch, other.new_branch),
            _sum(self.opening_branch, other.opening_branch),
            [
                *self.first_commits,
                *other.first_commits,
            ],
        )


def _sum(first: Mapping[int, Odds], second: Mapping[int, Odds]) -> dict[int, Odds]:
    return {key: first.get(key, Odds()).plus(second.get(key, Odds())) for key in sorted({*first, *second})}


def _weighted_median(pairs: Sequence[tuple[float, float]]) -> float | None:
    kept = sorted((value, weight) for value, weight in pairs if weight > 0)
    total = sum(weight for _, weight in kept)
    running = 0.0
    for value, weight in kept:
        running += weight
        if running >= total / 2:
            return value
    return None


def stall_bar(hazard: Mapping[int, Odds], new_branch: Mapping[int, Odds]) -> int:
    """The stop bar the curves set: the first ``k`` from :data:`MIN_STALL_TESTS` whose ``p(k)`` is below ``q``; where
    ``p(k)`` cannot be read, that ``k`` and :data:`DEFAULT_STALL_TESTS` at least; :data:`DEFAULT_STALL_TESTS` without
    ``q``."""

    q = within(new_branch, 1)
    if q is None:
        return DEFAULT_STALL_TESTS
    for stall in range(MIN_STALL_TESTS, MAX_STALL_TESTS + 1):
        p = within(hazard, stall + 1)
        if p is None:
            return min(MAX_STALL_TESTS, max(stall, DEFAULT_STALL_TESTS))
        if p < q:
            return stall
    return MAX_STALL_TESTS


def _workload(task: TaskConditions) -> str:
    return task.labels.get(WORKLOAD_LABEL, DEFAULT_WORKLOAD)


def stall_odds(history: Sequence[SearchTree], task: TaskConditions, *, session: SearchTree | None = None) -> StallOdds:
    """The odds for a session on ``task`` from the archived ``history``, with the running ``session``'s own tree in
    full when given; an archived tree of the session itself is left out."""

    model_type = task.labels.get(MODEL_TYPE_LABEL)
    archived = _Tally()
    sessions = same_model = 0
    for tree in history:
        if (session is not None and tree.session_id == session.session_id) or _workload(tree.task) != _workload(task):
            continue
        same = model_type is not None and tree.task.labels.get(MODEL_TYPE_LABEL) == model_type
        archived.add(tree, 1.0 if same else OTHER_MODEL_WEIGHT)
        sessions += 1
        same_model += same
    tally = archived.capped()
    own = _Tally()
    if session is not None:
        own.add(session, 1.0)
        tally = tally.plus(own)
    return StallOdds(
        sessions=sessions,
        same_model=same_model,
        hazard=dict(sorted(tally.hazard.items())),
        new_branch=dict(sorted(tally.new_branch.items())),
        opening_branch=dict(sorted(tally.opening_branch.items())),
        first_commit_minutes=_weighted_median(tally.first_commits),
        stall_tests=stall_bar(tally.hazard, tally.new_branch),
        session_tests=round(sum(odds.seen for odds in own.hazard.values())),
    )


def session_stall_odds(snapshot: NotebookSnapshot, history: Sequence[SearchTree], metric: str) -> StallOdds | None:
    """The odds for the session ``snapshot`` shows, its own branches counted in; ``None`` before its profile and
    baseline."""

    try:
        tree = tree_from_snapshot(snapshot, metric)
    except (NotRecorded, ValueError):
        return None
    return stall_odds(history, tree.task, session=tree)


def _share(rate: float | None) -> str:
    return "too few" if rate is None else f"{rate * 100:.0f}%"


def _bar_reason(odds: StallOdds) -> str:
    kept, new = odds.after_stall(odds.stall_tests), odds.new_branch_odds
    if kept is None or new is None:
        return (
            f"where the archive cannot tell yet ({DEFAULT_STALL_TESTS} at least); what branches drawn for probation do "
            "past it will tell"
        )
    if kept < new:
        return f"the first stall from {MIN_STALL_TESTS} at which a kept branch is less likely to commit than a new one"
    return "the most the rule allows, a kept branch staying likelier to commit than a new one"


def _lines(odds: StallOdds) -> list[str]:
    source = f"{odds.sessions} archived sessions ({odds.same_model} of this model type)"
    if odds.session_tests:
        source += f" and this session's {odds.session_tests} judged tests"
    hazard = [
        f"{position}{'+' if position == MAX_POSITION else ''}: {_share(item.rate)} of {item.seen:.0f}"
        for position, item in odds.hazard.items()
        if item.rate is not None
    ]
    stalls = {stall: odds.after_stall(stall) for stall in range(MAX_STALL_TESTS + 1)}
    known = [f"k={stall} {_share(kept)}" for stall, kept in stalls.items() if kept is not None]
    rest = "; too few tests to tell at the other k" if len(known) < len(stalls) else ""
    lines = [
        f"Stall odds, from {source}. How often a branch's j-th judged test past its last commit committed: "
        f"{', '.join(hazard) or 'too few tests yet'}.",
        f"So a branch stalled k tests commits within its next {HORIZON_TESTS}: {', '.join(known) or 'too few tests yet'}"
        f"{rest}.",
        f"A branch opened mid-session commits within its first {HORIZON_TESTS} judged tests: "
        f"{_share(odds.new_branch_odds)}.",
    ]
    if odds.first_commit_minutes is not None:
        lines.append(f"Its first commit came {odds.first_commit_minutes:.0f} min after it opened (median).")
    lines.append(
        f"Stop bar: {odds.stall_tests} judged tests past a branch's last commit that did not fail to merge, "
        f"{_bar_reason(odds)}."
    )
    return lines


__all__ = [
    "DEFAULT_STALL_TESTS",
    "DEFAULT_WORKLOAD",
    "HORIZON_TESTS",
    "MAX_CHARS",
    "MAX_POSITION",
    "MAX_STALL_TESTS",
    "MIN_SEEN",
    "MIN_STALL_TESTS",
    "MODEL_TYPE_LABEL",
    "OPENING_SECONDS",
    "OTHER_MODEL_WEIGHT",
    "PRIOR_TESTS",
    "WORKLOAD_LABEL",
    "Odds",
    "StallOdds",
    "hazard_observations",
    "judged_tests",
    "new_branch_observations",
    "opened_with",
    "session_stall_odds",
    "stall_bar",
    "stall_odds",
    "within",
]
