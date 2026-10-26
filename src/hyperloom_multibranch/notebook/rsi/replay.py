# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay: how an archived session would have gone under a scheduling playbook, the playbook's offline evaluation.

:func:`simulate` replays one archived session under a playbook, or under none to follow the archive. A branch's judged
tests are the replay's events. Where the playbook agrees with the archive, the replay follows what happened. Where it
does not:

* a branch the playbook stops earlier than the archive did loses what it did after, which is exact;
* a branch the judge stopped and the playbook keeps goes on with sampled tests, one every interval it kept, each
  committing with ``h(k)`` of the stall odds (``g(i)`` before its first commit) and adding a gain drawn from the
  commits of the other archived sessions, of the same model type where they are enough;
* a slot the playbook frees opens the next branch of the archive not opened yet, its record shifted to start then,
  from the integration gain then, or a sampled branch once the archive runs out; none opens in the last
  ``last_open_minutes`` of the session, by default the median time a new branch took to its first commit.

A branch a cause other than the judge ended, its specialist giving up or failing to start, ends where it did; no
playbook keeps it. The integration branch is taken as the best gain any branch has reached, so the replay's measure
is the best gain at the end, with the GPU hours its tests cost at the rate each branch's record gives.

Every statistic a session is replayed with is read off the other sessions (:func:`pace_for`), so no session is judged
on its own outcome. Beyond what they recorded, ``h(k)`` and ``g(i)`` hold their last readable value
(:data:`OPTIMISTIC`) or fall to 0 (:data:`PESSIMISTIC`); a conclusion has to hold both ways. Each draw is keyed by the
session, the run and the branch, test by test, so every playbook meets the same draws, and :func:`compare` reads
paired differences against the archive, with an interval bootstrapped over sessions.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import statistics
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.notebook.rsi.history import (
    DEFAULT_WORKLOAD,
    MAX_POSITION,
    MAX_STALL_TESTS,
    MIN_STALL_TESTS,
    MODEL_TYPE_LABEL,
    WORKLOAD_LABEL,
    Odds,
    judged_tests,
    stall_odds,
)
from hyperloom_multibranch.notebook.rsi.playbook import H0, Playbook, PlaybookNumbers, replaced
from hyperloom_multibranch.notebook.rsi.report import number, table
from hyperloom_multibranch.notebook.rsi.stops import JUDGE_CAUSE

REPLAY_RUNS = 200
BOOTSTRAP_DRAWS = 1000
#: How ``h(k)`` and ``g(i)`` read where the other sessions recorded too few tests.
OPTIMISTIC = "optimistic"
PESSIMISTIC = "pessimistic"
#: The fewest commits of the same model type the gains are drawn from; with fewer, every model type's count.
MIN_GAINS = 10
#: The minutes to a first commit when the other sessions recorded none.
DEFAULT_FIRST_COMMIT_MINUTES = 42.0
#: The judged-test interval of a branch or a session that recorded fewer than two.
DEFAULT_INTERVAL_SECONDS = 900.0
#: The archived session's own label in a comparison.
ARCHIVE = "archive"
#: The probation shares and branch caps :func:`variants` tries beside H0.
PROBATION_SHARES = (0.125, 0.5)
CAPS = (2, 4, 6)
#: The old rule in a replay: the judge stopped the branches it scored lowest, at a median of three judged tests past
#: their last commit over the archived sessions (the stop audit), and kept no probation.
OLD_RULE = "old-rule"
OLD_RULE_STALL = 3
_SESSION_END = "session_end"
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0
_INTERVAL_PERCENT = 95.0


@dataclass(frozen=True)
class Pace:
    """What a session is replayed with, read off the other sessions: ``h(k)`` and ``g(i)`` by position, the first at
    index 0; the gains a commit added, sorted; the stop bar their stall odds set; the median seconds a new branch
    took to its first commit."""

    hazard: tuple[float, ...]
    new_branch: tuple[float, ...]
    gains: tuple[float, ...]
    bar: int
    first_commit_seconds: float


def _curve(curve: Mapping[int, Odds], beyond: str) -> tuple[float, ...]:
    rates: list[float] = []
    last = 0.0
    for position in range(1, MAX_POSITION + 1):
        rate = curve.get(position, Odds()).rate
        if rate is None:
            rate = last if beyond == OPTIMISTIC else 0.0
        else:
            last = rate
        rates.append(rate)
    return tuple(rates)


def _own(branch: TreeBranch) -> list[tuple[float, float]]:
    """The branch's own commits that did not fail to merge, as (when, the gain reached), in order."""

    return sorted(
        (commit.t_s, commit.gain_pct)
        for commit in branch.commits
        if commit.origin is CommitOrigin.COMMIT and not commit.merge_failed
    )


def _increments(branch: TreeBranch) -> list[float]:
    reached, before = [], branch.fork_gain_pct
    for _, gain in _own(branch):
        reached.append(gain - before)
        before = gain
    return reached


def _label(tree: SearchTree, key: str, default: str = "") -> str:
    return tree.task.labels.get(key, default)


def pace_for(tree: SearchTree, history: Sequence[SearchTree], beyond: str = OPTIMISTIC) -> Pace:
    """What ``tree`` is replayed with: the stall odds and the commit gains of ``history`` less ``tree`` itself."""

    others = tuple(item for item in history if item.session_id != tree.session_id)
    odds = stall_odds(others, tree.task)
    workload = _label(tree, WORKLOAD_LABEL, DEFAULT_WORKLOAD)
    alike = [item for item in others if _label(item, WORKLOAD_LABEL, DEFAULT_WORKLOAD) == workload]
    same = [item for item in alike if _label(item, MODEL_TYPE_LABEL) == _label(tree, MODEL_TYPE_LABEL)]

    def gains(trees: Sequence[SearchTree]) -> list[float]:
        return [gain for item in trees for branch in item.branches if branch.steered for gain in _increments(branch)]

    drawn = gains(same)
    minutes = odds.first_commit_minutes if odds.first_commit_minutes is not None else DEFAULT_FIRST_COMMIT_MINUTES
    return Pace(
        hazard=_curve(odds.hazard, beyond),
        new_branch=_curve(odds.new_branch, beyond),
        gains=tuple(sorted(drawn if len(drawn) >= MIN_GAINS else gains(alike))),
        bar=odds.stall_tests,
        first_commit_seconds=minutes * _SECONDS_PER_MINUTE,
    )


@dataclass(frozen=True)
class RunOutcome:
    """One replay: the best gain reached, the GPU hours spent, the branches stopped and the branches opened."""

    best_gain_pct: float
    gpu_hours: float
    stops: int
    opened: int


def _cause(branch: TreeBranch) -> str:
    cause, said, _ = branch.stop_reason.partition(":")
    return cause.strip() if said else ""


def _steered(tree: SearchTree) -> list[TreeBranch]:
    return sorted((branch for branch in tree.branches if branch.steered), key=lambda branch: branch.opened_s)


def _end(branch: TreeBranch, tree: SearchTree) -> float:
    return branch.closed_s if branch.closed_s is not None else tree.budget_seconds


def archive_cap(tree: SearchTree) -> int:
    """The most steered branches the archived session ran at once."""

    edges = sorted(
        (when, step) for branch in _steered(tree) for when, step in ((branch.opened_s, 1), (_end(branch, tree), -1))
    )
    running = most = 0
    for _, step in edges:
        running += step
        most = max(most, running)
    return max(1, most)


def _median_gap(times: Sequence[float]) -> float | None:
    gaps = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    return statistics.median(gaps) if gaps else None


def _gpu_per_test(branch: TreeBranch) -> float:
    tests = len(judged_tests(branch)) or len(branch.test_times)
    return branch.gpu_seconds / tests if tests else 0.0


def _seeded(*parts: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class _Test:
    at: float
    gain: float | None = None


@dataclass
class _Branch:
    """One branch as the replay runs it: its tests from here on, the one due next, how it stands, and where its
    record ends."""

    key: str
    opened: float
    tests: Iterator[_Test]
    gpu_per_test: float
    hard_end: float
    gain: float = 0.0
    next_test: _Test = field(default_factory=lambda: _Test(math.inf))
    position: int = 0
    judged: int = 0
    commits: int = 0


@dataclass
class _Run:
    """One replay of ``tree``: what it has reached so far, its branches running, and the tests due, in order."""

    tree: SearchTree
    pace: Pace
    playbook: Playbook | None
    run: int
    seed: str
    best: float = 0.0
    gpu_seconds: float = 0.0
    stops: int = 0
    opened: int = 0
    synthetic: int = 0
    events: list[tuple[float, int, str]] = field(default_factory=list)
    running: dict[str, _Branch] = field(default_factory=dict)
    order: int = 0
    session_gap: float = field(init=False)
    session_gpu: float = field(init=False)
    archive_best: list[tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        steered = _steered(self.tree)
        gaps = [gap for gap in (_median_gap(judged_tests(branch)) for branch in steered) if gap is not None]
        self.session_gap = statistics.median(gaps) if gaps else DEFAULT_INTERVAL_SECONDS
        rates = [rate for rate in map(_gpu_per_test, steered) if rate > 0]
        self.session_gpu = statistics.median(rates) if rates else 0.0
        self.archive_best = sorted((when, gain) for branch in steered for when, gain in _own(branch))

    def play(self) -> None:
        raise NotImplementedError

    def archive_gain_at(self, t: float) -> float:
        return max((gain for when, gain in self.archive_best if when <= t), default=0.0)

    def draws(self, key: str) -> random.Random:
        return _seeded(self.seed, self.tree.session_id, self.run, key)

    def start(self, branch: _Branch) -> None:
        self.running[branch.key] = branch
        self.opened += 1
        self.schedule(branch)

    def schedule(self, branch: _Branch) -> None:
        """Queue ``branch``'s next test, or end it where its record or the budget runs out."""

        self.order += 1
        upcoming = next(branch.tests, None)
        end = min(branch.hard_end, self.tree.budget_seconds)
        if upcoming is None or upcoming.at > end:
            self.running.pop(branch.key, None)
            self.ended(end)
            return
        branch.next_test = upcoming
        heapq.heappush(self.events, (upcoming.at, self.order, branch.key))

    def ended(self, at: float) -> None:
        """What follows a branch's end at ``at``; the playbook's replay opens into the slot."""


class _ArchiveRun(_Run):
    """The archive followed: every branch opens and ends as it did."""

    def play(self) -> None:
        for source in _steered(self.tree):
            recorded = _recorded(source, offset=0.0, base=source.fork_gain_pct, until=_last_test(source))
            tests = (_Test(at, gain) for at, gain in recorded)
            self.start(_Branch(source.name, source.opened_s, tests, _gpu_per_test(source), math.inf))
            if _cause(source) == JUDGE_CAUSE:
                self.stops += 1
        while self.events:
            _, _, key = heapq.heappop(self.events)
            branch = self.running[key]
            self.gpu_seconds += branch.gpu_per_test
            if branch.next_test.gain is not None:
                self.best = max(self.best, branch.next_test.gain)
            self.schedule(branch)


def _last_test(branch: TreeBranch) -> float:
    judged = judged_tests(branch)
    return judged[-1] if judged else branch.opened_s


def _recorded(source: TreeBranch, *, offset: float, base: float, until: float) -> Iterator[tuple[float, float | None]]:
    """``source``'s judged tests up to ``until``, shifted by ``offset``, each with the gain reached by the last commit
    it made, counted from ``base`` rather than the fork gain, or ``None``: a commit goes to the last test judged by
    then, or the first."""

    judged = [when for when in judged_tests(source) if when <= until]
    commits = _own(source)
    for index, when in enumerate(judged):
        after = judged[index + 1] if index + 1 < len(judged) else math.inf
        landed = [gain for at, gain in commits if (index == 0 or when <= at) and at < after]
        yield when + offset, (base + landed[-1] - source.fork_gain_pct) if landed else None


@dataclass
class _PlaybookRun(_Run):
    """The playbook's replay: it stops on the stall, opens into freed slots, and samples what the archive lacks.
    ``bar`` is the stop bar, ``cap`` the branches at once, ``last_open`` the last moment a branch opens, and
    ``queue`` the archive's branches not opened yet."""

    numbers: PlaybookNumbers = field(default_factory=PlaybookNumbers)
    bar: int = 0
    cap: int = 1
    last_open: float = 0.0
    queue: deque[TreeBranch] = field(default_factory=deque)

    def play(self) -> None:
        if self.playbook is None:
            raise ValueError("a playbook's replay needs a playbook")
        numbers = self.numbers = self.playbook.numbers
        self.bar = numbers.stall_tests if numbers.stall_tests is not None else self.pace.bar
        self.cap = numbers.max_open if numbers.max_open is not None else archive_cap(self.tree)
        late = numbers.last_open_minutes
        self.last_open = self.tree.budget_seconds - (
            late * _SECONDS_PER_MINUTE if late is not None else self.pace.first_commit_seconds
        )
        self.queue = deque(_steered(self.tree))
        start = self.queue[0].opened_s if self.queue else 0.0
        while len(self.running) < self.cap and self.open(start):
            pass
        while self.events:
            at, _, key = heapq.heappop(self.events)
            branch = self.running.get(key)
            if branch is None:
                continue
            if self.tested(branch, at):
                self.schedule(branch)
            else:
                self.running.pop(key)
                self.stops += 1
                self.refill(at)

    def ended(self, at: float) -> None:
        self.refill(at)

    def refill(self, at: float) -> None:
        while len(self.running) < self.cap and at <= self.last_open and self.open(at):
            pass

    def open(self, at: float) -> bool:
        """Open a branch at ``at``: the archive's next, shifted, or a sampled one; ``False`` when none can open."""

        if at >= self.tree.budget_seconds:
            return False
        if self.queue:
            self.start(self.borrowed(self.queue.popleft(), at))
        else:
            self.synthetic += 1
            self.start(self.sampled(f"sampled-{self.synthetic}", at, None))
        return True

    def borrowed(self, source: TreeBranch, at: float) -> _Branch:
        offset = at - source.opened_s
        base = source.fork_gain_pct + self.best - self.archive_gain_at(source.opened_s)
        cause = _cause(source)
        recorded = list(_recorded(source, offset=offset, base=base, until=_last_test(source)))
        extendable = cause in (JUDGE_CAUSE, _SESSION_END, "") or source.closed_s is None
        hard_end = math.inf if extendable else _end(source, self.tree) + offset
        gap = _median_gap([when for when, _ in recorded]) or self.session_gap
        rate = _gpu_per_test(source) or self.session_gpu
        branch = _Branch(source.name, at, iter(()), rate, hard_end, gain=base)
        tail = self.extension(branch, recorded[-1][0] if recorded else at, gap) if extendable else iter(())
        branch.tests = _chain((_Test(when, gain) for when, gain in recorded), tail)
        return branch

    def sampled(self, key: str, at: float, gap: float | None) -> _Branch:
        branch = _Branch(key, at, iter(()), self.session_gpu, math.inf, gain=self.best)
        branch.tests = self.extension(branch, at, gap or self.session_gap)
        return branch

    def extension(self, branch: _Branch, after: float, gap: float) -> Iterator[_Test]:
        """Sampled tests every ``gap`` from ``after``: each commits with ``h(k)``, or ``g(i)`` before the branch's
        first commit, adding a gain drawn from :attr:`Pace.gains`."""

        draws, gains = self.draws(branch.key), self.pace.gains
        at = after
        while True:
            at += gap
            committed, picked = draws.random(), draws.random()
            if branch.commits:
                odds = self.pace.hazard[min(branch.position + 1, MAX_POSITION) - 1]
            else:
                odds = self.pace.new_branch[min(branch.judged + 1, MAX_POSITION) - 1]
            if committed < odds and gains:
                yield _Test(at, branch.gain + gains[min(int(picked * len(gains)), len(gains) - 1)])
            else:
                yield _Test(at)

    def tested(self, branch: _Branch, at: float) -> bool:
        """Count the test due at ``at`` and say whether the branch keeps running after it."""

        test = branch.next_test
        self.gpu_seconds += branch.gpu_per_test
        branch.judged += 1
        if test.gain is not None:
            branch.gain, branch.position = test.gain, 0
            branch.commits += 1
            self.best = max(self.best, test.gain)
            return True
        branch.position += 1
        return not self.stopped(branch)

    def stopped(self, branch: _Branch) -> bool:
        """Whether ``branch`` is stopped here: past the bar and its grace, and not held by a probation draw."""

        numbers = self.numbers
        if branch.judged < numbers.grace_tests or branch.position < self.bar:
            return False
        drawn = _seeded(self.seed, self.tree.session_id, self.run, branch.key, "probation", branch.commits).random()
        return not (drawn < numbers.probation_share and branch.position < self.bar + numbers.probation_tests)


def _chain(first: Iterator[_Test], then: Iterator[_Test]) -> Iterator[_Test]:
    yield from first
    yield from then


def simulate(tree: SearchTree, playbook: Playbook | None, pace: Pace, run: int = 0, *, seed: str = "") -> RunOutcome:
    """One replay of ``tree`` under ``playbook``, ``None`` to follow the archive, with the draws of ``run``."""

    replay: _Run = (_ArchiveRun if playbook is None else _PlaybookRun)(tree, pace, playbook, run, seed)
    replay.play()
    return RunOutcome(replay.best, replay.gpu_seconds / _SECONDS_PER_HOUR, replay.stops, replay.opened)


@dataclass(frozen=True)
class PolicyRow:
    """One playbook over the compared sessions: the mean of each session's mean best gain, GPU hours and stops; the
    mean paired difference in best gain against the comparison's baseline, run by run, with its interval
    bootstrapped over sessions."""

    label: str
    best_gain_pct: float
    gain_difference: float
    difference_low: float
    difference_high: float
    gpu_hours: float
    stops: float

    @property
    def gain_per_gpu_hour(self) -> float | None:
        return self.best_gain_pct / self.gpu_hours if self.gpu_hours > 0 else None


@dataclass(frozen=True)
class SchedulingComparison:
    """How each playbook replays over the sessions, beside the archive, ``runs`` runs a session, each read against
    ``baseline``. The archive opens its branches when it did, every playbook into the slots it frees, so a
    playbook's difference from another says more than its difference from the archive."""

    rows: tuple[PolicyRow, ...]
    sessions: tuple[str, ...]
    runs: int
    beyond: str
    baseline: str = ARCHIVE

    def render(self) -> str:
        rows = [
            (
                row.label,
                number(row.best_gain_pct),
                f"{number(row.gain_difference)} ({number(row.difference_low)} to {number(row.difference_high)})",
                number(row.gpu_hours, 1),
                number(row.gain_per_gpu_hour, 3),
                number(row.stops, 1),
            )
            for row in self.rows
        ]
        headers = ("playbook", "best gain %", f"vs {self.baseline} (95%)", "GPU h", "gain per GPU h", "stops")
        title = (
            f"Scheduling replay over {len(self.sessions)} sessions, {self.runs} runs each, h(k) {self.beyond} beyond "
            "the data:"
        )
        return "\n".join([title, "", *table(headers, rows)])


def variants() -> dict[str, Playbook]:
    """The playbooks a scheduling replay compares by default: H0; the old rule, which stopped a branch at the
    archive's median stall, :data:`OLD_RULE_STALL` tests, with no probation; the stop bars from 4 to 10; probation for
    an eighth and for half of the stop candidates; and caps of 2, 4 and 6 branches."""

    found = {H0.label: H0, OLD_RULE: replaced(H0, stall_tests=OLD_RULE_STALL, probation_share=0.0)}
    found.update((f"stall-{bar}", replaced(H0, stall_tests=bar)) for bar in range(MIN_STALL_TESTS, MAX_STALL_TESTS + 1))
    found.update((f"probation-{share:.0%}", replaced(H0, probation_share=share)) for share in PROBATION_SHARES)
    found.update((f"cap-{cap}", replaced(H0, max_open=cap)) for cap in CAPS)
    return found


def _interval(values: Sequence[float], draws: int, seed: str) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sampler = _seeded(seed, "bootstrap")
    means = sorted(statistics.fmean(sampler.choices(values, k=len(values))) for _ in range(draws))
    tail = (100.0 - _INTERVAL_PERCENT) / 200.0
    return means[int(tail * (draws - 1))], means[int((1.0 - tail) * (draws - 1))]


def compare(
    trees: Sequence[SearchTree],
    playbooks: Mapping[str, Playbook],
    *,
    runs: int = REPLAY_RUNS,
    beyond: str = OPTIMISTIC,
    seed: str = "",
    draws: int = BOOTSTRAP_DRAWS,
    baseline: str = H0.label,
) -> SchedulingComparison:
    """Each of ``playbooks`` replayed ``runs`` times on every tree, beside the archive, and read against
    ``baseline``, one of them, or the archive when it is none of them."""

    against = baseline if baseline in playbooks else ARCHIVE
    outcomes: dict[str, list[tuple[float, float, float, float]]] = {ARCHIVE: [], **{label: [] for label in playbooks}}
    for tree in trees:
        pace = pace_for(tree, trees, beyond)
        kept = simulate(tree, None, pace)
        played = {ARCHIVE: [kept] * runs}
        played.update(
            (label, [simulate(tree, playbook, pace, run, seed=seed) for run in range(runs)])
            for label, playbook in playbooks.items()
        )
        base = played[against]
        for label, results in played.items():
            outcomes[label].append(
                (
                    statistics.fmean(item.best_gain_pct for item in results),
                    statistics.fmean(item.gpu_hours for item in results),
                    statistics.fmean(item.stops for item in results),
                    statistics.fmean(item.best_gain_pct - other.best_gain_pct for item, other in zip(results, base)),
                )
            )
    rows = []
    for label, per_session in outcomes.items():
        differences = [item[3] for item in per_session]
        low, high = _interval(differences, draws, f"{seed}:{label}")
        rows.append(
            PolicyRow(
                label,
                statistics.fmean(item[0] for item in per_session) if per_session else 0.0,
                statistics.fmean(differences) if differences else 0.0,
                low,
                high,
                statistics.fmean(item[1] for item in per_session) if per_session else 0.0,
                statistics.fmean(item[2] for item in per_session) if per_session else 0.0,
            )
        )
    return SchedulingComparison(tuple(rows), tuple(tree.session_id for tree in trees), runs, beyond, against)


__all__ = [
    "ARCHIVE",
    "BOOTSTRAP_DRAWS",
    "CAPS",
    "DEFAULT_FIRST_COMMIT_MINUTES",
    "DEFAULT_INTERVAL_SECONDS",
    "MIN_GAINS",
    "OLD_RULE",
    "OLD_RULE_STALL",
    "OPTIMISTIC",
    "PESSIMISTIC",
    "PROBATION_SHARES",
    "REPLAY_RUNS",
    "Pace",
    "PolicyRow",
    "RunOutcome",
    "SchedulingComparison",
    "archive_cap",
    "compare",
    "pace_for",
    "simulate",
    "variants",
]
