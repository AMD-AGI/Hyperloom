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

A playbook's rules only a judge reads, so given ``decide``, a judge's answer to a prompt, the replay asks it every
:data:`ROUND_SECONDS` which running branches to stop, as a session asks its judge, instead of stopping them at the
stall bar; it refuses the stop of a branch on probation, as the scheduler does. Opening and K follow the numbers
either way.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import statistics
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
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
#: The shortest judged-test interval a replay keeps: a server takes longer to launch, so a tree whose tests sit closer
#: was not measured, and its replay would sample without end.
MIN_INTERVAL_SECONDS = 60.0
#: The playbook's numbers a replay plays; a change to any other number replays as its parent.
REPLAYED_NUMBERS = frozenset(
    {"stall_tests", "probation_share", "probation_tests", "grace_tests", "max_open", "last_open_minutes"}
)
#: The archived session's own label in a comparison.
ARCHIVE = "archive"
#: The probation shares and branch caps :func:`variants` tries beside H0.
PROBATION_SHARES = (0.125, 0.5)
CAPS = (2, 4, 6)
#: The old rule in a replay: the judge stopped the branches it scored lowest, at a median of three judged tests past
#: their last commit over the archived sessions (the stop audit), and kept no probation.
OLD_RULE = "old-rule"
OLD_RULE_STALL = 3
#: How often a judged replay asks its judge, as a session's judge is asked.
ROUND_SECONDS = 600.0
#: How a judge's reply names the branches to stop.
STOP_REPLY = (
    'Reply with one JSON object {"stop": [{"branch": "<name>", "reason": "<why>"}]}, its list empty to stop none.'
)
ON_PROBATION = "on probation"
_STOP_KEY = "stop"
_SESSION_END = "session_end"
#: The key of a branch's opening among a replay's events, before its name.
_OPENS = "opens:"
#: The key of a judge's round among a judged replay's events.
_ROUND = "round"
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0
_INTERVAL_PERCENT = 95.0


@dataclass(frozen=True)
class Draws:
    """How a replay draws: ``runs`` runs a session, each draw keyed by ``seed``, ``h(k)`` and ``g(i)`` ``beyond`` the
    data, a comparison's intervals bootstrapped ``draws`` times; with ``decide``, the judge each playbook's replay asks
    every round which branches to stop."""

    runs: int = REPLAY_RUNS
    beyond: str = OPTIMISTIC
    seed: str = ""
    draws: int = BOOTSTRAP_DRAWS
    decide: Callable[[str], str] | None = None


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
    return max(MIN_INTERVAL_SECONDS, statistics.median(gaps)) if gaps else None


def _gpu_per_test(branch: TreeBranch) -> float:
    tests = len(judged_tests(branch)) or len(branch.test_times)
    return branch.gpu_seconds / tests if tests else 0.0


def _seeded(*parts: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def stops_in(reply: str) -> tuple[str, ...]:
    """The branches a judge's reply stops, as :data:`STOP_REPLY` asks; none when it holds no JSON object with a stop
    list."""

    start, end = reply.find("{"), reply.rfind("}")
    try:
        document = json.loads(reply[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return ()
    orders = document.get(_STOP_KEY) if isinstance(document, dict) else None
    return tuple(str(item.get("branch")) for item in orders or () if isinstance(item, Mapping))


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

    def start(self, branch: _Branch) -> bool:
        """Run ``branch`` from its first test; ``False`` when none lands within the budget and it ended as it opened."""

        self.running[branch.key] = branch
        self.opened += 1
        self.schedule(branch)
        return branch.key in self.running

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
            if key == _ROUND:
                self.round(at)
                continue
            branch = self.running.get(key)
            if branch is None:
                continue
            if self.tested(branch, at):
                self.schedule(branch)
            else:
                self.running.pop(key)
                self.stops += 1
                self.refill(at)

    def round(self, at: float) -> None:
        """A judge's round at ``at``; the numbers need none."""

    def ended(self, at: float) -> None:
        self.refill(at)

    def refill(self, at: float) -> None:
        while len(self.running) < self.cap and at <= self.last_open and self.open(at):
            pass

    def open(self, at: float) -> bool:
        """Open a branch at ``at``: the archive's next, shifted, or a sampled one; ``False`` when none can open, or
        the one opened has no test left in the budget, so no other would."""

        if at >= self.tree.budget_seconds:
            return False
        if self.queue:
            return self.start(self.borrowed(self.queue.popleft(), at))
        self.synthetic += 1
        return self.start(self.sampled(f"sampled-{self.synthetic}", at, None))

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

    def on_probation(self, branch: _Branch) -> bool:
        """Whether ``branch``'s stall past the bar was drawn for probation and its probation runs still."""

        drawn = _seeded(self.seed, self.tree.session_id, self.run, branch.key, "probation", branch.commits).random()
        numbers = self.numbers
        return drawn < numbers.probation_share and branch.position < self.bar + numbers.probation_tests

    def stopped(self, branch: _Branch) -> bool:
        """Whether ``branch`` is stopped here: past the bar and its grace, and not held by a probation draw."""

        if branch.judged < self.numbers.grace_tests or branch.position < self.bar:
            return False
        return not self.on_probation(branch)


@dataclass
class _JudgedRun(_PlaybookRun):
    """The playbook's replay with ``decide`` asked every :data:`ROUND_SECONDS` which running branches to stop, and the
    stall bar stopping none: what a judge makes of the playbook's rules. A stop of a branch on probation is
    refused."""

    decide: Callable[[str], str] | None = None

    def play(self) -> None:
        for index in range(1, int(self.tree.budget_seconds // ROUND_SECONDS) + 1):
            self.order += 1
            heapq.heappush(self.events, (index * ROUND_SECONDS, self.order, _ROUND))
        super().play()

    def stopped(self, branch: _Branch) -> bool:
        return False

    def standing(self, branch: _Branch) -> str:
        if branch.judged < self.numbers.grace_tests:
            return "in its grace"
        if branch.position < self.bar:
            return "below the stop bar"
        return ON_PROBATION if self.on_probation(branch) else "a stop candidate"

    def view(self, at: float) -> str:
        """What the judge reads in the round at ``at``: the clock, each running branch's stall evidence, the
        playbook."""

        if self.playbook is None:
            raise ValueError("a judged replay needs a playbook")
        return "\n".join(
            [
                f"Session clock {at / _SECONDS_PER_MINUTE:.0f} of {self.tree.budget_seconds / _SECONDS_PER_MINUTE:.0f}"
                f" minutes; the best gain so far {self.best:+.2f}%.",
                "Running branches:",
                *(
                    f"- {branch.key}: {branch.position} judged tests since its last commit (stop bar {self.bar}), "
                    f"{branch.commits} commits, at {branch.gain:+.2f}%; {self.standing(branch)}"
                    for branch in self.running.values()
                ),
                "",
                self.playbook.render(),
                "",
                STOP_REPLY,
            ]
        )

    def round(self, at: float) -> None:
        if not self.running or self.decide is None:
            return
        for key in stops_in(self.decide(self.view(at))):
            branch = self.running.get(key)
            if branch is not None and self.standing(branch) != ON_PROBATION:
                self.running.pop(key)
                self.stops += 1
        self.refill(at)


def _chain(first: Iterator[_Test], then: Iterator[_Test]) -> Iterator[_Test]:
    yield from first
    yield from then


@dataclass
class _CutRun(_Run):
    """The archive's decisions and test times, its outcomes sampled after ``cut``: what the replay's model of a
    test's outcome makes of a future the archive knows."""

    cut: float = 0.0
    sources: dict[str, TreeBranch] = field(default_factory=dict)

    def play(self) -> None:
        for source in _steered(self.tree):
            self.sources[source.name] = source
            self.order += 1
            heapq.heappush(self.events, (source.opened_s, self.order, f"{_OPENS}{source.name}"))
        while self.events:
            at, _, key = heapq.heappop(self.events)
            if key.startswith(_OPENS):
                self.open(self.sources[key.removeprefix(_OPENS)], at)
                continue
            branch = self.running[key]
            self.gpu_seconds += branch.gpu_per_test
            branch.judged += 1
            if branch.next_test.gain is not None:
                branch.gain, branch.position = branch.next_test.gain, 0
                branch.commits += 1
                self.best = max(self.best, branch.next_test.gain)
            else:
                branch.position += 1
            self.schedule(branch)

    def open(self, source: TreeBranch, at: float) -> None:
        """Open ``source`` when it opened; after the cut, from the replay's integration gain rather than the
        archive's."""

        base = source.fork_gain_pct + (self.best - self.archive_gain_at(at) if at > self.cut else 0.0)
        branch = _Branch(source.name, at, iter(()), _gpu_per_test(source), math.inf, gain=base)
        branch.tests = self.outcomes(branch, source, base)
        self.start(branch)

    def outcomes(self, branch: _Branch, source: TreeBranch, base: float) -> Iterator[_Test]:
        draws, gains = self.draws(branch.key), self.pace.gains
        for at, gain in _recorded(source, offset=0.0, base=base, until=_last_test(source)):
            committed, picked = draws.random(), draws.random()
            if at <= self.cut:
                yield _Test(at, gain)
                continue
            if branch.commits:
                odds = self.pace.hazard[min(branch.position + 1, MAX_POSITION) - 1]
            else:
                odds = self.pace.new_branch[min(branch.judged + 1, MAX_POSITION) - 1]
            sampled = gains[min(int(picked * len(gains)), len(gains) - 1)] if gains else 0.0
            yield _Test(at, branch.gain + sampled if committed < odds and gains else None)


@dataclass(frozen=True)
class CutCheck:
    """One session cut at ``cut_minutes``: its real best gain, the replays' median, and the share of replays below
    the real one, ties counting half (its probability integral transform)."""

    session_id: str
    cut_minutes: float
    real_gain_pct: float
    median_gain_pct: float
    rank: float


@dataclass(frozen=True)
class ReplayCheck:
    """The replay's checks: whether following the archive reproduced each session, where each real outcome fell in
    the replays cut at a random time, and whether the playbooks' differences from H0 keep their sign whether
    ``h(k)`` holds or falls beyond the data."""

    reproduced: tuple[str, ...]
    not_reproduced: tuple[str, ...]
    cuts: tuple[CutCheck, ...]
    agreeing: tuple[str, ...]
    disagreeing: tuple[str, ...]

    @property
    def mean_rank(self) -> float | None:
        return statistics.fmean(item.rank for item in self.cuts) if self.cuts else None

    def render(self) -> str:
        rows = [
            (
                item.session_id,
                f"{item.cut_minutes:.0f}",
                number(item.real_gain_pct),
                number(item.median_gain_pct),
                number(item.rank),
            )
            for item in self.cuts
        ]
        lines = [
            f"Following the archive reproduces {len(self.reproduced)} of "
            f"{len(self.reproduced) + len(self.not_reproduced)} sessions"
            + (f"; not {', '.join(self.not_reproduced)}." if self.not_reproduced else "."),
            "",
            f"Cut at a random time, the real best gain's rank among the replays averages {number(self.mean_rank)} "
            "(0.5 when the replay is neither optimistic nor pessimistic):",
            "",
            *table(("session", "cut at (min)", "real best %", "replays' median %", "rank"), rows),
            "",
            f"The sign of each playbook's difference from H0 holds both ways beyond the data for "
            f"{len(self.agreeing)} of {len(self.agreeing) + len(self.disagreeing)}"
            + (f"; not for {', '.join(self.disagreeing)}." if self.disagreeing else "."),
        ]
        return "\n".join(lines)


def _reproduced(tree: SearchTree) -> bool:
    steered = _steered(tree)
    best = max((gain for branch in steered for _, gain in _own(branch)), default=0.0)
    gpu = sum(branch.gpu_seconds for branch in steered if judged_tests(branch)) / _SECONDS_PER_HOUR
    kept = simulate(tree, None, pace_for(tree, (tree,)))
    return math.isclose(kept.best_gain_pct, best) and math.isclose(kept.gpu_hours, gpu, rel_tol=1e-9, abs_tol=1e-9)


def cut_check(tree: SearchTree, pace: Pace, cut: float, *, runs: int = REPLAY_RUNS, seed: str = "") -> CutCheck:
    """Where ``tree``'s real best gain falls among ``runs`` replays that keep its decisions and test times and sample
    every outcome after ``cut``."""

    real = simulate(tree, None, pace).best_gain_pct
    replays = []
    for run in range(runs):
        replay = _CutRun(tree, pace, None, run, seed, cut=cut)
        replay.play()
        replays.append(replay.best)
    below = sum(1 for gain in replays if gain < real) + 0.5 * sum(1 for gain in replays if gain == real)
    return CutCheck(tree.session_id, cut / _SECONDS_PER_MINUTE, real, statistics.median(replays), below / runs)


def check(
    trees: Sequence[SearchTree], *, runs: int = REPLAY_RUNS, seed: str = "", draws: int = BOOTSTRAP_DRAWS
) -> ReplayCheck:
    """The replay's checks over ``trees``: each cut at a time drawn between a fifth and four fifths of its span."""

    reproduced = [tree.session_id for tree in trees if _reproduced(tree)]
    cuts = []
    for tree in trees:
        steered = _steered(tree)
        if not steered:
            continue
        start, end = steered[0].opened_s, tree.budget_seconds
        cut = start + (end - start) * (0.2 + 0.6 * _seeded(seed, tree.session_id, "cut").random())
        cuts.append(cut_check(tree, pace_for(tree, trees), cut, runs=runs, seed=seed))
    held, dropped = (
        {
            row.label: row
            for row in compare(trees, variants(), drawn=Draws(runs=runs, beyond=beyond, seed=seed, draws=draws)).rows
        }
        for beyond in (OPTIMISTIC, PESSIMISTIC)
    )
    signs = {
        label: (held[label].gain_difference > 0) == (dropped[label].gain_difference > 0)
        for label in held
        if label not in (ARCHIVE, H0.label)
    }
    return ReplayCheck(
        reproduced=tuple(reproduced),
        not_reproduced=tuple(tree.session_id for tree in trees if tree.session_id not in reproduced),
        cuts=tuple(cuts),
        agreeing=tuple(label for label, same in signs.items() if same),
        disagreeing=tuple(label for label, same in signs.items() if not same),
    )


def simulate(
    tree: SearchTree, playbook: Playbook | None, pace: Pace, *, run: int = 0, drawn: Draws = Draws()
) -> RunOutcome:
    """One replay of ``tree`` under ``playbook``, ``None`` to follow the archive, with the draws of ``run`` from
    ``drawn``'s seed; with its ``decide``, the judge it asks every round which branches to stop."""

    replay: _Run
    if playbook is None:
        replay = _ArchiveRun(tree, pace, playbook, run, drawn.seed)
    elif drawn.decide is None:
        replay = _PlaybookRun(tree, pace, playbook, run, drawn.seed)
    else:
        replay = _JudgedRun(tree, pace, playbook, run, drawn.seed, decide=drawn.decide)
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


@dataclass(frozen=True)
class _SessionMeans:
    """One playbook's runs on one session: the means of their best gain, GPU hours and stops, and of their best gain
    less the baseline's, run by run."""

    best_gain_pct: float
    gpu_hours: float
    stops: float
    gain_difference: float

    @classmethod
    def of(cls, results: Sequence[RunOutcome], base: Sequence[RunOutcome]) -> _SessionMeans:
        return cls(
            statistics.fmean(item.best_gain_pct for item in results),
            statistics.fmean(item.gpu_hours for item in results),
            statistics.fmean(item.stops for item in results),
            statistics.fmean(item.best_gain_pct - other.best_gain_pct for item, other in zip(results, base)),
        )


def _session_means(
    tree: SearchTree, trees: Sequence[SearchTree], playbooks: Mapping[str, Playbook], against: str, drawn: Draws
) -> dict[str, _SessionMeans]:
    """Each of ``playbooks`` and the archive on ``tree``, paced by ``trees``, read against ``against``."""

    pace = pace_for(tree, trees, drawn.beyond)
    played = {ARCHIVE: [simulate(tree, None, pace)] * drawn.runs}
    played.update(
        (label, [simulate(tree, playbook, pace, run=run, drawn=drawn) for run in range(drawn.runs)])
        for label, playbook in playbooks.items()
    )
    return {label: _SessionMeans.of(results, played[against]) for label, results in played.items()}


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _row(label: str, sessions: Sequence[_SessionMeans], drawn: Draws) -> PolicyRow:
    differences = [item.gain_difference for item in sessions]
    low, high = _interval(differences, drawn.draws, f"{drawn.seed}:{label}")
    return PolicyRow(
        label,
        _mean([item.best_gain_pct for item in sessions]),
        _mean(differences),
        low,
        high,
        _mean([item.gpu_hours for item in sessions]),
        _mean([item.stops for item in sessions]),
    )


def compare(
    trees: Sequence[SearchTree],
    playbooks: Mapping[str, Playbook],
    *,
    baseline: str = H0.label,
    drawn: Draws = Draws(),
) -> SchedulingComparison:
    """Each of ``playbooks`` replayed on every tree as ``drawn`` says, beside the archive, and read against
    ``baseline``, one of them, or the archive when it is none of them."""

    against = baseline if baseline in playbooks else ARCHIVE
    sessions: dict[str, list[_SessionMeans]] = {ARCHIVE: [], **{label: [] for label in playbooks}}
    for tree in trees:
        for label, means in _session_means(tree, trees, playbooks, against, drawn).items():
            sessions[label].append(means)
    rows = tuple(_row(label, means, drawn) for label, means in sessions.items())
    return SchedulingComparison(rows, tuple(tree.session_id for tree in trees), drawn.runs, drawn.beyond, against)


__all__ = [
    "ARCHIVE",
    "BOOTSTRAP_DRAWS",
    "CAPS",
    "DEFAULT_FIRST_COMMIT_MINUTES",
    "DEFAULT_INTERVAL_SECONDS",
    "MIN_INTERVAL_SECONDS",
    "MIN_GAINS",
    "OLD_RULE",
    "OLD_RULE_STALL",
    "ON_PROBATION",
    "OPTIMISTIC",
    "PESSIMISTIC",
    "PROBATION_SHARES",
    "REPLAYED_NUMBERS",
    "REPLAY_RUNS",
    "ROUND_SECONDS",
    "STOP_REPLY",
    "CutCheck",
    "Draws",
    "Pace",
    "PolicyRow",
    "ReplayCheck",
    "RunOutcome",
    "SchedulingComparison",
    "archive_cap",
    "check",
    "compare",
    "cut_check",
    "pace_for",
    "simulate",
    "stops_in",
    "variants",
]
