# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How well the frontier scores foretold the gains that followed, on archived sessions: their calibration (MB2).

Each archived session is replayed round by round, every ``round_seconds`` of its clock. At each round every specialist
branch open then is scored with the one formula (:func:`frontier.score_at`) on what it showed by then
(:meth:`TreeBranch.as_of`), against the other archived sessions as the prior (a session never counts towards its own),
and set beside the gain it went on to make: over the next ``horizon_seconds``, or by default until it stopped or the
budget ran out. Two numbers say how well a policy ranks the branches:

* ``rank_correlation``: over the rounds with two or more branches whose scores and gains both differ, the mean Spearman
  correlation of the scores with the gains that followed;
* ``top_hit_rate``: over the rounds with two or more branches whose gains differ, how often a branch scored highest
  went on to gain the most.

A branch the scheduler stopped gains nothing after it stopped, so the scores the scheduler went by favour themselves a
little. The archive does not date claims, so a kernel direction counts as under observation only while one of its tests
was requested within the observation period.

:func:`calibrate` sets the current policy, sized to each session's budget, beside :data:`VARIATIONS` of one parameter
each, beside other scores given as :class:`Baseline` rows (such as the island bandit of :mod:`rsi.bandit`), and beside
the scores the sessions logged when their logs are given. It recommends a variation, or a baseline that may be advice,
only when it beats the current policy's rank correlation by :data:`MIN_IMPROVEMENT` over at least
:data:`MIN_SESSIONS` sessions and :data:`MIN_ROUNDS` rounds whose scores and later gains both differ; otherwise it
says why not. The advice is for the scheduler's owner; nothing here changes a parameter.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import FrontierScore
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.notebook.rsi.frontier import (
    NEW_BRANCH_FLOOR_PCT,
    FrontierPolicy,
    Readings,
    ScoringContext,
    headroom_pct,
    score_at,
)
from hyperloom_multibranch.notebook.rsi.report import number, table

#: The judge's cadence, about: how often a replayed session's branches are scored.
ROUND_SECONDS = 600.0
MIN_SESSIONS = 3
MIN_ROUNDS = 20
#: How much higher a variation's rank correlation must be than the current policy's to be recommended.
MIN_IMPROVEMENT = 0.05
CURRENT = "current"
LOGGED = "as logged"
_SAME = 1e-9
_PERCENT = 100.0
_HALF, _DOUBLE = 0.5, 2.0


@dataclass(frozen=True)
class Variation:
    """One change to a session's policy: its windows scaled, or the new-branch floor or the similarity bar replaced."""

    label: str
    window_scale: float = 1.0
    observation_scale: float = 1.0
    new_branch_floor_pct: float | None = None
    min_similarity: float | None = None

    def apply(self, policy: FrontierPolicy) -> FrontierPolicy:
        floor, bar = self.new_branch_floor_pct, self.min_similarity
        return replace(
            policy,
            window_seconds=policy.window_seconds * self.window_scale,
            observation_seconds=policy.observation_seconds * self.observation_scale,
            new_branch_floor_pct=policy.new_branch_floor_pct if floor is None else floor,
            min_similarity=policy.min_similarity if bar is None else bar,
        )


#: The current policy first, then one parameter changed at a time.
VARIATIONS = (
    Variation(CURRENT),
    Variation("window x0.5", window_scale=_HALF),
    Variation("window x2", window_scale=_DOUBLE),
    Variation("observation x0.5", observation_scale=_HALF),
    Variation("observation x2", observation_scale=_DOUBLE),
    Variation("no new-branch floor", new_branch_floor_pct=0.0),
    Variation("new-branch floor x2", new_branch_floor_pct=_DOUBLE * NEW_BRANCH_FLOOR_PCT),
    Variation("similarity bar 0.3", min_similarity=0.3),
    Variation("similarity bar 0.7", min_similarity=0.7),
)


#: A score of ``branch`` of ``tree`` at ``t_s`` on the session clock, against the other archived sessions.
Scorer = Callable[[SearchTree, TreeBranch, float, tuple[SearchTree, ...]], float]


@dataclass(frozen=True)
class Baseline:
    """Another score set beside the frontier score's; ``advisable`` when it could replace it, and so be advice."""

    label: str
    score: Scorer
    advisable: bool = False


@dataclass(frozen=True)
class CalibrationSettings:
    """How the sessions are replayed (every ``round_seconds``, gains over ``horizon_seconds`` or until each branch
    stopped) and the evidence advice needs (``min_sessions`` sessions and ``min_rounds`` rounds whose scores and later
    gains both differ)."""

    round_seconds: float = ROUND_SECONDS
    horizon_seconds: float | None = None
    min_sessions: int = MIN_SESSIONS
    min_rounds: int = MIN_ROUNDS


@dataclass(frozen=True)
class PolicyCalibration:
    """How one policy ranked the branches: over how many rounds of how many sessions, and how well.

    ``rounds`` had two or more branches open; ``compared_rounds`` of them had scores and later gains that both differ,
    the only rounds a rank correlation is drawn from.
    """

    label: str
    rank_correlation: float | None
    top_hit_rate: float | None
    rounds: int
    compared_rounds: int
    sessions: int


@dataclass(frozen=True)
class CalibrationReport:
    """Every policy's calibration, the current one first; ``recommended`` only when the evidence carries it."""

    rows: tuple[PolicyCalibration, ...]
    horizon_seconds: float | None
    recommended: PolicyCalibration | None
    note: str

    def render(self) -> str:
        """The report as markdown: a table of the policies, then the advice."""

        horizon = "until each branch stopped" if self.horizon_seconds is None else f"{self.horizon_seconds:.0f} s"
        rows = [
            (
                row.label,
                number(row.rank_correlation),
                number(row.top_hit_rate),
                row.rounds,
                row.compared_rounds,
                row.sessions,
            )
            for row in self.rows
        ]
        lines = table(("policy", "rank correlation", "top hit rate", "rounds", "compared", "sessions"), rows)
        return "\n".join(
            [f"Frontier calibration, against the gains that followed {horizon}:", "", *lines, "", self.note]
        )


@dataclass(frozen=True)
class _Round:
    correlation: float | None
    top_hit: bool | None


def session_policy(tree: SearchTree, variation: Variation = VARIATIONS[0]) -> FrontierPolicy:
    """The policy the scoring would have used in ``tree``'s session, sized to its budget, changed by ``variation``."""

    base = FrontierPolicy.for_budget("", tree.budget_seconds) if tree.budget_seconds > 0 else FrontierPolicy("")
    return variation.apply(base)


def score_archived(
    tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...], policy: FrontierPolicy
) -> FrontierScore:
    """``branch`` of the archived ``tree`` scored at ``t_s`` on what the tree shows by then, as the live scoring does."""

    then = branch.as_of(t_s)
    since = t_s - policy.observation_seconds
    ceiling = None if tree.roofline_ceiling_pct is None else 1.0 + tree.roofline_ceiling_pct / _PERCENT
    return score_at(
        then,
        t_s,
        ScoringContext(tree.task, history, policy),
        Readings(
            observing=any(since <= when <= t_s for when in then.kernel_test_times),
            headroom_pct=headroom_pct(ceiling, 1.0 + then.gain_at(t_s) / _PERCENT),
        ),
    )


def gain_after(tree: SearchTree, branch: TreeBranch, t_s: float, horizon_seconds: float | None = None) -> float:
    """The gain ``branch`` made after ``t_s``: until it stopped or the budget ran out, or within ``horizon_seconds``."""

    end = branch.closed_s if branch.closed_s is not None else tree.budget_seconds
    if horizon_seconds is not None:
        end = min(end, t_s + horizon_seconds)
    return branch.gain_at(end) - branch.gain_at(t_s)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] - values[order[start]] <= _SAME:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / _DOUBLE
        start = end + 1
    return ranks


def rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """The Spearman correlation of ``xs`` with ``ys``, ties ranked alike; ``None`` while either has no spread."""

    try:
        return statistics.correlation(_ranks(xs), _ranks(ys))
    except statistics.StatisticsError:
        return None


def _judged(scores: Sequence[float], gains: Sequence[float]) -> _Round:
    if max(gains) - min(gains) <= _SAME:
        return _Round(rank_correlation(scores, gains), None)
    top, best = max(scores), max(gains)
    hit = any(score >= top - _SAME and gain >= best - _SAME for score, gain in zip(scores, gains))
    return _Round(rank_correlation(scores, gains), hit)


def _open_at(tree: SearchTree, t_s: float) -> list[TreeBranch]:
    return [
        branch
        for branch in tree.branches
        if branch.name != INTEGRATION_BRANCH
        and branch.opened_s <= t_s
        and (branch.closed_s is None or branch.closed_s > t_s)
    ]


@dataclass(frozen=True)
class _PolicyScore:
    variation: Variation

    def __call__(self, tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...]) -> float:
        return score_archived(tree, branch, t_s, history, session_policy(tree, self.variation)).expected_gain_pct


def _session_rounds(
    tree: SearchTree,
    history: tuple[SearchTree, ...],
    scorer: Scorer,
    round_seconds: float,
    horizon_seconds: float | None,
) -> list[_Round]:
    rounds = []
    for step in range(1, math.ceil(tree.budget_seconds / round_seconds)):
        t_s = step * round_seconds
        branches = _open_at(tree, t_s)
        if len(branches) < 2:
            continue
        scores = [scorer(tree, branch, t_s, history) for branch in branches]
        rounds.append(_judged(scores, [gain_after(tree, branch, t_s, horizon_seconds) for branch in branches]))
    return rounds


def _logged_round(tree: SearchTree, line: Mapping[str, Any], horizon_seconds: float | None) -> _Round | None:
    branches = {branch.name: branch for branch in tree.branches}
    t_s = float(line["t_s"])
    scored = [
        (float(entry["frontier"]["expected_gain_pct"]), branches[entry["branch"]])
        for entry in line["branches"]
        if entry["branch"] in branches and entry.get("frontier")
    ]
    if len(scored) < 2:
        return None
    gains = [gain_after(tree, branch, t_s, horizon_seconds) for _, branch in scored]
    return _judged([score for score, _ in scored], gains)


def _summary(label: str, rounds: Mapping[str, Sequence[_Round]]) -> PolicyCalibration:
    every = [item for found in rounds.values() for item in found]
    correlations = [item.correlation for item in every if item.correlation is not None]
    hits = [item.top_hit for item in every if item.top_hit is not None]
    return PolicyCalibration(
        label,
        statistics.fmean(correlations) if correlations else None,
        sum(hits) / len(hits) if hits else None,
        len(every),
        len(correlations),
        sum(1 for found in rounds.values() if found),
    )


def _advice(
    rows: Sequence[PolicyCalibration], eligible: frozenset[str], min_sessions: int, min_rounds: int
) -> tuple[PolicyCalibration | None, str]:
    current = rows[0]
    if current.sessions < min_sessions or current.compared_rounds < min_rounds:
        return None, (
            f"Too little evidence for advice: {current.sessions} sessions and {current.compared_rounds} rounds whose "
            f"scores and later gains both differ, of {current.rounds} with two or more branches open; "
            f"{min_sessions} and {min_rounds} needed."
        )
    if current.rank_correlation is None:
        return None, "No round had branches whose scores and later gains both differed; nothing to compare."
    bar = current.rank_correlation + MIN_IMPROVEMENT
    better = [
        row
        for row in rows[1:]
        if row.label in eligible and row.rank_correlation is not None and row.rank_correlation >= bar
    ]
    if not better:
        return None, f"Keep the current policy: no variation ranks the branches better by {MIN_IMPROVEMENT} or more."
    best = max(better, key=lambda row: row.rank_correlation if row.rank_correlation is not None else -math.inf)
    return best, (
        f"Consider '{best.label}': rank correlation {number(best.rank_correlation)} against "
        f"{number(current.rank_correlation)} for the current policy. Advice only; the scheduler's owner decides."
    )


def calibrate(
    trees: Sequence[SearchTree],
    *,
    variations: Sequence[Variation] = VARIATIONS,
    logs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    settings: CalibrationSettings = CalibrationSettings(),
    baselines: Sequence[Baseline] = (),
) -> CalibrationReport:
    """Every variation's calibration over ``trees``, the first variation being the current policy, then each of
    ``baselines``.

    ``logs`` maps a session id to its scores log (``frontier.jsonl``, one decoded line per round), whose scores are set
    beside the gains its archived tree went on to make.
    """

    horizon_seconds = settings.horizon_seconds
    scorers: list[tuple[str, Scorer]] = [(variation.label, _PolicyScore(variation)) for variation in variations]
    scorers += [(baseline.label, baseline.score) for baseline in baselines]
    rows = []
    for label, scorer in scorers:
        rounds = {
            tree.session_id: _session_rounds(
                tree,
                tuple(other for other in trees if other.session_id != tree.session_id),
                scorer,
                settings.round_seconds,
                horizon_seconds,
            )
            for tree in trees
        }
        rows.append(_summary(label, rounds))
    if logs:
        by_id = {tree.session_id: tree for tree in trees}
        logged = {
            session_id: [
                found
                for line in lines
                if (found := _logged_round(by_id[session_id], line, horizon_seconds)) is not None
            ]
            for session_id, lines in logs.items()
            if session_id in by_id
        }
        rows.append(_summary(LOGGED, logged))
    eligible = frozenset(
        [variation.label for variation in variations[1:]]
        + [baseline.label for baseline in baselines if baseline.advisable]
    )
    recommended, note = _advice(rows, eligible, settings.min_sessions, settings.min_rounds)
    return CalibrationReport(tuple(rows), horizon_seconds, recommended, note)


__all__ = [
    "CURRENT",
    "LOGGED",
    "MIN_IMPROVEMENT",
    "MIN_ROUNDS",
    "MIN_SESSIONS",
    "ROUND_SECONDS",
    "VARIATIONS",
    "Baseline",
    "CalibrationReport",
    "CalibrationSettings",
    "PolicyCalibration",
    "Scorer",
    "Variation",
    "calibrate",
    "gain_after",
    "rank_correlation",
    "score_archived",
    "session_policy",
]
