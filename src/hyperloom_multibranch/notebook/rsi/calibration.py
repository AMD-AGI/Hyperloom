# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How well the frontier scores foretold the gains that followed, on archived sessions: their calibration (MB2).

Each archived session is replayed round by round, every ``round_seconds`` of its clock. At each round every specialist
branch open then is scored on what it showed by then (:meth:`TreeBranch.as_of`, and :meth:`SearchTree.as_of` for a
scorer that sets a branch beside its siblings), against the other archived sessions as the prior (a session never
counts towards its own), and set beside the gain it went on to make: over the next ``horizon_seconds``, or by default
until it stopped or the budget ran out. Two numbers say how well a policy ranks the branches:

* ``rank_correlation``: over the rounds with two or more branches whose scores and gains both differ, the mean Spearman
  correlation of the scores with the gains that followed;
* ``top_hit_rate``: over the same rounds, how often a branch scored highest went on to gain the most.

A branch the scheduler stopped gains nothing after it stopped, so the scores the scheduler went by favour themselves a
little. The archive does not date claims, so a kernel direction counts as under observation only while one of its tests
was requested within the observation period.

:func:`calibrate` sets the current policy, ``frontier-v1`` sized to each session's budget, beside :data:`VARIATIONS` of
one parameter each, beside other scores given as :class:`Baseline` rows (another frontier scorer, by
:func:`scorer_baseline`, or the island bandit's rows of :mod:`rsi.bandit`), and beside the scores the sessions logged
when their logs are given. A baseline with ``since`` is judged only on the sessions that started after it: one written
after the archive could have been fitted to it. Each candidate, a variation or a baseline that may be advice, is
compared with the current policy on the same rounds (:class:`Comparison`), and recommended only when it beats the
current policy there by :data:`MIN_IMPROVEMENT` over at least :data:`MIN_SESSIONS` sessions and :data:`MIN_ROUNDS`
rounds; otherwise the report says why not. :func:`frontier_scorer_for` is the one place a recommendation changes how a
session scores: a session starting on a task scores with the variation, one step from the defaults, or the scorer that
the calibration over its similar sessions recommends.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import FrontierScore
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, similar_trees
from hyperloom_multibranch.notebook.rsi.frontier import (
    FRONTIER_V1,
    NEW_BRANCH_FLOOR_PCT,
    FrontierPolicy,
    FrontierScorer,
    Readings,
    ScoringContext,
    headroom_pct,
    paced,
    score_at,
    score_with,
)
from hyperloom_multibranch.notebook.rsi.report import number, table
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions

log = logging.getLogger(__name__)

#: The judge's cadence, about: how often a replayed session's branches are scored.
ROUND_SECONDS = 600.0
MIN_SESSIONS = 3
MIN_ROUNDS = 20
#: How much higher a candidate's rank correlation must be than the current policy's, on the same rounds, to be
#: recommended.
MIN_IMPROVEMENT = 0.05
CURRENT = "current"
LOGGED = "as logged"
#: The label of a session that scores with the defaults.
DEFAULT_POLICY = "default"
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


#: A score of ``branch`` of ``tree`` at ``t_s`` on the session clock, against the other archived sessions; the replay
#: passes the tree as it stood at ``t_s``.
Scorer = Callable[[SearchTree, TreeBranch, float, tuple[SearchTree, ...]], float]


@dataclass(frozen=True)
class Baseline:
    """Another score set beside the frontier score's; ``advisable`` when it could replace it, and so be advice;
    ``since``, in seconds since the epoch, when only the sessions that started after it are evidence for it."""

    label: str
    score: Scorer
    advisable: bool = False
    since: float | None = None


@dataclass(frozen=True)
class CalibrationSettings:
    """How the sessions are replayed (every ``round_seconds``, gains over ``horizon_seconds`` or until each branch
    stopped) and the evidence advice needs (``min_sessions`` sessions and ``min_rounds`` rounds that compare a
    candidate with the current policy)."""

    round_seconds: float = ROUND_SECONDS
    horizon_seconds: float | None = None
    min_sessions: int = MIN_SESSIONS
    min_rounds: int = MIN_ROUNDS


@dataclass(frozen=True)
class PolicyCalibration:
    """How one policy ranked the branches: over how many rounds of how many sessions, and how well.

    ``rounds`` had two or more branches open; ``compared_rounds`` of them had scores and later gains that both differ,
    the only rounds a rank correlation and a top hit are drawn from. ``failure`` says why a baseline has no numbers.
    """

    label: str
    rank_correlation: float | None
    top_hit_rate: float | None
    rounds: int
    compared_rounds: int
    sessions: int
    failure: str = ""


@dataclass(frozen=True)
class Comparison:
    """A candidate beside the current policy on the same rounds: those whose later gains differ and where the scores of
    either differ, a round where one's scores all tie counting 0 for it; ``sessions`` they came from."""

    label: str
    rounds: int
    sessions: int
    rank_correlation: float | None
    current_correlation: float | None


@dataclass(frozen=True)
class CalibrationReport:
    """Every policy's calibration, the current one first, and every candidate's comparison with it; ``recommended``
    only when the evidence carries it."""

    rows: tuple[PolicyCalibration, ...]
    horizon_seconds: float | None
    recommended: PolicyCalibration | None
    note: str
    comparisons: tuple[Comparison, ...] = ()

    def comparison(self, label: str) -> Comparison | None:
        return next((item for item in self.comparisons if item.label == label), None)

    def render(self) -> str:
        """The report as markdown: a table of the policies, one of the candidates beside the current policy, then the
        advice."""

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
        lines = [
            f"Frontier calibration, against the gains that followed {horizon}:",
            "",
            *table(("policy", "rank correlation", "top hit rate", "rounds", "compared", "sessions"), rows),
        ]
        if self.comparisons:
            paired = [
                (
                    item.label,
                    number(item.rank_correlation),
                    number(item.current_correlation),
                    item.rounds,
                    item.sessions,
                )
                for item in self.comparisons
            ]
            heading = ("candidate", "rank correlation", "current policy", "rounds", "sessions")
            lines += ["", "Each candidate beside the current policy, on the same rounds:", "", *table(heading, paired)]
        failed = [f"'{row.label}' failed: {row.failure}" for row in self.rows if row.failure]
        return "\n".join([*lines, *([""] + failed if failed else []), "", self.note])


@dataclass(frozen=True)
class _Round:
    session_id: str
    t_s: float
    correlation: float | None
    top_hit: bool | None
    scores_differ: bool
    gains_differ: bool


def session_policy(tree: SearchTree, variation: Variation = VARIATIONS[0]) -> FrontierPolicy:
    """The policy the scoring would have used in ``tree``'s session, sized to its budget, changed by ``variation`` and
    paced by its longest root test, in the order the live scoring takes them."""

    base = FrontierPolicy.for_budget("", tree.budget_seconds) if tree.budget_seconds > 0 else FrontierPolicy("")
    return paced(variation.apply(base), tree.longest_root_test_seconds)


@dataclass(frozen=True)
class _Point:
    branch: TreeBranch
    context: ScoringContext
    readings: Readings


def _point(
    tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...], policy: FrontierPolicy
) -> _Point:
    """What the live scoring would have read about ``branch`` at ``t_s``; ``tree`` stands as it does, the replay passing
    it as of ``t_s``."""

    then = branch.as_of(t_s)
    since = t_s - policy.observation_seconds
    ceiling = None if tree.roofline_ceiling_pct is None else 1.0 + tree.roofline_ceiling_pct / _PERCENT
    return _Point(
        then,
        ScoringContext(tree.task, history, policy, tree),
        Readings(
            observing=any(since <= when <= t_s for when in then.kernel_test_times),
            headroom_pct=headroom_pct(ceiling, 1.0 + then.gain_at(t_s) / _PERCENT),
        ),
    )


def score_archived(
    tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...], policy: FrontierPolicy
) -> FrontierScore:
    """``branch`` of the archived ``tree`` scored at ``t_s`` on what the tree shows by then, as the live scoring does."""

    point = _point(tree, branch, t_s, history, policy)
    return score_at(point.branch, t_s, point.context, point.readings)


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


def _spread(values: Sequence[float]) -> bool:
    return max(values) - min(values) > _SAME


def _judged(session_id: str, t_s: float, scores: Sequence[float], gains: Sequence[float]) -> _Round:
    scores_differ, gains_differ = _spread(scores), _spread(gains)
    if not (scores_differ and gains_differ):
        return _Round(session_id, t_s, None, None, scores_differ, gains_differ)
    top, best = max(scores), max(gains)
    hit = any(score >= top - _SAME and gain >= best - _SAME for score, gain in zip(scores, gains))
    return _Round(session_id, t_s, rank_correlation(scores, gains), hit, True, True)


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


@dataclass(frozen=True)
class _ScorerScore:
    scorer: FrontierScorer

    def __call__(self, tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...]) -> float:
        point = _point(tree, branch, t_s, history, session_policy(tree))
        return score_with(self.scorer, point.branch, t_s, point.context, point.readings).expected_gain_pct


def scorer_baseline(scorer: FrontierScorer, since: float | None = None) -> Baseline:
    """``scorer`` as a row that may be advice: each archived session scored with it within the envelope, under the
    session's own policy, as a live session would score under the defaults; ``since`` as for :class:`Baseline`."""

    return Baseline(scorer.label, _ScorerScore(scorer), advisable=True, since=since)


def _session_rounds(
    tree: SearchTree,
    history: tuple[SearchTree, ...],
    scorer: Scorer,
    settings: CalibrationSettings,
) -> list[_Round]:
    rounds = []
    for step in range(1, math.ceil(tree.budget_seconds / settings.round_seconds)):
        t_s = step * settings.round_seconds
        branches = _open_at(tree, t_s)
        if len(branches) < 2:
            continue
        then = tree.as_of(t_s)
        standing = {branch.name: branch for branch in then.branches}
        scores = [scorer(then, standing[branch.name], t_s, history) for branch in branches]
        gains = [gain_after(tree, branch, t_s, settings.horizon_seconds) for branch in branches]
        rounds.append(_judged(tree.session_id, t_s, scores, gains))
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
    return _judged(tree.session_id, t_s, [score for score, _ in scored], gains)


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


def _credit(item: _Round) -> float:
    """What a round counts for a policy beside another: its correlation, or 0 where its scores told no branch apart."""

    return item.correlation if item.correlation is not None else 0.0


def _compared(label: str, candidate: Sequence[_Round], current: Sequence[_Round]) -> Comparison:
    by_round = {(item.session_id, item.t_s): item for item in current}
    paired = [
        (item, by_round[(item.session_id, item.t_s)]) for item in candidate if (item.session_id, item.t_s) in by_round
    ]
    kept = [
        (mine, theirs) for mine, theirs in paired if mine.gains_differ and (mine.scores_differ or theirs.scores_differ)
    ]
    if not kept:
        return Comparison(label, 0, 0, None, None)
    return Comparison(
        label,
        len(kept),
        len({mine.session_id for mine, _ in kept}),
        statistics.fmean(_credit(mine) for mine, _ in kept),
        statistics.fmean(_credit(theirs) for _, theirs in kept),
    )


def _gain_over_current(item: Comparison) -> float:
    if item.rank_correlation is None or item.current_correlation is None:
        return -math.inf
    return item.rank_correlation - item.current_correlation


def _advice(
    current: PolicyCalibration, comparisons: Sequence[Comparison], settings: CalibrationSettings
) -> tuple[str | None, str]:
    """The label of the candidate to recommend, if any, and the note that says why or why not."""

    if not comparisons:
        return None, "No candidate to set beside the current policy."
    enough = [
        item for item in comparisons if item.sessions >= settings.min_sessions and item.rounds >= settings.min_rounds
    ]
    if not enough:
        most = max(comparisons, key=lambda item: (item.rounds, item.sessions))
        return None, (
            f"Too little evidence for advice: at most {most.sessions} sessions and {most.rounds} rounds compare a "
            f"candidate with the current policy, of {current.rounds} with two or more branches open; "
            f"{settings.min_sessions} and {settings.min_rounds} needed."
        )
    best = max(enough, key=_gain_over_current)
    if _gain_over_current(best) < MIN_IMPROVEMENT:
        return None, (
            f"Keep the current policy: no candidate ranks the branches better by {MIN_IMPROVEMENT} or more on the same "
            "rounds."
        )
    return best.label, (
        f"Consider '{best.label}': rank correlation {number(best.rank_correlation)} against "
        f"{number(best.current_correlation)} for the current policy on the same {best.rounds} rounds of "
        f"{best.sessions} sessions. Advice only; the scheduler's owner decides."
    )


def _counts(tree: SearchTree, since: float | None) -> bool:
    """Whether ``tree`` is evidence for a row registered at ``since``: it started later, or the row was registered
    before any archive it is judged on."""

    return since is None or (tree.started_at is not None and tree.started_at > since)


def _rows(
    trees: Sequence[SearchTree], scorer: Scorer, since: float | None, settings: CalibrationSettings
) -> dict[str, list[_Round]]:
    return {
        tree.session_id: _session_rounds(
            tree, tuple(other for other in trees if other.session_id != tree.session_id), scorer, settings
        )
        for tree in trees
        if _counts(tree, since)
    }


def _baseline_rows(
    trees: Sequence[SearchTree], baseline: Baseline, settings: CalibrationSettings
) -> tuple[dict[str, list[_Round]], str]:
    """A baseline's rounds, or none and why: a scorer that fails is left out, not the calibration of the others."""

    try:
        return _rows(trees, baseline.score, baseline.since, settings), ""
    except Exception as exc:
        log.exception("the frontier calibration could not score %s", baseline.label)
        return {}, f"{type(exc).__name__}: {exc}"


def calibrate(
    trees: Sequence[SearchTree],
    *,
    variations: Sequence[Variation] = VARIATIONS,
    logs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    settings: CalibrationSettings = CalibrationSettings(),
    baselines: Sequence[Baseline] = (),
) -> CalibrationReport:
    """Every variation's calibration over ``trees``, the first variation being the current policy, then each of
    ``baselines``, and every candidate beside the current policy.

    ``logs`` maps a session id to its scores log (``frontier.jsonl``, one decoded line per round), whose scores are set
    beside the gains its archived tree went on to make.
    """

    rows: list[PolicyCalibration] = []
    rounds: dict[str, list[_Round]] = {}
    for variation in variations:
        scored = _rows(trees, _PolicyScore(variation), None, settings)
        rows.append(_summary(variation.label, scored))
        rounds[variation.label] = [item for items in scored.values() for item in items]
    for baseline in baselines:
        scored, failure = _baseline_rows(trees, baseline, settings)
        rows.append(replace(_summary(baseline.label, scored), failure=failure))
        rounds[baseline.label] = [item for items in scored.values() for item in items]
    if logs:
        by_id = {tree.session_id: tree for tree in trees}
        logged = {
            session_id: [
                found
                for line in lines
                if (found := _logged_round(by_id[session_id], line, settings.horizon_seconds)) is not None
            ]
            for session_id, lines in logs.items()
            if session_id in by_id
        }
        rows.append(_summary(LOGGED, logged))
    eligible = [variation.label for variation in variations[1:]] + [
        baseline.label for baseline in baselines if baseline.advisable
    ]
    current = rows[0]
    comparisons = tuple(
        _compared(label, rounds[label], rounds[current.label])
        for label in eligible
        if not next(row for row in rows if row.label == label).failure
    )
    chosen, note = _advice(current, comparisons, settings)
    recommended = next((row for row in rows if row.label == chosen), None)
    return CalibrationReport(tuple(rows), settings.horizon_seconds, recommended, note, comparisons)


@dataclass(frozen=True)
class PolicyChoice:
    """How a session scores: with ``scorer`` under ``policy``. ``label`` is :data:`DEFAULT_POLICY`, a variation's or a
    scorer's, which the manifest's ``policy_versions`` records, and ``evidence`` one line on why, for the judge's
    prompt and the frontier log."""

    policy: FrontierPolicy
    label: str
    evidence: str
    scorer: FrontierScorer = FRONTIER_V1


@dataclass(frozen=True)
class Registered:
    """A scorer a session may choose; ``since`` as for :class:`Baseline`: when it was registered, so that only the
    sessions that started after it are evidence for it."""

    scorer: FrontierScorer
    since: float | None = None


def frontier_scorer_for(
    history: Sequence[SearchTree],
    task: TaskConditions,
    defaults: FrontierPolicy,
    scorers: Sequence[Registered] = (),
) -> PolicyChoice:
    """How a session on ``task`` scores: the variation, one step from ``defaults``, or the one of ``scorers`` that the
    calibration over the archived sessions similar to it recommends, or ``frontier-v1`` under ``defaults``, with the
    reason, when the evidence carries none.

    An archive the calibration cannot read gives the defaults too, so a session never fails to start over it.
    """

    try:
        similar = tuple(item.tree for item in similar_trees(history, task))
        report = calibrate(similar, baselines=tuple(scorer_baseline(item.scorer, item.since) for item in scorers))
    except (ArithmeticError, ValueError) as exc:
        return PolicyChoice(defaults, DEFAULT_POLICY, f"{DEFAULT_POLICY}: the calibration failed: {exc}")
    best = report.recommended
    compared = report.comparison(best.label) if best is not None else None
    if compared is None:
        return PolicyChoice(defaults, DEFAULT_POLICY, f"{DEFAULT_POLICY}: {report.note}")
    evidence = (
        f"{compared.label}: rank correlation {number(compared.rank_correlation)} against "
        f"{number(compared.current_correlation)} for the defaults, over {compared.rounds} compared rounds in "
        f"{compared.sessions} similar sessions"
    )
    variation = next((item for item in VARIATIONS[1:] if item.label == compared.label), None)
    if variation is not None:
        return PolicyChoice(variation.apply(defaults), variation.label, evidence)
    chosen = next(item.scorer for item in scorers if item.scorer.label == compared.label)
    return PolicyChoice(defaults, chosen.label, evidence, chosen)


def frontier_policy_for(history: Sequence[SearchTree], task: TaskConditions, defaults: FrontierPolicy) -> PolicyChoice:
    """:func:`frontier_scorer_for` with :data:`VARIATIONS` as the only candidates: a parameter one step from the
    defaults, never another formula."""

    return frontier_scorer_for(history, task, defaults, ())


__all__ = [
    "CURRENT",
    "DEFAULT_POLICY",
    "LOGGED",
    "MIN_IMPROVEMENT",
    "MIN_ROUNDS",
    "MIN_SESSIONS",
    "ROUND_SECONDS",
    "VARIATIONS",
    "Baseline",
    "CalibrationReport",
    "CalibrationSettings",
    "Comparison",
    "PolicyCalibration",
    "PolicyChoice",
    "Registered",
    "Scorer",
    "Variation",
    "calibrate",
    "frontier_policy_for",
    "frontier_scorer_for",
    "gain_after",
    "rank_correlation",
    "score_archived",
    "scorer_baseline",
    "session_policy",
]
