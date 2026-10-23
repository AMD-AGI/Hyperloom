# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Calibrating the frontier scores on archived sessions: how well each policy ranked the gains that followed."""

from __future__ import annotations

import dataclasses
import math

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver
from hyperloom_multibranch.notebook.rsi import calibration
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, session_start, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.calibration import (
    CURRENT,
    DEFAULT_POLICY,
    LOGGED,
    VARIATIONS,
    Baseline,
    Registered,
    Variation,
    calibrate,
    frontier_policy_for,
    frontier_scorer_for,
    gain_after,
    rank_correlation,
    ranked_alike,
    rankings,
    score_archived,
    session_policy,
)
from hyperloom_multibranch.notebook.rsi.frontier import (
    FRONTIER_V1,
    Estimate,
    FrontierPolicy,
    Readings,
    ScoringContext,
    score_progress,
)
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree
from hyperloom_multibranch.notebook.tests.sessions import METRIC

BUDGET = 2 * HOUR
# A commit of +0.2% every ten minutes: under the new-branch floor over any 15-minute window.
STEADY = [(600.0 * step, 0.2 * step) for step in range(1, 13)]
NO_FLOOR = Variation("no new-branch floor", new_branch_floor_pct=0.0)


def _session(n: int) -> SearchTree:
    fresh = branch("fresh", [], opened_s=5400.0)
    unlike = TaskConditions(labels={"model": f"model-{n}"})
    return dataclasses.replace(tree(f"sess-{n}", branch("steady", STEADY), fresh, task=unlike), budget_seconds=BUDGET)


SESSIONS = tuple(_session(n) for n in range(7))


def _tied(n: int) -> SearchTree:
    """Two branches alike until one commits at 90 minutes: before, their scores tie while their later gains differ;
    after, their gains tie."""

    unlike = TaskConditions(labels={"model": f"tied-{n}"})
    branches = (branch("a", [(5400.0, 2.0)]), branch("b", []))
    return dataclasses.replace(tree(f"tied-{n}", *branches, task=unlike), budget_seconds=BUDGET)


def test_an_archived_branch_scores_as_the_live_scoring_scored_it(snapshot: NotebookSnapshot) -> None:
    policy = FrontierPolicy(METRIC)
    live = {row.branch: row.frontier for row in score_progress(snapshot, (), policy)}["s1"]
    searched = tree_from_snapshot(snapshot, METRIC)
    s1 = next(item for item in searched.branches if item.name == "s1")

    archived = score_archived(searched, s1, snapshot.taken_at - session_start(snapshot), (), policy)

    assert live is not None and archived.basis is live.basis
    assert archived.expected_gain_pct == pytest.approx(live.expected_gain_pct)
    assert archived.components == pytest.approx(live.components)


def test_a_variation_that_ranks_the_gains_better_is_recommended() -> None:
    report = calibrate(SESSIONS, variations=(Variation(CURRENT), NO_FLOOR))
    current, unfloored = report.rows

    # Two of each session's three rounds rank the fresh branch, floored at 0.5%, above the steady one.
    assert (current.rounds, current.sessions) == (21, 7)
    assert current.rank_correlation == pytest.approx(-1 / 3) and current.top_hit_rate == pytest.approx(1 / 3)
    assert unfloored.rank_correlation == pytest.approx(1.0) and unfloored.top_hit_rate == pytest.approx(1.0)
    assert report.recommended == unfloored and report.note.startswith("Consider 'no new-branch floor'")


def test_no_variation_better_by_enough_keeps_the_current_policy() -> None:
    report = calibrate(SESSIONS, variations=(dataclasses.replace(NO_FLOOR, label=CURRENT), Variation("floored")))

    assert report.recommended is None and report.note.startswith("Keep the current policy")


def test_rounds_whose_scores_and_gains_do_not_both_differ_are_no_evidence_for_advice() -> None:
    # sess-0 has 3 rounds that compare its two branches; each tied session has 11 rounds that compare none, 8 of them
    # with later gains that differ.
    report = calibrate((SESSIONS[0], *(_tied(n) for n in range(6))), variations=(Variation(CURRENT), NO_FLOOR))
    current = report.rows[0]

    assert (current.rounds, current.compared_rounds, current.sessions) == (69, 3, 7)
    assert report.recommended is None and report.note.startswith("Too little evidence")


def test_too_few_sessions_or_rounds_give_no_advice() -> None:
    report = calibrate(SESSIONS[:2])

    assert report.recommended is None and report.note.startswith("Too little evidence")
    assert [row.label for row in report.rows][:2] == [CURRENT, "window x0.5"]


def test_the_scores_a_session_logged_are_set_beside_the_gains_its_tree_went_on_to_make() -> None:
    line = {
        "t_s": 5400.0,
        "branches": [
            {"branch": "steady", "frontier": {"expected_gain_pct": 0.4}},
            {"branch": "fresh", "frontier": {"expected_gain_pct": 0.5}},
            {"branch": "gone", "frontier": {"expected_gain_pct": 9.0}},
        ],
    }

    report = calibrate(SESSIONS, variations=(Variation(CURRENT),), logs={"sess-0": [line], "other": [line]})
    logged = report.rows[-1]

    assert (logged.label, logged.rank_correlation, logged.rounds, logged.sessions) == (LOGGED, -1.0, 1, 1)
    assert report.recommended is None


def test_the_gain_that_followed_runs_to_the_stop_or_the_horizon() -> None:
    steady = SESSIONS[0].branches[1]

    assert gain_after(SESSIONS[0], steady, 5400.0) == pytest.approx(0.6)
    assert gain_after(SESSIONS[0], steady, 5400.0, 600.0) == pytest.approx(0.2)
    assert session_policy(SESSIONS[0]).window_seconds == 900.0
    assert session_policy(dataclasses.replace(SESSIONS[0], longest_root_test_seconds=600.0)).window_seconds == 2400.0


def _on_the_gpu(n: int, session: SearchTree) -> SearchTree:
    """``session`` on the new session's GPU, with a framework and an architecture no other archived session has: like
    the new session, and too unlike any other archived one to be its prior."""

    labels = {"gpu": "mi355x", "framework": f"framework-{n}", "architecture": f"architecture-{n}"}
    return dataclasses.replace(session, task=TaskConditions(labels=labels))


def test_a_session_scores_with_the_variation_its_similar_sessions_recommend_or_else_the_defaults() -> None:
    archived = tuple(_on_the_gpu(n, session) for n, session in enumerate(SESSIONS))
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)

    chosen = frontier_policy_for(archived, TaskConditions(labels={"gpu": "mi355x"}), defaults)
    alone = frontier_policy_for(SESSIONS, TaskConditions(labels={"model": "model-0"}), defaults)

    variations = {item.label: item for item in VARIATIONS[1:]}
    assert chosen.label in variations and chosen.policy == variations[chosen.label].apply(defaults)
    assert chosen.evidence.startswith(f"{chosen.label}: rank correlation ") and "7 similar sessions" in chosen.evidence
    assert (alone.policy, alone.label) == (defaults, DEFAULT_POLICY)
    assert alone.evidence.startswith(f"{DEFAULT_POLICY}: Too little evidence")


def test_an_archive_the_calibration_cannot_read_leaves_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(*_args: object, **_kwargs: object) -> calibration.CalibrationReport:
        raise ValueError("a tree without a baseline")

    monkeypatch.setattr(calibration, "calibrate", unreadable)
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)

    choice = frontier_policy_for(SESSIONS, TaskConditions(labels={"model": "model-0"}), defaults)

    assert (choice.policy, choice.label) == (defaults, DEFAULT_POLICY)
    assert choice.evidence == f"{DEFAULT_POLICY}: the calibration failed: a tree without a baseline"


def test_a_round_whose_scores_all_tie_gives_no_top_hit() -> None:
    current = calibrate(tuple(_tied(n) for n in range(3)), variations=(Variation(CURRENT),)).rows[0]

    assert (current.compared_rounds, current.top_hit_rate, current.rank_correlation) == (0, None, None)


@dataclasses.dataclass(frozen=True)
class _Names:
    """A baseline that scores the branch named ``best`` above the others."""

    best: str

    def __call__(self, tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...]) -> float:
        return 1.0 if branch.name == self.best else 0.0


def test_a_candidate_beside_the_current_policy_counts_the_rounds_where_only_it_told_the_branches_apart() -> None:
    picks = Baseline("a first", _Names("a"), advisable=True)

    report = calibrate(tuple(_tied(n) for n in range(3)), variations=(Variation(CURRENT),), baselines=(picks,))
    (compared,) = report.comparisons

    # Before a's commit at 90 minutes the current policy ties the two branches, while a went on to gain.
    assert (compared.label, compared.rounds, compared.sessions) == ("a first", 24, 3)
    assert compared.rank_correlation == pytest.approx(1.0) and compared.current_correlation == 0.0
    assert report.recommended is not None and report.recommended.label == "a first"
    assert "| a first | 1.00 | 0.00 | 1.00 to 1.00 | 24 | 3 |" in report.render()


def test_a_baseline_registered_later_is_judged_only_on_the_sessions_that_started_after_it() -> None:
    sessions = (
        *(dataclasses.replace(_tied(n), started_at=1000.0 * n) for n in range(4)),
        dataclasses.replace(_tied(4), started_at=None),
    )
    later = Baseline("a first", _Names("a"), advisable=True, since=1500.0)

    report = calibrate(sessions, variations=(Variation(CURRENT),), baselines=(later,))
    row = next(item for item in report.rows if item.label == "a first")
    (compared,) = report.comparisons

    assert (row.sessions, compared.sessions, compared.rounds) == (2, 2, 16)
    assert report.recommended is None and report.note.startswith("Too little evidence for advice: at most 2 sessions")


def test_a_baseline_that_fails_is_left_out_with_why_and_the_others_stand() -> None:
    def broken(*_: object) -> float:
        raise RuntimeError("no estimate")

    report = calibrate(
        SESSIONS[:2], variations=(Variation(CURRENT), NO_FLOOR), baselines=(Baseline("broken", broken, True),)
    )
    failed = report.rows[-1]

    assert (failed.label, failed.failure, failed.rounds) == ("broken", "RuntimeError: no estimate", 0)
    assert [item.label for item in report.comparisons] == [NO_FLOOR.label]
    assert "'broken' failed: RuntimeError: no estimate" in report.render()


@dataclasses.dataclass(frozen=True)
class _Prefers:
    """A frontier scorer that expects 2 points of the branch named ``best`` and nothing of the others."""

    best: str
    label: str = "prefers-a"

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        return Estimate(2.0 if branch.name == self.best else 0.0)


def test_a_session_scores_with_the_scorer_its_similar_sessions_recommend_once_they_are_evidence_for_it() -> None:
    archived = tuple(_on_the_gpu(n, _tied(n)) for n in range(3))
    task, defaults, scorer = (
        TaskConditions(labels={"gpu": "mi355x"}),
        FrontierPolicy.for_budget(METRIC, BUDGET),
        _Prefers("a"),
    )

    chosen = frontier_scorer_for(archived, task, defaults, (Registered(scorer),))
    unproven = frontier_scorer_for(archived, task, defaults, (Registered(scorer, since=math.inf),))

    assert (chosen.label, chosen.policy, chosen.scorer) == ("prefers-a", defaults, scorer)
    assert chosen.evidence.startswith("prefers-a: rank correlation 1.00 against 0.00 for the defaults")
    assert "3 similar sessions" in chosen.evidence
    assert (unproven.label, unproven.policy, unproven.scorer) == (DEFAULT_POLICY, defaults, FRONTIER_V1)


def test_a_lead_that_varies_from_session_to_session_needs_its_interval_above_0() -> None:
    sessions = tuple(_tied(n) for n in range(4))
    # The baseline picks the branch that went on to gain in two sessions, and the wrong one in the other two.
    picks = Baseline(
        "flip", lambda tree, branch, t_s, _: float((branch.name == "a") == (tree.session_id < "tied-2")), True
    )

    report = calibrate(sessions, variations=(Variation(CURRENT),), baselines=(picks,))
    (compared,) = report.comparisons

    assert (compared.rounds, compared.sessions) == (32, 4)
    assert compared.rank_correlation == pytest.approx(0.0)
    assert compared.difference_low is not None and compared.difference_low < 0 < (compared.difference_high or 0)
    assert report.recommended is None and report.note.startswith("Keep the current policy")


def test_a_champion_is_kept_until_a_challenger_ranks_better_and_frontier_v1_can_take_its_place_back() -> None:
    archived = tuple(_on_the_gpu(n, _tied(n)) for n in range(3))
    task, defaults = TaskConditions(labels={"gpu": "mi355x"}), FrontierPolicy.for_budget(METRIC, BUDGET)
    wrong, right = _Prefers("b", "prefers-b"), _Prefers("a")

    kept = frontier_scorer_for(archived, task, defaults, (), Registered(right))
    displaced = frontier_scorer_for(archived, task, defaults, (Registered(right),), Registered(wrong))
    back = frontier_scorer_for(archived, task, defaults, (), Registered(wrong))

    assert (kept.label, kept.scorer) == ("prefers-a", right)
    assert kept.evidence.startswith("prefers-a: Keep prefers-a: no candidate ranks the branches better")
    assert (displaced.label, displaced.scorer) == ("prefers-a", right)
    assert displaced.evidence.startswith("prefers-a: rank correlation 1.00 against -1.00 for prefers-b")
    assert (back.label, back.scorer) == (DEFAULT_POLICY, FRONTIER_V1)
    assert back.evidence.startswith("frontier-v1: rank correlation 0.00 against -1.00 for prefers-b")


def test_rankings_tell_two_scorers_that_rank_the_branches_alike() -> None:
    sessions = tuple(_tied(n) for n in range(2))

    double = rankings(sessions, _Prefers("a"))
    same, other = rankings(sessions, _Prefers("a", "prefers-a-again")), rankings(sessions, _Prefers("b", "prefers-b"))

    assert len(double) == 2 * 11 and all(len(ranks) == 2 for ranks in double.values())
    assert ranked_alike(double, same) == 1.0 and ranked_alike(double, other) < 0.5
    assert ranked_alike(double, {}) == 0.0


def test_a_branch_an_autonomous_agent_drove_is_in_no_round() -> None:
    agent = branch("agent", [(1800.0, 5.0)], driver=BranchDriver.AUTONOMOUS)
    sessions = tuple(_tied(n) for n in range(2))
    joined = tuple(dataclasses.replace(item, branches=(*item.branches, agent)) for item in sessions)
    alone = dataclasses.replace(_tied(9), branches=(*_tied(9).branches[:2], agent))

    assert rankings(joined, _Prefers("a")) == rankings(sessions, _Prefers("a"))
    assert rankings((alone,), _Prefers("a")) == {}


def test_rank_correlation_ranks_ties_alike_and_needs_a_spread() -> None:
    assert rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert rank_correlation([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.8660254)
    assert rank_correlation([1.0, 1.0], [1.0, 2.0]) is None


def test_the_report_renders_a_table_and_the_advice() -> None:
    text = calibrate(SESSIONS[:2], variations=(Variation(CURRENT),)).render()

    assert text.splitlines()[0] == "Frontier calibration, against the gains that followed until each branch stopped:"
    assert "| policy | rank correlation | top hit rate | rounds | compared | sessions |" in text
    assert "| current | -0.33 | 0.33 | 6 | 6 | 2 |" in text
    assert text.endswith("No candidate to set beside the current policy.")
