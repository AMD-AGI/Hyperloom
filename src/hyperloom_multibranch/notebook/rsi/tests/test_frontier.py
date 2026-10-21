# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frontier score of each running specialist branch: trend, spend, headroom, Amdahl ceiling, prior, observation."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import Purpose, TestKind, Validity, Verdict
from hyperloom_multibranch.contracts.notebook.records import (
    LineageEvent,
    NotebookSnapshot,
    PendingCandidate,
    TestRecord,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, FrontierBasis, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, longest_root_test_seconds
from hyperloom_multibranch.notebook.rsi.frontier import (
    FALLBACK_COMPONENT,
    FALLBACK_LIMIT,
    FRONTIER_LOG_FORMAT,
    FRONTIER_V1_LABEL,
    OBSERVATION_SECONDS,
    SETTLED_TESTS,
    WINDOW_SECONDS,
    Estimate,
    FrontierPolicy,
    FrontierScoring,
    Readings,
    ScorerError,
    ScorerGuard,
    ScoringContext,
    checked_estimate,
    paced,
    remaining_prior,
    score_at,
    score_progress,
    score_with,
)
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, task_conditions, task_similarity
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, judgement, profile, request, result
from hyperloom_multibranch.notebook.tests.sessions import GEMM_AGAIN, METRIC, scripted_session

POLICY = FrontierPolicy(METRIC)
TASK = task_conditions(profile())
PAYS_LATE = tree(
    "late", branch("s1", [(600.0, 2.0), (1800.0, 6.0)], test_times=(0.0, 600.0, 1200.0, 1800.0)), task=TASK
)
PAYS_SOON = tree("soon", branch("s1", [(1800.0, 3.0)], test_times=(0.0, 900.0, 1800.0)), task=TASK)
# The same GPU on another framework: 3 of 5 label weights match.
SAME_GPU = TaskConditions(labels={"gpu": TASK.labels["gpu"], "framework": "sglang"})
# The same GPU and nothing else: 3 of 7.
ONLY_THE_GPU = TaskConditions(
    labels={"gpu": TASK.labels["gpu"], "framework": "sglang", "model": "Qwen/Qwen3-8B", "precision": "bf16"}
)


def _sessions(task: TaskConditions, count: int) -> tuple[SearchTree, ...]:
    gains = [(1800.0, 3.0)]
    return tuple(tree(f"t{n}", branch("s1", gains, test_times=(0.0, 900.0, 1800.0)), task=task) for n in range(count))


def _scores(snapshot: NotebookSnapshot, *history: object, policy: FrontierPolicy = POLICY) -> dict[str, object]:
    return {row.branch: row.frontier for row in score_progress(snapshot, tuple(history), policy)}


def test_a_running_specialist_branch_is_scored_and_the_others_are_not(snapshot: NotebookSnapshot) -> None:
    scores = _scores(snapshot)

    s1 = scores["s1"]
    assert scores["s2"] is None and scores[INTEGRATION_BRANCH] is None
    assert s1 is not None and s1.expected_gain_pct == pytest.approx(4.0)
    assert s1.basis is FrontierBasis.TREND
    assert s1.components["trend_pct"] == pytest.approx(4.0)
    assert (s1.components["tests"], s1.components["gpu_hours"], s1.components["observing"]) == (2.0, 1200 / 3600, 1.0)
    assert s1.components["headroom_pct"] == pytest.approx((1500.0 / 1040.0 - 1.0) * 100.0)
    assert "prior_remaining_pct" not in s1.components


def test_the_history_prior_replaces_the_trend_once_enough_similar_sessions_back_it(snapshot: NotebookSnapshot) -> None:
    backed = _scores(snapshot, PAYS_LATE, PAYS_SOON)["s1"]
    alone = _scores(snapshot, PAYS_LATE)["s1"]

    assert backed is not None and backed.expected_gain_pct == pytest.approx(3.5)
    assert backed.basis is FrontierBasis.PRIOR
    assert backed.components["prior_effective_sessions"] == pytest.approx(2.0)
    assert alone is not None and alone.expected_gain_pct == pytest.approx(4.0)
    assert alone.basis is FrontierBasis.TREND


def test_a_prior_needs_sessions_at_least_half_similar_that_add_up_to_one_and_a_half(
    snapshot: NotebookSnapshot,
) -> None:
    two, three = remaining_prior(_sessions(SAME_GPU, 2), TASK, 2), remaining_prior(_sessions(SAME_GPU, 3), TASK, 2)
    short = _scores(snapshot, *_sessions(SAME_GPU, 2))["s1"]
    backed = _scores(snapshot, *_sessions(SAME_GPU, 3))["s1"]

    assert task_similarity(TASK, SAME_GPU) == pytest.approx(0.6)
    assert task_similarity(TASK, ONLY_THE_GPU) == pytest.approx(3 / 7)
    assert remaining_prior(_sessions(ONLY_THE_GPU, 5), TASK, 2) is None
    assert two is not None and two.effective_sessions == pytest.approx(1.2)
    assert three is not None and three.effective_sessions == pytest.approx(1.8)
    assert short is not None and short.basis is FrontierBasis.TREND
    assert backed is not None and backed.basis is FrontierBasis.PRIOR and backed.expected_gain_pct == pytest.approx(3.0)


async def test_a_new_branch_without_a_commit_keeps_a_floor_until_its_tests_settle(tmp_path: Path) -> None:
    clock = FakeClock()
    root = tmp_path / "notebook"
    observer = await scripted_session(root, clock)
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"), clock=clock)
    toolbox, gate = (LocalNotebook(root, Writer(role, role.value), clock=clock) for role in (Role.TOOLBOX, Role.GATE))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s3", from_commit="c1", reason="fill"))

    async def judge_flat(request_id: str) -> None:
        await toolbox.write_request(request(request_id, intent("s3", "c1", target=request_id)))
        await toolbox.write_result(result(request_id))
        flat = judgement(0.2, (request_id,), verdict=Verdict.FLAT)
        await gate.write_judgement(request_id, flat, Conclusion.RECORDED_ONLY)

    fresh = _scores(await observer.snapshot())["s3"]
    clock.advance(WINDOW_SECONDS)
    waited = _scores(await observer.snapshot())["s3"]
    for n in range(SETTLED_TESTS - 1):
        await judge_flat(f"n{n}")
    unsettled = _scores(await observer.snapshot())["s3"]
    await judge_flat("last")
    settled = _scores(await observer.snapshot())["s3"]

    assert fresh is not None and fresh.expected_gain_pct == POLICY.new_branch_floor_pct
    assert fresh.basis is FrontierBasis.NEW and fresh.components["trend_pct"] == 0.0
    assert waited is not None and waited.basis is FrontierBasis.NEW
    assert unsettled is not None and unsettled.expected_gain_pct == POLICY.new_branch_floor_pct
    assert unsettled.basis is FrontierBasis.NEW
    assert settled is not None and settled.expected_gain_pct == 0.0 and settled.basis is FrontierBasis.TREND
    assert SETTLED_TESTS == 10


async def test_the_floor_ends_with_the_judges_grace_at_the_settled_explore_tests_failed_ones_included(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    root = tmp_path / "notebook"
    observer = await scripted_session(root, clock)
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"), clock=clock)
    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s3", from_commit="c1", reason="fill"))
    clock.advance(WINDOW_SECONDS)

    async def ended(request_id: str, validity: Validity, **kinds: object) -> None:
        await toolbox.write_request(request(request_id, intent("s3", "c1", target=request_id), **kinds))
        await toolbox.write_result(result(request_id, validity=validity))

    for n in range(SETTLED_TESTS - 1):
        await ended(f"f{n}", Validity.CANDIDATE_FAILED)
    await ended("p1", Validity.SUCCEEDED, kind=TestKind.PROFILE, purpose=Purpose.PROFILE)
    await toolbox.write_request(request("w1", intent("s3", "c1", target="w1")))
    await toolbox.write_withdrawal(Withdrawal("w1", "branch stopped"))
    unsettled = _scores(await observer.snapshot())["s3"]
    await ended("last", Validity.MEASUREMENT_FAILED)
    settled = _scores(await observer.snapshot())["s3"]

    assert unsettled is not None and unsettled.basis is FrontierBasis.NEW
    assert settled is not None and settled.basis is FrontierBasis.TREND


def test_an_archived_branch_without_judged_times_keeps_the_floor_for_one_window_only() -> None:
    context = ScoringContext(TASK, (), POLICY)
    late = WINDOW_SECONDS + 1.0

    assert score_at(branch("s9", []), 60.0, context).basis is FrontierBasis.NEW
    assert score_at(branch("s9", []), late, context).basis is FrontierBasis.TREND
    assert score_at(branch("s9", [], judged_times=()), late, context).basis is FrontierBasis.NEW


def test_a_candidate_waiting_for_its_confirm_lifts_the_score_to_its_gain_in_points_of_the_baseline(
    snapshot: NotebookSnapshot,
) -> None:
    waiting = PendingCandidate("r6", "r6c", "c1", METRIC, 1092.0, 5.0)
    rows = tuple(
        dataclasses.replace(row, pending=(waiting,)) if row.branch == "s1" else row for row in snapshot.progress
    )
    pending = dataclasses.replace(snapshot, progress=rows)

    lifted = _scores(pending)["s1"]
    capped = _scores(dataclasses.replace(pending, profile=profile(roofline_ceiling={METRIC: 1050.0})))["s1"]

    assert lifted is not None and lifted.basis is FrontierBasis.PENDING
    assert lifted.expected_gain_pct == pytest.approx(5.2) and lifted.components["pending_pct"] == pytest.approx(5.2)
    assert capped is not None and capped.basis is FrontierBasis.HEADROOM


async def test_the_branchs_test_count_leaves_out_the_tests_still_waiting_for_a_result(tmp_path: Path) -> None:
    clock = FakeClock()
    observer = await scripted_session(tmp_path / "notebook", clock)
    toolbox = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    await toolbox.write_request(request("r6", GEMM_AGAIN))

    scored = _scores(await observer.snapshot())["s1"]

    assert scored is not None and scored.components["tests"] == 2.0


def test_the_window_holds_four_of_the_longest_root_tests_when_that_is_longer(snapshot: NotebookSnapshot) -> None:
    root_test = request("b2", intent(INTEGRATION_BRANCH, "c0"), RecipeDelta(), purpose=Purpose.BASELINE)
    timed = TestRecord(root_test, dataclasses.replace(result("b2"), started_at=100.0, finished_at=700.0))
    with_root = dataclasses.replace(snapshot, records=(*snapshot.records, timed))

    assert (longest_root_test_seconds(snapshot), longest_root_test_seconds(with_root)) == (None, 600.0)
    assert paced(POLICY, 600.0).window_seconds == 4 * 600.0
    assert paced(POLICY, 300.0) == POLICY and paced(POLICY, None) == POLICY


def test_the_prior_counts_only_branches_that_spent_as_many_tests() -> None:
    assert remaining_prior((PAYS_LATE, PAYS_SOON), TASK, 4) == remaining_prior((PAYS_LATE,), TASK, 4)
    assert remaining_prior((PAYS_SOON,), TASK, 4) is None
    from_the_start = remaining_prior((PAYS_SOON,), TASK, 0)
    assert from_the_start is not None and from_the_start.remaining_pct == pytest.approx(3.0)


def test_the_headroom_caps_the_score_and_observation_keeps_a_floor(snapshot: NotebookSnapshot) -> None:
    tight = dataclasses.replace(snapshot, profile=profile(roofline_ceiling={METRIC: 1050.0}))
    capped = _scores(tight)["s1"]
    stalled = _scores(snapshot, policy=dataclasses.replace(POLICY, window_seconds=0.0))["s1"]
    unwatched_policy = dataclasses.replace(
        POLICY, window_seconds=0.0, observation_seconds=0.0, new_branch_floor_pct=0.0
    )
    unwatched = _scores(snapshot, policy=unwatched_policy)["s1"]

    assert capped is not None and capped.expected_gain_pct == pytest.approx((1050.0 / 1040.0 - 1.0) * 100.0)
    assert capped.basis is FrontierBasis.HEADROOM
    assert stalled is not None and stalled.expected_gain_pct == POLICY.observation_floor_pct
    assert stalled.basis is FrontierBasis.OBSERVING
    assert unwatched is not None and unwatched.expected_gain_pct == 0.0
    assert unwatched.basis is FrontierBasis.TREND


async def test_the_largest_op_on_the_profiled_head_caps_the_score_at_its_amdahl_ceiling(tmp_path: Path) -> None:
    clock = FakeClock()
    observer = await scripted_session(tmp_path / "notebook", clock)
    toolbox = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    s1 = next(row for row in (await observer.snapshot()).progress if row.branch == "s1")
    await toolbox.notify_tool_result(
        ToolResultPointer("s1", "job-1", "/a", "trace analysed", s1.start_commit, "", {"gemm": 0.5})
    )
    unprofiled = _scores(await observer.snapshot())["s1"]
    await toolbox.notify_tool_result(
        ToolResultPointer("s1", "job-2", "/a", "trace analysed", s1.head_commit, "", {"gemm": 3.0, "norm": 1.0})
    )

    capped = _scores(await observer.snapshot())["s1"]

    assert unprofiled is not None and unprofiled.basis is FrontierBasis.TREND
    assert "amdahl_ceiling_pct" not in unprofiled.components
    assert capped is not None and capped.basis is FrontierBasis.AMDAHL
    assert capped.expected_gain_pct == pytest.approx(300.0 / 97.0)
    assert capped.components["amdahl_ceiling_pct"] == pytest.approx(300.0 / 97.0)
    assert capped.components["headroom_pct"] > capped.expected_gain_pct


def test_a_shorter_session_looks_back_and_observes_for_less_and_a_longer_one_keeps_the_defaults() -> None:
    two_hours = FrontierPolicy.for_budget(METRIC, 2 * 3600.0)
    eight_hours = FrontierPolicy.for_budget(METRIC, 8 * 3600.0)

    assert (two_hours.window_seconds, two_hours.observation_seconds) == (900.0, 1800.0)
    assert (eight_hours.window_seconds, eight_hours.observation_seconds) == (WINDOW_SECONDS, OBSERVATION_SECONDS)
    assert two_hours.metric == METRIC and two_hours.observation_floor_pct == POLICY.observation_floor_pct
    with pytest.raises(ValueError, match="positive"):
        FrontierPolicy.for_budget(METRIC, 0.0)


def test_without_a_profile_nothing_is_scored(snapshot: NotebookSnapshot) -> None:
    unscored = score_progress(dataclasses.replace(snapshot, profile=None), (), POLICY)

    assert all(row.frontier is None for row in unscored)


async def test_the_scoring_is_the_notebooks_progress_query(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    scoring = FrontierScoring(notebook, POLICY, (PAYS_LATE, PAYS_SOON))

    rows = {row.branch: row for row in await scoring.progress()}

    assert rows["s1"].frontier is not None and rows["s1"].frontier.expected_gain_pct == pytest.approx(3.5)
    assert (await scoring.lineage()) == (await notebook.lineage())


async def test_with_a_log_every_round_that_scored_a_branch_is_appended(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    path = tmp_path / "session" / "frontier.jsonl"
    policy = FrontierPolicy.for_budget(METRIC, 2 * 3600.0)
    scoring = FrontierScoring(notebook, policy, log=path)

    await scoring.progress()
    await scoring.progress()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(lines) == 2 and lines[0]["at"] < lines[1]["at"]
    first = lines[0]
    assert (first["format"], first["metric"], first["window_seconds"]) == (FRONTIER_LOG_FORMAT, METRIC, 900.0)
    assert (first["new_branch_floor_pct"], first["min_similarity"]) == (
        policy.new_branch_floor_pct,
        policy.min_similarity,
    )
    assert first["t_s"] > 0
    assert [row["branch"] for row in first["branches"]] == ["s1"]
    s1 = first["branches"][0]
    assert (s1["tests"], s1["commits"], s1["head_value"]) == (2, 1, {METRIC: 1040.0})
    assert s1["frontier"]["expected_gain_pct"] == pytest.approx(4.0) and s1["frontier"]["basis"] == "trend"
    assert first["scorer"] == s1["frontier"]["scorer"] == FRONTIER_V1_LABEL


@dataclasses.dataclass(frozen=True)
class _Fixed:
    """A scorer that expects ``gain`` of every branch, resting on ``basis``."""

    gain: float
    basis: FrontierBasis = FrontierBasis.CUSTOM
    components: tuple[tuple[str, float], ...] = ()
    label: str = "fixed-v1"

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        return Estimate(self.gain, dict(self.components), self.basis)


class _Failing:
    label = "failing-v1"

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        raise RuntimeError("no estimate today")


def test_any_scorer_is_held_to_the_envelope_and_named_in_its_score() -> None:
    context = ScoringContext(TASK, (), POLICY)
    young, settled = branch("s9", []), branch("s9", [], ended_times=(0.0,) * SETTLED_TESTS)

    floored = score_with(_Fixed(-1.0), young, 60.0, context)
    resting = score_with(_Fixed(-1.0, FrontierBasis.PRIOR), young, 60.0, context)
    capped = score_with(_Fixed(9.0), settled, 60.0, context, Readings(headroom_pct=2.0, pending_pct=1.0))
    lifted = score_with(_Fixed(0.2), settled, 60.0, context, Readings(pending_pct=1.0))

    assert (floored.expected_gain_pct, floored.basis) == (POLICY.new_branch_floor_pct, FrontierBasis.NEW)
    assert floored.scorer == "fixed-v1" and score_at(settled, 60.0, context).scorer == FRONTIER_V1_LABEL
    assert (resting.expected_gain_pct, resting.basis) == (0.0, FrontierBasis.PRIOR)
    assert (capped.expected_gain_pct, capped.basis) == (2.0, FrontierBasis.HEADROOM)
    assert capped.components == {"pending_pct": 1.0, "headroom_pct": 2.0}
    assert (lifted.expected_gain_pct, lifted.basis) == (1.0, FrontierBasis.PENDING)


@pytest.mark.parametrize(
    ("scorer", "why"),
    [
        (_Fixed(math.nan), "not a finite number"),
        (_Fixed(1.0, components=(("gpu_hours", math.inf),)), "not finite numbers: gpu_hours"),
        (_Fixed(1.0, components=(("headroom_pct", 1.0),)), "components the envelope writes: headroom_pct"),
        (_Fixed(1.0, FrontierBasis.NEW), "the envelope's to give"),
    ],
)
def test_an_estimate_that_breaks_the_scorer_contract_is_refused(scorer: _Fixed, why: str) -> None:
    with pytest.raises(ScorerError, match=why):
        score_with(scorer, branch("s9", []), 60.0, ScoringContext(TASK, (), POLICY))
    with pytest.raises(ScorerError, match="not an Estimate"):
        checked_estimate("fixed-v1", 1.0)


def test_the_guard_stands_frontier_v1_in_for_a_failing_scorer_then_scores_the_rest_of_the_session_with_it() -> None:
    told: list[str] = []
    guard = ScorerGuard(_Failing(), on_fallback=told.append)
    context = ScoringContext(TASK, (), POLICY)
    settled = branch("s9", [(30.0, 1.0)], ended_times=(0.0,) * SETTLED_TESTS)

    first = guard.score(settled, 60.0, context, Readings())
    for _ in range(FALLBACK_LIMIT - 1):
        guard.score(settled, 60.0, context, Readings())
    after = guard.score(settled, 60.0, context, Readings())

    assert first.scorer == FRONTIER_V1_LABEL and first.components[FALLBACK_COMPONENT] == 1.0
    assert first.expected_gain_pct == score_at(settled, 60.0, context).expected_gain_pct
    assert (guard.failures, guard.label, guard.chosen.label) == (FALLBACK_LIMIT, FRONTIER_V1_LABEL, "failing-v1")
    assert FALLBACK_COMPONENT not in after.components
    assert told == [
        f"the frontier scorer failing-v1 failed {FALLBACK_LIMIT} times; {FRONTIER_V1_LABEL} scores the rest of the "
        "session"
    ]


async def test_the_scoring_scores_with_its_guards_scorer_and_logs_its_label(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    path = tmp_path / "frontier.jsonl"
    scoring = FrontierScoring(notebook, POLICY, log=path, guard=ScorerGuard(_Fixed(3.0)))

    rows = {row.branch: row for row in await scoring.progress()}
    line = json.loads(path.read_text(encoding="utf-8"))

    assert rows["s1"].frontier is not None and rows["s1"].frontier.scorer == "fixed-v1"
    assert rows["s1"].frontier.expected_gain_pct == 3.0
    assert line["scorer"] == "fixed-v1" and line["branches"][0]["frontier"]["basis"] == "custom"


async def test_a_round_that_scored_nothing_writes_no_line(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    path = tmp_path / "frontier.jsonl"
    unmeasured = FrontierPolicy("a metric the baseline never recorded")

    await FrontierScoring(notebook, unmeasured, log=path).progress()

    assert not path.exists()
