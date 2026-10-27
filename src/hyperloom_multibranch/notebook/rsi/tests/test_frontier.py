# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frontier score of each running specialist branch: trend, spend, headroom, Amdahl ceiling, observation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import Purpose, TestKind, Validity, Verdict
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    LineageEvent,
    NotebookSnapshot,
    PendingCandidate,
    TestRecord,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver, Conclusion, FrontierBasis, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.rsi.archive import longest_root_test_seconds
from hyperloom_multibranch.notebook.rsi.frontier import (
    CANDIDATE_COMPONENT,
    FRONTIER_LOG_FORMAT,
    FRONTIER_V1_LABEL,
    MERGED_COMPONENT,
    MERGED_GAIN_COMPONENT,
    NEAR_BAR_COMPONENT,
    OBSERVATION_SECONDS,
    PROBATION_COMPONENT,
    PROBATION_SHARE,
    PROBATION_TESTS,
    SETTLED_TESTS,
    STALL_BAR_COMPONENT,
    STALLED_COMPONENT,
    STRANDED_COMPONENT,
    WINDOW_SECONDS,
    Estimate,
    FrontierPolicy,
    FrontierScoring,
    Readings,
    StallBar,
    StopRule,
    drawn_for_probation,
    enveloped,
    is_young,
    paced,
    probation_event,
    score_at,
    score_progress,
    stall_evidence,
    trend_estimate,
    trend_start,
)
from hyperloom_multibranch.notebook.rsi.history import DEFAULT_STALL_TESTS
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, judgement, profile, request, result
from hyperloom_multibranch.notebook.tests.sessions import GEMM_AGAIN, METRIC, scripted_session

POLICY = FrontierPolicy(METRIC)
TASK = task_conditions(profile())
PAYS_LATE = tree(
    "late", branch("s1", [(600.0, 2.0), (1800.0, 6.0)], test_times=(0.0, 600.0, 1200.0, 1800.0)), task=TASK
)
PAYS_SOON = tree("soon", branch("s1", [(1800.0, 3.0)], test_times=(0.0, 900.0, 1800.0)), task=TASK)


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


def test_a_branch_an_autonomous_agent_drives_is_not_scored(snapshot: NotebookSnapshot) -> None:
    lineage = snapshot.lineage
    nodes = tuple(
        dataclasses.replace(node, driver=BranchDriver.AUTONOMOUS) if node.branch == "s1" else node
        for node in lineage.branches
    )
    driven = dataclasses.replace(snapshot, lineage=dataclasses.replace(lineage, branches=nodes))

    assert _scores(snapshot)["s1"] is not None and _scores(driven)["s1"] is None


def test_the_built_in_reads_the_trend_however_many_similar_sessions_the_archive_holds(
    snapshot: NotebookSnapshot,
) -> None:
    backed = _scores(snapshot, PAYS_LATE, PAYS_SOON)["s1"]

    assert backed is not None and backed.expected_gain_pct == pytest.approx(4.0)
    assert backed.basis is FrontierBasis.TREND and "prior_remaining_pct" not in backed.components


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
    assert SETTLED_TESTS == 5


async def test_the_floor_ends_with_the_judges_grace_at_the_explore_tests_that_succeeded_and_were_judged(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    root = tmp_path / "notebook"
    observer = await scripted_session(root, clock)
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"), clock=clock)
    toolbox, gate = (LocalNotebook(root, Writer(role, role.value), clock=clock) for role in (Role.TOOLBOX, Role.GATE))
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
    await ended("mf1", Validity.MEASUREMENT_FAILED)
    failed = _scores(await observer.snapshot())["s3"]
    for n in range(SETTLED_TESTS - 1):
        await ended(f"j{n}", Validity.SUCCEEDED)
        await gate.write_judgement(f"j{n}", judgement(0.1, (f"j{n}",), verdict=Verdict.FLAT), Conclusion.RECORDED_ONLY)
    await ended("unjudged", Validity.SUCCEEDED)
    unjudged = _scores(await observer.snapshot())["s3"]
    await gate.write_judgement(
        "unjudged", judgement(0.1, ("unjudged",), verdict=Verdict.FLAT), Conclusion.RECORDED_ONLY
    )
    settled = _scores(await observer.snapshot())["s3"]

    assert failed is not None and failed.basis is FrontierBasis.NEW
    assert unjudged is not None and unjudged.basis is FrontierBasis.NEW
    assert settled is not None and settled.basis is FrontierBasis.TREND


def test_a_tree_kept_before_settled_tests_counts_its_ended_tests_for_the_floor() -> None:
    failing = branch("s9", [], ended_times=(0.0,) * SETTLED_TESTS, settled_times=(0.0,))
    older = branch("s9", [], ended_times=(0.0,) * SETTLED_TESTS)

    assert is_young(failing, 60.0, POLICY)
    assert not is_young(older, 60.0, POLICY)


def test_the_trend_reaches_back_to_the_branchs_last_four_ended_explore_tests() -> None:
    # c8927ba5's b09 at its stop: a long profile left one explore result in the 37-minute window.
    minute = 60.0
    b09 = branch(
        "b09",
        [(312 * minute, 41.49), (370 * minute, 43.83), (395 * minute, 49.59), (419 * minute, 53.89)],
        ended_times=tuple(when * minute for when in (300, 361, 385, 405, 430)),
    )
    # b03 kept testing and stalled: its last four results all fall inside the window.
    b03 = branch("b03", [(61 * minute, 1.11)], ended_times=tuple(when * minute for when in (137, 144, 153, 159)))
    window = FrontierPolicy(METRIC, window_seconds=37 * minute)

    assert trend_start(b09, 467 * minute, window.window_seconds) == 361 * minute
    assert trend_estimate(b09, 467 * minute, window, Readings()).components["trend_pct"] == pytest.approx(12.40)
    assert trend_estimate(b03, 163 * minute, window, Readings()).components["trend_pct"] == 0.0
    assert trend_start(branch("old", []), 467 * minute, window.window_seconds) == 430 * minute


def _row(snapshot: NotebookSnapshot, name: str = "s1") -> BranchProgress:
    return next(row for row in snapshot.progress if row.branch == name)


def test_each_scored_branch_carries_its_stall_evidence(snapshot: NotebookSnapshot) -> None:
    s1 = _scores(snapshot)["s1"]

    assert s1 is not None
    components = s1.components
    assert (components[STALLED_COMPONENT], components[STALL_BAR_COMPONENT]) == (1.0, float(DEFAULT_STALL_TESTS))
    assert (components[MERGED_COMPONENT], components[MERGED_GAIN_COMPONENT]) == (1.0, pytest.approx(4.0))
    assert (components[NEAR_BAR_COMPONENT], components[CANDIDATE_COMPONENT]) == (0.0, 0.0)
    assert PROBATION_COMPONENT not in components and STRANDED_COMPONENT not in components


def test_a_stalled_branch_is_a_stop_candidate_unless_a_candidate_or_its_grace_holds_it(
    snapshot: NotebookSnapshot,
) -> None:
    row, never_drawn = _row(snapshot), StallBar(StopRule(probation_share=0.0), 1)
    waiting = dataclasses.replace(row, pending=(PendingCandidate("r9", "r9c", "c1", METRIC, 1100.0, 5.0),))

    assert stall_evidence(snapshot, row, METRIC, never_drawn, young=False)[CANDIDATE_COMPONENT] == 1.0
    assert stall_evidence(snapshot, row, METRIC, never_drawn, young=True)[CANDIDATE_COMPONENT] == 0.0
    assert stall_evidence(snapshot, waiting, METRIC, never_drawn, young=False)[CANDIDATE_COMPONENT] == 0.0
    assert (
        stall_evidence(snapshot, row, METRIC, StallBar(StopRule(probation_share=0.0), 2), young=False)[
            CANDIDATE_COMPONENT
        ]
        == 0.0
    )


def test_a_stall_drawn_for_probation_runs_its_extra_tests_first_the_same_draw_every_round(
    snapshot: NotebookSnapshot,
) -> None:
    row = _row(snapshot)
    always, short = (
        StallBar(StopRule(probation_share=1.0), 1),
        StallBar(StopRule(probation_share=1.0, probation_tests=0), 1),
    )
    drawn = stall_evidence(snapshot, row, METRIC, always, young=False)

    assert (drawn[PROBATION_COMPONENT], drawn[CANDIDATE_COMPONENT]) == (1.0, 0.0)
    assert stall_evidence(snapshot, row, METRIC, short, young=False)[CANDIDATE_COMPONENT] == 1.0
    stalls = [f"s{n}" for n in range(400)]
    draws = [drawn_for_probation(snapshot, name, "c1", PROBATION_SHARE) for name in stalls]
    assert draws == [drawn_for_probation(snapshot, name, "c1", PROBATION_SHARE) for name in stalls]
    assert 60 < sum(draws) < 140
    assert drawn_for_probation(snapshot, "s1", "c1", 1.0) and not drawn_for_probation(snapshot, "s1", "c1", 0.0)


def test_a_branch_on_probation_gives_the_scheduler_the_same_event_every_round_of_its_stall(
    snapshot: NotebookSnapshot,
) -> None:
    row = _row(snapshot)
    drawn = stall_evidence(snapshot, row, METRIC, StallBar(StopRule(probation_share=1.0), 1), young=False)
    held = dataclasses.replace(row, frontier=FrontierScore(0.0, drawn, FrontierBasis.TREND, FRONTIER_V1_LABEL))
    longer = StopRule(probation_tests=6)

    assert probation_event(snapshot, held, longer) == LineageEvent(
        LineageKind.PROBATION,
        "s1",
        commit_id="c1",
        reason="stalled past the stop bar since c1, drawn for probation",
        tests=6,
    )
    assert probation_event(snapshot, held) == probation_event(snapshot, held)
    assert probation_event(snapshot, held).tests == PROBATION_TESTS
    assert probation_event(snapshot, dataclasses.replace(held, frontier=None)) is None


def test_a_commit_that_failed_to_merge_does_not_end_the_stall(snapshot: NotebookSnapshot) -> None:
    failed = tuple(
        dataclasses.replace(item, merged_into=None, first_failed_at=1.0) if item.commit_id == "c1" else item
        for item in snapshot.committed
    )
    unmerged = dataclasses.replace(snapshot, committed=failed)
    evidence = stall_evidence(unmerged, _row(unmerged), METRIC, StallBar(StopRule(), 6), young=False)

    assert (evidence[STALLED_COMPONENT], evidence[MERGED_COMPONENT]) == (2.0, 0.0)


def test_a_near_miss_among_the_last_tests_holds_the_branch(snapshot: NotebookSnapshot) -> None:
    near = judgement(0.8, ("r4",), verdict=Verdict.FLAT, threshold_pct=1.0)
    records = tuple(
        dataclasses.replace(record, judgement=near) if record.request_id == "r4" else record
        for record in snapshot.records
    )
    missed = dataclasses.replace(snapshot, records=records)
    evidence = stall_evidence(missed, _row(missed), METRIC, StallBar(StopRule(probation_share=0.0), 1), young=False)

    assert (evidence[NEAR_BAR_COMPONENT], evidence[CANDIDATE_COMPONENT]) == (1.0, 0.0)


def test_an_archived_branch_without_judged_times_keeps_the_floor_for_one_window_only() -> None:
    late = WINDOW_SECONDS + 1.0

    assert score_at(branch("s9", []), 60.0, POLICY).basis is FrontierBasis.NEW
    assert score_at(branch("s9", []), late, POLICY).basis is FrontierBasis.TREND
    assert score_at(branch("s9", [], judged_times=()), late, POLICY).basis is FrontierBasis.NEW


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


async def test_a_young_repair_branch_scores_at_least_the_gain_of_the_blocked_candidate_it_repairs(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    root = tmp_path / "notebook"
    observer = await scripted_session(root, clock)
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"), clock=clock)
    toolbox, gate = (LocalNotebook(root, Writer(role, role.value), clock=clock) for role in (Role.TOOLBOX, Role.GATE))

    async def judged(request_id: str, branch: str, verdict: Verdict, conclusion: Conclusion, value: float) -> None:
        await toolbox.write_request(request(request_id, intent(branch, "c1", target=request_id)))
        await toolbox.write_result(result(request_id))
        found = judgement(0.2, (request_id,), verdict=verdict, recorded={METRIC: value})
        await gate.write_judgement(request_id, found, conclusion)

    await judged("x1", "s1", Verdict.BLOCKED, Conclusion.REPAIR_REQUESTED, 1092.0)
    opened = LineageEvent(LineageKind.OPEN, "s3", from_commit="c1", reason="repair branch of s1", repairs="x1")
    await scheduler.record_lineage(opened)
    young = _scores(await observer.snapshot())["s3"]
    for n in range(SETTLED_TESTS):
        await judged(f"n{n}", "s3", Verdict.FLAT, Conclusion.RECORDED_ONLY, 1040.0)
    settled = _scores(await observer.snapshot())["s3"]

    assert young is not None and young.basis is FrontierBasis.REPAIR
    assert young.expected_gain_pct == pytest.approx(5.2) and young.components["repairing_pct"] == pytest.approx(5.2)
    assert settled is not None and settled.basis is FrontierBasis.TREND and "repairing_pct" not in settled.components


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


def test_a_baseline_read_from_the_prelude_childs_run_paces_the_window_by_that_run(snapshot: NotebookSnapshot) -> None:
    assert snapshot.baseline is not None
    child_run = dataclasses.replace(snapshot.baseline, request_ids=(), source="prelude-child:r1", wall_seconds=5288.0)

    longest = longest_root_test_seconds(dataclasses.replace(snapshot, baselines=(child_run,)))

    assert longest == 5288.0 and paced(POLICY, longest).window_seconds == 4 * 5288.0


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

    assert rows["s1"].frontier is not None and rows["s1"].frontier.expected_gain_pct == pytest.approx(4.0)
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
    assert first["new_branch_floor_pct"] == policy.new_branch_floor_pct
    assert first["t_s"] > 0
    assert [row["branch"] for row in first["branches"]] == ["s1"]
    s1 = first["branches"][0]
    assert (s1["tests"], s1["commits"], s1["head_value"]) == (2, 1, {METRIC: 1040.0})
    assert s1["frontier"]["expected_gain_pct"] == pytest.approx(4.0) and s1["frontier"]["basis"] == "trend"
    assert first["scorer"] == s1["frontier"]["scorer"] == FRONTIER_V1_LABEL


def test_the_envelope_floors_a_young_branchs_own_trend_lifts_it_to_a_pending_gain_and_caps_it() -> None:
    young, settled = branch("s9", []), branch("s9", [], ended_times=(0.0,) * SETTLED_TESTS)

    floored = enveloped(Estimate(-1.0), young, 60.0, POLICY, Readings())
    observed = enveloped(Estimate(0.2, basis=FrontierBasis.OBSERVING), young, 60.0, POLICY, Readings())
    capped = enveloped(Estimate(9.0), settled, 60.0, POLICY, Readings(headroom_pct=2.0, pending_pct=1.0))
    lifted = enveloped(Estimate(0.2), settled, 60.0, POLICY, Readings(pending_pct=1.0))

    assert (floored.expected_gain_pct, floored.basis) == (POLICY.new_branch_floor_pct, FrontierBasis.NEW)
    assert (observed.expected_gain_pct, observed.basis) == (0.2, FrontierBasis.OBSERVING)
    assert (capped.expected_gain_pct, capped.basis) == (2.0, FrontierBasis.HEADROOM)
    assert capped.components == {"pending_pct": 1.0, "headroom_pct": 2.0}
    assert (lifted.expected_gain_pct, lifted.basis) == (1.0, FrontierBasis.PENDING)
    assert score_at(settled, 60.0, POLICY).scorer == FRONTIER_V1_LABEL


async def test_a_round_that_scored_nothing_writes_no_line(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    path = tmp_path / "frontier.jsonl"
    unmeasured = FrontierPolicy("a metric the baseline never recorded")

    await FrontierScoring(notebook, unmeasured, log=path).progress()

    assert not path.exists()
