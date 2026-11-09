# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_INTVTY_P50_CHECKPOINT,
    GRADED_OUTPUT,
    GRADED_REQUESTS_TOTAL,
    GRADED_UNSERVED_REQUESTS,
    TOTAL_TOKEN_THROUGHPUT,
)
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Verdict
from hyperloom_multibranch.rulegate.agentx import (
    BELOW_THRESHOLD,
    P50_BELOW_THRESHOLD,
    SPEED_GATES,
    TOTAL_BELOW_THRESHOLD,
    AgentXObjective,
    compare,
    confirm,
)
from hyperloom_multibranch.rulegate.readings import GRADED_SHORT_REQUESTS

P50, TOTAL = AgentXObjective.P50_INTERACTIVITY, AgentXObjective.TOTAL_TOKEN_THROUGHPUT
PARENT = {
    GRADED_INTVTY_P50: 100.0,
    GRADED_INTVTY: 80.0,
    GRADED_OUTPUT: 1000.0,
    GRADED_DURATION: 600.0,
    GRADED_ERROR_RATE: 0.2,
    TOTAL_TOKEN_THROUGHPUT: 5000.0,
}
P90_BROKEN = {GRADED_INTVTY_P50: 106.0, GRADED_INTVTY: 75.2}


@pytest.fixture(autouse=True)
def _default_noise_band(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)


@pytest.mark.parametrize(
    ("changes", "verdict", "refused_by"),
    [
        ({GRADED_INTVTY_P50: 104.0}, Verdict.IMPROVED, ()),
        ({GRADED_INTVTY_P50: 102.5}, Verdict.IMPROVED, ()),
        ({GRADED_INTVTY_P50: 101.5}, Verdict.FLAT, (P50_BELOW_THRESHOLD,)),
        ({GRADED_INTVTY_P50: 98.5}, Verdict.FLAT, (P50_BELOW_THRESHOLD,)),
        ({GRADED_INTVTY_P50: 97.5}, Verdict.REGRESSED, (P50_BELOW_THRESHOLD,)),
        (P90_BROKEN, Verdict.NEEDS_CONFIRM, ("p90_regressed",)),
        ({GRADED_INTVTY_P50: 104.0, TOTAL_TOKEN_THROUGHPUT: 4700.0}, Verdict.NEEDS_CONFIRM, ("total_regressed",)),
        ({GRADED_INTVTY_P50: 104.0, GRADED_OUTPUT: 900.0}, Verdict.IMPROVED, ()),
        ({GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 1.2}, Verdict.NEEDS_CONFIRM, ("extra_failed_requests",)),
        ({GRADED_INTVTY_P50: 101.5, GRADED_INTVTY: 75.2}, Verdict.REGRESSED, ("p90_regressed", P50_BELOW_THRESHOLD)),
        ({GRADED_INTVTY_P50: 104.0, GRADED_DURATION: 660.0}, Verdict.REGRESSED, ("duration_drift",)),
    ],
)
def test_a_measurement_against_its_parent(changes, verdict, refused_by):
    candidate = {**PARENT, **changes}
    judgement = compare(candidate, PARENT, "r1", P50)
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.threshold_pct == pytest.approx(2.0)
    assert judgement.recorded_value == candidate
    assert judgement.request_ids == ("r1",)


THROUGHPUT_PARENT = PARENT
TOTAL_UP = {TOTAL_TOKEN_THROUGHPUT: 5200.0}


@pytest.mark.parametrize(
    ("objective", "verdict", "refused_by"),
    [(TOTAL, Verdict.IMPROVED, ()), (P50, Verdict.FLAT, (P50_BELOW_THRESHOLD,))],
    ids=["total_token_throughput", "p50_interactivity"],
)
def test_a_total_throughput_gain_with_p50_flat_is_kept_only_on_the_total_throughput_objective(
    objective, verdict, refused_by
):
    candidate = {**THROUGHPUT_PARENT, **TOTAL_UP}
    judgement = compare(candidate, THROUGHPUT_PARENT, "r1", objective)
    assert (judgement.verdict, judgement.refused_by) == (verdict, refused_by)
    assert judgement.recorded_value[TOTAL_TOKEN_THROUGHPUT] == 5200.0


def test_a_p50_gain_with_total_throughput_flat_is_not_kept_on_the_total_throughput_objective():
    judgement = compare({**THROUGHPUT_PARENT, GRADED_INTVTY_P50: 110.0}, THROUGHPUT_PARENT, "r1", TOTAL)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.FLAT, (TOTAL_BELOW_THRESHOLD,))


@pytest.mark.parametrize("objective", [P50, TOTAL])
def test_the_p90_guard_holds_a_gain_back_under_either_objective(objective):
    gain = {TOTAL_TOKEN_THROUGHPUT: 5200.0, GRADED_INTVTY_P50: 104.0, GRADED_INTVTY: 75.2}
    judgement = compare({**THROUGHPUT_PARENT, **gain}, THROUGHPUT_PARENT, "r1", objective)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.NEEDS_CONFIRM, ("p90_regressed",))


@pytest.mark.parametrize("objective", [P50, TOTAL])
def test_a_gain_of_two_and_a_half_percent_on_the_objective_is_kept(objective):
    gain = {GRADED_INTVTY_P50: 102.5} if objective is P50 else {TOTAL_TOKEN_THROUGHPUT: 5125.0}
    judgement = compare({**PARENT, **gain}, PARENT, "r1", objective)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.IMPROVED, ())
    assert judgement.threshold_pct == pytest.approx(2.0) == objective.bar_pct


def test_the_p50_guard_holds_a_total_throughput_gain_bought_with_p50_interactivity():
    gain = {TOTAL_TOKEN_THROUGHPUT: 5200.0, GRADED_INTVTY_P50: 94.0}
    judgement = compare({**THROUGHPUT_PARENT, **gain}, THROUGHPUT_PARENT, "r1", TOTAL)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.NEEDS_CONFIRM, ("p50_regressed",))


@pytest.mark.parametrize(("objective", "gain"), [(P50, {GRADED_INTVTY_P50: 104.0}), (TOTAL, TOTAL_UP)])
def test_output_throughput_guards_neither_objective_and_is_still_recorded(objective, gain):
    candidate = {**PARENT, **gain, GRADED_OUTPUT: 800.0}
    judgement = compare(candidate, PARENT, "r1", objective)
    assert (judgement.verdict, judgement.refused_by) == (Verdict.IMPROVED, ())
    assert judgement.recorded_value[GRADED_OUTPUT] == 800.0


def test_each_objective_guards_p90_and_the_other_objective_s_axis():
    assert [(guard.refusal, guard.metric) for guard in P50.guards] == [
        ("p90_regressed", GRADED_INTVTY),
        ("total_regressed", TOTAL_TOKEN_THROUGHPUT),
    ]
    assert [(guard.refusal, guard.metric) for guard in TOTAL.guards] == [
        ("p90_regressed", GRADED_INTVTY),
        ("p50_regressed", GRADED_INTVTY_P50),
    ]


def test_the_speed_gates_are_every_objective_s_bar_and_guard_refusals():
    assert SPEED_GATES == {*BELOW_THRESHOLD, "p90_regressed", "total_regressed", "p50_regressed"}


def test_a_run_without_its_objective_metric_is_not_compared():
    parent = {key: value for key, value in PARENT.items() if key != TOTAL_TOKEN_THROUGHPUT}
    with pytest.raises(ValueError, match=f"cannot compare a {TOTAL_TOKEN_THROUGHPUT}"):
        compare({**parent, GRADED_INTVTY_P50: 110.0}, parent, "r1", TOTAL)


def test_the_error_rate_bar_blocks_a_gain_and_fails_the_accuracy_gate():
    judgement = compare({**PARENT, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 12.0}, PARENT, "r1", P50)
    assert judgement.verdict is Verdict.NEEDS_CONFIRM
    assert judgement.accuracy is AccuracyVerdict.FAILED


@pytest.mark.parametrize(
    ("retest", "verdict", "refused_by"),
    [
        ({GRADED_INTVTY_P50: 105.0, GRADED_INTVTY: 74.0}, Verdict.BLOCKED, ("p90_regressed",)),
        (
            {GRADED_INTVTY_P50: 105.0, TOTAL_TOKEN_THROUGHPUT: 4700.0},
            Verdict.BLOCKED,
            ("p90_regressed", "total_regressed"),
        ),
        ({GRADED_INTVTY_P50: 105.0}, Verdict.INCONCLUSIVE, ("p90_regressed",)),
        (
            {GRADED_INTVTY_P50: 101.0, GRADED_INTVTY: 74.0},
            Verdict.INCONCLUSIVE,
            ("p90_regressed", P50_BELOW_THRESHOLD),
        ),
    ],
)
def test_a_retest_settles_a_blocked_gain(retest, verdict, refused_by):
    first = compare({**PARENT, **P90_BROKEN}, PARENT, "r1", P50)
    judgement = confirm(first, {**PARENT, **retest}, PARENT, "r2", P50)
    assert judgement.verdict is verdict
    assert judgement.refused_by == refused_by
    assert judgement.request_ids == ("r1", "r2")
    assert judgement.recorded_value[GRADED_INTVTY_P50] == pytest.approx((106.0 + retest[GRADED_INTVTY_P50]) / 2)


def test_only_a_provisional_judgement_is_retested():
    with pytest.raises(ValueError, match="provisional"):
        confirm(compare({**PARENT, GRADED_INTVTY_P50: 104.0}, PARENT, "r1", P50), PARENT, PARENT, "r2", P50)


def test_a_pair_missing_a_comparability_input_counts_as_regressed():
    candidate = {key: value for key, value in PARENT.items() if key != GRADED_ERROR_RATE}
    judgement = compare({**candidate, GRADED_INTVTY_P50: 104.0}, PARENT, "r1", P50)
    assert judgement.verdict is Verdict.REGRESSED
    assert judgement.refused_by == ("comparability_inputs_missing",)


@pytest.mark.parametrize(
    ("objective", "guard", "broken"),
    [
        (P50, GRADED_INTVTY, "p90_regressed"),
        (P50, TOTAL_TOKEN_THROUGHPUT, "total_regressed"),
        (TOTAL, GRADED_INTVTY, "p90_regressed"),
        (TOTAL, GRADED_INTVTY_P50, "p50_regressed"),
    ],
)
@pytest.mark.parametrize("side", ["candidate", "parent"])
def test_a_guard_axis_either_side_did_not_report_counts_as_broken(objective, guard, broken, side):
    gain = {**PARENT, GRADED_INTVTY_P50: 104.0, TOTAL_TOKEN_THROUGHPUT: 5200.0}
    unreported = {key: value for key, value in (gain if side == "candidate" else PARENT).items() if key != guard}
    candidate, parent = (unreported, PARENT) if side == "candidate" else (gain, unreported)

    judgement = compare(candidate, parent, "r1", objective)

    assert judgement.verdict is Verdict.NEEDS_CONFIRM
    assert judgement.refused_by == (broken,)


def test_a_run_without_a_p50_is_not_compared():
    candidate = {key: value for key, value in PARENT.items() if key != GRADED_INTVTY_P50}
    with pytest.raises(ValueError, match="cannot compare"):
        compare(candidate, PARENT, "r1", P50)


COUNTED = {**PARENT, GRADED_UNSERVED_REQUESTS: 1.0, GRADED_SHORT_REQUESTS: 0.0, GRADED_REQUESTS_TOTAL: 200.0}


def test_the_request_classes_are_recorded_on_the_judgement():
    candidate = {**COUNTED, GRADED_INTVTY_P50: 104.0, GRADED_SHORT_REQUESTS: 2.0}
    judgement = compare(candidate, COUNTED, "r1", P50)
    assert judgement.recorded_value[GRADED_SHORT_REQUESTS] == 2.0
    assert judgement.recorded_value[GRADED_UNSERVED_REQUESTS] == 1.0
    assert judgement.recorded_value[GRADED_REQUESTS_TOTAL] == 200.0


def test_more_short_answers_alone_never_refuse_a_candidate():
    candidate = {**COUNTED, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 5.0, GRADED_SHORT_REQUESTS: 30.0}
    judgement = compare(candidate, COUNTED, "r1", P50)
    assert judgement.verdict is Verdict.IMPROVED
    assert judgement.refused_by == ()


@pytest.mark.parametrize(
    ("candidate", "parent"),
    [
        ({**COUNTED, GRADED_UNSERVED_REQUESTS: 3.0}, COUNTED),
        ({key: value for key, value in COUNTED.items() if key != GRADED_UNSERVED_REQUESTS}, COUNTED),
        (COUNTED, PARENT),
    ],
    ids=["more-unserved", "candidate-uncounted", "parent-uncounted"],
)
def test_more_failed_requests_refuse_unless_the_unserved_count_shows_none_more(candidate, parent):
    judgement = compare({**candidate, GRADED_INTVTY_P50: 104.0, GRADED_ERROR_RATE: 5.0}, parent, "r1", P50)
    assert judgement.verdict is Verdict.NEEDS_CONFIRM
    assert judgement.refused_by == ("extra_failed_requests",)


@pytest.mark.parametrize(
    ("retest", "verdict"), [({GRADED_INTVTY_P50: 93.0}, Verdict.BLOCKED), ({}, Verdict.INCONCLUSIVE)]
)
def test_a_total_throughput_gain_the_p50_guard_held_back_is_retested(retest, verdict):
    first = compare({**PARENT, TOTAL_TOKEN_THROUGHPUT: 5200.0, GRADED_INTVTY_P50: 94.0}, PARENT, "r1", TOTAL)
    judgement = confirm(first, {**PARENT, TOTAL_TOKEN_THROUGHPUT: 5200.0, **retest}, PARENT, "r2", TOTAL)
    assert first.verdict is Verdict.NEEDS_CONFIRM
    assert (judgement.verdict, judgement.refused_by) == (verdict, ("p50_regressed",))
    assert judgement.threshold_pct == pytest.approx(2.0)


def test_the_checkpoint_median_is_recorded_so_a_committed_candidate_is_the_next_ones_reference():
    candidate = {**PARENT, GRADED_INTVTY_P50: 104.0, GRADED_INTVTY_P50_CHECKPOINT: 52.0}
    judgement = compare(candidate, PARENT, "r1", P50)
    assert judgement.verdict is Verdict.IMPROVED
    assert judgement.recorded_value[GRADED_INTVTY_P50_CHECKPOINT] == 52.0


def test_a_retest_averages_the_checkpoint_median_like_every_recorded_axis():
    first = compare({**PARENT, **P90_BROKEN, GRADED_INTVTY_P50_CHECKPOINT: 50.0}, PARENT, "r1", P50)
    retest = {**PARENT, GRADED_INTVTY_P50: 105.0, GRADED_INTVTY: 74.0, GRADED_INTVTY_P50_CHECKPOINT: 54.0}
    judgement = confirm(first, retest, PARENT, "r2", P50)
    assert judgement.recorded_value[GRADED_INTVTY_P50_CHECKPOINT] == pytest.approx(52.0)
