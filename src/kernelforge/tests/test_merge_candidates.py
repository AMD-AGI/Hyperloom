# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Selecting rejected candidates that are worth measuring stacked."""

from __future__ import annotations

from kernelforge.loop.merge_candidates import (
    MERGE_PLAN_PREFIX,
    MergeCandidate,
    attempted_pairs,
    case_spreads,
    cases_beating_reference,
    eligible_candidates,
    merge_plan,
    select_merge_pair,
)


INCUMBENT = {"prefill": 1.0, "decode": 2.0, "mixed": 4.0}


def _meta(
    iteration: int,
    *,
    speedup: float,
    case_times: dict[str, float],
    decision: str = "REVERT_PERF",
    plan: str = "",
) -> dict:
    """One archived candidate, its three runs agreeing to within 0.05%."""
    return {
        "iteration": iteration,
        "decision": decision,
        "mean_case_speedup": speedup,
        "plan": plan or f"plan {iteration}",
        "bench": {
            "case_times": case_times,
            "measurements": [
                {
                    "success": True,
                    "case_times": {
                        case_id: time_ms * jitter
                        for case_id, time_ms in case_times.items()
                        if isinstance(time_ms, (int, float)) and time_ms > 0.0
                    },
                    "unscored_cases": [],
                }
                for jitter in (1.0, 1.0005, 0.9995)
            ],
        },
    }


def _candidate(iteration: int, speedup: float, cases: set[str]) -> MergeCandidate:
    return MergeCandidate(
        iteration=iteration,
        plan=f"plan {iteration}",
        mean_case_speedup=speedup,
        winning_cases=frozenset(cases),
    )


def test_a_case_is_owned_only_when_it_beat_the_incumbent_time():
    metas = [
        _meta(
            1,
            speedup=1.01,
            case_times={
                "prefill": 0.9,
                "decode": 2.0,
                "mixed": 5.0,
            },
        )
    ]

    eligible = eligible_candidates(metas, INCUMBENT)

    assert [item.winning_cases for item in eligible] == [frozenset({"prefill"})]


def test_an_unmeasured_or_impossible_case_is_not_owned():
    """A missing or non-positive time is no evidence, not a win."""
    owned = cases_beating_reference(
        {"prefill": 0.0, "decode": None, "mixed": -1.0},
        INCUMBENT,
        {"prefill": 0.01, "decode": 0.01, "mixed": 0.01},
    )

    assert owned == frozenset()


def test_only_measured_gains_the_gate_turned_down_are_eligible():
    """A regression stays rejected, and a KEEP is the incumbent, not a candidate."""
    metas = [
        _meta(1, speedup=1.003, case_times={"prefill": 0.9}),
        _meta(2, speedup=0.97, case_times={"decode": 2.1}),
        _meta(3, speedup=1.20, case_times={"mixed": 3.0}, decision="KEEP"),
    ]

    eligible = eligible_candidates(metas, INCUMBENT)

    assert [item.iteration for item in eligible] == [1]


def test_a_candidate_that_no_longer_beats_the_incumbent_is_dropped():
    """The incumbent moves on; an old gain below it is no longer a gain.

    No separate staleness test does this. The reference the candidate is
    measured against is the live incumbent, so a gain the campaign has since
    banked stops being ground anyone owns.
    """
    metas = [_meta(1, speedup=1.02, case_times={"prefill": 0.9})]

    assert eligible_candidates(metas, {"prefill": 0.5}) == []


def test_a_candidate_winning_no_case_is_not_worth_stacking():
    """Every case a wash against the incumbent leaves nothing to combine."""
    metas = [_meta(1, speedup=1.01, case_times={"prefill": 1.0, "decode": 2.0})]

    assert eligible_candidates(metas, INCUMBENT) == []


def test_the_selected_pair_each_own_ground_the_other_loses():
    pair = select_merge_pair(
        [
            _candidate(1, 1.01, {"prefill"}),
            _candidate(2, 1.02, {"decode"}),
        ]
    )

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 2}


def test_a_candidate_whose_wins_are_covered_is_never_paired():
    """Stacking a subset re-measures what the better candidate already showed."""
    assert (
        select_merge_pair(
            [
                _candidate(1, 1.01, {"prefill"}),
                _candidate(2, 1.05, {"prefill", "decode"}),
            ]
        )
        is None
    )


def test_coverage_outranks_a_faster_pair_that_covers_less():
    """1+2 score highest together but leave 'mixed' untouched; 1+4 covers all."""
    pair = select_merge_pair(
        [
            _candidate(1, 1.30, {"prefill"}),
            _candidate(2, 1.30, {"decode"}),
            _candidate(3, 1.01, {"prefill", "mixed"}),
            _candidate(4, 1.01, {"decode", "mixed"}),
        ]
    )

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 4}


def test_selection_is_stable_for_two_runs_reading_one_archive():
    candidates = [
        _candidate(1, 1.01, {"prefill"}),
        _candidate(2, 1.01, {"decode"}),
        _candidate(3, 1.01, {"mixed"}),
    ]

    assert select_merge_pair(candidates) == select_merge_pair(list(reversed(candidates)))


def test_no_pair_exists_when_every_candidate_owns_the_same_case():
    assert select_merge_pair([_candidate(1, 1.01, {"prefill"})]) is None


def test_a_pair_already_measured_is_not_measured_again():
    """Otherwise a stall keeps re-buying the same answer for the same price."""
    first = _candidate(1, 1.01, {"prefill"})
    second = _candidate(2, 1.02, {"decode"})

    assert select_merge_pair([first, second]) is not None
    assert (
        select_merge_pair(
            [first, second],
            already_attempted=attempted_pairs([merge_plan(first, second)]),
        )
        is None
    )


def test_the_recorded_plan_survives_a_round_trip_in_either_order():
    """The archive line is the only record that a combination was tried."""
    first = _candidate(7, 1.01, {"prefill"})
    second = _candidate(3, 1.02, {"decode"})

    assert attempted_pairs([merge_plan(first, second)]) == frozenset({frozenset({3, 7})})
    assert attempted_pairs([merge_plan(second, first)]) == frozenset({frozenset({3, 7})})


def test_an_ordinary_plan_is_never_read_as_an_attempted_pair():
    assert attempted_pairs(["vectorize the epilogue stores", "", "stacked x+y: n"]) == (frozenset())


def test_a_prefixed_plan_naming_anything_but_two_iterations_is_not_a_pair():
    """Only the two-integer form records a measured stack.

    Reading a malformed line as a pair would retire a combination that was never
    measured, so anything that is not exactly two integers is discarded rather
    than guessed at.
    """
    assert (
        attempted_pairs(
            [
                f"{MERGE_PLAN_PREFIX} 1+2+3: three at once",
                f"{MERGE_PLAN_PREFIX} first+second: named, not numbered",
                f"{MERGE_PLAN_PREFIX} 4: a lone iteration",
            ]
        )
        == frozenset()
    )


def test_a_case_without_a_usable_reference_time_can_never_be_owned():
    """A missing or non-positive reference leaves nothing to have run faster than."""
    owned = cases_beating_reference(
        {"prefill": 0.5, "decode": 0.5, "mixed": 0.5},
        {"prefill": 0.0, "decode": None, "mixed": 4.0},
        {"prefill": 0.01, "decode": 0.01, "mixed": 0.01},
    )

    assert owned == frozenset({"mixed"})


def test_a_covered_pair_is_rejected_whichever_side_does_the_covering():
    """Complementarity is mutual, so the subset test cannot depend on order."""
    superset_first = [
        _candidate(1, 1.05, {"prefill", "decode"}),
        _candidate(2, 1.01, {"prefill"}),
    ]
    subset_first = [
        _candidate(1, 1.05, {"prefill"}),
        _candidate(2, 1.01, {"prefill", "decode"}),
    ]

    assert select_merge_pair(superset_first) is None
    assert select_merge_pair(subset_first) is None


# ── Per-case near misses: a candidate that beat the incumbent on one case ─────

# The 2026-08 GQA campaign's incumbent, per case. `m3-prefill-b2-q8073p60` is
# one of the two cases carrying the whole deficit against the competing agent;
# iteration 21 won 2.0% on it, lost the equal-weight mean, and left no trace.
GQA_INCUMBENT = {
    "m3-decode-q61": 0.010369,
    "m3-prefill-b2-q8131p60": 0.700000,
    "m3-prefill-b2-q8073p60": 0.706000,
}


def _per_case_meta(iteration: int, *, speedup: float, runs: list[dict]) -> dict:
    """One archived REVERT_PERF carrying its three independent measurements."""
    return {
        "iteration": iteration,
        "decision": "REVERT_PERF",
        "mean_case_speedup": speedup,
        "plan": f"plan {iteration}",
        "bench": {
            "case_times": {case_id: sum(run[case_id] for run in runs) / len(runs) for case_id in runs[0]},
            "measurements": [{"success": True, "case_times": run, "unscored_cases": []} for run in runs],
        },
    }


def test_a_case_is_won_only_when_the_gain_clears_that_case_s_own_spread():
    runs = [
        {"heavy": 1.00, "light": 0.50},
        {"heavy": 1.02, "light": 0.51},
        {"heavy": 0.98, "light": 0.49},
    ]
    spreads = case_spreads([{"case_times": run} for run in runs])
    measured = {"heavy": 1.00, "light": 0.50}

    # `heavy` moved 2% against a 2% spread; `light` moved 10% against the same.
    assert cases_beating_reference(measured, {"heavy": 1.02, "light": 0.556}, spreads) == frozenset({"light"})


def test_a_case_measured_only_once_can_never_be_won():
    """Three runs that never agreed about a case are not a measurement of it."""
    spreads = case_spreads([{"case_times": {"only": 1.0}}])

    assert spreads == {}
    assert cases_beating_reference({"only": 0.1}, {"only": 1.0}, spreads) == frozenset()


def test_a_revert_beating_the_incumbent_on_one_case_beyond_noise_is_eligible():
    """Iteration 21: 2.0% on q8073, the mean lost to a third case, no trace."""
    runs = [
        {
            "m3-decode-q61": 0.010369 * factor,
            "m3-prefill-b2-q8131p60": 0.700000 * factor,
            # 2.0% under the incumbent's 0.706, against a 0.07% spread.
            "m3-prefill-b2-q8073p60": 0.692 * factor,
        }
        for factor in (1.0, 1.0007, 0.9993)
    ]
    metas = [_per_case_meta(21, speedup=0.995, runs=runs)]

    eligible = eligible_candidates(metas, GQA_INCUMBENT)

    assert [item.iteration for item in eligible] == [21]
    assert [item.winning_cases for item in eligible] == [frozenset({"m3-prefill-b2-q8073p60"})]


def test_a_revert_winning_only_inside_its_own_spread_stays_out():
    """0.1% on a case whose three runs disagree by 0.2% is not a measurement."""
    runs = [
        {
            "m3-decode-q61": 0.010369,
            "m3-prefill-b2-q8131p60": 0.700000,
            "m3-prefill-b2-q8073p60": q8073,
        }
        # A mean 0.1% under the incumbent, drawn from runs spread over 0.28%.
        for q8073 in (0.7033, 0.7053, 0.7073)
    ]
    metas = [_per_case_meta(21, speedup=0.995, runs=runs)]

    assert eligible_candidates(metas, GQA_INCUMBENT) == []


def test_a_candidate_that_regresses_every_case_is_still_dropped():
    """Eligibility admits gains, not a second chance at a regression."""
    runs = [{case_id: time_ms * 1.05 for case_id, time_ms in GQA_INCUMBENT.items()} for _ in range(3)]
    runs[1] = {case_id: time_ms * 1.051 for case_id, time_ms in GQA_INCUMBENT.items()}
    metas = [_per_case_meta(7, speedup=0.95, runs=runs)]

    assert eligible_candidates(metas, GQA_INCUMBENT) == []


def _incumbent_runs(*, faster_case: str, factor: float) -> list[dict]:
    """Three runs level with the incumbent everywhere but on ``faster_case``."""
    return [
        {
            case_id: time_ms * (factor if case_id == faster_case else 1.0) * jitter
            for case_id, time_ms in GQA_INCUMBENT.items()
        }
        for jitter in (1.0, 1.0007, 0.9993)
    ]


def test_an_aggregate_winner_owns_what_it_took_from_the_incumbent():
    """Not the broad pristine ground it shares with every candidate in the archive.

    Both of these beat pristine on every case, which is what any candidate looks
    like once the campaign has banked a few KEEPs. Ranking them on that shared
    set leaves neither owning ground the other lacks.
    """
    metas = [
        _per_case_meta(
            21,
            speedup=1.01,
            runs=_incumbent_runs(faster_case="m3-prefill-b2-q8073p60", factor=0.98),
        ),
        _per_case_meta(
            24,
            speedup=1.02,
            runs=_incumbent_runs(faster_case="m3-prefill-b2-q8131p60", factor=0.98),
        ),
    ]

    eligible = eligible_candidates(metas, GQA_INCUMBENT)

    assert [item.winning_cases for item in eligible] == [
        frozenset({"m3-prefill-b2-q8073p60"}),
        frozenset({"m3-prefill-b2-q8131p60"}),
    ]


def test_an_aggregate_winner_holding_no_ground_of_its_own_is_dropped():
    """What the incumbent reference costs, stated as a test.

    Its conservative mean led the incumbent, but no single case moved further
    than that case's own runs disagreed, so it brings a stack nothing to
    combine. On the 2026-08 archives this loses one pair; measuring ownership
    against the incumbent gains twenty-three.
    """
    runs = [
        {case_id: time_ms * jitter for case_id, time_ms in GQA_INCUMBENT.items()}
        # Every case level with the incumbent to within its own 0.07% spread.
        for jitter in (0.9995, 1.0002, 0.9998)
    ]
    metas = [_per_case_meta(21, speedup=1.01, runs=runs)]

    assert eligible_candidates(metas, GQA_INCUMBENT) == []


def test_two_reverts_beating_the_incumbent_on_different_cases_form_a_pair():
    """The reason this mechanism never ran, admission through to selection.

    Both candidates are twice as fast as pristine on every case, so a
    pristine-relative ownership hands them identical sets and the selector's
    mutual-complementarity test rejects the pair. Measured against the
    incumbent they own one case each, which is the pair the mechanism exists to
    measure.
    """
    metas = [
        _per_case_meta(
            21,
            speedup=0.995,
            runs=_incumbent_runs(faster_case="m3-prefill-b2-q8073p60", factor=0.98),
        ),
        _per_case_meta(
            24,
            speedup=0.996,
            runs=_incumbent_runs(faster_case="m3-prefill-b2-q8131p60", factor=0.98),
        ),
    ]
    pristine = {case_id: time_ms * 2 for case_id, time_ms in GQA_INCUMBENT.items()}

    assert select_merge_pair(eligible_candidates(metas, pristine)) is None

    eligible = eligible_candidates(metas, GQA_INCUMBENT)

    assert [item.iteration for item in eligible] == [21, 24]

    pair = select_merge_pair(eligible)

    assert pair is not None
    assert {item.iteration for item in pair} == {21, 24}


def test_a_stack_that_reverted_is_not_itself_stackable():
    """Two is the cap, and a reverted stack is archived like any other REVERT.

    Without this the pair selector picks up a previous stack and produces three
    diffs under a record naming two, which is exactly what ``merge_attempt_
    staged`` and ``merge_attempt_kept`` are counting. Nothing measurable is
    given up: across the thirty archived runs of 2026-08-22 and 08-23, a
    mutually-complementary triple exists at 2 of the 121 consulted iterations,
    and at neither does it cover more cases than the best available pair.
    """
    metas = [
        _meta(1, speedup=1.003, case_times={"prefill": 0.9}),
        _meta(2, speedup=1.004, case_times={"decode": 1.8}),
        _meta(
            3,
            speedup=1.006,
            case_times={"prefill": 0.9, "decode": 1.8},
            plan=merge_plan(_candidate(1, 1.003, {"prefill"}), _candidate(2, 1.004, {"decode"})),
        ),
    ]

    eligible = eligible_candidates(metas, INCUMBENT)

    assert [item.iteration for item in eligible] == [1, 2]
