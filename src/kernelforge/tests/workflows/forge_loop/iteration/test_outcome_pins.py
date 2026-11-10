# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an iteration's outcome pins for retrieval: near misses and gains over the pristine score, never regressions."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.campaign.run_state import BestRecord
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.evaluation.scoring import passes_keep_threshold
from kernelforge.workflows.forge_loop.iteration import outcome_runtime
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _reduction_loop


def _reverted_candidate(iteration, mean_case_speedup):
    """One correct candidate that was rejected by the KEEP threshold."""
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=1.0,
        mean_case_speedup=mean_case_speedup,
        snr_db=40.0,
        kept=False,
        bench_detail={"case_times": {"case": 1.0}},
    )


def test_a_near_miss_is_pinned_so_the_retrieval_map_points_at_it(
    tmp_path,
    monkeypatch,
):
    """A gain the KEEP gate rejected is still the best lead the run has.

    ``REVERT_PERF`` covers both a regression and a real gain that landed in the
    band between the incumbent and the accept threshold. The long-horizon prompt
    carries a retrieval map instead of the candidate diffs, and only KEEPs were
    ever pinned, so the most promising rejected work sat in the archive with
    nothing pointing at it and was re-derived from scratch.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    # A real 0.5% gain on every measurement, but spread widely enough that its
    # mean does not clear the t bound on its own scatter.
    scores = [1.00100, 1.00520, 1.00950]

    assert not passes_keep_threshold(scores, best_mean_case_speedup=1.0)

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, min(scores)),
        plan="stage the scales through LDS",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == [1]


def test_a_regression_is_not_pinned(tmp_path, monkeypatch):
    """The other half of the split: a candidate that lost is not a lead.

    The absence of a pin is the whole verdict. A failed candidate is
    deliberately not recorded as a spent direction anywhere else either: it does
    not become a permanent search constraint, and the trajectory already carries
    what happened as fact.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    loop._record_iteration_outcome(
        _reverted_candidate(1, 0.972),
        plan="stage the scales through LDS",
        decision_label="REVERT_PERF",
    )

    assert loop.run_state.pinned_iterations == []


def test_a_run_of_near_misses_cannot_evict_the_best_lineage_pin(
    tmp_path,
    monkeypatch,
):
    """The pin the retrieval map is built around outlives later near-misses.

    Near-misses are pinned into the same list as the KEEP behind the current
    best, and a run produces far more of them than KEEPs, so eviction purely by
    age drops the best lineage after eight later pins. It is only released once
    another KEEP takes its place.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.run_state.best = BestRecord(
        iteration=3,
        mean_case_speedup=1.2,
        commit_hash="kept",
        source="iteration",
    )
    outcome_runtime.pin_iteration(loop.run_state, 3)

    for iteration in range(4, 14):
        loop._record_iteration_outcome(
            _reverted_candidate(iteration, 1.003),
            plan=f"near miss {iteration}",
            decision_label="REVERT_PERF",
        )
    while_best = list(loop.run_state.pinned_iterations)

    loop.run_state.best = BestRecord(
        iteration=14,
        mean_case_speedup=1.4,
        commit_hash="newer",
        source="iteration",
    )
    outcome_runtime.pin_iteration(loop.run_state, 14)

    assert while_best == [3, 7, 8, 9, 10, 11, 12, 13]
    assert loop.run_state.pinned_iterations == [7, 8, 9, 10, 11, 12, 13, 14]


def test_an_unscored_candidate_is_not_pinned(tmp_path, monkeypatch):
    """A candidate that measured nothing cannot claim to have beaten anything.

    The pin gate reads the KEEP score, so a missing one has to fail
    closed. Reading it as a gain would point the retrieval map at work that was
    never shown to be worth re-reading.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, None),
        plan="fuse the two passes",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == []


def test_gain_over_pristine_is_pinned_before_the_first_keep(
    tmp_path,
    monkeypatch,
):
    """A missing incumbent is the pristine 1.0, matching the KEEP gate.

    Reading it as "no gain is possible" would lose every near miss of the
    cold-start iterations, where a real gain over pristine is necessarily still
    below the threshold.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.best_mean_case_speedup = None

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, 1.003),
        plan="hoist the mask computation",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == [1]


def test_a_regression_before_the_first_keep_is_not_pinned(
    tmp_path,
    monkeypatch,
):
    """The pristine fallback is a real bar, not a waiver.

    Paired with the test above: together they show a missing incumbent is read
    as 1.0 rather than as "anything qualifies", which would pin every cold-start
    candidate including the ones slower than the kernel they started from.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.best_mean_case_speedup = None

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, 0.971),
        plan="widen the accumulator",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == []
