# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Measurement guards: the KEEP margin, aggregate consistency, drift."""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.knowledge import experience_integration as integration
from kernelforge.loop.baseline_reference import (
    BASELINE_DRIFT_TOLERANCE,
    BaselineReferenceError,
    check_baseline_against_reference,
    load_reference_case_times,
)
from kernelforge.loop.recovery import publish_warm_start_recovery
from kernelforge.loop.run_state import LoopStateStore, RunState, apply_iteration
from kernelforge.loop import runner as runner_module
from kernelforge.loop.runner import IterationResult, _decision_label
from kernelforge.loop.scoring import (
    KEEP_MEASUREMENT_COUNT,
    KEEP_MIN_MARGIN_FRACTION,
    SIGMA_REMEASURE_BATCH,
    SIGMA_REMEASURE_MAX_ROUNDS,
    aggregate_regression_detail,
    attribute_sigma,
    keep_t_critical,
    measurement_sigma,
    passes_keep_threshold,
    required_keep_speedup,
    rescaled_sigma,
    warm_start_improvement_flags,
)
from kernelforge.tests.test_loop_runner import (
    _make_loop,
    _measurement_loop,
    _no_change_agent,
    _unused_supervisor,
)

# A modest speedup, well inside anything the loop has ever argued about.
MODEST_SPEEDUP = 5.72

# The 2026-08-18 run on vllm_triton_paged_attention_2d_minimax_m3: the incumbent
# the campaign froze at, and the three independent measurements of the candidate
# it kept rejecting. The competing agent won that kernel with 23.884x.
FROZEN_INCUMBENT = 19.920933
INCIDENT_SCORES = [24.405855, 24.392908, 24.39891]

# The same kernel measured quietly: three scores whose relative sample sigma is
# 0.022%, carrying a 0.15% gain over the frozen incumbent.
QUIET_SCORES = [19.94561, 19.95, 19.95439]

# mla_decode_grouped from the same 2026-08-18 batch, an order of magnitude
# noisier at 0.281% relative sigma, with its incumbent. Every score beats that
# incumbent, but the 4.385408 candidate does so by under one sigma.
NOISY_INCUMBENT = 4.375031
NOISY_SCORES = [4.377112, 4.385408, 4.401372]


# ── Guard 1: the noise-relative KEEP margin ───────────────────────────────────


def test_a_high_speedup_measurement_is_believed(monkeypatch):
    """39 candidates scoring 19.92x-24.39x were thrown away as impossible.

    17 of them beat the 23.884x the arena scored as a legitimate PASS on the
    same kernel, and the best of them measured a raw mean of 0.045 ms against
    the winner's 0.0467 ms. Three measurements agreeing to within 0.1% are a
    measurement, not a broken timing path.
    """
    assert passes_keep_threshold(
        INCIDENT_SCORES,
        best_mean_case_speedup=FROZEN_INCUMBENT,
    )
    # These three agree to 0.033%, so the noise term is 0.055% of the incumbent
    # and the 0.1% floor is what sets the bar: 19.940854x, asked of a candidate
    # measuring 24.39x. Not a near thing under the new rule either.
    assert required_keep_speedup(FROZEN_INCUMBENT, INCIDENT_SCORES) == pytest.approx(
        FROZEN_INCUMBENT * (1 + KEEP_MIN_MARGIN_FRACTION), abs=1e-6
    )
    assert required_keep_speedup(FROZEN_INCUMBENT, INCIDENT_SCORES) == pytest.approx(19.940854, abs=1e-6)


def test_the_keep_margin_is_charged_in_units_of_the_measured_noise():
    """Neither fixed rule could be right for both of these kernels.

    ``best * 1.005`` asked 20.020537x of the frozen incumbent while nothing over
    20.0x could then be believed, so the campaign held 19.920933x for 60
    iterations and 30 hours. ``best + 0.005`` replaced it with a 0.025% relative
    bar at that incumbent, under the 0.168% median noise, so the incumbent could
    ratchet on noise alone. The margin is now the one-sided 95% Student-t bound
    on the mean of the candidate's own scores, which is 0.475% of the noisy
    kernel and, on the quiet one, small enough that the floor is what holds it.
    """
    quiet_margin = required_keep_speedup(FROZEN_INCUMBENT, QUIET_SCORES)
    quiet_margin -= FROZEN_INCUMBENT
    noisy_margin = required_keep_speedup(NOISY_INCUMBENT, NOISY_SCORES)
    noisy_margin -= NOISY_INCUMBENT

    # The quiet kernel repeats to 0.022%, so t sigma / sqrt(n) is 0.037% of the
    # incumbent -- under the 0.1% floor, which therefore sets its bar. That is
    # the floor doing the job it exists for and not the noise term failing: at
    # this scatter the candidate's 0.15% gain is t = 8.8, certainly real, and
    # what the floor decides is whether a gain that small is worth a KEEP.
    assert (
        keep_t_critical(KEEP_MEASUREMENT_COUNT) * measurement_sigma(QUIET_SCORES) / math.sqrt(KEEP_MEASUREMENT_COUNT)
        < FROZEN_INCUMBENT * KEEP_MIN_MARGIN_FRACTION
    )
    assert quiet_margin == pytest.approx(FROZEN_INCUMBENT * KEEP_MIN_MARGIN_FRACTION)
    # The noisy one is an order of magnitude wider, so its own scatter sets the
    # bar and asks over four times the relative gain of the quiet kernel's
    # floor -- 0.475% against 0.1%.
    assert noisy_margin == pytest.approx(
        keep_t_critical(KEEP_MEASUREMENT_COUNT) * measurement_sigma(NOISY_SCORES) / math.sqrt(KEEP_MEASUREMENT_COUNT)
    )
    assert noisy_margin / NOISY_INCUMBENT > 4 * (quiet_margin / FROZEN_INCUMBENT)


def test_a_quiet_kernel_earns_a_gain_the_old_multiplier_refused():
    """0.15% on a kernel that repeats to 0.022% is a certain improvement.

    ``best * 1.005`` demanded 20.020537x of these scores and rejected all three.
    Their own spread is quiet enough that the noise term falls under the floor,
    so the bar is the floor: 19.940854x, which their mean of 19.950000x clears.
    """
    assert passes_keep_threshold(
        QUIET_SCORES,
        best_mean_case_speedup=FROZEN_INCUMBENT,
    )
    assert min(QUIET_SCORES) < FROZEN_INCUMBENT * 1.005


def test_a_noisy_kernel_is_refused_the_same_relative_gain():
    """0.24% on mla_decode_grouped is under one sigma of its own spread.

    Every one of these scores beats the incumbent, so the old rules both kept
    it; here the candidate has to out-measure its own scatter and does not.
    """
    assert min(NOISY_SCORES) > NOISY_INCUMBENT
    assert not passes_keep_threshold(
        NOISY_SCORES,
        best_mean_case_speedup=NOISY_INCUMBENT,
    )


def test_near_identical_measurements_fall_back_to_the_floor():
    """Three scores agreeing to 1e-6 would otherwise set a bar of zero.

    The floor is the only thing between a freak-quiet measurement and an
    incumbent that advances on nothing.
    """
    freak_quiet = [2.500300, 2.500301, 2.500302]

    assert (
        keep_t_critical(KEEP_MEASUREMENT_COUNT) * measurement_sigma(freak_quiet) / math.sqrt(KEEP_MEASUREMENT_COUNT)
        < 2.5 * KEEP_MIN_MARGIN_FRACTION
    )
    assert required_keep_speedup(2.5, freak_quiet) == pytest.approx(2.5 * (1.0 + KEEP_MIN_MARGIN_FRACTION))
    assert not passes_keep_threshold(freak_quiet, best_mean_case_speedup=2.5)


# The 2026-08-23 21:28 iteration on sglang_tilelang_dsa_sparse_mla_glm5, which
# the old rule reverted with `mean case speedup=1.396574x not better than
# best=1.393438x`. Every measurement beat the incumbent; the weakest missed the
# bar by 0.00033x.
DOUBLE_CHARGED_INCUMBENT = 1.393438
DOUBLE_CHARGED_SCORES = [1.398627, 1.398520, 1.396574]


def test_a_single_low_draw_is_not_charged_twice():
    """The 1.396574 draw was the score *and* the thing that raised the bar over it.

    Under ``all(score >= best + 3 sigma)`` the weakest measurement stood in as
    the candidate's score while also widening the sigma that set the margin
    above it, so one unlucky draw was paid for twice. The mean is charged once:
    the low draw pulls it down and widens the spread, and that is the whole of
    its effect. This candidate carried a +0.32% mean gain at t = 6.70 and was
    thrown away.
    """
    mean_gain = statistics.fmean(DOUBLE_CHARGED_SCORES) / DOUBLE_CHARGED_INCUMBENT - 1
    sigma = measurement_sigma(DOUBLE_CHARGED_SCORES)
    t_statistic = (statistics.fmean(DOUBLE_CHARGED_SCORES) - DOUBLE_CHARGED_INCUMBENT) / (
        sigma / math.sqrt(KEEP_MEASUREMENT_COUNT)
    )

    assert mean_gain > 0.003
    assert t_statistic > 6.0
    # The old rule's own arithmetic, reproduced: the weakest score missed
    # best + 3 sigma by 0.00033x while the other two cleared it.
    old_bar = DOUBLE_CHARGED_INCUMBENT + 3.0 * sigma
    assert min(DOUBLE_CHARGED_SCORES) < old_bar
    assert sorted(DOUBLE_CHARGED_SCORES)[1] > old_bar

    assert passes_keep_threshold(
        DOUBLE_CHARGED_SCORES,
        best_mean_case_speedup=DOUBLE_CHARGED_INCUMBENT,
    )


def test_a_sigma_estimated_from_more_samples_is_charged_the_df_it_earned():
    """A re-measure buys degrees of freedom; charging df = 2 wastes what it bought.

    The extra benches run the whole suite, so every scored case reaches the same
    count and the df is exact. An unlisted count is charged the largest
    tabulated df at or below it, so it is never charged less than it earned.
    """
    assert keep_t_critical(3) > keep_t_critical(6) > keep_t_critical(9)
    # Between tabulated points, and below and above the table.
    assert keep_t_critical(8) == keep_t_critical(6)
    assert keep_t_critical(50) == keep_t_critical(9)
    assert keep_t_critical(1) == keep_t_critical(3)

    scores = [1.010, 1.014, 1.021]
    sigma = measurement_sigma(scores)
    nine = required_keep_speedup(1.0, scores, sigma=sigma, sigma_sample_size=9)
    three = required_keep_speedup(1.0, scores, sigma=sigma, sigma_sample_size=3)

    assert nine < three
    # The sample size changes the critical value only. The standard error stays
    # over the three scores the protocol took: a bought measurement informs the
    # bar and is never admitted as evidence of a gain.
    assert nine - 1.0 == pytest.approx(keep_t_critical(9) * sigma / math.sqrt(len(scores)))


def test_a_candidate_inside_the_margin_is_still_refused():
    scores = [19.925933, 19.925933, 19.925932]

    assert min(scores) > FROZEN_INCUMBENT
    assert not passes_keep_threshold(
        scores,
        best_mean_case_speedup=FROZEN_INCUMBENT,
    )


def test_the_sample_standard_deviation_is_the_one_that_is_used():
    """The population form divides by 3 rather than 2 at this measurement count.

    It understates the spread enough that a simulation using it produced a
    higher false-accept rate at k = 2 than at k = 3, which cannot be true of a
    bar that only gets stricter.
    """
    assert measurement_sigma(INCIDENT_SCORES) == pytest.approx(statistics.stdev(INCIDENT_SCORES))
    assert measurement_sigma(INCIDENT_SCORES) > statistics.pstdev(INCIDENT_SCORES)


def test_an_unmeasured_spread_leaves_the_bar_at_the_floor():
    """A crashed bench reports no scores, and no scores is not zero noise."""
    assert measurement_sigma([]) is None
    assert required_keep_speedup(2.5, []) == pytest.approx(2.5 * (1.0 + KEEP_MIN_MARGIN_FRACTION))
    assert not passes_keep_threshold([], best_mean_case_speedup=2.5)


def test_the_printed_bar_is_the_bar_that_was_enforced(monkeypatch, capsys):
    """A log naming a threshold other than the enforced one is worse than none.

    The bar is now derived from the scores on the same line, so the operator
    reading a REVERT can tell a weak candidate from a noisy measurement -- but
    only if the printed sigma and the printed bar are the ones that decided it.
    """
    scores = [1.002, 1.006, 1.012]
    loop, _calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": 1.0 / scores[1],
            "case_times": {"small": 1.0 / scores[1], "large": 1.0 / scores[1]},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {"small": 1.0 / score, "large": 1.0 / score},
                    "unscored_cases": [],
                }
                for score in scores
            ],
            "message": "three measurements",
        },
    )

    result = asyncio.run(loop.run_one_iteration(1))
    bench_line = next(
        line for line in capsys.readouterr().out.splitlines() if "[bench] pristine-relative scores=" in line
    )

    required = required_keep_speedup(1.0, scores)
    assert f"sigma={measurement_sigma(scores):.6f}" in bench_line
    assert f"required={required:.6f}x" in bench_line
    # The scores straddle the bar they set, so the printed number is load-bearing
    # rather than trivially cleared.
    assert statistics.fmean(scores) < required <= max(scores)
    assert result.kept is False


def test_no_iteration_outcome_is_labelled_implausible(monkeypatch):
    """The label is gone: a fast measurement is a KEEP or it is a REVERT_PERF."""
    candidate_ms = 0.001572
    loop, _calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": candidate_ms,
            "case_times": {"small": candidate_ms, "large": candidate_ms},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {
                        "small": candidate_ms,
                        "large": candidate_ms,
                    },
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "three measurements",
        },
    )

    result = asyncio.run(loop.run_one_iteration(1))

    assert result.kept is True
    assert _decision_label(result) == "KEEP"


# The load-independent timing floor every case collapsed onto, and the one
# genuinely expensive case whose 1.1232 ms divided by that floor reads 714.60x.
TIMING_FLOOR_MS = 0.001572
EXPENSIVE_PRISTINE_MS = 1.1232


def _floored_case_times() -> tuple[dict[str, float], dict[str, float]]:
    """Pristine and candidate suites whose per-case mean reads 19.29x.

    One case reads 714.60x; the other 38 already ran at the floor and read 1.0x.
    """
    pristine = {"k001": EXPENSIVE_PRISTINE_MS}
    pristine.update({f"k{index:03d}": TIMING_FLOOR_MS for index in range(2, 40)})
    return pristine, {case_id: TIMING_FLOOR_MS for case_id in pristine}


def test_kb_warm_start_adopts_a_high_scoring_prior_solution(
    monkeypatch,
    tmp_path,
):
    """The KB write path scores a 19.29x warm start on its measurement alone.

    It used to refuse this one and record no measured value against the record
    it came from, which is how a real result reached the next day as no result
    at all.
    """
    pristine, candidate = _floored_case_times()
    monkeypatch.setattr(integration, "_apply_candidate_patch", lambda *_a, **_k: "")
    monkeypatch.setattr(integration, "_force_jit_rebuild", lambda *_a, **_k: None)
    monkeypatch.setattr(integration, "_correctness_once", lambda *_a, **_k: True)
    monkeypatch.setattr(
        integration,
        "_bench_once",
        lambda *_a, **_k: {
            "success": True,
            "median_ms": TIMING_FLOOR_MS,
            "case_times": dict(candidate),
            "unscored_cases": [],
        },
    )

    trial = integration._try_apply_candidate(
        {"solution_slug": "prior-solution"},
        kernel=str(tmp_path / "kernel.py"),
        driver=str(tmp_path / "driver.py"),
        workspace_dir=str(tmp_path),
        snr_threshold=30.0,
        source_files=None,
        # A real pristine bench reports both halves of its measurement; the
        # aggregate is dominated by the one expensive case.
        pristine_bench={
            "case_times": pristine,
            "median_ms": EXPENSIVE_PRISTINE_MS,
        },
        allowed_paths=None,
        pre_untracked=set(),
    )

    assert trial.reject_reason == ""
    assert trial.adoptable_ms == TIMING_FLOOR_MS
    assert trial.adoptable_mean_case_speedup == pytest.approx(19.29, abs=0.01)
    assert trial.measured_mean_case_speedup == trial.adoptable_mean_case_speedup


def test_kb_warm_start_still_adopts_a_modest_prior_solution(
    monkeypatch,
    tmp_path,
):
    pristine, _ = _floored_case_times()
    candidate = {case_id: baseline_ms / MODEST_SPEEDUP for case_id, baseline_ms in pristine.items()}
    monkeypatch.setattr(integration, "_apply_candidate_patch", lambda *_a, **_k: "")
    monkeypatch.setattr(integration, "_force_jit_rebuild", lambda *_a, **_k: None)
    monkeypatch.setattr(integration, "_correctness_once", lambda *_a, **_k: True)
    monkeypatch.setattr(
        integration,
        "_bench_once",
        lambda *_a, **_k: {
            "success": True,
            "median_ms": TIMING_FLOOR_MS,
            "case_times": dict(candidate),
            "unscored_cases": [],
        },
    )

    trial = integration._try_apply_candidate(
        {"solution_slug": "prior-solution"},
        kernel=str(tmp_path / "kernel.py"),
        driver=str(tmp_path / "driver.py"),
        workspace_dir=str(tmp_path),
        snr_threshold=30.0,
        source_files=None,
        # Both halves of the pristine measurement, agreeing at 5.72x: the
        # candidate has to beat the aggregate as well as the per-case mean.
        pristine_bench={
            "case_times": pristine,
            "median_ms": TIMING_FLOOR_MS * MODEST_SPEEDUP,
        },
        allowed_paths=None,
        pre_untracked=set(),
    )

    assert trial.reject_reason == ""
    assert trial.adoptable_ms == TIMING_FLOOR_MS
    assert trial.adoptable_mean_case_speedup == pytest.approx(MODEST_SPEEDUP)
    assert trial.adoptable_bench["mean_case_speedup"] == (trial.adoptable_mean_case_speedup)
    assert trial.measured_mean_case_speedup == trial.adoptable_mean_case_speedup


def test_a_high_scoring_measurement_reaches_the_event_stream_as_a_keep(
    tmp_path,
    monkeypatch,
):
    """The whole point of the change: 24.39x is published as a KEEP."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def fast_agent(kernel_path, _history, session_sink):
        with open(kernel_path, "w") as handle:
            handle.write("def kernel():\n    return 2\n")
        session_sink["plan"] = "a very fast kernel"
        return "rewrote the kernel"

    async def fast_iteration(iteration, _plan="", **_kwargs):
        return IterationResult(
            iteration=iteration,
            duration_sec=1.0,
            validation_passed=True,
            validation_summary="PASS",
            wall_ms=0.045,
            mean_case_speedup=min(INCIDENT_SCORES),
            kept=True,
        )

    monkeypatch.setattr(loop, "run_one_iteration", fast_iteration)

    asyncio.run(loop.run(agent_fn=fast_agent, supervisor_fn=_unused_supervisor))

    events = [
        event for event in LoopStateStore(str(workspace)).read_events() if event.get("type") == "iteration_result"
    ]
    assert [event.get("decision") for event in events] == ["KEEP"]
    assert "REVERT_IMPLAUSIBLE" not in json.dumps(events)


def test_a_candidate_that_was_merely_slow_advances_the_stall_streak():
    """A REVERT the search can learn from still counts against it."""
    state = RunState()

    apply_iteration(
        state,
        iteration=1,
        decision="REVERT_PERF",
        kept=False,
        commit_hash="",
        wall_ms=1.0,
        mean_case_speedup=1.001,
        plan="a real but insufficient gain",
        baseline_wall_ms=1.0,
        best_wall_ms=1.0,
    )

    assert state.stall.no_improvement_iters == 1


# ── Guard 2: best_ms < baseline_ms invariant ──────────────────────────────────


def test_a_result_slower_than_baseline_is_not_reported_as_improved():
    """A landed report claimed speedup 1.211 while being 1.93x slower overall.

    The score is an equal-weight mean of per-case speedups, so two cheap winners
    outvoted one collapsing expensive case and nothing checked the wall times.
    """
    detail = aggregate_regression_detail(
        baseline_ms=0.0303,
        best_ms=0.0586,
        mean_case_speedup=1.211,
    )
    assert detail
    assert "0.0586" in detail and "0.0303" in detail


def test_a_consistent_result_records_no_aggregate_regression():
    assert not aggregate_regression_detail(
        baseline_ms=0.0586,
        best_ms=0.0303,
        mean_case_speedup=1.211,
    )


def test_an_unmeasured_aggregate_is_not_reported_as_a_regression():
    """A run with no best yet must stay silent rather than invent a violation."""
    assert not aggregate_regression_detail(
        baseline_ms=0.0303,
        best_ms=None,
        mean_case_speedup=None,
    )
    assert not aggregate_regression_detail(
        baseline_ms=None,
        best_ms=0.0586,
        mean_case_speedup=1.211,
    )


def test_a_result_that_never_claimed_improvement_is_not_flagged():
    """REVERT-only runs legitimately report the pristine time as the best."""
    assert not aggregate_regression_detail(
        baseline_ms=0.0303,
        best_ms=0.0303,
        mean_case_speedup=1.0,
    )


def test_a_warm_start_slower_in_aggregate_claims_no_improvement():
    """A warm start ships a result JSON, a checkpoint and a manifest.

    The manifest already withheld the badge on the aggregate invariant while
    the other two hardcoded it, so the same run answered "did this improve?"
    differently depending on which artifact was read.
    """
    flags = warm_start_improvement_flags(
        pristine_ms=0.0303,
        best_ms=0.0586,
        mean_case_speedup=1.211,
    )

    assert flags["improved"] is False
    assert flags["total_improved"] is False
    assert "is not faster than the pristine baseline" in flags["aggregate_regression"]


def test_a_warm_start_faster_in_aggregate_keeps_its_improvement():
    flags = warm_start_improvement_flags(
        pristine_ms=10.0,
        best_ms=5.0,
        mean_case_speedup=2.0,
    )

    assert flags == {
        "aggregate_regression": "",
        "improved": True,
        "total_improved": True,
    }


def test_a_warm_start_that_gained_nothing_claims_nothing():
    """Adopting a prior solution that only matched pristine is not an improvement."""
    flags = warm_start_improvement_flags(
        pristine_ms=10.0,
        best_ms=10.0,
        mean_case_speedup=1.0,
    )

    assert flags["improved"] is False
    assert flags["total_improved"] is False
    assert flags["aggregate_regression"] == ""


def test_a_warm_start_without_a_pristine_aggregate_refuses_to_adopt(
    monkeypatch,
    tmp_path,
):
    """The aggregate gate must not go quiet when it has nothing to compare to.

    ``aggregate_regression_detail`` reports no contradiction when either wall
    time is unknown, which is right for a run holding no best yet but wrong as
    an adoption verdict: a pristine bench missing its aggregate would let this
    candidate through on a silent "" rather than on a comparison. The per-case
    half of the same measurement is already mandatory, so both halves are.
    """
    pristine, _ = _floored_case_times()
    candidate = {case_id: baseline_ms / MODEST_SPEEDUP for case_id, baseline_ms in pristine.items()}
    discarded: list[str] = []
    monkeypatch.setattr(integration, "_apply_candidate_patch", lambda *_a, **_k: "")
    monkeypatch.setattr(integration, "_force_jit_rebuild", lambda *_a, **_k: None)
    monkeypatch.setattr(integration, "_correctness_once", lambda *_a, **_k: True)
    monkeypatch.setattr(
        integration,
        "_bench_once",
        lambda *_a, **_k: {
            "success": True,
            "median_ms": TIMING_FLOOR_MS,
            "case_times": dict(candidate),
            "unscored_cases": [],
        },
    )
    monkeypatch.setattr(
        integration,
        "_git_discard_worktree",
        lambda workspace, **_k: discarded.append(str(workspace)),
    )

    trial = integration._try_apply_candidate(
        {"solution_slug": "prior-solution"},
        kernel=str(tmp_path / "kernel.py"),
        driver=str(tmp_path / "driver.py"),
        workspace_dir=str(tmp_path),
        snr_threshold=30.0,
        source_files=None,
        pristine_bench={"case_times": pristine},
        allowed_paths=None,
        pre_untracked=set(),
    )

    # Named apart from aggregate_regression: nothing was compared at all.
    assert trial.reject_reason == "pristine_aggregate_missing"
    assert trial.adoptable_ms is None
    assert trial.adoptable_mean_case_speedup is None
    assert trial.adoptable_bench is None
    # The suite still ran, so the record this candidate came from is still owed
    # the amendment; only the adoption is refused.
    assert trial.measured_mean_case_speedup == pytest.approx(MODEST_SPEEDUP)
    assert discarded == [str(tmp_path)]


def _committed_warm_start_workspace(tmp_path) -> tuple[str, str]:
    """A repo whose HEAD is one adopted warm-start patch past its base."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for command in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "guards@example.com"],
        ["git", "config", "user.name", "Guards"],
    ):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("pristine\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "pristine"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    kernel.write_text("prior solution\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return str(workspace), base_commit


def test_every_warm_start_artifact_agrees_on_the_aggregate_verdict(tmp_path):
    """The manifest, the checkpoint and the caller's result come from one run.

    The manifest withheld the badge on the aggregate invariant while the other
    two hardcoded it, so which artifact a reader opened decided the answer.
    """
    workspace, base_commit = _committed_warm_start_workspace(tmp_path)
    checkpoints: dict[str, dict] = {}

    class Tracker:
        @staticmethod
        def set_checkpoint(experiment_id: str, checkpoint: dict) -> None:
            checkpoints[experiment_id] = checkpoint

    result_json = tmp_path / "caller-result.json"
    result = publish_warm_start_recovery(
        workspace_dir=workspace,
        base_commit=base_commit,
        warm={
            "applied": True,
            "pristine_ms": 0.0303,
            "keep_baseline_ms": 0.0586,
            "mean_case_speedup": 1.211,
            "case_times": {"cheap": 0.001, "expensive": 0.0576},
            "solution_slug": "prior-solution",
        },
        caller_experiment_id="consumer-run",
        experience_id="producer-run",
        tracker=Tracker(),
        result_json=str(result_json),
    )

    manifest = json.loads((Path(workspace) / "forge_experiments" / "best" / "manifest.json").read_text())
    checkpoint = checkpoints["consumer-run"]
    verdicts = [
        manifest["total_improved"],
        checkpoint["improved"],
        checkpoint["total_improved"],
        result["improved"],
        result["total_improved"],
        json.loads(result_json.read_text())["improved"],
    ]

    assert verdicts == [False] * len(verdicts)
    assert "is not faster than the pristine baseline" in (checkpoint["aggregate_regression"])
    assert checkpoint["aggregate_regression"] == manifest["aggregate_regression"]


# ── Guard 3: pristine baseline vs the task's shipped reference ────────────────


# The operator-facing name of the drift override. Every runbook that widens the
# tolerance for a machine the reference was not measured on types this string.
DRIFT_TOLERANCE_ENV = "FORGE_BASELINE_DRIFT_TOLERANCE"


def _write_reference(workspace, cases: dict[str, float]) -> None:
    lines = ["test_cases:"]
    for case_id, ms in cases.items():
        lines.extend(
            [
                f"- test_case_id: {case_id}",
                f"  execution_time_ms: {ms!r}",
                "  shape:",
                "  - (64,8,128) bf16",
            ]
        )
    (workspace / "baseline_perf.yaml").write_text("\n".join(lines) + "\n")


def _write_partly_readable_reference(
    workspace,
    readable: dict[str, float],
    unreadable: list[str],
) -> None:
    """A reference where only ``readable`` carries the timing field.

    The ``unreadable`` entries misspell ``execution_time_ms``, which is what a
    file written against a schema no sample in this repo pins looks like.
    """
    lines = ["test_cases:"]
    for case_id, ms in readable.items():
        lines.extend(
            [
                f"- test_case_id: {case_id}",
                f"  execution_time_ms: {ms!r}",
            ]
        )
    for case_id in unreadable:
        lines.extend(
            [
                f"- test_case_id: {case_id}",
                "  execution_time_msec: 1.0",
            ]
        )
    (workspace / "baseline_perf.yaml").write_text("\n".join(lines) + "\n")


def test_baseline_drift_from_the_shipped_reference_fails_loudly(tmp_path):
    """Forge measured 0.162733 ms where the reference says 0.043476 ms.

    A 3.7x inflated denominator inflates every ratio in the run; the other ten
    kernels that day were within 1% of their medians, so it was the timing path
    degrading from CUDA-graph to per-launch event timing, not the machine.
    """
    _write_reference(tmp_path, {"vllm-verified-mhc-fused-k001": 0.043476})

    with pytest.raises(BaselineReferenceError) as excinfo:
        check_baseline_against_reference(
            str(tmp_path),
            {"vllm-verified-mhc-fused-k001": 0.162733},
        )

    assert "0.162733" in str(excinfo.value)
    assert "0.043476" in str(excinfo.value)


def test_a_baseline_within_tolerance_is_accepted(tmp_path):
    _write_reference(tmp_path, {"case": 1.0})
    within = 1.0 * (1.0 + BASELINE_DRIFT_TOLERANCE * 0.9)

    check_baseline_against_reference(str(tmp_path), {"case": within})


def test_an_absent_reference_never_breaks_a_run(tmp_path):
    """Most task workspaces ship no reference; that must stay a no-op."""
    assert load_reference_case_times(str(tmp_path)) is None

    check = check_baseline_against_reference(str(tmp_path), {"case": 1.0})

    assert "ships no" in check.unverified_reason


def test_a_checked_baseline_reports_how_many_cases_backed_it(tmp_path):
    """Only 5 of the 36 kernels in daily CI ship a reference.

    An inactive check and a passing check are indistinguishable to an operator
    unless the count comes back, so the caller can name which one happened.
    """
    _write_reference(tmp_path, {"case-a": 1.0, "case-b": 2.0, "unrelated": 9.0})

    checked = check_baseline_against_reference(str(tmp_path), {"case-a": 1.0, "case-b": 2.0})

    assert checked.compared_case_count == 2


def test_reference_case_ids_follow_the_driver_underscore_convention(tmp_path):
    """Drivers emit ``case_ms: <id with spaces replaced by underscores>``."""
    _write_reference(tmp_path, {"decode graph k001": 0.2777})

    assert load_reference_case_times(str(tmp_path)).case_times == {"decode_graph_k001": 0.2777}


def test_an_unusable_reference_is_not_silently_ignored(tmp_path):
    """A present-but-unreadable reference would disable the check invisibly.

    It leaves the anchor unverified, which is what a missing file leaves it, so
    it does not end the run -- but it has to be named, because an inactive check
    reads to an operator exactly like a check that passed.
    """
    (tmp_path / "baseline_perf.yaml").write_text("test_cases: []\n")

    check = check_baseline_against_reference(str(tmp_path), {"case": 1.0})

    assert "no usable test case" in check.unverified_reason


def test_a_reference_sharing_no_case_with_the_run_is_loud(tmp_path):
    """Naming none of this run's cases means the check could not run at all.

    That is not evidence the baseline drifted, and the case ids come from a
    schema this repository does not produce, so refusing to start would put a
    whole campaign behind a naming mismatch nothing here can validate.
    """
    _write_reference(tmp_path, {"other-kernel-k001": 1.0})

    check = check_baseline_against_reference(str(tmp_path), {"case": 1.0})

    assert "names no case this run measured" in check.unverified_reason
    assert "other-kernel-k001" in check.unverified_reason
    assert "case" in check.unverified_reason


def test_a_partly_readable_reference_reports_the_entries_it_dropped(tmp_path):
    """Skipping 11 of 12 entries makes this a partial silent no-op.

    The surviving case still answers "the anchor was checked", so the entries
    that dropped out have to reach the caller: a check covering one twelfth of
    the file it was handed reads exactly like a check that passed.
    """
    _write_partly_readable_reference(
        tmp_path,
        {"k001": 1.0},
        [f"k{index:03d}" for index in range(2, 13)],
    )

    check = check_baseline_against_reference(str(tmp_path), {"k001": 1.0})

    assert len(check.unusable_entries) == 11
    assert any("k002" in entry for entry in check.unusable_entries)


def test_a_fully_unreadable_reference_names_every_entry_it_lost(tmp_path):
    """Nothing left to compare is a no-op, and it must not read as a thin check."""
    _write_partly_readable_reference(tmp_path, {}, ["k001", "k002"])

    check = check_baseline_against_reference(str(tmp_path), {"k001": 1.0})

    assert "k001" in check.unverified_reason
    assert check.compared_case_count == 0


def test_only_a_measured_disagreement_stops_the_run(tmp_path):
    """The asymmetry this whole check turns on, pinned in one place.

    A drift verdict is evidence the anchor is wrong and every speedup divided by
    it would be a lie, so it fails closed. Every way of failing to reach a
    verdict costs one layer of protection; refusing to start costs a twelve-hour
    campaign at second zero.
    """
    _write_reference(tmp_path, {"case": 1.0})

    with pytest.raises(BaselineReferenceError):
        check_baseline_against_reference(str(tmp_path), {"case": 10.0})

    for workspace in (tmp_path / "absent", tmp_path / "empty", tmp_path / "other"):
        workspace.mkdir()
    (tmp_path / "empty" / "baseline_perf.yaml").write_text("test_cases: []\n")
    _write_reference(tmp_path / "other", {"a-case-nobody-measured": 1.0})

    assert all(
        check_baseline_against_reference(str(tmp_path / name), {"case": 1.0}).unverified_reason
        for name in ("absent", "empty", "other")
    )


def test_a_check_reports_the_coverage_behind_its_numerator(tmp_path):
    """One backed case out of twelve measured is not a verified anchor.

    Every case the reference does not name divides its own speedups by an
    unchecked denominator, so the count of compared cases is meaningless
    without the count of measured ones.
    """
    _write_reference(tmp_path, {"k001": 1.0})
    measured = {f"k{index:03d}": 1.0 for index in range(1, 13)}

    check = check_baseline_against_reference(str(tmp_path), measured)

    assert check.compared_case_count == 1
    assert check.measured_case_count == 12
    assert check.reference_case_count == 1


def test_a_drifted_baseline_names_the_way_to_proceed(tmp_path):
    """The reference was measured on one machine and one image.

    A different GPU SKU can exceed the tolerance with nothing wrong, and the
    failure aborts the campaign at startup, so the message has to name the
    override instead of leaving the operator to grep for one.
    """
    _write_reference(tmp_path, {"case": 1.0})

    with pytest.raises(BaselineReferenceError) as excinfo:
        check_baseline_against_reference(str(tmp_path), {"case": 1.4})

    assert DRIFT_TOLERANCE_ENV in str(excinfo.value)


def test_a_widened_drift_tolerance_is_honored_and_reported(tmp_path, monkeypatch):
    _write_reference(tmp_path, {"case": 1.0})
    monkeypatch.setenv(DRIFT_TOLERANCE_ENV, "0.5")

    check = check_baseline_against_reference(str(tmp_path), {"case": 1.4})

    assert check.drift_tolerance == 0.5
    assert check.tolerance_overridden


def test_an_unreadable_drift_override_fails_instead_of_defaulting(
    tmp_path,
    monkeypatch,
):
    """An override that quietly does nothing is the failure mode being fixed."""
    _write_reference(tmp_path, {"case": 1.0})
    monkeypatch.setenv(DRIFT_TOLERANCE_ENV, "40%")

    with pytest.raises(BaselineReferenceError) as excinfo:
        check_baseline_against_reference(str(tmp_path), {"case": 1.0})

    assert DRIFT_TOLERANCE_ENV in str(excinfo.value)


def test_the_default_drift_tolerance_stays_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv(DRIFT_TOLERANCE_ENV, raising=False)
    _write_reference(tmp_path, {"case": 1.0})

    check = check_baseline_against_reference(str(tmp_path), {"case": 1.0})

    assert check.drift_tolerance == BASELINE_DRIFT_TOLERANCE
    assert not check.tolerance_overridden


def test_loop_startup_rejects_a_drifted_pristine_baseline(tmp_path, monkeypatch):
    """Fail before the agent burns the budget optimizing against a bad anchor."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _write_reference(workspace, {"case": 0.25})

    with pytest.raises(BaselineReferenceError):
        asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))


def test_loop_startup_accepts_a_matching_pristine_baseline(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _write_reference(workspace, {"case": 1.02})

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ] == ["NO_CHANGES"]


def test_loop_startup_says_when_the_anchor_is_unverified(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A run against an unverified anchor must not look like a checked one."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert "the pristine anchor every speedup divides by is unverified" in (capsys.readouterr().out)


def test_loop_startup_reports_how_many_of_its_cases_the_reference_backed(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A numerator with no denominator reads like full coverage."""
    measured = {f"k{index:03d}": 1.0 for index in range(1, 13)}
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=measured)
    _write_reference(workspace, {"k001": 1.0})

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert "agrees with the task reference on 1 of 12 measured case(s)" in (capsys.readouterr().out)


def test_loop_startup_names_the_reference_entries_it_could_not_read(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A thinned-out reference must not print as a reference that was read."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _write_partly_readable_reference(workspace, {"case": 1.0}, ["ghost"])

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert "could not read 1 of the 2 entries" in capsys.readouterr().out


def test_loop_startup_announces_a_widened_drift_tolerance(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The escape hatch has to be as visible as the failure it suppresses."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _write_reference(workspace, {"case": 1.4})
    monkeypatch.setenv(DRIFT_TOLERANCE_ENV, "0.5")

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    output = capsys.readouterr().out
    assert "drift tolerance widened to 50%" in output
    assert DRIFT_TOLERANCE_ENV in output


# ── Guard 5: which case set the bar ───────────────────────────────────────────
#
# The KEEP rule below is unchanged. What changed is the estimate it is charged
# to: sigma was taken over the aggregate score, so a 10 us case supplying 87% of
# it set the bar for everything, and the same 0.92% gain was kept once and
# reverted once purely on which side of a 0.32%-8.42% range that case drew.

# The 2026-08 GQA campaign's pristine per-case baselines, from its run_state.json.
GQA_BASELINE = {
    "m3-decode-q61": 0.0871,
    "m3-prefill-b2-q8131p60": 0.716913,
    "m3-prefill-b2-q8073p60": 0.7232,
}
# A typical candidate on that suite: ~8.4x on the 10 us decode case, ~1.02x on
# the two prefills that carry 94% of the wall time.
GQA_CANDIDATE = {
    "m3-decode-q61": 0.010369,
    "m3-prefill-b2-q8131p60": 0.700000,
    "m3-prefill-b2-q8073p60": 0.706000,
}
# Median relative spreads decomposed from that run's archived measurement
# groups: 1.15% on the decode case against 0.30% and 0.33% on the prefills.
GQA_SPREAD = {
    "m3-decode-q61": 0.0115,
    "m3-prefill-b2-q8131p60": 0.0030,
    "m3-prefill-b2-q8073p60": 0.0033,
}


def _gqa_runs(scale: float = 1.0, *, level: float = 1.0) -> list[dict[str, float]]:
    """Three measurements of the GQA candidate at ``level`` times its spread."""
    return [
        {
            case_id: GQA_CANDIDATE[case_id] * scale * (1.0 + step * level * GQA_SPREAD[case_id])
            for case_id in GQA_CANDIDATE
        }
        for step in (-1.0, 0.0, 1.0)
    ]


# An incumbent inside the near-miss band of the default `_gqa_runs()` profile:
# the candidate's three scores mean 3.483106 and the measured sigma draws a bar
# of t * sigma / sqrt(3) = 0.057921 over the incumbent, so sigma decides the
# verdict only for 3.4252 < incumbent < 3.4814. That band is the only state a
# re-measure is bought in: under it the candidate is already a KEEP at the
# measured sigma and a second draw can only take that away, over it it reverts
# at every sigma including zero, where the floor alone still refuses it. Every
# test that exercises the purchase therefore starts its loop here. 3.45 rather
# than anywhere in the band because it also leaves room for the sigma a quiet
# re-measure comes back with to put the bar low enough to change the verdict
# and not merely to move it. The band is a property of the profile and not a
# constant of the gate.
GQA_NEAR_MISS_INCUMBENT = 3.45


def _bench(runs: list[dict[str, float]]) -> dict:
    return {
        "success": True,
        "median_ms": statistics.fmean(sum(run.values()) for run in runs),
        "case_times": {case_id: statistics.fmean([run[case_id] for run in runs]) for case_id in runs[0]},
        "unscored_cases": [],
        "measurement_count": len(runs),
        "measurements": [{"success": True, "case_times": dict(run), "unscored_cases": []} for run in runs],
        "message": "three measurements",
    }


def _case_scores(runs: list[dict[str, float]], baseline: dict[str, float]) -> list[float]:
    return [sum(baseline[case_id] / run[case_id] for case_id in baseline) / len(baseline) for run in runs]


def _attributed_loop(monkeypatch, baseline, runs, extra_rounds=()):
    """A measurement loop whose re-measure rounds return ``extra_rounds`` in turn.

    With no ``extra_rounds`` a re-measure returns the same profile again, which
    is what a case that is simply that noisy looks like.
    """
    loop, calls = _measurement_loop(monkeypatch, _bench(runs))
    loop._baseline_case_times = dict(baseline)
    loop._best_case_times = dict(baseline)
    rounds = [list(extra) for extra in extra_rounds]

    async def fake_benchmark(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _bench(runs)
        if not rounds:
            return _bench(runs)
        return _bench(rounds[min(len(calls) - 2, len(rounds) - 1)])

    monkeypatch.setattr(runner_module, "measure_wallclock", fake_benchmark)
    return loop, calls


def _bench_line(capsys) -> str:
    return next(line for line in capsys.readouterr().out.splitlines() if "[bench] pristine-relative scores=" in line)


# Four suites whose per-case noise is uniform in the sense that matters: no
# single case supplies a majority of the objective's variance while carrying
# less than its equal share of the wall time. Every one of them must produce
# the number the gate produced before per-case attribution existed.
UNIFORM_NOISE_SHAPES = [
    (
        "two equal cases moving together",
        {"small": 1.0, "large": 1.0},
        [{"small": 1.0 / score, "large": 1.0 / score} for score in (1.004, 1.006, 1.008)],
    ),
    (
        "three equal-cost cases with comparable spreads",
        {"a": 2.0, "b": 2.0, "c": 2.0},
        [
            {"a": 1.90, "b": 1.91, "c": 1.92},
            {"a": 1.92, "b": 1.93, "c": 1.90},
            {"a": 1.91, "b": 1.92, "c": 1.91},
        ],
    ),
    (
        "the noisy case is also the expensive one",
        {"cheap": 0.1, "heavy": 4.0},
        [
            {"cheap": 0.0500, "heavy": 3.0},
            {"cheap": 0.0501, "heavy": 3.2},
            {"cheap": 0.04995, "heavy": 2.9},
        ],
    ),
    (
        "a quiet cheap case beside a noisy big one",
        {"cheap": 0.05, "big": 1.0},
        [
            {"cheap": 0.0400, "big": 0.800},
            {"cheap": 0.04002, "big": 0.812},
            {"cheap": 0.03999, "big": 0.795},
        ],
    ),
    (
        # One case holds all of the variance and all of the wall time, so a
        # single-case suite can never buy a bench to sharpen itself against.
        "a single-case suite",
        {"only": 1.0},
        [{"only": 0.900}, {"only": 0.912}, {"only": 0.895}],
    ),
    (
        "four cases, the cheapest the noisiest without holding a majority",
        {"tiny": 0.05, "small": 0.4, "mid": 1.0, "big": 2.0},
        [
            {"tiny": 0.0400, "small": 0.300, "mid": 0.800, "big": 1.60},
            {"tiny": 0.0404, "small": 0.303, "mid": 0.806, "big": 1.63},
            {"tiny": 0.0397, "small": 0.298, "mid": 0.795, "big": 1.58},
        ],
    ),
]


@pytest.mark.parametrize(
    "shape, baseline, runs",
    UNIFORM_NOISE_SHAPES,
    ids=[shape for shape, _baseline, _runs in UNIFORM_NOISE_SHAPES],
)
def test_uniform_per_case_noise_reproduces_todays_bar_exactly(monkeypatch, capsys, shape, baseline, runs):
    """The no-op case. Nothing is bought, nothing moves, no verdict changes."""
    loop, calls = _attributed_loop(monkeypatch, baseline, runs)

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, baseline)

    assert len(calls) == 1
    assert f"sigma={measurement_sigma(scores):.6f}" in line
    assert f"required={required_keep_speedup(1.0, scores):.6f}x" in line
    assert "sigma attributed" not in line
    # The split was established and came out even; it did not fail to establish.
    assert "not attributed" not in line
    assert result.kept is passes_keep_threshold(scores, best_mean_case_speedup=1.0)


def test_per_case_times_that_resolve_no_split_say_so_rather_than_fall_back_quietly(monkeypatch, capsys):
    """A degraded estimate is still the aggregate one, and must not read as the new path.

    Three byte-identical runs leave no variance to divide, so attribution
    declines. The bar is then today's floor-driven bar, which is correct -- but
    a reader who cannot tell "no case dominated" from "the split could not be
    taken" cannot tell a healthy suite from a driver whose resolution swallowed
    every case.
    """
    runs = [dict(GQA_CANDIDATE) for _ in range(KEEP_MEASUREMENT_COUNT)]
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs)

    asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, GQA_BASELINE)

    assert len(calls) == 1
    assert "sigma not attributed per case" in line
    assert f"required={required_keep_speedup(1.0, scores):.6f}x" in line


def test_a_re_measure_bench_that_failed_is_reported_as_bought_not_as_measured(monkeypatch, capsys):
    """The worst outcome here is reporting an unmeasured thing as measured.

    The round paid for the bench either way, so it is reported as bought; its
    samples never reached the estimate, so the sample count and the sigma both
    stand where the three KEEP measurements left them.
    """
    runs = _gqa_runs()
    loop, calls = _measurement_loop(monkeypatch, _bench(runs))
    loop._baseline_case_times = dict(GQA_BASELINE)
    loop._best_case_times = dict(GQA_BASELINE)
    loop.best_mean_case_speedup = GQA_NEAR_MISS_INCUMBENT

    async def fake_benchmark(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _bench(runs)
        return {"success": False, "message": "driver aborted", "measurements": []}

    monkeypatch.setattr(runner_module, "measure_wallclock", fake_benchmark)

    asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, GQA_BASELINE)

    assert len(calls) == 2
    assert "bought 1 extra bench(es)" in line
    assert f"sigma over {KEEP_MEASUREMENT_COUNT} samples per case" in line
    assert "stopped early: re-measure bench failed" in line
    assert f"sigma={measurement_sigma(scores):.6f}" in line
    assert f"required={required_keep_speedup(GQA_NEAR_MISS_INCUMBENT, scores):.6f}x" in line


def test_the_extra_benches_are_charged_to_the_round_that_bought_them(monkeypatch):
    """Round admission prices the next round from what this one spent measuring.

    A re-measure the budget cannot see would let one iteration buy three
    whole-suite benches while telling the admission check that an iteration
    costs one, and the campaign would keep dispatching rounds it cannot finish.
    """
    clock = [1000.0]
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(time=lambda: clock[0]))

    def charged(baseline, runs, extra_rounds=(), *, incumbent=None):
        loop, calls = _attributed_loop(monkeypatch, baseline, runs, extra_rounds)
        if incumbent is not None:
            loop.best_mean_case_speedup = incumbent
        inner = runner_module.measure_wallclock

        async def metered(**kwargs):
            clock[0] += 100.0
            return await inner(**kwargs)

        monkeypatch.setattr(runner_module, "measure_wallclock", metered)
        loop._round_started_at = clock[0]
        loop._round_measurement_sec = 0.0
        asyncio.run(loop.run_one_iteration(1))
        return len(calls), loop._round_measurement_sec

    quiet = _gqa_runs(level=0.02)
    # The uniform shape needs no incumbent: no case dominates its sigma, so the
    # purchase is declined before the verdict band is consulted at all.
    plain_benches, plain_sec = charged(*UNIFORM_NOISE_SHAPES[1][1:])
    bought_benches, bought_sec = charged(
        GQA_BASELINE,
        _gqa_runs(),
        [quiet, quiet],
        incumbent=GQA_NEAR_MISS_INCUMBENT,
    )

    assert (plain_benches, plain_sec) == (1, pytest.approx(100.0))
    assert bought_benches == 1 + SIGMA_REMEASURE_MAX_ROUNDS
    assert bought_sec == pytest.approx(100.0 * bought_benches)


def test_a_cheap_dominant_case_is_re_measured_and_the_bar_comes_down(monkeypatch, capsys):
    """q61 held 87% of sigma on 5.7% of the wall time; six more runs settled it.

    Settled it in the verdict's sense, not only the bar's: the three scores the
    aggregate estimate refused clear the bar the nine-sample per-case estimate
    draws over the same incumbent. The scores themselves never move -- the six
    bought runs are data about the spread and are never admitted as evidence of
    a gain.
    """
    runs = _gqa_runs()
    quiet = _gqa_runs(level=0.02)
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs, [quiet, quiet])
    # An incumbent just under the candidate's weakest score: a candidate the
    # re-measure is for, and the only kind that pays for one.
    loop.best_mean_case_speedup = GQA_NEAR_MISS_INCUMBENT

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, GQA_BASELINE)
    aggregate_bar = required_keep_speedup(GQA_NEAR_MISS_INCUMBENT, scores)

    assert len(calls) == 1 + SIGMA_REMEASURE_MAX_ROUNDS
    assert "sigma attributed to case 'm3-decode-q61'" in line
    assert f"bought {SIGMA_REMEASURE_MAX_ROUNDS} extra bench(es)" in line
    assert (
        f"sigma over "
        f"{KEEP_MEASUREMENT_COUNT + SIGMA_REMEASURE_MAX_ROUNDS * SIGMA_REMEASURE_BATCH}"
        " samples per case" in line
    )
    assert "did not lower its spread" not in line
    bar = float(line.split("required=")[1].split("x;")[0])
    # The aggregate estimate put the bar out of this candidate's reach; the
    # per-case one, taken over nine samples, brings it back under every score.
    assert aggregate_bar > min(scores)
    assert bar < aggregate_bar
    # The rule is untouched: the bar is still the incumbent plus k sigma.
    assert bar > GQA_NEAR_MISS_INCUMBENT
    assert result.kept is True


def test_a_case_that_stays_unstable_keeps_the_inflated_bar_and_says_so(monkeypatch, capsys):
    """Genuinely unstable, not merely cheap. The honest outcome is to say so."""
    runs = _gqa_runs()
    wilder = _gqa_runs(level=2.5)
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs, [wilder, wilder])
    # An incumbent in the band where sigma still decides, so the bar -- and only
    # the bar -- decides.
    loop.best_mean_case_speedup = GQA_NEAR_MISS_INCUMBENT

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, GQA_BASELINE)

    assert len(calls) == 1 + SIGMA_REMEASURE_MAX_ROUNDS
    assert "did not lower its spread" in line
    assert "inflated by one case rather than by this candidate" in line
    bar = float(line.split("required=")[1].split("x;")[0])
    assert bar > required_keep_speedup(GQA_NEAR_MISS_INCUMBENT, scores)
    assert statistics.fmean(scores) > GQA_NEAR_MISS_INCUMBENT
    assert result.kept is False


def test_the_re_measure_loop_terminates_on_a_pathologically_noisy_case(monkeypatch, capsys):
    """Every round comes back worse; the bound, not convergence, ends it."""
    runs = _gqa_runs()
    rounds = [_gqa_runs(level=level) for level in (8.0, 40.0, 200.0)]
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs, rounds)
    # A near-miss incumbent, so the purchase is made at all. There are more
    # rounds staged here than the bound allows, and each is wilder than the one
    # before, so nothing but the bound can stop this.
    loop.best_mean_case_speedup = GQA_NEAR_MISS_INCUMBENT

    result = asyncio.run(loop.run_one_iteration(1))

    assert len(calls) == 1 + SIGMA_REMEASURE_MAX_ROUNDS
    assert result.kept is False
    assert "did not lower its spread" in _bench_line(capsys)


def test_a_candidate_that_cannot_be_kept_at_any_sigma_buys_no_measurements(monkeypatch, capsys):
    """The bar is always strictly above the incumbent, so this is cost, not policy."""
    runs = _gqa_runs(scale=2.0)
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs)
    loop.best_mean_case_speedup = 4.0

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)

    assert len(calls) == 1
    assert "reverted at every sigma" in line
    assert result.kept is False


def test_a_candidate_already_clearing_the_bar_buys_no_measurements(monkeypatch, capsys):
    """Above the bar sigma is not deciding, it is being drawn a second time.

    Replaying 1240 archived candidates, the floor-only gate charged 28% of them
    for the estimate while only 6% could gain from it; the difference is
    entirely candidates in this state, which cannot be helped and can only be
    taken away.
    """
    runs = _gqa_runs()
    loop, calls = _attributed_loop(monkeypatch, GQA_BASELINE, runs)
    loop.best_mean_case_speedup = 1.0

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)

    assert len(calls) == 1
    assert "kept at the measured sigma" in line
    assert result.kept is True


def test_an_aggregate_gain_carried_by_a_regressing_case_is_untouched(monkeypatch, capsys):
    """A 2.5x win paid for with a 0.6x collapse. Nothing here is about sigma.

    Attribution re-estimates the objective's spread; it never re-decides which
    cases the objective is taken over. The regressed case is averaged in at full
    weight before and after, and the verdict is the aggregate rule's.
    """
    baseline = {"won": 4.25, "lost": 1.0}
    runs = [
        {"won": 1.700, "lost": 1.600},
        {"won": 1.704, "lost": 1.610},
        {"won": 1.697, "lost": 1.595},
    ]
    loop, calls = _attributed_loop(monkeypatch, baseline, runs)

    result = asyncio.run(loop.run_one_iteration(1))
    line = _bench_line(capsys)
    scores = _case_scores(runs, baseline)

    assert len(calls) == 1
    assert "sigma attributed" not in line
    assert f"required={required_keep_speedup(1.0, scores):.6f}x" in line
    assert result.kept is passes_keep_threshold(scores, best_mean_case_speedup=1.0)
    assert result.bench_detail["case_times"]["lost"] > baseline["lost"]
    assert scores == pytest.approx(result.bench_detail["measurement_mean_case_speedups"])


def _old_rule_passes(incumbent: float, scores: list[float]) -> bool:
    """``all(score >= best + 3 sigma)``, the rule the t test replaced."""
    margin = max(3.0 * measurement_sigma(scores), incumbent * KEEP_MIN_MARGIN_FRACTION)
    return all(score >= incumbent + margin for score in scores)


def test_the_gate_trades_no_false_accepts_for_the_power_it_gains():
    """The measurement the k = 3 calibration never made.

    That replay scored rules by false-accept rate on a zero-gain candidate and
    by whether recoveries were >= 3 sigma gains. Both are one-sided: a stricter
    rule wins the first by construction, and the second makes 3 sigma its own
    ground truth. Neither can report a rule that is too strict, so neither
    measured power, and k = 3 survived being 1.78x stricter than the number it
    cited.

    Both rules are simulated here as the loop actually runs them, incumbent
    included -- the old one ratcheting on a minimum, the new one on a mean --
    because that offset is exactly what cancels.
    """
    rng = random.Random(20260824)
    sigma = 0.0017  # the 2026-08 batch's median relative spread

    def trial(true_gain_sigmas: float) -> tuple[bool, bool]:
        incumbent = [rng.gauss(1.0, sigma) for _ in range(KEEP_MEASUREMENT_COUNT)]
        true = 1.0 + true_gain_sigmas * sigma
        scores = [rng.gauss(true, sigma) for _ in range(KEEP_MEASUREMENT_COUNT)]
        return (
            _old_rule_passes(min(incumbent), scores),
            passes_keep_threshold(scores, best_mean_case_speedup=statistics.fmean(incumbent)),
        )

    trials = 20_000
    null = [trial(0.0) for _ in range(trials)]
    real = [trial(3.0) for _ in range(trials)]
    old_false = sum(old for old, _new in null) / trials
    new_false = sum(new for _old, new in null) / trials
    old_power = sum(old for old, _new in real) / trials
    new_power = sum(new for _old, new in real) / trials

    # Against a measured incumbent the two rules admit noise at the same rate:
    # the old rule's extra strictness was spent undoing its own minimum.
    assert abs(new_false - old_false) < 0.01
    # What it buys is the whole of the change. A 3 sigma true gain -- 0.51% on
    # this kernel -- went from a coin flip to near certain.
    assert old_power < 0.65 < 0.85 < new_power


def test_the_one_place_the_new_rule_is_looser_is_the_first_keep():
    """Against pristine there is no minimum to cancel, so the level is nominal.

    Before the first KEEP the incumbent is the pristine 1.0 exactly, by
    construction rather than by measurement. The old rule's minimum offset was
    on one side only, which is where its cited 1.05% false-accept figure came
    from; the t test charges its nominal 5%, less whatever the floor clips off
    the tail. At the 0.17% sigma used here the 0.1% floor takes it to 3.9%. This
    is stated rather than fixed: it is one decision per campaign, it is the
    decision the campaign is least able to make without it, and a false one
    raises the incumbent onto a noisy mean that every later candidate then has
    to beat.
    """
    rng = random.Random(20260825)
    sigma = 0.0017
    trials = 20_000
    old_false = new_false = 0
    for _ in range(trials):
        scores = [rng.gauss(1.0, sigma) for _ in range(KEEP_MEASUREMENT_COUNT)]
        old_false += _old_rule_passes(1.0, scores)
        new_false += passes_keep_threshold(scores, best_mean_case_speedup=1.0)

    assert old_false / trials == pytest.approx(0.0105, abs=0.004)
    assert new_false / trials == pytest.approx(0.039, abs=0.01)
    # Still under the nominal 5% the t test would charge on its own, because the
    # floor is what refuses the marginal draws here.
    assert new_false / trials < 0.05


def test_the_gqa_campaign_bar_stops_being_a_lottery():
    """The regression fixture: 23 candidates on the archived GQA noise profile.

    Every candidate here is identical -- same kernel, same true times, same
    noise. Only the draw differs. Under the aggregate sigma the bar they each
    face spans a factor of 52, which is how iteration 14 was reverted at +0.923%
    against a 2.14% bar while iteration 19 was kept at +0.914% against 0.32%.
    Attributing sigma to the case that supplies it and re-measuring collapses
    that spread, without moving the objective or the rule.
    """
    rng = random.Random(20260821)

    def draw(count):
        return {
            case_id: [GQA_CANDIDATE[case_id] * (1.0 + rng.gauss(0.0, GQA_SPREAD[case_id])) for _ in range(count)]
            for case_id in GQA_BASELINE
        }

    aggregate_bars: list[float] = []
    attributed_bars: list[float] = []
    for _ in range(23):
        series = draw(KEEP_MEASUREMENT_COUNT)
        extended = {case_id: list(times) for case_id, times in series.items()}
        scores = [
            sum(GQA_BASELINE[case_id] / series[case_id][index] for case_id in GQA_BASELINE) / len(GQA_BASELINE)
            for index in range(KEEP_MEASUREMENT_COUNT)
        ]
        sigma = measurement_sigma(scores)
        # An incumbent a hair under the weakest score, so every candidate is a
        # contender and the bar is the only thing deciding it.
        incumbent = min(scores) * 0.999
        aggregate_bars.append((required_keep_speedup(incumbent, scores) - incumbent) / incumbent)

        base = attribute_sigma(series, GQA_BASELINE)
        assert base.dominant_case == "m3-decode-q61"
        current = base
        for _round in range(SIGMA_REMEASURE_MAX_ROUNDS):
            if current.dominant_case is None:
                break
            for case_id, times in draw(SIGMA_REMEASURE_BATCH).items():
                extended[case_id].extend(times)
            current = attribute_sigma(extended, GQA_BASELINE)
        refined = rescaled_sigma(sigma, base, current)
        attributed_bars.append((required_keep_speedup(incumbent, scores, sigma=refined) - incumbent) / incumbent)

    aggregate_range = max(aggregate_bars) / min(aggregate_bars)
    attributed_range = max(attributed_bars) / min(attributed_bars)

    assert aggregate_range > 20.0
    assert attributed_range < 5.0
    assert max(attributed_bars) < max(aggregate_bars)
    assert min(attributed_bars) > min(aggregate_bars)


# ── Guard 6: a rejected candidate that beat the incumbent on one case ─────────


def _revert_result(iteration: int, runs: list[dict[str, float]]) -> IterationResult:
    return IterationResult(
        iteration=iteration,
        duration_sec=1.0,
        validation_passed=True,
        validation_summary="PASS",
        mean_case_speedup=0.995,
        kept=False,
        bench_detail=_bench(runs),
    )


def _pin_after(runs: list[dict[str, float]], incumbent: dict[str, float]) -> list[int]:
    state = RunState()
    runner_module.IterationLoop._record_direction_verdict(
        state,
        iteration=21,
        decision_label="REVERT_PERF",
        mean_case_speedup=0.995,
        best_mean_case_speedup=1.0,
        bench_detail=_revert_result(21, runs).bench_detail,
        incumbent_case_times=incumbent,
    )
    return state.pinned_iterations


def test_a_revert_beating_the_incumbent_on_one_case_beyond_spread_is_pinned():
    """Iteration 21 won 2.0% on q8073 and left no trace under the aggregate test."""
    runs = [
        {
            "m3-decode-q61": GQA_CANDIDATE["m3-decode-q61"] * factor,
            "m3-prefill-b2-q8131p60": 0.700000 * factor,
            "m3-prefill-b2-q8073p60": 0.692000 * factor,
        }
        for factor in (0.9993, 1.0, 1.0007)
    ]
    incumbent = dict(GQA_CANDIDATE)

    assert _pin_after(runs, incumbent) == [21]
    # The gate is untouched: this candidate is a REVERT either way.
    scores = _case_scores(runs, GQA_BASELINE)
    assert not passes_keep_threshold(scores, best_mean_case_speedup=max(scores) + 1.0)


def test_a_revert_winning_only_inside_its_own_spread_is_not_pinned():
    """0.1% on a case whose own runs disagree by 0.28% is not a measurement."""
    runs = [
        {
            "m3-decode-q61": GQA_CANDIDATE["m3-decode-q61"],
            "m3-prefill-b2-q8131p60": 0.700000,
            "m3-prefill-b2-q8073p60": q8073,
        }
        for q8073 in (0.7033, 0.7053, 0.7073)
    ]
    incumbent = dict(GQA_CANDIDATE)

    assert _pin_after(runs, incumbent) == []
    scores = _case_scores(runs, GQA_BASELINE)
    assert not passes_keep_threshold(scores, best_mean_case_speedup=max(scores) + 1.0)


def test_a_candidate_with_no_per_case_detail_falls_back_to_the_aggregate_test():
    """A journal replay carries scalars only; it must not start pinning nothing."""
    state = RunState()
    runner_module.IterationLoop._record_direction_verdict(
        state,
        iteration=21,
        decision_label="REVERT_PERF",
        mean_case_speedup=0.995,
        best_mean_case_speedup=1.0,
    )

    assert state.pinned_iterations == []


def test_attribution_declines_rather_than_guesses():
    """Every uncertain input returns None, which is what keeps today's sigma."""
    baseline = {"a": 1.0, "b": 1.0}

    # One measurement: no spread was measured at all.
    assert attribute_sigma({"a": [0.5], "b": [0.5]}, baseline) is None
    # A case the baseline scores but this candidate never timed.
    assert attribute_sigma({"a": [0.5, 0.51]}, baseline) is None
    # Measurements of unequal length: the runs are not comparable groups.
    assert attribute_sigma({"a": [0.5, 0.51], "b": [0.5]}, baseline) is None
    # Three identical runs: a real zero, and nothing to attribute it to.
    assert attribute_sigma({"a": [0.5] * 3, "b": [0.5] * 3}, baseline) is None
    # A non-positive timing, which the scoring path already refuses to divide by.
    assert attribute_sigma({"a": [0.5, 0.0, 0.5], "b": [0.5] * 3}, baseline) is None


def test_a_case_that_carries_the_wall_time_is_never_re_measured():
    """Noise on the case the campaign is optimising is the objective's own noise."""
    baseline = {"cheap": 0.1, "heavy": 4.0}
    series = {"cheap": [0.05, 0.0501, 0.04995], "heavy": [3.0, 3.2, 2.9]}

    attribution = attribute_sigma(series, baseline)

    assert attribution.variance_shares["heavy"] > 0.9
    assert attribution.wall_shares["heavy"] > 0.5
    assert attribution.dominant_case is None


def test_a_larger_sample_can_raise_the_bar_as_well_as_lower_it():
    """Re-measuring sharpens the estimate; it does not lean on one direction."""
    baseline = {"cheap": 0.1, "big": 4.0}
    quiet = {"cheap": [0.0500, 0.05001, 0.04999], "big": [2.0, 2.001, 1.999]}
    base = attribute_sigma(quiet, baseline)
    wide = attribute_sigma(
        {case_id: list(times) + [times[1] * 1.05, times[1] * 0.95, times[1]] for case_id, times in quiet.items()},
        baseline,
    )

    assert rescaled_sigma(0.01, base, wide) > 0.01
    assert rescaled_sigma(0.01, wide, base) < 0.01
