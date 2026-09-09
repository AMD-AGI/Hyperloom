# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grading chokepoint reads candidate and reference off ONE axis, interactivity under AgentX."""

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_OUTPUT,
    VERDICT_KEEP,
    VERDICT_RECORDED,
    VERDICT_REVERT,
    graded_axes_of,
)
from hyperloom.orchestrator.state.shared_state import (
    ANCHOR_DEGRADED,
    resolve_graded_comparison,
    resolve_grading_anchor_tput,
)


# A measured AgentX round: prefill dominates, so the two axes cannot be confused.
# e2e_norm_intvty_p90 is P10 of per-request OSL/E2EL_s (the slow-tail users).
_ANCHOR = {
    "tput": 183.44,
    "input_throughput": 25801.36,
    "output_throughput": 183.44,
    "total_throughput": 25984.80,
    "e2e_norm_intvty_p90": 22.56,
}


class _State:
    """Minimal SharedState double: the attributes the resolvers read."""

    def __init__(
        self,
        *,
        current_best=None,
        baseline_perf=None,
        baseline_tput=0.0,
        framework="vllm",
        benchmark_mode="",
    ):
        self.current_best = current_best if current_best is not None else {}
        self.baseline_perf = baseline_perf if baseline_perf is not None else {}
        self.baseline_tput = baseline_tput
        self.framework = framework
        self.benchmark_mode = benchmark_mode


def _agentx(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


def _synthetic(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _full_measurement(*, total: float, output: float, intvty: float) -> dict[str, float]:
    return {
        "output_throughput": output,
        "input_throughput": total - output,
        "total_token_throughput": total,
        "e2e_norm_intvty_p90": intvty,
    }


# --- both sides, one axis ---


def test_agentx_reads_both_sides_on_the_intvty_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(
        state, _full_measurement(total=26500.0, output=190.0, intvty=24.0), keep_threshold_pct=2.0
    )

    assert graded.objective == GRADED_INTVTY
    assert graded.candidate == pytest.approx(24.0)  # intvty
    assert graded.reference == pytest.approx(_ANCHOR["e2e_norm_intvty_p90"])
    assert graded.degrade_reason == ""


def test_agentx_keep_verdict_when_intvty_clears_threshold(monkeypatch):
    """A candidate that improves interactivity >= 2% and tput within band is KEEP."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    # +10% interactivity, same tput
    graded = resolve_graded_comparison(
        state, _full_measurement(total=25984.0, output=183.0, intvty=24.8), keep_threshold_pct=2.0
    )
    assert graded.verdict == VERDICT_KEEP


def test_agentx_revert_when_both_axes_worse(monkeypatch):
    """Both interactivity AND tput regressed beyond the noise band -> REVERT."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(
        state, _full_measurement(total=20000.0, output=130.0, intvty=15.0), keep_threshold_pct=2.0
    )
    assert graded.verdict == VERDICT_REVERT


def test_agentx_recorded_when_neither_dominates(monkeypatch):
    """Intvty gain < 2% and tput within band -> RECORDED (not promoted)."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    # +1% intvty (below 2% floor), tput roughly same
    graded = resolve_graded_comparison(
        state, _full_measurement(total=25984.0, output=183.0, intvty=22.79), keep_threshold_pct=2.0
    )
    assert graded.verdict == VERDICT_RECORDED


def test_a_degraded_round_stays_on_the_output_axis(monkeypatch):
    """``ANCHOR_DEGRADED`` must not re-resolve the session anchor.

    A round degrades when a KEEP cannot supply the graded axes. Passing ``None``
    reads as "no anchor supplied" and falls back to ``current_best``, which
    would grade later variants on interactivity against the round's opening
    state while they stack on top of a KEEP graded on output.
    """
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    meas = _full_measurement(total=26000.0, output=190.0, intvty=30.0)

    graded = resolve_graded_comparison(state, meas, anchor_perf=ANCHOR_DEGRADED, anchor_tput=185.0)
    assert graded.objective == GRADED_OUTPUT
    assert graded.reference == pytest.approx(185.0)
    assert graded.degrade_reason == "round_degraded"

    # None keeps the old "not supplied" meaning.
    assert resolve_graded_comparison(state, meas, anchor_perf=None).objective == GRADED_INTVTY


def test_a_candidate_without_the_graded_axes_degrades_both_sides_together(monkeypatch):
    """Degrading one side alone would grade an output figure against interactivity."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, {"output_throughput": 190.0})

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == "candidate_axes_missing"


def test_an_anchor_without_the_graded_axes_degrades_both_sides_together(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best={"tput": 183.44}, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=24.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason


def test_a_synthetic_run_grades_output_against_output(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=24.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == ""


def test_a_scriptable_framework_keeps_output_grading_under_agentx(monkeypatch):
    """A scriptable framework reports an image-quality gate, not a total axis."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0, framework="xdit")
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=24.0))

    assert graded.objective == GRADED_OUTPUT


# --- the interactivity constraint travels with the objective ---


def test_an_interactivity_regression_with_tput_win_is_recorded(monkeypatch):
    """Intvty regresses BUT tput wins -> RECORDED (neither dominates the other).

    Under the 2-D rule REVERT requires BOTH axes to be worse.  A candidate
    that trades interactivity for throughput is 'RECORDED' — stored and visible,
    but not promoted to the optimization stack.  This mirrors the Pareto
    semantics: it is a different point on the frontier, not a dominated one.
    """
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    # Large tput win but interactivity crashes
    graded = resolve_graded_comparison(
        state, _full_measurement(total=40000.0, output=190.0, intvty=10.0), keep_threshold_pct=2.0
    )

    assert graded.graded_on_intvty
    assert graded.verdict == VERDICT_RECORDED  # not REVERT — tput won


def test_both_axes_worse_is_revert(monkeypatch):
    """Both interactivity AND throughput regress -> hard REVERT."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    # Both regress strongly beyond the 5% noise band
    graded = resolve_graded_comparison(
        state, _full_measurement(total=18000.0, output=100.0, intvty=10.0), keep_threshold_pct=2.0
    )
    assert graded.graded_on_intvty
    assert graded.verdict == VERDICT_REVERT


def test_a_collapsed_interactivity_does_not_block_a_synthetic_run(monkeypatch):
    """The interactivity axis belongs to the AgentX objective and only to it."""
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=40000.0, output=190.0, intvty=1.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.verdict == VERDICT_KEEP


# --- grading against the session baseline ---


def test_cumulative_gain_reads_the_baseline_on_the_graded_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(baseline_perf=_ANCHOR, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=24.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_INTVTY
    assert graded.reference == pytest.approx(_ANCHOR["e2e_norm_intvty_p90"])


def test_cumulative_gain_falls_back_to_baseline_tput_together(monkeypatch):
    """A resumed session that never wrote ``baseline_perf`` still grades one axis."""
    _agentx(monkeypatch)
    state = _State(baseline_perf={}, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=24.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason == "baseline_axes_missing"


# --- the anchor chokepoint stays on the output axis ---


@pytest.mark.parametrize("agentx", [True, False])
def test_the_anchor_chokepoint_is_the_output_axis_on_every_session(monkeypatch, agentx):
    """It seeds ``base_tput``, backs the drift check, and answers the two objective resolvers, whose targets are operator-supplied output figures."""
    _agentx(monkeypatch) if agentx else _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(_ANCHOR["tput"])


def test_the_anchor_falls_back_to_the_baseline_before_any_layer_lands(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best={}, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(180.0)


# --- a KEEP must not strip the axes off the anchor ---


def test_graded_axes_survive_a_winner_record(monkeypatch):
    """The defect this guards: AgentX grading dies after the first KEEP, because ``current_best`` is the next
    round's anchor and a winner record carrying no graded axes writes an anchor with none.
    """
    _agentx(monkeypatch)
    measurement = _full_measurement(total=26500.0, output=190.0, intvty=24.0)
    winner = {"name": "v", "tput": 190.0, **graded_axes_of(measurement)}

    graded = resolve_graded_comparison(_State(current_best=winner, baseline_tput=180.0), measurement)
    assert graded.objective == GRADED_INTVTY
    assert graded.reference == pytest.approx(24.0)


def test_graded_axes_of_omits_what_was_not_measured():
    assert graded_axes_of({"output_throughput": 190.0}) == {}
    assert graded_axes_of({"total_token_throughput": 26500.0}) == {"total_throughput": 26500.0}
    assert graded_axes_of(None) == {}


# --- the persisted marker reaches the chokepoint ---


def test_a_round_without_the_env_var_still_grades_on_intvty(monkeypatch):
    """A re-baseline or integrate round can be driven from a shell that never saw it; the seeded mode covers it."""
    _synthetic(monkeypatch)
    state = _State(current_best=dict(_ANCHOR), benchmark_mode="agentx")
    graded = resolve_graded_comparison(
        state, _full_measurement(total=27000.0, output=190.0, intvty=23.0), keep_threshold_pct=2.0
    )
    assert graded.objective == GRADED_INTVTY
    assert graded.candidate == pytest.approx(23.0)
    assert graded.reference == pytest.approx(_ANCHOR["e2e_norm_intvty_p90"])


def test_a_synthetic_session_is_untouched_by_the_marker_check(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=dict(_ANCHOR))
    graded = resolve_graded_comparison(state, _full_measurement(total=27000.0, output=190.0, intvty=23.0))
    assert graded.objective == GRADED_OUTPUT
    assert graded.reference == pytest.approx(resolve_grading_anchor_tput(state))
