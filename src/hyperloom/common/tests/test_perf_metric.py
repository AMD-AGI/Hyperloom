# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    output_tput_of,
    parse_intvty_noise_pct,
    passes_intvty_gate,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_grading_enabled,
    total_tput_of,
)

_KEEP_THRESHOLD_PCT = 1.0

# Shaped like a measured AgentX round: prefill dominates the token budget, so
# total is essentially input and output-only grading would see ~1% of it.
_BASELINE = {
    "input_throughput": 25801.36,
    "output_throughput": 183.44,
    "total_throughput": 25984.80,
    "intvty_p90": 447.20,
}


def _measured(**pct: float) -> dict[str, float]:
    """Baseline with the named axes scaled, leaving total to be re-derived."""
    out = {k: v for k, v in _BASELINE.items() if k != "total_throughput"}
    for axis, delta in pct.items():
        out[axis] = _BASELINE[axis] * (1.0 + delta / 100.0)
    return out


def _graded_gain(candidate: dict[str, float], anchor: dict[str, float]) -> float | None:
    """Compose the primitives the way the decision round does."""
    cand = perf_snapshot_from_mapping(candidate)
    base = perf_snapshot_from_mapping(anchor)
    assert cand and base
    if not passes_intvty_gate(cand, base):
        return None
    return gain_pct(total_tput_of(cand), total_tput_of(base))


def test_objective_is_total_token_throughput():
    snap = perf_snapshot_from_mapping(_BASELINE)
    assert snap is not None
    assert total_tput_of(snap) == pytest.approx(_BASELINE["total_throughput"])


def test_total_falls_back_to_input_plus_output():
    snap = perf_snapshot_from_mapping(_measured())
    assert snap is not None
    assert total_tput_of(snap) == pytest.approx(_BASELINE["input_throughput"] + _BASELINE["output_throughput"])


def test_snapshot_requires_both_graded_axes():
    assert perf_snapshot_from_mapping({"output_throughput": 1.0}) is None
    assert perf_snapshot_from_mapping({"total_throughput": 1.0}) is None
    assert perf_snapshot_from_mapping({"intvty_p90": 1.0}) is None
    assert perf_snapshot_from_mapping(None) is None


def test_degenerate_axis_is_not_a_snapshot():
    assert perf_snapshot_from_mapping({**_BASELINE, "intvty_p90": 0.0}) is None
    # A total that cannot be derived either.
    assert perf_snapshot_from_mapping({"intvty_p90": 447.2, "output_throughput": 183.44}) is None


def test_unusable_total_falls_back_to_input_plus_output():
    for bad in (0.0, -1.0, None, "n/a"):
        snap = perf_snapshot_from_mapping({**_BASELINE, "total_throughput": bad})
        assert snap is not None
        assert total_tput_of(snap) == pytest.approx(_BASELINE["input_throughput"] + _BASELINE["output_throughput"])


def test_total_lift_grades_like_a_throughput_delta():
    """The graded gain must be the plain percentage delta of the objective.

    This is what makes it comparable against ``keep_threshold_pct`` on the same
    terms run_grid and integrate_patch use for a throughput delta.
    """
    for pct in (1.0, 3.0, 10.0):
        candidate = _measured(input_throughput=pct, output_throughput=pct)
        assert _graded_gain(candidate, _BASELINE) == pytest.approx(pct)


def test_one_percent_lift_is_keepable():
    candidate = _measured(input_throughput=1.0, output_throughput=1.0)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and gain >= _KEEP_THRESHOLD_PCT


def test_sub_threshold_lift_is_refused_by_the_threshold_alone():
    candidate = _measured(input_throughput=0.5, output_throughput=0.5)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and 0.0 < gain < _KEEP_THRESHOLD_PCT


def test_trading_input_for_output_is_not_a_win():
    """Prefill carries the token budget, so giving it up cannot read as a gain.

    Output-only grading called this variant a +8% win.
    """
    candidate = _measured(input_throughput=-10.0, output_throughput=8.0)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and gain < 0.0
    assert gain_pct(candidate["output_throughput"], _BASELINE["output_throughput"]) == pytest.approx(8.0)


def test_intvty_gate_vetoes_regression_past_band():
    candidate = perf_snapshot_from_mapping(_measured(intvty_p90=-6.0))
    anchor = perf_snapshot_from_mapping(_BASELINE)
    assert candidate and anchor
    assert passes_intvty_gate(candidate, anchor) is False


def test_intvty_gate_allows_movement_within_band():
    candidate = perf_snapshot_from_mapping(_measured(intvty_p90=-4.0))
    anchor = perf_snapshot_from_mapping(_BASELINE)
    assert candidate and anchor
    assert passes_intvty_gate(candidate, anchor) is True


def test_vetoed_candidate_is_never_graded():
    candidate = _measured(intvty_p90=-20.0, input_throughput=50.0)
    assert _graded_gain(candidate, _BASELINE) is None


def test_default_band_matches_upstream_measured_noise(monkeypatch):
    """Upstream records run-to-run noise on this workload as 1-5%."""
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    assert parse_intvty_noise_pct() == pytest.approx(5.0)


@pytest.mark.parametrize("raw,expected", [("2.5", 2.5), ("0", 0.0), ("", 5.0), ("nonsense", 5.0)])
def test_band_env_override(monkeypatch, raw, expected):
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", raw)
    assert parse_intvty_noise_pct() == pytest.approx(expected)


def test_an_agentx_run_grades_on_total_without_being_asked(monkeypatch):
    """The corpus this grading exists for must not need an opt-in to get it."""
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert total_tput_grading_enabled() is True


def test_a_synthetic_run_still_grades_on_output(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert total_tput_grading_enabled() is False


def test_an_explicit_metric_overrides_the_agentx_default(monkeypatch):
    """The escape hatch has to work in both directions, not just to turn it on."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert total_tput_grading_enabled() is False


def test_an_explicit_metric_still_opts_a_synthetic_run_in(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    assert total_tput_grading_enabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "nonsense"])
def test_agentx_off_tokens_do_not_enable_grading(monkeypatch, raw):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", raw)
    assert total_tput_grading_enabled() is False


# --- resolve_grading_anchor_perf ---


class _State:
    """Minimal state double."""

    def __init__(self, current_best=None, baseline_perf=None):
        self.current_best = current_best
        self.baseline_perf = baseline_perf


def test_anchor_perf_uses_current_best_when_axes_present():
    state = _State(current_best=_BASELINE, baseline_perf={})
    snap, reason = resolve_grading_anchor_perf(state)
    assert reason == ""
    assert snap is not None
    assert snap["total_throughput"] == pytest.approx(_BASELINE["total_throughput"])


def test_anchor_perf_does_not_fall_through_to_baseline_when_current_best_lacks_axes():
    """A current_best without graded axes must not silently anchor on baseline_perf."""
    bad_best = {"action": "explore", "tput": 200.0}  # no intvty_p90 / total_throughput
    good_baseline = _BASELINE
    state = _State(current_best=bad_best, baseline_perf=good_baseline)
    snap, reason = resolve_grading_anchor_perf(state)
    assert snap is None
    assert reason == "current_best_axes_missing"


def test_anchor_perf_uses_baseline_when_current_best_empty():
    state = _State(current_best={}, baseline_perf=_BASELINE)
    snap, reason = resolve_grading_anchor_perf(state)
    assert reason == ""
    assert snap is not None
    assert snap["total_throughput"] == pytest.approx(_BASELINE["total_throughput"])


def test_anchor_perf_returns_missing_reason_when_both_absent():
    state = _State(current_best={}, baseline_perf=None)
    snap, reason = resolve_grading_anchor_perf(state)
    assert snap is None
    assert reason == "baseline_perf_missing"


# --- output_tput_of ---


def test_output_tput_of_prefers_the_measurement_field_over_tput():
    assert output_tput_of({**_BASELINE, "tput": 1.0}) == pytest.approx(_BASELINE["output_throughput"])


def test_output_tput_of_falls_back_to_tput():
    assert output_tput_of({"tput": 183.0}) == pytest.approx(183.0)


def test_output_tput_of_reports_zero_when_absent():
    assert output_tput_of({}) == 0.0
    assert output_tput_of(None) == 0.0
