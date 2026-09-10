# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import (
    INTVTY_V1,
    intvty_grading_enabled,
    intvty_of,
    intvty_serving_grading_enabled,
    output_tput_of,
    parse_intvty_noise_pct,
    passes_intvty_gate,
    passes_tput_guard,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_of,
)

_KEEP_THRESHOLD_PCT = 1.0

# Shaped like a measured AgentX round: prefill dominates the token budget (~114k prompt / ~806 output tokens), so
# total is essentially input. e2e_norm_intvty_p90 is the slow tail, P10 of per-request OSL/E2EL_s.
_BASELINE = {
    "input_throughput": 25801.36,
    "output_throughput": 183.44,
    "total_throughput": 25984.80,
    "e2e_norm_intvty_p90": 22.56,  # realistic Kimi-K3 p10 value
}


def _measured(**pct: float) -> dict[str, float]:
    """Baseline with named axes scaled; total re-derived on demand."""
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
    return gain_pct(intvty_of(cand), intvty_of(base))


def test_snapshot_carries_both_graded_axes():
    snap = perf_snapshot_from_mapping(_BASELINE)
    assert snap is not None
    assert snap["e2e_norm_intvty_p90"] == pytest.approx(_BASELINE["e2e_norm_intvty_p90"])
    assert snap["total_throughput"] == pytest.approx(_BASELINE["total_throughput"])


def test_intvty_of_reads_slow_tail_field():
    snap = perf_snapshot_from_mapping(_BASELINE)
    assert snap is not None
    assert intvty_of(snap) == pytest.approx(_BASELINE["e2e_norm_intvty_p90"])


def test_total_falls_back_to_input_plus_output():
    data = {k: v for k, v in _BASELINE.items() if k != "total_throughput"}
    snap = perf_snapshot_from_mapping(data)
    assert snap is not None
    assert total_tput_of(snap) == pytest.approx(_BASELINE["input_throughput"] + _BASELINE["output_throughput"])


def test_snapshot_requires_both_graded_axes():
    # Missing e2e_norm_intvty_p90 -> None
    assert perf_snapshot_from_mapping({"output_throughput": 1.0, "total_throughput": 100.0}) is None
    assert perf_snapshot_from_mapping({"e2e_norm_intvty_p90": 22.56}) is None
    assert perf_snapshot_from_mapping(None) is None


def test_degenerate_axis_is_not_a_snapshot():
    assert perf_snapshot_from_mapping({**_BASELINE, "e2e_norm_intvty_p90": 0.0}) is None
    # total absent and cannot be derived
    assert perf_snapshot_from_mapping({"e2e_norm_intvty_p90": 22.56, "output_throughput": 183.44}) is None


def test_unusable_total_falls_back_to_input_plus_output():
    for bad in (0.0, -1.0, None, "n/a"):
        snap = perf_snapshot_from_mapping({**_BASELINE, "total_throughput": bad})
        assert snap is not None
        assert total_tput_of(snap) == pytest.approx(_BASELINE["input_throughput"] + _BASELINE["output_throughput"])


def test_intvty_lift_is_keepable():
    """A +3% interactivity improvement is above the 2% AgentX floor."""
    candidate = _measured(e2e_norm_intvty_p90=3.0, input_throughput=0.0, output_throughput=0.0)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and gain >= 2.0


def test_sub_threshold_lift_is_a_gain_but_below_floor():
    candidate = _measured(e2e_norm_intvty_p90=1.0, input_throughput=0.5, output_throughput=0.5)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and 0.0 < gain < 2.0


def test_trading_input_for_output_with_intvty_unchanged_is_neutral():
    """Interactivity unchanged: gain == 0, not a loss."""
    candidate = _measured(input_throughput=-10.0, output_throughput=8.0)
    gain = _graded_gain(candidate, _BASELINE)
    assert gain is not None and gain == pytest.approx(0.0)


def test_intvty_gate_vetoes_regression_past_band():
    candidate = perf_snapshot_from_mapping(_measured(e2e_norm_intvty_p90=-6.0))
    anchor = perf_snapshot_from_mapping(_BASELINE)
    assert candidate and anchor
    assert passes_intvty_gate(candidate, anchor) is False


def test_intvty_gate_allows_movement_within_band():
    candidate = perf_snapshot_from_mapping(_measured(e2e_norm_intvty_p90=-4.0))
    anchor = perf_snapshot_from_mapping(_BASELINE)
    assert candidate and anchor
    assert passes_intvty_gate(candidate, anchor) is True


def test_tput_guard_allows_within_band():
    cand = perf_snapshot_from_mapping({**_BASELINE, "total_throughput": _BASELINE["total_throughput"] * 0.97})
    anch = perf_snapshot_from_mapping(_BASELINE)
    assert cand and anch
    assert passes_tput_guard(cand, anch) is True


def test_tput_guard_rejects_regression_past_band():
    cand = perf_snapshot_from_mapping({**_BASELINE, "total_throughput": _BASELINE["total_throughput"] * 0.90})
    anch = perf_snapshot_from_mapping(_BASELINE)
    assert cand and anch
    assert passes_tput_guard(cand, anch) is False


def test_vetoed_candidate_is_never_graded():
    candidate = _measured(e2e_norm_intvty_p90=-20.0, input_throughput=50.0)
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
    assert intvty_grading_enabled() is True


def test_a_synthetic_run_still_grades_on_output(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_grading_enabled() is False


def test_an_explicit_metric_overrides_the_agentx_default(monkeypatch):
    """The escape hatch has to work in both directions."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert intvty_grading_enabled() is False


def test_an_explicit_metric_still_opts_a_synthetic_run_in(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", INTVTY_V1)
    assert intvty_grading_enabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "nonsense"])
def test_agentx_off_tokens_do_not_enable_grading(monkeypatch, raw):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", raw)
    assert intvty_grading_enabled() is False


# --- the persisted marker ---


def test_the_persisted_benchmark_mode_enables_grading_without_the_env_var(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_grading_enabled(benchmark_mode="agentx") is True
    assert intvty_serving_grading_enabled(benchmark_mode="AgentX") is True


def test_a_synthetic_benchmark_mode_does_not_enable_grading(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    for mode in ("", "synthetic", "sweep", None):
        assert intvty_grading_enabled(benchmark_mode=mode or "") is False


def test_an_explicit_metric_still_outranks_the_persisted_marker(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert intvty_grading_enabled(benchmark_mode="agentx") is False


def test_a_scriptable_framework_still_grades_on_output_under_the_marker(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_serving_grading_enabled(scriptable=True, benchmark_mode="agentx") is False


# --- resolve_grading_anchor_perf ---


class _State:
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
    bad_best = {"action": "explore", "tput": 200.0}  # no e2e_norm_intvty_p90 / total_throughput
    state = _State(current_best=bad_best, baseline_perf=_BASELINE)
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
