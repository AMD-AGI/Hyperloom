# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import pytest

from hyperloom.common.perf_metric import (
    GRADED_AXIS_KEYS,
    GRADED_INTVTY,
    GRADED_INTVTY_P90,
    GRADED_OUTPUT_PER_GPU,
    INTVTY_V1,
    agentx_active,
    agentx_policy_config,
    agentx_snapshot_of,
    graded_axes_of,
    intvty_grading_enabled,
    intvty_of,
    intvty_p90_of,
    intvty_serving_grading_enabled,
    output_tput_of,
    output_tput_per_gpu_of,
    parse_intvty_noise_pct,
    passes_intvty_gate,
    passes_tput_guard,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_of,
)


_BASELINE = {
    "input_throughput": 800.0,
    "output_throughput": 200.0,
    "total_throughput": 1000.0,
    "e2e_intvty_p50": 100.0,
    "e2e_intvty_p90": 80.0,
    "output_tput_per_gpu": 25.0,
    "duration_s": 100.0,
    "request_error_rate": 1.0,
    "submission_valid": True,
    "accuracy_passed": True,
    "ttft_p50_ms": 1000.0,
    "ttft_p90_ms": 1500.0,
    "tpot_p50_ms": 10.0,
    "tpot_p90_ms": 15.0,
}


@pytest.mark.parametrize(
    "mode,env,expected",
    [
        ("", "", False),
        ("synthetic", "0", False),
        ("agentx", "0", True),
        (" AgentX ", "", True),
        ("synthetic", "true", True),
        (None, "1", True),
    ],
)
def test_agentx_active_uses_workload_identity_not_grading_override(monkeypatch, mode, env, expected):
    monkeypatch.setenv("HYPERLOOM_AGENTX", env)
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert agentx_active(benchmark_mode=mode) is expected


def test_snapshot_carries_every_policy_anchor_field_and_display_metric():
    snap = perf_snapshot_from_mapping(_BASELINE)

    assert snap is not None
    assert snap[GRADED_INTVTY] == 100.0
    assert snap[GRADED_INTVTY_P90] == 80.0
    assert snap[GRADED_OUTPUT_PER_GPU] == 25.0
    assert snap["duration_s"] == 100.0
    assert snap["request_error_rate"] == 1.0
    assert snap["ttft_p50_ms"] == 1000.0
    assert snap["ttft_p90_ms"] == 1500.0
    assert snap["tpot_p50_ms"] == 10.0
    assert snap["tpot_p90_ms"] == 15.0


def test_snapshot_accepts_legacy_flat_names_for_compatibility():
    snap = perf_snapshot_from_mapping(
        {
            "e2e_norm_intvty_p50": 100.0,
            "e2e_norm_intvty_p90": 80.0,
            "output_tput_per_gpu": 25.0,
            "duration": 100.0,
            "request_error_rate": 1.0,
        }
    )

    assert snap is not None
    assert snap[GRADED_INTVTY] == 100.0
    assert snap[GRADED_INTVTY_P90] == 80.0
    assert snap[GRADED_OUTPUT_PER_GPU] == 25.0


@pytest.mark.parametrize(
    "missing",
    [GRADED_INTVTY, GRADED_INTVTY_P90, GRADED_OUTPUT_PER_GPU, "duration_s", "request_error_rate"],
)
def test_snapshot_requires_every_anchor_input(missing):
    source = dict(_BASELINE)
    source.pop(missing)
    assert perf_snapshot_from_mapping(source) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, None, "n/a"])
def test_positive_policy_axis_rejects_invalid_values(bad):
    source = {**_BASELINE, GRADED_INTVTY: bad}
    assert perf_snapshot_from_mapping(source) is None


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan, None, "n/a"])
def test_error_rate_rejects_invalid_values_but_allows_zero(bad):
    source = {**_BASELINE, "request_error_rate": bad}
    assert perf_snapshot_from_mapping(source) is None
    assert perf_snapshot_from_mapping({**_BASELINE, "request_error_rate": 0.0}) is not None


def test_metric_accessors_use_the_new_agentx_axes():
    assert intvty_of(_BASELINE) == 100.0
    assert intvty_p90_of(_BASELINE) == 80.0
    assert output_tput_per_gpu_of(_BASELINE) == 25.0
    assert output_tput_of(_BASELINE) == 200.0
    assert total_tput_of(_BASELINE) == 1000.0


def test_graded_axes_are_the_seven_user_facing_metrics_with_explicit_subset_only():
    assert GRADED_AXIS_KEYS == (
        "e2e_intvty_p50",
        "e2e_intvty_p90",
        "output_tput_per_gpu",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
    )
    assert graded_axes_of(_BASELINE) == {key: _BASELINE[key] for key in GRADED_AXIS_KEYS}
    assert graded_axes_of(None) == {}


def test_agentx_snapshot_keeps_policy_flags_beside_metrics():
    snap = agentx_snapshot_of(_BASELINE)
    assert snap["submission_valid"] is True
    assert snap["accuracy_passed"] is True


def test_legacy_band_helpers_now_read_p50_and_output_per_gpu():
    assert passes_intvty_gate({**_BASELINE, GRADED_INTVTY: 96.0}, _BASELINE)
    assert not passes_intvty_gate({**_BASELINE, GRADED_INTVTY: 94.0}, _BASELINE)
    assert passes_tput_guard({**_BASELINE, GRADED_OUTPUT_PER_GPU: 24.0}, _BASELINE)
    assert not passes_tput_guard({**_BASELINE, GRADED_OUTPUT_PER_GPU: 23.0}, _BASELINE)


def test_fixed_agentx_policy_is_serializable_and_names_every_gate():
    assert agentx_policy_config() == {
        "mode": "all_of",
        "anchor_priority": ["current_best", "baseline"],
        "submission_valid": True,
        "request_error_rate_max": "anchor",
        "accuracy_passed": True,
        "duration_max_abs_delta_pct": 5.0,
        "e2e_intvty_p50_min_delta_pct": 3.0,
        "e2e_intvty_p90_min_delta_pct": -5.0,
        "output_tput_per_gpu_min_delta_pct": -5.0,
    }


def test_default_band_matches_upstream_measured_noise(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    assert parse_intvty_noise_pct() == pytest.approx(5.0)


@pytest.mark.parametrize("raw,expected", [("2.5", 2.5), ("0", 0.0), ("", 5.0), ("nonsense", 5.0)])
def test_band_env_override(monkeypatch, raw, expected):
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", raw)
    assert parse_intvty_noise_pct() == pytest.approx(expected)


def test_an_agentx_run_enables_interactivity_grading(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert intvty_grading_enabled() is True


def test_a_synthetic_run_still_grades_on_output(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_grading_enabled() is False


def test_an_explicit_metric_overrides_the_agentx_default(monkeypatch):
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


def test_the_persisted_benchmark_mode_enables_grading_without_the_env_var(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_grading_enabled(benchmark_mode="agentx") is True
    assert intvty_serving_grading_enabled(benchmark_mode="AgentX") is True


def test_a_scriptable_framework_still_grades_on_output_under_the_marker(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert intvty_serving_grading_enabled(scriptable=True, benchmark_mode="agentx") is False


class _State:
    def __init__(self, current_best=None, baseline_perf=None):
        self.current_best = current_best
        self.baseline_perf = baseline_perf


def test_anchor_perf_uses_current_best_when_complete():
    snap, reason = resolve_grading_anchor_perf(_State(current_best=_BASELINE, baseline_perf={}))
    assert reason == ""
    assert snap == agentx_snapshot_of(_BASELINE)


def test_anchor_perf_does_not_fall_through_when_current_best_is_incomplete():
    incomplete = dict(_BASELINE)
    incomplete.pop(GRADED_INTVTY)
    snap, reason = resolve_grading_anchor_perf(_State(current_best=incomplete, baseline_perf=_BASELINE))
    assert snap is None
    assert reason == "current_best_axes_missing"


def test_anchor_perf_uses_baseline_when_current_best_is_absent():
    snap, reason = resolve_grading_anchor_perf(_State(current_best={}, baseline_perf=_BASELINE))
    assert reason == ""
    assert snap == agentx_snapshot_of(_BASELINE)


def test_anchor_perf_reports_when_baseline_is_missing():
    snap, reason = resolve_grading_anchor_perf(_State(current_best={}, baseline_perf=None))
    assert snap is None
    assert reason == "baseline_perf_missing"


def test_output_tput_of_prefers_measurement_field_then_legacy_tput():
    assert output_tput_of({"output_throughput": 183.0, "tput": 1.0}) == 183.0
    assert output_tput_of({"tput": 183.0}) == 183.0
    assert output_tput_of(None) == 0.0
