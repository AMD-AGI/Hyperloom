# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measured window's GPU energy: InferenceX's integral, its validity reasons, and J per token."""

from __future__ import annotations

import pytest

from hyperloom_common.power_energy import (
    EnergyInputs,
    TokenCounts,
    Window,
    counter_delta,
    energy_block,
    integrate,
)

_COUNTS = TokenCounts(successful_queries=100, input_tokens=40_000, output_tokens=10_000)


def _window(start: float = 10.0, end: float = 20.0) -> Window:
    return Window("measured", ((start, end),), 1_000.0 + start, 1_000.0 + end, "phase_marks")


def _flat(watts: float, start: float = 9.5, end: float = 20.5, step: float = 1.0) -> list[tuple[float, float | None]]:
    count = int(round((end - start) / step)) + 1
    return [(start + i * step, watts) for i in range(count)]


def _inputs(power, *, window=None, scope=(0, 1), serving=None, accumulator=None, **flags) -> EnergyInputs:
    return EnergyInputs(
        window=_window() if window is None else window,
        power=power,
        accumulator=accumulator or {},
        scope=scope,
        serving=scope if serving is None else serving,
        expected_scope=flags.get("expected_scope", True),
        external_server=flags.get("external_server", False),
        aborted=flags.get("aborted", False),
    )


def test_the_integral_interpolates_both_window_boundaries():
    # 0 W at t=0 rising to 100 W at t=10: the window [2.5, 7.5] averages 50 W over 5 s.
    ramp = [(0.0, 0.0), (5.0, 50.0), (10.0, 100.0)]
    assert integrate(ramp, 2.5, 7.5) == pytest.approx(250.0)
    assert integrate(ramp, -1.0, 7.5) is None


def test_a_valid_window_reports_total_energy_power_and_joules_per_token():
    block = energy_block(_inputs({0: _flat(500.0), 1: _flat(300.0)}), _COUNTS)
    assert block["power_valid"] == 1 and block["reasons"] == []
    assert block["total_gpu_energy_j"] == pytest.approx(8_000.0)
    assert block["avg_total_gpu_power_w"] == pytest.approx(800.0)
    assert block["avg_power_w"] == pytest.approx(400.0)
    assert block["p90_total_gpu_power_w"] == pytest.approx(800.0, rel=1e-6)
    assert block["joules_per_output_token"] == pytest.approx(0.8)
    assert block["joules_per_input_token"] == pytest.approx(0.2)
    assert block["joules_per_total_token"] == pytest.approx(0.16)
    assert block["joules_per_successful_query"] == pytest.approx(80.0)
    assert block["per_gpu"]["0"] == {"integral_j": 5000.0, "accumulator_delta_j": None, "max_gap_s": 1.0, "samples": 10}
    assert block["window"]["duration_s"] == 10.0 and block["window"]["start_unix"] == 1010.0


def test_the_p75_and_p90_are_time_weighted_quantiles_of_the_summed_curve():
    # One card at 100 W for 9 s, then 1000 W for 1 s: p75 stays at the low level, p90 sits on the step.
    curve = [*_flat(100.0, start=9.0, end=19.0), (19.001, 1000.0), (20.0, 1000.0), (21.0, 1000.0)]
    block = energy_block(_inputs({0: curve}, scope=(0,)), _COUNTS)
    assert block["p75_power_w"] == pytest.approx(100.0, abs=1e-3)
    assert block["p90_power_w"] == pytest.approx(100.0, abs=1.0)
    assert block["avg_power_w"] == pytest.approx(190.0, rel=1e-3)


def test_a_sampling_gap_over_three_seconds_invalidates_the_energy():
    gappy = [(9.5, 400.0), (11.0, 400.0), (14.5, 400.0), (20.5, 400.0)]
    block = energy_block(_inputs({0: _flat(400.0), 1: gappy}), _COUNTS)
    assert block["power_valid"] == 0
    assert "sampling_gap_exceeded" in block["reasons"]
    assert block["total_gpu_energy_j"] is None and block["joules_per_output_token"] is None
    assert block["per_gpu"]["1"]["max_gap_s"] == 6.0


def test_a_gap_across_a_window_boundary_counts():
    late_first = [(5.0, 400.0), (13.5, 400.0), (20.5, 400.0)]
    block = energy_block(_inputs({0: late_first}, scope=(0,)), _COUNTS)
    assert block["reasons"] == ["sampling_gap_exceeded"]


def test_samples_that_do_not_bracket_the_window_invalidate_it():
    block = energy_block(_inputs({0: _flat(400.0), 1: _flat(400.0, start=10.5)}), _COUNTS)
    assert block["reasons"] == ["benchmark_window_not_bracketed"]


def test_no_window_is_an_invalid_window():
    block = energy_block(_inputs({0: _flat(400.0)}, window=Window("measured", (), 0.0, 0.0, "phase_marks")), _COUNTS)
    assert block["power_valid"] == 0 and "invalid_benchmark_window" in block["reasons"]


def test_a_configured_card_holding_no_model_invalidates_the_energy():
    block = energy_block(_inputs({0: _flat(400.0), 1: _flat(90.0)}, serving=(0,)), _COUNTS)
    assert block["reasons"] == ["expected_gpu_count_mismatch"]


def test_cards_outside_the_configured_scope_holding_a_model_do_not_invalidate_the_energy():
    power = {gpu: _flat(400.0) for gpu in range(4)}
    block = energy_block(_inputs(power, scope=(0, 1), serving=(0, 1, 2, 3)), _COUNTS)
    assert block["power_valid"] == 1
    assert block["avg_power_w"] == 400.0


def test_a_configured_card_never_sampled_invalidates_the_energy():
    block = energy_block(_inputs({0: _flat(400.0)}), _COUNTS)
    assert block["reasons"] == ["expected_gpu_count_mismatch"]


def test_a_failed_power_read_inside_the_window_is_an_invalid_sample():
    failed = [*_flat(400.0)[:4], (13.0, None), *_flat(400.0)[5:]]
    block = energy_block(_inputs({0: failed}, scope=(0,)), _COUNTS)
    assert block["reasons"] == ["invalid_power_sample"]


def test_the_accumulator_is_compared_over_its_own_read_span():
    accumulator = {0: [(10.2, 1_000.0), (20.2, 5_000.0)], 1: [(10.2, 0.0), (20.2, 4_000.0)]}
    block = energy_block(_inputs({0: _flat(400.0), 1: _flat(400.0)}, accumulator=accumulator), _COUNTS)
    check = block["accumulator_check"]
    assert check["accumulator_delta_j"] == 8_000.0 and check["integrated_j"] == pytest.approx(8_000.0)
    assert check["within_tolerance"] is True and block["warnings"] == []
    assert block["per_gpu"]["1"]["accumulator_delta_j"] == 4_000.0


def test_an_accumulator_off_by_more_than_five_percent_is_a_warning_not_an_invalid_energy():
    accumulator = {0: [(10.2, 0.0), (20.2, 4_600.0)]}
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,), accumulator=accumulator), _COUNTS)
    assert block["power_valid"] == 1
    assert block["warnings"] == ["accumulator_mismatch"]
    assert block["accumulator_check"]["relative_error"] == pytest.approx(600 / 4600, rel=1e-4)


def test_a_missing_closing_accumulator_read_is_named():
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,), accumulator={0: [(10.2, 0.0)]}), _COUNTS)
    assert block["power_valid"] == 1 and block["accumulator_check"] is None
    assert block["warnings"] == ["missing_end_snapshot", "accumulator_check_unavailable"]


def test_an_external_server_round_measures_no_energy():
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,), external_server=True), _COUNTS)
    assert block["power_valid"] == 0 and block["reasons"] == ["external_server"]
    assert block["total_gpu_energy_j"] is None and block["joules_per_total_token"] is None and block["per_gpu"] == {}


def test_an_aborted_round_is_invalid():
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,), aborted=True), _COUNTS)
    assert block["reasons"] == ["aborted"]


def test_a_missing_token_count_nulls_only_its_own_figure():
    counts = TokenCounts(successful_queries=100, input_tokens=None, output_tokens=10_000)
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,)), counts)
    assert block["power_valid"] == 1 and block["total_gpu_energy_j"] == pytest.approx(4_000.0)
    assert block["reasons"] == ["invalid_input_token_count"]
    assert block["joules_per_input_token"] is None and block["joules_per_total_token"] is None
    assert block["joules_per_output_token"] == pytest.approx(0.4)


def test_a_load_that_counts_no_tokens_has_no_per_token_figure():
    block = energy_block(_inputs({0: _flat(400.0)}, scope=(0,)), None)
    assert block["power_valid"] == 1
    assert block["reasons"] == [
        "invalid_successful_query_count",
        "invalid_input_token_count",
        "invalid_output_token_count",
    ]
    assert block["counts"] == {"successful_queries": None, "input_tokens": None, "output_tokens": None}


def test_a_window_of_several_intervals_sums_them():
    window = Window("measured", ((10.0, 12.0), (15.0, 18.0)), 1010.0, 1018.0, "server_log_poll")
    block = energy_block(_inputs({0: _flat(100.0)}, window=window, scope=(0,)), _COUNTS)
    assert block["total_gpu_energy_j"] == pytest.approx(500.0)
    assert block["avg_power_w"] == pytest.approx(100.0)
    assert block["window"]["intervals"] == 2 and block["window"]["source"] == "server_log_poll"


def test_a_counter_delta_unwraps_at_its_range_and_refuses_a_reset():
    readings = [(0.0, 900.0), (10.0, 100.0)]
    assert counter_delta(readings, 0.0, 10.0, wrap=1000.0) == pytest.approx(200.0)
    assert counter_delta(readings, 0.0, 10.0) is None
    assert counter_delta([(0.0, 0.0), (10.0, 1000.0)], 2.0, 4.0) == pytest.approx(200.0)
