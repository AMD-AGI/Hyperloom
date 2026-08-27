# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from hyperloom.common.perf_metric import (
    composite_score,
    noise_adjusted_delta,
    passes_intvty_gate,
    perf_snapshot_from_mapping,
    score_gain_pct,
)

_BASELINE = {
    "input_throughput": 24252.0,
    "output_throughput": 170.0,
    "intvty_p90": 745.19,
}


def test_default_noise_is_zero_uses_raw_delta():
    candidate = dict(_BASELINE)
    candidate["output_throughput"] = _BASELINE["output_throughput"] * 1.02
    score = composite_score(candidate, _BASELINE)
    assert abs(score - 0.15 * 0.02) < 1e-9


def test_noise_floor_zeros_small_output_gain():
    candidate = dict(_BASELINE)
    candidate["output_throughput"] = _BASELINE["output_throughput"] * 1.02
    score = composite_score(candidate, _BASELINE, noise_pct=(2.0, 2.0, 2.0))
    assert score < 1e-9


def test_input_gain_dominates_output_noise():
    candidate = dict(_BASELINE)
    candidate["input_throughput"] = _BASELINE["input_throughput"] * 1.10
    candidate["output_throughput"] = _BASELINE["output_throughput"] * 1.05
    score = composite_score(
        candidate,
        _BASELINE,
        weights=(0.55, 0.30, 0.15),
        noise_pct=(2.0, 2.0, 2.0),
    )
    assert score > 0.04


def test_intvty_gate_blocks_regression():
    anchor = dict(_BASELINE)
    candidate = dict(_BASELINE)
    candidate["intvty_p90"] = anchor["intvty_p90"] * 0.95
    assert passes_intvty_gate(candidate, anchor, noise_pct=2.0) is False


def test_score_gain_pct_incremental():
    anchor = dict(_BASELINE)
    candidate = dict(_BASELINE)
    candidate["input_throughput"] *= 1.10
    gain = score_gain_pct(candidate, anchor, _BASELINE)
    assert gain is not None
    assert gain > 0.0


def test_perf_snapshot_requires_all_axes():
    assert perf_snapshot_from_mapping({"output_throughput": 1.0}) is None
    assert perf_snapshot_from_mapping(_BASELINE) == _BASELINE


def test_noise_adjusted_delta():
    assert abs(noise_adjusted_delta(0.05, 2.0) - 0.03) < 1e-9
