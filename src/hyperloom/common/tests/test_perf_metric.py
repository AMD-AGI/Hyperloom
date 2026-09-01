# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import (
    composite_score,
    noise_adjusted_delta,
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


def test_score_gain_pct_incremental():
    anchor = dict(_BASELINE)
    candidate = dict(_BASELINE)
    candidate["input_throughput"] *= 1.10
    gain = score_gain_pct(candidate, anchor, _BASELINE)
    assert gain is not None
    assert gain > 0.0


def test_keep_gain_pct_falls_back_to_output_tput():
    from hyperloom.common.perf_metric import keep_gain_pct

    gain, used = keep_gain_pct(
        {"output_throughput": 110.0},
        base_tput=100.0,
    )
    assert used is False
    assert gain == pytest.approx(10.0)


def test_keep_gain_pct_uses_composite_when_flag_on(monkeypatch):
    from types import SimpleNamespace

    from hyperloom.common.perf_metric import keep_gain_pct

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    state = SimpleNamespace(
        framework="sglang",
        baseline_perf=dict(_BASELINE),
        current_best=dict(_BASELINE),
    )
    candidate = dict(_BASELINE)
    candidate["input_throughput"] = _BASELINE["input_throughput"] * 1.20
    gain, used = keep_gain_pct(candidate, state=state, base_tput=_BASELINE["output_throughput"])
    assert used is True
    assert gain is not None
    assert gain > 1.0


def test_perf_snapshot_requires_all_axes():
    assert perf_snapshot_from_mapping({"output_throughput": 1.0}) is None
    assert perf_snapshot_from_mapping(_BASELINE) == _BASELINE


def test_noise_adjusted_delta():
    assert abs(noise_adjusted_delta(0.05, 2.0) - 0.03) < 1e-9


def test_session_gain_pct_falls_back_to_output_tput():
    from hyperloom.common.perf_metric import session_gain_pct

    gain, used = session_gain_pct({"output_throughput": 110.0}, base_tput=100.0)
    assert used is False
    assert gain == pytest.approx(10.0)


def test_session_gain_pct_uses_score_vs_baseline(monkeypatch):
    from types import SimpleNamespace

    from hyperloom.common.perf_metric import session_gain_pct

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    state = SimpleNamespace(framework="sglang", baseline_perf=dict(_BASELINE), current_best=dict(_BASELINE))
    candidate = dict(_BASELINE)
    candidate["input_throughput"] = _BASELINE["input_throughput"] * 1.20
    gain, used = session_gain_pct(candidate, state=state, base_tput=_BASELINE["output_throughput"])
    assert used is True
    assert gain == pytest.approx(11.0)


def test_session_gain_from_measurement_reads_current_best_axes(monkeypatch):
    from types import SimpleNamespace

    from hyperloom.common.perf_metric import session_gain_from_measurement

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    state = SimpleNamespace(
        framework="sglang",
        baseline_tput=_BASELINE["output_throughput"],
        baseline_perf=dict(_BASELINE),
        current_best={
            "tput": _BASELINE["output_throughput"],
            "input_throughput": _BASELINE["input_throughput"] * 1.20,
            "intvty_p90": _BASELINE["intvty_p90"],
        },
    )
    gain, used = session_gain_from_measurement(
        _BASELINE["output_throughput"],
        state=state,
        base_tput=_BASELINE["output_throughput"],
    )
    assert used is True
    assert gain == pytest.approx(11.0)


def test_session_composite_score_none_when_flag_off():
    from types import SimpleNamespace

    from hyperloom.common.perf_metric import session_composite_score

    state = SimpleNamespace(framework="sglang", baseline_perf=dict(_BASELINE), current_best=dict(_BASELINE))
    assert session_composite_score(state) is None


def test_composite_watermark_levels_treat_missing_last_score_as_zero(monkeypatch):
    from types import SimpleNamespace

    from hyperloom.common.perf_metric import composite_watermark_levels, session_composite_score

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    candidate = dict(_BASELINE)
    candidate["input_throughput"] = _BASELINE["input_throughput"] * 1.20
    state = SimpleNamespace(
        framework="sglang",
        baseline_perf=dict(_BASELINE),
        current_best=candidate,
        last_roofline_score=None,
    )
    assert session_composite_score(state) == pytest.approx(0.11)
    cur, last = composite_watermark_levels(state)
    assert cur == pytest.approx(1.11)
    assert last == pytest.approx(1.0)
    state.last_roofline_score = 0.11
    cur, last = composite_watermark_levels(state)
    assert cur / last == pytest.approx(1.0)
