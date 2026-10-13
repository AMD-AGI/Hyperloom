# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether two runs were in the same host state while they measured, read off their own requests."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors import report as rp
from hyperloom.orchestrator.measurement import intvty_samples as isx


def _records(ttft_values, itl_values=None):
    itl_values = itl_values if itl_values is not None else [4.0] * len(ttft_values)
    return [
        {"time_to_first_token": {"value": t}, "inter_token_latency": {"value": i}}
        for t, i in zip(ttft_values, itl_values)
    ]


def test_a_stalled_run_shows_up_as_mean_over_median_inflation():
    # Medians barely move; the stalls land in the mean. This is the one ratio that separated the two host states
    # on the GLM-5.3 workload: 1.6-1.7 healthy, 3.4-5.4 stalled, with medians within 10%.
    healthy = isx.regime_fingerprint(_records([430.0] * 90 + [700.0] * 10))
    stalled = isx.regime_fingerprint(_records([430.0] * 90 + [20000.0] * 10))

    assert healthy is not None and stalled is not None
    assert healthy["ttft_p50_ms"] == stalled["ttft_p50_ms"]
    assert healthy["ttft_inflation"] < 1.2 < 3.0 < stalled["ttft_inflation"]


def test_two_runs_in_one_state_are_not_called_cross_regime():
    a = isx.regime_fingerprint(_records([430.0] * 90 + [900.0] * 10))
    b = isx.regime_fingerprint(_records([440.0] * 90 + [950.0] * 10))

    ratio = isx.regimes_differ(a, b)
    assert ratio is not None and ratio < isx.CROSS_REGIME_RATIO


def test_a_stalled_run_against_a_healthy_one_is_cross_regime():
    healthy = isx.regime_fingerprint(_records([430.0] * 90 + [700.0] * 10))
    stalled = isx.regime_fingerprint(_records([430.0] * 90 + [20000.0] * 10))

    assert isx.regimes_differ(stalled, healthy) >= isx.CROSS_REGIME_RATIO


def test_the_fingerprint_also_carries_the_decode_side():
    out = isx.regime_fingerprint(_records([430.0] * 100, [4.0] * 90 + [60.0] * 10))

    assert out is not None
    assert out["itl_p50_ms"] == 4.0
    assert out["itl_inflation"] > 2.0


def test_too_few_requests_carry_no_fingerprint():
    assert isx.regime_fingerprint(_records([430.0] * 5)) is None
    assert isx.regime_fingerprint([]) is None
    assert isx.regimes_differ(None, {"ttft_inflation": 1.6}) is None
    assert isx.regimes_differ({"ttft_inflation": 0.0}, {"ttft_inflation": 1.6}) is None


def test_comparison_regimes_is_absent_when_neither_run_can_be_read(tmp_path):
    assert isx.comparison_regimes(tmp_path / "absent", tmp_path / "gone") is None


def test_the_report_states_both_sides_and_flags_a_cross_regime_pair():
    out = rp._format_comparison_regimes(
        {
            "candidate": {"ttft_inflation": 1.64},
            "reference": {"ttft_inflation": 3.78},
            "inflation_ratio": 2.305,
            "cross_regime": True,
        }
    )

    body = "\n".join(out)
    assert "`1.64`" in body and "`3.78`" in body and "`2.31`" in body
    assert "different host states" in body


def test_the_report_is_quiet_when_both_runs_sat_in_one_state():
    out = rp._format_comparison_regimes(
        {
            "candidate": {"ttft_inflation": 1.64},
            "reference": {"ttft_inflation": 1.70},
            "inflation_ratio": 1.037,
            "cross_regime": False,
        }
    )

    assert "⚠" not in "\n".join(out)
    assert rp._format_comparison_regimes(None) == []
    assert rp._format_comparison_regimes({"inflation_ratio": None}) == []
