# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiperf export -> InferenceX-schema mapping.

The mapping must emit exactly the keys Magpie's
``ResultParser.parse_inferencex_result`` reads, from the aiperf
``profile_export_aiperf.json`` metric shape (each metric is a dict with at
least ``avg``; latency metrics also carry ``p50``/``p99``/``std``).
"""

from __future__ import annotations

from hyperloom.inference_optimizer.agentx.mapping import map_aiperf, stat


def _metric(avg, **pct):
    d = {"unit": "ms", "avg": avg}
    d.update(pct)
    return d


def _sample():
    return {
        "request_throughput": {"unit": "req/s", "avg": 3.0},
        "output_token_throughput": {"unit": "tok/s", "avg": 500.0},
        "input_token_throughput": {"unit": "tok/s", "avg": 1500.0},
        "total_token_throughput": {"unit": "tok/s", "avg": 2000.0},
        "request_count": {"unit": "req", "avg": 42},
        "input_sequence_length": {"unit": "tok", "avg": 100.0},
        "total_isl": {"unit": "tok", "avg": 4200.0},
        "total_output_tokens": {"unit": "tok", "avg": 2100.0},
        "benchmark_duration": {"unit": "s", "avg": 14.0},
        "time_to_first_token": _metric(120.0, p50=110.0, p99=200.0, std=15.0),
        "inter_token_latency": _metric(20.0, p50=18.0, p99=40.0, std=5.0),
        "request_latency": _metric(900.0, p50=850.0, p99=1500.0, std=120.0),
        "theoretical_prefix_cache_hit": {"unit": "%", "avg": 0.73},
    }


def test_stat_reads_sub_key_and_default():
    m = {"x": {"avg": 1.0, "p99": 9.0}}
    assert stat(m, "x") == 1.0
    assert stat(m, "x", "p99") == 9.0
    assert stat(m, "missing") == 0.0
    assert stat(m, "x", "p50") == 1.0  # falls back to avg when sub absent


def test_map_core_throughput_and_counts():
    r = map_aiperf(_sample())
    assert r["request_throughput"] == 3.0
    assert r["output_throughput"] == 500.0
    assert r["total_token_throughput"] == 2000.0
    assert r["completed"] == 42
    assert r["total_input_tokens"] == 4200
    assert r["total_output_tokens"] == 2100
    assert r["duration"] == 14.0


def test_map_latency_fields():
    r = map_aiperf(_sample())
    assert r["mean_ttft_ms"] == 120.0
    assert r["median_ttft_ms"] == 110.0
    assert r["p99_ttft_ms"] == 200.0
    assert r["std_ttft_ms"] == 15.0
    assert r["mean_itl_ms"] == 20.0
    assert r["p99_itl_ms"] == 40.0
    # tpot mirrors inter_token_latency in the aiperf schema
    assert r["mean_tpot_ms"] == 20.0
    assert r["mean_e2el_ms"] == 900.0
    assert r["p99_e2el_ms"] == 1500.0


def test_map_prefix_cache_hit():
    r = map_aiperf(_sample())
    assert r["theoretical_prefix_cache_hit"] == 0.73


def test_map_total_tput_fallback_from_in_plus_out():
    s = _sample()
    del s["total_token_throughput"]
    r = map_aiperf(s)
    assert r["total_token_throughput"] == 2000.0  # 1500 in + 500 out


def test_map_accepts_metrics_wrapped():
    r = map_aiperf({"metrics": _sample()})
    assert r["output_throughput"] == 500.0


def test_map_missing_metric_defaults_zero():
    r = map_aiperf({"output_token_throughput": {"avg": 10.0}})
    assert r["output_throughput"] == 10.0
    assert r["mean_ttft_ms"] == 0.0
    assert r["completed"] == 0


def test_vendored_asset_fallback_matches_package(monkeypatch):
    """The deployed asset vendors a fallback map_aiperf for when the package is
    not importable; guard it against drifting from the package implementation."""
    import importlib.util
    import sys

    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    asset = agentx_asset_dir() / "map_aiperf.py"
    # Force the asset's `from ...mapping import map_aiperf` to raise so the
    # vendored fallback branch is the one exercised.
    monkeypatch.setitem(sys.modules, "hyperloom.inference_optimizer.agentx.mapping", None)
    spec = importlib.util.spec_from_file_location("_asset_map_aiperf", str(asset))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.map_aiperf(_sample()) == map_aiperf(_sample())
