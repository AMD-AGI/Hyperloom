# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a request pays per input token before the engine sees it."""

from __future__ import annotations

import json

from hyperloom.orchestrator.measurement import host_input_cost as hic


def _workspace(tmp_path, pairs, *, name: str = "ws"):
    """Write a per-request export holding ``(input tokens, ttft ms)`` for the scored phase."""
    artifacts = tmp_path / name / "benchmark_1" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "metadata": {"benchmark_phase": "profiling"},
                "metrics": {
                    "input_sequence_length": {"value": isl},
                    "time_to_first_token": {"value": ttft},
                },
            }
        )
        for isl, ttft in pairs
    ]
    (artifacts / "profile_export.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / name


def _linear(n: int, *, ms_per_100k: float, base_ms: float = 200.0):
    """Requests whose TTFT grows with input length at a known rate."""
    return [(10_000 + i * 2_000, base_ms + (10_000 + i * 2_000) / 100_000 * ms_per_100k) for i in range(n)]


def test_the_slope_is_reported_in_ms_per_100k_tokens(tmp_path):
    ws = _workspace(tmp_path, _linear(200, ms_per_100k=200.0))

    out = hic.input_cost(ws)

    assert out is not None
    assert abs(out["ms_per_100k_tokens"] - 200.0) < 1.0
    assert out["requests"] == 200
    assert out["low_isl"] < out["high_isl"]
    assert out["flagged"] is True


def test_a_front_end_that_costs_nothing_per_token_is_not_flagged(tmp_path):
    ws = _workspace(tmp_path, _linear(200, ms_per_100k=5.0))

    out = hic.input_cost(ws)

    assert out["ms_per_100k_tokens"] < hic.FLAG_MS_PER_100K
    assert out["flagged"] is False


def test_a_queued_tail_does_not_move_the_slope(tmp_path):
    """TTFT carries requests that waited; medians on each half are not moved by them."""
    clean = _linear(200, ms_per_100k=50.0)
    queued = [(isl, ttft * 40) for isl, ttft in clean[:20]]
    ws = _workspace(tmp_path, clean + queued)

    out = hic.input_cost(ws)

    assert abs(out["ms_per_100k_tokens"] - 50.0) < 15.0


def test_too_few_requests_carry_no_slope(tmp_path):
    ws = _workspace(tmp_path, _linear(hic.MIN_REQUESTS - 1, ms_per_100k=200.0))

    assert hic.input_cost(ws) is None


def test_a_run_over_one_prompt_length_carries_no_slope(tmp_path):
    ws = _workspace(tmp_path, [(100_000, 400.0 + i) for i in range(200)])

    assert hic.input_cost(ws) is None


def test_an_absent_run_reports_nothing(tmp_path):
    assert hic.input_cost(tmp_path / "absent") is None
    assert hic.input_cost(None) is None
    assert hic.input_cost("") is None


def test_only_scored_requests_that_reported_both_are_paired(tmp_path):
    ws = _workspace(tmp_path, [(1000, 10.0)])
    export = ws / "benchmark_1" / "aiperf_artifacts" / "profile_export.jsonl"
    warmup = {
        "metadata": {"benchmark_phase": "warmup"},
        "metrics": {"input_sequence_length": {"value": 9}, "time_to_first_token": {"value": 9.0}},
    }
    failed = {"metadata": {"benchmark_phase": "profiling"}, "metrics": {"error_isl": {"value": 5000}}}
    export.write_text(export.read_text() + json.dumps(warmup) + "\n" + json.dumps(failed) + "\n")

    assert hic._paired(hic.profiling_records(ws)) == [(1000, 10.0)]
