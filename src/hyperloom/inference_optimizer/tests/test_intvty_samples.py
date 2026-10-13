# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The graded interactivity axes are recomputable from the per-request export, and carry an interval."""

from __future__ import annotations

import json
import random

from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.orchestrator.actions.executors import report as rp
from hyperloom.orchestrator.measurement import intvty_samples as isx


def _write_export(workspace, values, *, warmup=()):
    artifacts = workspace / "benchmark_sglang_1" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    lines = []
    for phase, series in (("profiling", values), ("warmup", warmup)):
        for value in series:
            lines.append(
                json.dumps(
                    {
                        "metadata": {"benchmark_phase": phase},
                        "metrics": {"e2e_output_token_throughput": {"value": value, "unit": "tokens/sec/user"}},
                    }
                )
            )
    (artifacts / "profile_export.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return workspace


def test_the_graded_axes_are_the_p50_and_the_slow_tail_of_the_same_sample():
    # A real run reproduces exactly: e2e_norm_intvty_p50 is the median, and the axis named p90 is the 10th
    # percentile, because it is the 90th percentile of latency.
    samples = [float(v) for v in range(1, 101)]

    assert isx.graded_quantile(samples, GRADED_INTVTY_P50) == 50.5
    assert isx.graded_quantile(samples, GRADED_INTVTY) == 10.9
    assert isx.graded_quantile(samples, GRADED_OUTPUT) is None


def test_only_the_scored_phase_is_read(tmp_path):
    workspace = _write_export(tmp_path / "ws", [10.0, 20.0, 30.0], warmup=[999.0, 999.0])

    assert isx.profiling_samples(workspace) == [10.0, 20.0, 30.0]


def test_a_workspace_with_no_export_yields_nothing(tmp_path):
    (tmp_path / "empty").mkdir()

    assert isx.profiling_samples(tmp_path / "empty") == []
    assert isx.profiling_samples(None) == []
    assert isx.profiling_samples(tmp_path / "absent") == []


def test_a_clear_difference_produces_an_interval_that_excludes_zero():
    rng = random.Random(7)
    reference = [rng.gauss(100.0, 8.0) for _ in range(400)]
    candidate = [rng.gauss(130.0, 8.0) for _ in range(400)]

    out = isx.bootstrap_delta_ci(candidate, reference, GRADED_INTVTY_P50)

    assert out is not None
    assert out["crosses_zero"] is False
    assert out["ci_low_pct"] > 0 and out["ci_low_pct"] <= out["delta_pct"] <= out["ci_high_pct"]
    assert (out["n_candidate"], out["n_reference"]) == (400, 400)


def test_a_difference_inside_the_request_spread_is_not_separable():
    rng = random.Random(11)
    reference = [rng.gauss(100.0, 25.0) for _ in range(400)]
    candidate = [rng.gauss(101.0, 25.0) for _ in range(400)]

    out = isx.bootstrap_delta_ci(candidate, reference, GRADED_INTVTY_P50)

    assert out is not None and out["crosses_zero"] is True


def test_too_few_requests_carry_no_interval():
    assert isx.bootstrap_delta_ci([100.0] * 5, [100.0] * 5, GRADED_INTVTY_P50) is None
    assert isx.bootstrap_delta_ci([100.0] * 400, [100.0] * 400, GRADED_OUTPUT) is None


def test_the_interval_is_the_same_on_a_re_render():
    rng = random.Random(3)
    reference = [rng.gauss(100.0, 15.0) for _ in range(200)]
    candidate = [rng.gauss(112.0, 15.0) for _ in range(200)]

    first = isx.bootstrap_delta_ci(candidate, reference, GRADED_INTVTY_P50)
    second = isx.bootstrap_delta_ci(candidate, reference, GRADED_INTVTY_P50)

    assert first == second


def test_comparison_interval_reads_both_runs(tmp_path):
    rng = random.Random(5)
    ref_ws = _write_export(tmp_path / "ref", [rng.gauss(100.0, 6.0) for _ in range(200)])
    cand_ws = _write_export(tmp_path / "cand", [rng.gauss(140.0, 6.0) for _ in range(200)])

    out = isx.comparison_interval(cand_ws, ref_ws, GRADED_INTVTY_P50)

    assert out is not None and out["crosses_zero"] is False
    assert isx.comparison_interval(cand_ws, tmp_path / "absent", GRADED_INTVTY_P50) is None


def test_the_report_states_the_interval_and_what_it_omits():
    out = rp._format_comparison_interval(
        {
            "delta_pct": 9.5,
            "ci_low_pct": 4.2,
            "ci_high_pct": 14.1,
            "crosses_zero": False,
            "n_candidate": 564,
            "n_reference": 560,
            "confidence": 0.95,
        }
    )

    body = "\n".join(out)
    assert "[+4.20%, +14.10%]" in body
    assert "n=564 vs 560" in body
    assert "separable from no change" in body
    assert "run-to-run spread" in body


def test_the_report_flags_an_interval_that_crosses_zero():
    out = rp._format_comparison_interval(
        {"ci_low_pct": -3.1, "ci_high_pct": 5.4, "crosses_zero": True, "n_candidate": 100, "n_reference": 100}
    )

    assert "⚠ not separable from no change" in "\n".join(out)


def test_no_interval_renders_nothing():
    assert rp._format_comparison_interval(None) == []
    assert rp._format_comparison_interval({}) == []
