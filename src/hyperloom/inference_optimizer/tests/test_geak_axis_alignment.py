# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK must measure the axis Hyperloom keeps candidates on.

An interactivity-graded session keeps a candidate only on a gain in the median
``e2e_norm_intvty_p50``, with the p90 tail and output throughput as noise-band
guards; a fixed-ISL/OSL run is graded on output. GEAK measures whichever axis
``E2E_METRIC`` names and records the matching ``metric_basis``, so the flag has
to follow the grader rather than sit pinned to one value.

Failure modes these cover:

* Handing an AgentX session's GEAK a guard instead of the objective. On p90 it
  selects tail-only wins the rebench reverts; on total token throughput, which
  the verdict no longer reads, nothing can surface at all -- under a ~97% prefix
  cache it is ~99% input tokens that were never computed.
* Keying "graded on interactivity" on one percentile's label, which silently
  answers no the next time the recorded objective moves within that family.
* Reading the measurement back out of ``bench_summary.json`` by its output-named
  field. GEAK nulls ``output_throughput_tok_s_median`` off the output axis --
  deliberately, so nobody reads another axis under an "output" name -- which
  would make every rung of an agentic sweep report "no throughput".
* Trusting a summary recorded on a basis nobody requested. A GEAK build that
  cannot measure the axis may fall back to output, and that number would then
  rank the sweep's rungs.

Synthetic runs must come out byte-identical, so each case here has its
fixed-ISL/OSL twin.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    INTVTY_V1,
)
from hyperloom.orchestrator.actions.executors import _geak_sweep
from hyperloom.orchestrator.actions.executors._geak_sweep import sweep_via_geak
from hyperloom.orchestrator.actions.executors._workload_envs import (
    GEAK_METRIC_INTVTY,
    GEAK_METRIC_OUTPUT,
    build_agentx_workload_spec,
    geak_acceptance,
    geak_metric_axis,
)


def _clear_axis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)


@pytest.mark.parametrize(
    ("benchmark_mode", "agentx_env", "expected"),
    [
        ("", None, ("output", "aggregate_output_tok_s")),
        ("synthetic", None, ("output", "aggregate_output_tok_s")),
        ("agentx", None, ("e2e_norm_intvty_p50", "e2e_norm_intvty_p50")),
        # The persisted mode is the durable signal, but a round driven from a
        # subprocess that only inherited the env var must resolve the same way.
        ("", "1", ("e2e_norm_intvty_p50", "e2e_norm_intvty_p50")),
    ],
)
def test_the_axis_follows_the_grader(monkeypatch, benchmark_mode, agentx_env, expected):
    _clear_axis_env(monkeypatch)
    if agentx_env is not None:
        monkeypatch.setenv("HYPERLOOM_AGENTX", agentx_env)
    assert geak_metric_axis(benchmark_mode=benchmark_mode) == expected


def test_the_interactivity_handoff_names_the_objective_not_the_tail_guard(monkeypatch):
    """KEEP is decided on the p50 gain; p90 only has to hold within the noise band."""
    _clear_axis_env(monkeypatch)
    _, basis = geak_metric_axis(benchmark_mode="agentx")
    assert basis == GRADED_INTVTY_P50
    assert basis != GRADED_INTVTY


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        # ``seed_grading`` stamps the p90 name as the "graded on interactivity" label.
        (GRADED_INTVTY, GEAK_METRIC_INTVTY),
        (GRADED_INTVTY_P50, GEAK_METRIC_INTVTY),
        (GRADED_OUTPUT, GEAK_METRIC_OUTPUT),
    ],
)
def test_a_recorded_objective_is_read_as_a_family_not_one_label(monkeypatch, objective, expected):
    """Either interactivity percentile means the session is graded on interactivity.

    The env says AgentX throughout, so the output case also shows the recorded objective
    winning over the mode-based derivation.
    """
    _clear_axis_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert geak_metric_axis(benchmark_mode="agentx", grading={"objective": objective}) == expected


def test_an_explicit_override_wins_in_both_directions(monkeypatch):
    """An operator who names the metric is not overridden by the workload."""
    _clear_axis_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert geak_metric_axis(benchmark_mode="agentx")[0] == "output"

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", INTVTY_V1)
    assert geak_metric_axis(benchmark_mode="synthetic")[0] == "e2e_norm_intvty_p50"


def test_the_interactivity_handoff_carries_the_rule_the_rebench_keeps_on():
    """GEAK's "ok" has to mean the rebench's KEEP, or every sub-threshold win is a wasted rebench."""
    assert geak_acceptance(GEAK_METRIC_INTVTY[1], 7.5) == {
        "objective": "e2e_norm_intvty_p50",
        "min_gain_pct": AGENTX_KEEP_P50_THRESHOLD_PCT,
        "guard_max_drop_pct": {"e2e_norm_intvty_p90": 7.5, "aggregate_output_tok_s": 7.5},
    }
    assert geak_acceptance(GEAK_METRIC_OUTPUT[1], 7.5) is None


@pytest.mark.parametrize(
    ("backend_env", "objective", "rule_expected"),
    [
        ({}, GRADED_INTVTY, True),
        ({}, GRADED_OUTPUT, False),
        ({"HYPERLOOM_AGENTIC_BACKEND": "mlperf"}, GRADED_OUTPUT, False),
    ],
)
def test_the_workload_spec_publishes_the_rule_with_the_sessions_own_band(backend_env, objective, rule_expected):
    """The guards' band is the one the session was seeded with, not this process's default."""
    spec = build_agentx_workload_spec(
        {}, {}, model_path="/models/GLM-5.2-MXFP4", env=backend_env, grading={"objective": objective, "noise_pct": 7.5}
    )
    if rule_expected:
        assert spec["acceptance"] == geak_acceptance(spec["metric_basis"], 7.5)
    else:
        assert "acceptance" not in spec


def _sweep_with_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, Any],
    *,
    metric_axis: tuple[str, str] = GEAK_METRIC_OUTPUT,
):
    """Drive ``sweep_via_geak`` against a stubbed GEAK that writes *summary*.

    The replay env each point was launched with is appended to the returned list.
    """
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "1")
    launched: list[dict[str, str]] = []

    def _fake_run(_cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        launched.append(dict(kwargs["env"]))
        out = Path(kwargs["env"]["OUT_DIR"])
        (out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.CompletedProcess(_cmd, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    bench = tmp_path / "bench_e2e.sh"
    bench.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    async def _go():
        return await sweep_via_geak(
            result={
                "bench_script": str(bench),
                "output_dir": str(tmp_path),
                "validated_regimes": [{"num_warmups": 1, "seed": 1, "num_prompts": 8}],
                "accepted_config": {"flags": "", "env": ""},
            },
            conc_values=[1],
            isl_osl_configs=["16:16"],
            output_root=tmp_path / "sweep",
            variant_timeout_sec=30,
            repeats=1,
            metric_axis=metric_axis,
        )

    return _go, launched


def _benchmark_report(tmp_path: Path) -> dict[str, Any]:
    report = tmp_path / "sweep" / "variant_0_conc1_isl16_osl16" / "benchmark_report.json"
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_an_intvty_summary_is_ranked_on_its_axis_not_published_as_output_throughput(tmp_path, monkeypatch):
    """The interactivity median is a per-request rate, not a throughput.

    ``output_throughput`` is ``GRADED_OUTPUT``, which the perf snapshot reads, so
    publishing an interactivity score there would record it as the session's
    output throughput; the breakdown's ``benchmark_report.json`` has the same
    output-named field. The rung must still rank, or an agentic sweep reports
    nothing again.
    """
    go, launched = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "throughput_tok_s_median": 151.85,
            "output_throughput_tok_s_median": None,
            "metric_basis": "e2e_norm_intvty_p50",
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
        metric_axis=GEAK_METRIC_INTVTY,
    )
    result = await go()
    assert [env["E2E_METRIC"] for env in launched] == ["e2e_norm_intvty_p50"]
    assert result["status"] == "succeeded"
    point = result["points"][0]
    assert point["measured_value"] == pytest.approx(151.85)
    assert point["metric_basis"] == "e2e_norm_intvty_p50"
    assert "output_throughput" not in point
    assert result["promotion_measurement"]["measured_value"] == pytest.approx(151.85)
    report = _benchmark_report(tmp_path)
    assert report["success"] is True
    assert report["output_throughput_tok_s"] is None


@pytest.mark.parametrize(
    ("metric_axis", "recorded_basis"),
    [
        # A GEAK that cannot measure the median and falls back to output.
        (GEAK_METRIC_INTVTY, "aggregate_output_tok_s"),
        # The tail guard: a different interactivity axis, not a different name for this one.
        (GEAK_METRIC_INTVTY, "e2e_norm_intvty_p90"),
        # A replay that measured total under a synthetic session's output request.
        (GEAK_METRIC_OUTPUT, "aggregate_total_token_tok_s"),
    ],
)
@pytest.mark.asyncio
async def test_a_summary_on_a_basis_nobody_requested_is_refused(tmp_path, monkeypatch, metric_axis, recorded_basis):
    """A number on the wrong axis must not rank the rungs, however plausible it looks."""
    go, _ = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "throughput_tok_s_median": 445.7,
            "output_throughput_tok_s_median": None,
            "metric_basis": recorded_basis,
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
        metric_axis=metric_axis,
    )
    result = await go()
    assert result["status"] == "failed"
    point = result["points"][0]
    assert point["status"] == "failed"
    assert point["metric_basis"] == recorded_basis
    assert f"not the requested {metric_axis[1]}" in point["error"]
    assert "measured_value" not in point
    assert result["promotion_measurement"] == {}
    assert _benchmark_report(tmp_path)["success"] is False


@pytest.mark.asyncio
async def test_an_output_mode_summary_reads_the_same_number_as_before(tmp_path, monkeypatch):
    """Synthetic parity: in output mode both fields carry the same median.

    Confirmed against a real run's ``bench_summary.json``, where
    ``throughput_tok_s_median`` and ``output_throughput_tok_s_median`` were both
    167.259, so preferring the neutral field changes nothing here.
    """
    go, launched = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "throughput_tok_s_median": 167.259,
            "output_throughput_tok_s_median": 167.259,
            "metric_basis": "aggregate_output_tok_s",
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
    )
    result = await go()
    assert [env["E2E_METRIC"] for env in launched] == ["output"]
    assert result["status"] == "succeeded"
    point = result["points"][0]
    assert point["output_throughput"] == pytest.approx(167.259)
    assert point["measured_value"] == pytest.approx(167.259)
    assert _benchmark_report(tmp_path)["output_throughput_tok_s"] == pytest.approx(167.259)


@pytest.mark.asyncio
async def test_a_legacy_summary_without_the_neutral_field_still_reads(tmp_path, monkeypatch):
    """Summaries written before ``throughput_tok_s_median`` or ``metric_basis`` existed must still work."""
    go, _ = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "output_throughput_tok_s_median": 200.0,
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
    )
    result = await go()
    assert result["status"] == "succeeded"
    assert result["points"][0]["output_throughput"] == pytest.approx(200.0)
    assert result["points"][0]["metric_basis"] == "aggregate_output_tok_s"
