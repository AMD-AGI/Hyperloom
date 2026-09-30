# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK must measure the axis Hyperloom grades on.

An agentic replay is graded on interactivity and guarded on total token
throughput; a fixed-ISL/OSL run is graded on output. GEAK measures whichever
axis ``E2E_METRIC`` names and records the matching ``metric_basis``, so the flag
has to follow the workload rather than sit pinned to one value.

Three failure modes these cover:

* A pinned ``E2E_METRIC=output`` on an AgentX session points GEAK's search at a
  figure the session never scores. On this corpus the two axes run ~140x apart,
  and the load is prefill-dominated, so a kernel that lifts the decode-side
  output number need not lift the graded axes by the same margin.
* Sending an AgentX session at the *guard* instead of the objective. That is
  what this file asserted before GEAK could measure interactivity, and it is
  near-incompressible: under a ~97% prefix cache the guard is ~99% input tokens
  that were never computed, so no kernel win can surface on it.
* Reading the measurement back out of ``bench_summary.json`` by its output-named
  field. ``bench_e2e.sh`` sets ``output_throughput_tok_s_median`` to null off the
  output axis -- deliberately, so nobody reads another axis under an "output"
  name -- which would make every rung of an agentic sweep report "no throughput"
  the moment the flag flipped.

Synthetic runs must come out byte-identical, so each case here has its
fixed-ISL/OSL twin.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.perf_metric import INTVTY_V1
from hyperloom.orchestrator.actions.executors import _geak_sweep
from hyperloom.orchestrator.actions.executors._geak_sweep import sweep_via_geak
from hyperloom.orchestrator.actions.executors._workload_envs import geak_metric_axis


def _clear_axis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)


@pytest.mark.parametrize(
    ("benchmark_mode", "agentx_env", "expected"),
    [
        ("", None, ("output", "aggregate_output_tok_s")),
        ("synthetic", None, ("output", "aggregate_output_tok_s")),
        ("agentx", None, ("intvty", "e2e_norm_intvty_p90")),
        # The persisted mode is the durable signal, but a round driven from a
        # subprocess that only inherited the env var must resolve the same way.
        ("", "1", ("intvty", "e2e_norm_intvty_p90")),
    ],
)
def test_the_axis_follows_the_grader(monkeypatch, benchmark_mode, agentx_env, expected):
    _clear_axis_env(monkeypatch)
    if agentx_env is not None:
        monkeypatch.setenv("HYPERLOOM_AGENTX", agentx_env)
    assert geak_metric_axis(benchmark_mode=benchmark_mode) == expected


def test_an_explicit_override_wins_in_both_directions(monkeypatch):
    """An operator who names the metric is not overridden by the workload."""
    _clear_axis_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert geak_metric_axis(benchmark_mode="agentx")[0] == "output"

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", INTVTY_V1)
    assert geak_metric_axis(benchmark_mode="synthetic")[0] == "intvty"


def _sweep_with_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary: dict[str, Any]):
    """Drive ``sweep_via_geak`` against a stubbed GEAK that writes *summary*."""
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "1")

    def _fake_run(_cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
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
        )

    return _go


@pytest.mark.asyncio
async def test_a_total_mode_summary_still_reports_a_throughput(tmp_path, monkeypatch):
    """The case that used to record "no throughput" for every rung.

    Total mode nulls the output-named alias, so a reader keyed on it sees None,
    fails the ``tput > 0`` guard, and marks the variant failed even though GEAK
    measured the requested axis perfectly well.
    """
    go = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "throughput_tok_s_median": 23_697.0,
            "output_throughput_tok_s_median": None,
            "metric_basis": "aggregate_total_token_tok_s",
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
    )
    result = await go()
    assert result["status"] == "succeeded"
    points = result["points"]
    assert [p["status"] for p in points] == ["succeeded"]
    # The sweep row's field keeps its historical ``output_throughput`` name while
    # carrying the graded basis; ``metric_basis`` in the summary is what says
    # which axis it is. Renaming the row field would ripple through every
    # downstream sweep consumer, so it is left alone deliberately.
    assert points[0]["output_throughput"] == pytest.approx(23_697.0)


@pytest.mark.asyncio
async def test_an_intvty_mode_summary_is_not_published_as_output_throughput(tmp_path, monkeypatch):
    """The interactivity median is a per-request rate, not a throughput.

    ``output_throughput`` is ``GRADED_OUTPUT``, which the perf snapshot reads, so publishing
    an interactivity score there would record it as the session's output throughput. Total
    mode keeps the historical name (it really is a throughput); this axis does not.
    """
    go = _sweep_with_summary(
        tmp_path,
        monkeypatch,
        {
            "throughput_tok_s_median": 84.45,
            "output_throughput_tok_s_median": None,
            "metric_basis": "e2e_norm_intvty_p90",
            "guard_total_tok_s_median": 59_226.0,
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
        },
    )
    result = await go()
    assert result["status"] == "succeeded"
    point = result["points"][0]
    assert point["measured_value"] == pytest.approx(84.45)
    assert point["metric_basis"] == "e2e_norm_intvty_p90"
    assert "output_throughput" not in point
    # The rung must still be selectable, or an agentic sweep reports nothing again.
    assert result["promotion_measurement"]["measured_value"] == pytest.approx(84.45)


@pytest.mark.asyncio
async def test_an_output_mode_summary_reads_the_same_number_as_before(tmp_path, monkeypatch):
    """Synthetic parity: in output mode both fields carry the same median.

    Confirmed against a real run's ``bench_summary.json``, where
    ``throughput_tok_s_median`` and ``output_throughput_tok_s_median`` were both
    167.259, so preferring the neutral field changes nothing here.
    """
    go = _sweep_with_summary(
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
    assert result["status"] == "succeeded"
    assert result["points"][0]["output_throughput"] == pytest.approx(167.259)


@pytest.mark.asyncio
async def test_a_legacy_summary_without_the_neutral_field_still_reads(tmp_path, monkeypatch):
    """Summaries written before ``throughput_tok_s_median`` existed must still work."""
    go = _sweep_with_summary(
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
