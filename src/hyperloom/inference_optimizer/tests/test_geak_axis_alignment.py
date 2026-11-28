# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK must measure the axis Hyperloom keeps candidates on.

An interactivity-graded session keeps a candidate only on a gain in the median
``e2e_norm_intvty_p50``, with the p90 tail and total token throughput as noise-band
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

Synthetic runs must come out byte-identical, so each case here has its
fixed-ISL/OSL twin.
"""

from __future__ import annotations

import pytest

from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    INTVTY_V1,
)
from hyperloom.orchestrator.actions.executors.workload._agentx_switch import (
    GEAK_METRIC_INTVTY,
    GEAK_METRIC_OUTPUT,
    build_agentx_workload_spec,
    geak_acceptance,
    geak_metric_axis,
)


def _clear_axis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
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
def test_the_axis_follows_the_grader(native_benchmark, monkeypatch, benchmark_mode, agentx_env, expected):
    _clear_axis_env(monkeypatch)
    if agentx_env is not None:
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
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
def test_a_recorded_objective_is_read_as_a_family_not_one_label(native_benchmark, monkeypatch, objective, expected):
    """Either interactivity percentile means the session is graded on interactivity.

    The env says AgentX throughout, so the output case also shows the recorded objective
    winning over the mode-based derivation.
    """
    _clear_axis_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    assert geak_metric_axis(benchmark_mode="agentx", grading={"objective": objective}) == expected


def test_an_explicit_override_wins_in_both_directions(native_benchmark, monkeypatch):
    """An operator who names the metric is not overridden by the workload."""
    _clear_axis_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    assert geak_metric_axis(benchmark_mode="agentx")[0] == "output"

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", INTVTY_V1)
    assert geak_metric_axis(benchmark_mode="synthetic")[0] == "e2e_norm_intvty_p50"


def test_the_interactivity_handoff_carries_the_rule_the_rebench_keeps_on():
    """GEAK's "ok" has to mean the rebench's KEEP, or every sub-threshold win is a wasted rebench.

    Output throughput guards nothing, so GEAK must not refuse on it; GEAK records no total-throughput guard, and one
    it did not record would fail every candidate.
    """
    assert geak_acceptance(GEAK_METRIC_INTVTY[1], 7.5) == {
        "objective": "e2e_norm_intvty_p50",
        "min_gain_pct": AgentXObjective.P50_INTERACTIVITY.bar_pct,
        "guard_max_drop_pct": {"e2e_norm_intvty_p90": 7.5},
    }
    assert geak_acceptance(GEAK_METRIC_OUTPUT[1], 7.5) is None


@pytest.mark.parametrize(
    ("backend_env", "objective", "rule_expected"),
    [
        ({"HYPERLOOM_AGENTIC_BACKEND": "mlperf"}, GRADED_INTVTY, True),
        ({"HYPERLOOM_AGENTIC_BACKEND": "mlperf"}, GRADED_OUTPUT, False),
    ],
)
def test_the_workload_spec_publishes_the_rule_with_the_sessions_own_band(backend_env, objective, rule_expected):
    """The guards' band is the one the session was seeded with, not this process's default."""
    spec = build_agentx_workload_spec({}, env=backend_env, grading={"objective": objective, "noise_pct": 7.5})
    if rule_expected:
        assert spec["acceptance"] == geak_acceptance(spec["metric_basis"], 7.5)
    else:
        assert "acceptance" not in spec
