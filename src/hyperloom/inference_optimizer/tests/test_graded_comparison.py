# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX promotion is a fixed seven-condition all-of policy."""

from __future__ import annotations

from copy import deepcopy

import pytest

from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P90,
    GRADED_OUTPUT,
    GRADED_OUTPUT_PER_GPU,
    VERDICT_KEEP,
    VERDICT_REVERT,
    graded_axes_of,
)
from hyperloom.orchestrator.state.shared_state import (
    resolve_graded_comparison,
    resolve_grading_anchor_tput,
)


_ANCHOR = {
    "tput": 800.0,
    "output_throughput": 800.0,
    "input_throughput": 24000.0,
    "total_throughput": 24800.0,
    "e2e_intvty_p50": 100.0,
    "e2e_intvty_p90": 80.0,
    "output_tput_per_gpu": 100.0,
    "duration_s": 100.0,
    "request_error_rate": 1.0,
    "submission_valid": True,
    "accuracy_passed": True,
    "ttft_p50_ms": 1000.0,
    "ttft_p90_ms": 1400.0,
    "tpot_p50_ms": 10.0,
    "tpot_p90_ms": 15.0,
}


class _State:
    """Minimal SharedState double: only attributes read by the resolver."""

    def __init__(
        self,
        *,
        current_best=None,
        baseline_perf=None,
        baseline_tput=0.0,
        framework="vllm",
        benchmark_mode="",
        grading=None,
    ):
        self.current_best = current_best if current_best is not None else {}
        self.baseline_perf = baseline_perf if baseline_perf is not None else {}
        self.baseline_tput = baseline_tput
        self.framework = framework
        self.benchmark_mode = benchmark_mode
        self.grading = grading if grading is not None else {}


def _agentx(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


def _synthetic(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _candidate(**overrides) -> dict:
    measurement = {
        "output_throughput": 840.0,
        "input_throughput": 1.0,
        "total_token_throughput": 1.0,
        "e2e_intvty_p50": 103.0,
        "e2e_intvty_p90": 76.0,
        "output_tput_per_gpu": 95.0,
        "duration_s": 105.0,
        "request_error_rate": 1.0,
        "submission_valid": True,
        "accuracy_passed": True,
        "ttft_p50_ms": 900.0,
        "ttft_p90_ms": 1300.0,
        "tpot_p50_ms": 9.0,
        "tpot_p90_ms": 14.0,
    }
    measurement.update(overrides)
    return measurement


def test_agentx_keep_requires_and_accepts_all_seven_boundary_conditions(monkeypatch):
    _agentx(monkeypatch)
    graded = resolve_graded_comparison(
        _State(current_best=_ANCHOR),
        _candidate(),
        keep_threshold_pct=99.0,
    )

    assert graded.objective == GRADED_INTVTY
    assert graded.candidate == pytest.approx(103.0)
    assert graded.reference == pytest.approx(100.0)
    assert graded.tput_candidate == pytest.approx(95.0)
    assert graded.tput_reference == pytest.approx(100.0)
    assert graded.verdict == VERDICT_KEEP
    assert graded.anchor_source == "current_best"
    assert graded.failed_checks == ()
    assert all((graded.checks or {}).values())
    assert graded.deltas_pct == pytest.approx(
        {
            GRADED_INTVTY: 3.0,
            GRADED_INTVTY_P90: -5.0,
            GRADED_OUTPUT_PER_GPU: -5.0,
            "duration_s": 5.0,
        }
    )


@pytest.mark.parametrize(
    "overrides,failed",
    [
        ({"submission_valid": False}, "submission_valid"),
        ({"request_error_rate": 1.01}, "request_error_rate"),
        ({"accuracy_passed": False}, "accuracy_passed"),
        ({"duration_s": 105.01}, "duration_within_5pct"),
        ({"duration_s": 94.99}, "duration_within_5pct"),
        ({"e2e_intvty_p50": 102.99}, "e2e_intvty_p50_gain"),
        ({"e2e_intvty_p90": 75.99}, "e2e_intvty_p90_floor"),
        ({"output_tput_per_gpu": 94.99}, "output_tput_per_gpu_floor"),
    ],
)
def test_any_single_failed_condition_reverts(monkeypatch, overrides, failed):
    _agentx(monkeypatch)
    graded = resolve_graded_comparison(_State(current_best=_ANCHOR), _candidate(**overrides))

    assert graded.verdict == VERDICT_REVERT
    assert failed in graded.failed_checks
    assert graded.checks and graded.checks[failed] is False


@pytest.mark.parametrize(
    "field,failed",
    [
        ("submission_valid", "submission_valid"),
        ("request_error_rate", "request_error_rate"),
        ("accuracy_passed", "accuracy_passed"),
        ("duration_s", "duration_within_5pct"),
        ("e2e_intvty_p50", "e2e_intvty_p50_gain"),
        ("e2e_intvty_p90", "e2e_intvty_p90_floor"),
        ("output_tput_per_gpu", "output_tput_per_gpu_floor"),
    ],
)
def test_missing_policy_input_fails_closed(monkeypatch, field, failed):
    _agentx(monkeypatch)
    candidate = _candidate()
    candidate.pop(field)
    graded = resolve_graded_comparison(_State(current_best=_ANCHOR), candidate)

    assert graded.objective == GRADED_INTVTY
    assert graded.verdict == VERDICT_REVERT
    assert graded.comparable is False
    assert failed in graded.failed_checks
    assert field in graded.degrade_reason


def test_agentx_never_returns_recorded(monkeypatch):
    _agentx(monkeypatch)
    graded = resolve_graded_comparison(
        _State(current_best=_ANCHOR),
        _candidate(e2e_intvty_p50=101.0, e2e_intvty_p90=100.0, output_tput_per_gpu=120.0),
    )
    assert graded.verdict == VERDICT_REVERT
    assert graded.failed_checks == ("e2e_intvty_p50_gain",)


def test_total_throughput_no_longer_participates(monkeypatch):
    _agentx(monkeypatch)
    low_total = resolve_graded_comparison(
        _State(current_best=_ANCHOR),
        _candidate(total_token_throughput=1.0, input_throughput=0.1),
    )
    high_total = resolve_graded_comparison(
        _State(current_best=_ANCHOR),
        _candidate(total_token_throughput=999999.0, input_throughput=999000.0),
    )
    assert low_total.verdict == high_total.verdict == VERDICT_KEEP


def test_current_best_is_preferred_over_baseline(monkeypatch):
    _agentx(monkeypatch)
    current_best = {**_ANCHOR, "e2e_intvty_p50": 120.0}
    baseline = {**_ANCHOR, "e2e_intvty_p50": 90.0}
    graded = resolve_graded_comparison(
        _State(current_best=current_best, baseline_perf=baseline),
        _candidate(e2e_intvty_p50=110.0),
    )

    assert graded.anchor_source == "current_best"
    assert graded.reference == 120.0
    assert graded.verdict == VERDICT_REVERT


def test_baseline_is_used_only_when_current_best_is_absent(monkeypatch):
    _agentx(monkeypatch)
    graded = resolve_graded_comparison(_State(current_best={}, baseline_perf=_ANCHOR), _candidate())
    assert graded.anchor_source == "baseline"
    assert graded.verdict == VERDICT_KEEP


def test_incomplete_current_best_does_not_fall_through_to_baseline(monkeypatch):
    _agentx(monkeypatch)
    current_best = dict(_ANCHOR)
    current_best.pop("duration_s")
    graded = resolve_graded_comparison(
        _State(current_best=current_best, baseline_perf=_ANCHOR),
        _candidate(),
    )

    assert graded.anchor_source == "current_best"
    assert graded.verdict == VERDICT_REVERT
    assert graded.comparable is False
    assert "current_best_fields_missing:duration_s" in graded.degrade_reason


def test_explicit_round_anchor_is_used_by_explore(monkeypatch):
    _agentx(monkeypatch)
    round_anchor = {**_ANCHOR, "e2e_intvty_p50": 90.0}
    graded = resolve_graded_comparison(
        _State(current_best=_ANCHOR),
        _candidate(e2e_intvty_p50=93.0),
        anchor_perf=round_anchor,
    )
    assert graded.anchor_source == "round_anchor"
    assert graded.reference == 90.0
    assert graded.verdict == VERDICT_KEEP


def test_against_baseline_ignores_current_best(monkeypatch):
    _agentx(monkeypatch)
    current_best = {**_ANCHOR, "e2e_intvty_p50": 120.0}
    graded = resolve_graded_comparison(
        _State(current_best=current_best, baseline_perf=_ANCHOR),
        _candidate(),
        against_baseline=True,
    )
    assert graded.anchor_source == "baseline"
    assert graded.reference == 100.0


def test_policy_evidence_contains_checks_deltas_and_both_measurements(monkeypatch):
    _agentx(monkeypatch)
    graded = resolve_graded_comparison(_State(current_best=_ANCHOR), _candidate(request_error_rate=2.0))
    evidence = graded.policy_evidence()

    assert evidence["verdict"] == VERDICT_REVERT
    assert evidence["anchor_source"] == "current_best"
    assert evidence["checks"]["request_error_rate"] is False
    assert evidence["failed_checks"] == ["request_error_rate"]
    assert evidence["deltas_pct"][GRADED_INTVTY] == pytest.approx(3.0)
    assert evidence["candidate"]["request_error_rate"] == 2.0
    assert evidence["anchor"]["request_error_rate"] == 1.0


def test_synthetic_run_keeps_existing_output_throughput_rule(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=800.0)
    graded = resolve_graded_comparison(state, {"output_throughput": 808.0}, keep_threshold_pct=1.0)
    assert graded.objective == GRADED_OUTPUT
    assert graded.verdict == VERDICT_KEEP


def test_scriptable_framework_keeps_output_grading_under_agentx(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=800.0, framework="xdit")
    graded = resolve_graded_comparison(state, {"output_throughput": 808.0}, keep_threshold_pct=1.0)
    assert graded.objective == GRADED_OUTPUT
    assert graded.verdict == VERDICT_KEEP


def test_persisted_agentx_marker_reaches_the_chokepoint(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(
        current_best=deepcopy(_ANCHOR),
        benchmark_mode="agentx",
        grading={"objective": GRADED_INTVTY},
    )
    graded = resolve_graded_comparison(state, _candidate())
    assert graded.objective == GRADED_INTVTY
    assert graded.verdict == VERDICT_KEEP


def test_native_integrate_uses_the_same_agentx_verdict(monkeypatch):
    from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_perf=_ANCHOR, baseline_tput=800.0)
    state.optimization_stack = []

    kept = assess_integrate_performance(
        state,
        _candidate(),
        base_tput=800.0,
        keep_threshold_pct=99.0,
        stack_incremental_keep_threshold_pct=99.0,
    )
    reverted = assess_integrate_performance(
        state,
        _candidate(accuracy_passed=False),
        base_tput=800.0,
        keep_threshold_pct=0.0,
        stack_incremental_keep_threshold_pct=0.0,
    )

    assert kept.decision == "KEEP"
    assert reverted.decision == "REVERT"


def test_grading_anchor_tput_remains_the_legacy_output_axis_for_non_agentx_callers(monkeypatch):
    _synthetic(monkeypatch)
    assert resolve_grading_anchor_tput(_State(current_best=_ANCHOR, baseline_tput=700.0)) == 800.0


def test_winner_record_carries_all_user_facing_metrics(monkeypatch):
    _agentx(monkeypatch)
    winner = {"name": "candidate", **graded_axes_of(_candidate()), **_candidate()}
    graded = resolve_graded_comparison(_State(current_best=winner), _candidate(e2e_intvty_p50=106.09))
    assert graded.objective == GRADED_INTVTY
    assert graded.reference == 103.0
