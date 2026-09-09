# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grading chokepoint reads candidate and reference off ONE axis."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT, GRADED_TOTAL, graded_axes_of
from hyperloom.orchestrator.state.shared_state import (
    resolve_graded_comparison,
    resolve_grading_anchor_tput,
)


# A measured AgentX round: prefill dominates, so the two axes cannot be confused.
_ANCHOR = {
    "tput": 183.44,
    "input_throughput": 25801.36,
    "output_throughput": 183.44,
    "total_throughput": 25984.80,
    "intvty_p90": 447.20,
}


class _State:
    """Minimal SharedState double: the attributes the resolvers read."""

    def __init__(
        self,
        *,
        current_best=None,
        baseline_perf=None,
        baseline_tput=0.0,
        framework="vllm",
        benchmark_mode="",
    ):
        self.current_best = current_best if current_best is not None else {}
        self.baseline_perf = baseline_perf if baseline_perf is not None else {}
        self.baseline_tput = baseline_tput
        self.framework = framework
        self.benchmark_mode = benchmark_mode


def _agentx(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


def _synthetic(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _full_measurement(*, total: float, output: float, intvty: float) -> dict[str, float]:
    return {
        "output_throughput": output,
        "input_throughput": total - output,
        "total_token_throughput": total,
        "intvty_p90": intvty,
    }


# --- both sides, one axis ---


def test_agentx_reads_both_sides_on_the_total_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_TOTAL
    assert graded.candidate == pytest.approx(26500.0)
    assert graded.reference == pytest.approx(_ANCHOR["total_throughput"])
    assert graded.degrade_reason == ""
    assert graded.comparable is True


@pytest.mark.parametrize(
    "missing_axes",
    [
        pytest.param(("input_throughput", "total_token_throughput"), id="total"),
        pytest.param(("intvty_p90",), id="intvty"),
        pytest.param(("input_throughput", "total_token_throughput", "intvty_p90"), id="both"),
    ],
)
def test_a_candidate_without_the_graded_axes_degrades_both_sides_together(monkeypatch, missing_axes):
    """Missing required axes preserve output diagnostics, not total comparability."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    measurement = _full_measurement(total=26500.0, output=190.0, intvty=450.0)
    for axis in missing_axes:
        measurement.pop(axis)
    graded = resolve_graded_comparison(state, measurement)

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == "candidate_axes_missing"
    assert graded.comparable is False
    assert graded.vetoed is False


@pytest.mark.parametrize(
    "missing_axes",
    [
        pytest.param(("input_throughput", "total_throughput"), id="total"),
        pytest.param(("intvty_p90",), id="intvty"),
        pytest.param(("input_throughput", "total_throughput", "intvty_p90"), id="both"),
    ],
)
def test_an_anchor_without_the_graded_axes_degrades_both_sides_together(monkeypatch, missing_axes):
    _agentx(monkeypatch)
    anchor = dict(_ANCHOR)
    for axis in missing_axes:
        anchor.pop(axis)
    state = _State(current_best=anchor, baseline_perf=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason == "current_best_axes_missing"
    assert graded.comparable is False
    assert graded.vetoed is False


def test_a_synthetic_run_grades_output_against_output(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == ""
    assert graded.comparable is True


@pytest.mark.parametrize("mode", ["explicit-output", "synthetic", "scriptable"])
def test_output_only_measurements_are_comparable_when_total_is_not_requested(monkeypatch, mode):
    _agentx(monkeypatch)
    if mode == "explicit-output":
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    elif mode == "synthetic":
        _synthetic(monkeypatch)
    state = _State(current_best={"tput": 100.0}, framework="xdit" if mode == "scriptable" else "vllm")

    graded = resolve_graded_comparison(state, {"output_throughput": 200.0})

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == 200.0
    assert graded.reference == 100.0
    assert graded.degrade_reason == ""
    assert graded.comparable is True
    assert graded.vetoed is False


def test_a_scriptable_framework_keeps_output_grading_under_agentx(monkeypatch):
    """A scriptable framework reports an image-quality gate, not a total axis."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0, framework="xdit")
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT


# --- the interactivity constraint travels with the objective ---


def test_an_interactivity_regression_is_vetoed_not_scored(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=40000.0, output=190.0, intvty=200.0))

    assert graded.graded_on_total
    assert graded.vetoed is True
    assert graded.comparable is True


def test_the_veto_never_fires_on_the_output_axis(monkeypatch):
    """The constraint belongs to the total objective."""
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=40000.0, output=190.0, intvty=1.0))

    assert graded.vetoed is False


# --- grading against the session baseline ---


def test_cumulative_gain_reads_the_baseline_on_the_graded_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(baseline_perf=_ANCHOR, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=450.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_TOTAL
    assert graded.reference == pytest.approx(_ANCHOR["total_throughput"])


def test_cumulative_gain_falls_back_to_baseline_tput_together(monkeypatch):
    """A resumed session that never wrote ``baseline_perf`` still grades one axis."""
    _agentx(monkeypatch)
    state = _State(baseline_perf={}, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=450.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason == "baseline_axes_missing"
    assert graded.comparable is False
    assert graded.vetoed is False


@pytest.mark.parametrize("missing_from", ["candidate", "reference"])
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize(
    "output,stack",
    [
        pytest.param(200.0, False, id="large-output-gain"),
        pytest.param(50.0, False, id="output-regression"),
        pytest.param(100.75, True, id="stack-output-gain"),
    ],
)
def test_native_performance_requires_requested_axes(monkeypatch, missing_from, missing_axis, output, stack):
    from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

    _agentx(monkeypatch)
    anchor = {"tput": 100.0, **_full_measurement(total=1000.0, output=100.0, intvty=100.0)}
    state = _State(current_best=dict(anchor), baseline_perf=anchor, baseline_tput=100.0)
    state.current_best["action"] = "integrate" if stack else "baseline"
    state.optimization_stack = [{"action": "integrate"}] if stack else []
    measurement = _full_measurement(total=1100.0, output=output, intvty=100.0)
    incomplete = measurement if missing_from == "candidate" else state.current_best
    for axis in ("input_throughput", "total_token_throughput") if missing_axis == "total" else ("intvty_p90",):
        incomplete.pop(axis)

    performance = assess_integrate_performance(
        state,
        measurement,
        base_tput=100.0,
        keep_threshold_pct=1.0,
        stack_incremental_keep_threshold_pct=0.5,
    )

    assert performance.decision == "NEEDS_REVIEW"
    assert performance.stack_positive_keep is False
    assert performance.gain_pct == pytest.approx(output - 100.0)
    assert performance.stack_incremental_gain_pct == pytest.approx(output - 100.0)
    assert performance.graded.objective == GRADED_OUTPUT
    assert performance.graded.candidate == output
    assert performance.graded.reference == 100.0
    assert performance.graded.vetoed is False
    assert performance.graded.degrade_reason == (
        "candidate_axes_missing" if missing_from == "candidate" else "current_best_axes_missing"
    )


# --- the anchor chokepoint stays on the output axis ---


@pytest.mark.parametrize("agentx", [True, False])
def test_the_anchor_chokepoint_is_the_output_axis_on_every_session(monkeypatch, agentx):
    """It seeds ``base_tput``, backs the drift check, and answers the two objective resolvers, whose targets are operator-supplied output figures."""
    _agentx(monkeypatch) if agentx else _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(_ANCHOR["tput"])


def test_the_anchor_falls_back_to_the_baseline_before_any_layer_lands(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best={}, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(180.0)


# --- a KEEP must not strip the axes off the anchor ---


def test_graded_axes_survive_a_winner_record(monkeypatch):
    """The defect this guards: total grading dies after the first KEEP."""
    _agentx(monkeypatch)
    measurement = _full_measurement(total=26500.0, output=190.0, intvty=450.0)
    winner = {"name": "v", "tput": 190.0, **graded_axes_of(measurement)}

    graded = resolve_graded_comparison(_State(current_best=winner, baseline_tput=180.0), measurement)
    assert graded.objective == GRADED_TOTAL
    assert graded.reference == pytest.approx(26500.0)


def test_graded_axes_of_omits_what_was_not_measured():
    assert graded_axes_of({"output_throughput": 190.0}) == {}
    assert graded_axes_of({"total_token_throughput": 26500.0}) == {"total_throughput": 26500.0}
    assert graded_axes_of(None) == {}


# --- the persisted marker reaches the chokepoint ---


def test_a_round_without_the_env_var_still_grades_on_total(monkeypatch):
    """A re-baseline or integrate round can be driven from a shell that never saw it."""
    _synthetic(monkeypatch)
    state = _State(current_best=dict(_ANCHOR), benchmark_mode="agentx")
    graded = resolve_graded_comparison(state, _full_measurement(total=27000.0, output=190.0, intvty=447.20))
    assert graded.objective == GRADED_TOTAL
    assert graded.candidate == pytest.approx(27000.0)
    assert graded.reference == pytest.approx(_ANCHOR["total_throughput"])


def test_a_synthetic_session_is_untouched_by_the_marker_check(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=dict(_ANCHOR))
    graded = resolve_graded_comparison(state, _full_measurement(total=27000.0, output=190.0, intvty=447.20))
    assert graded.objective == GRADED_OUTPUT
    assert graded.reference == pytest.approx(resolve_grading_anchor_tput(state))


@pytest.fixture
def baseline_writer(monkeypatch, tmp_path):
    from hyperloom.orchestrator.loop.writeback import WritebackCollaborator, _PromoteOutcome
    from hyperloom.orchestrator.state.shared_state import SharedState

    _agentx(monkeypatch)
    state = SharedState(framework="vllm", benchmark_mode="agentx")
    writer = WritebackCollaborator(SimpleNamespace(shared_state=state, session_dir=tmp_path))
    monkeypatch.setattr(writer, "_refresh_gaps", AsyncMock(), raising=False)
    monkeypatch.setattr(writer, "_drain_queued_baselines", AsyncMock())
    monkeypatch.setattr(writer, "_should_run_prelude_bootstrap", lambda _tput: False)
    monkeypatch.setattr(state, "record_baseline_roofline_ceiling", Mock())
    return writer, _PromoteOutcome()


@pytest.mark.parametrize("baseline_enablement", [True, False], ids=["enablement", "validated-layer"])
async def test_baseline_perf_updates_without_resetting_the_stack(baseline_writer, tmp_path, baseline_enablement):
    writer, outcome = baseline_writer
    state = writer.shared_state
    state.baseline_tput = 0.0 if baseline_enablement else _ANCHOR["tput"]
    state.baseline_perf = {} if baseline_enablement else dict(_ANCHOR)
    stack = [
        {
            "action": "integrate_patch",
            "baseline_enablement": baseline_enablement,
            "extra_server_args": "--enable-prefix-caching",
            "extra_envs": {"VLLM_USE_V1": "1"},
        }
    ]
    current_best = {
        "action": "integrate_patch",
        "tput": 210.0,
        "extra_server_args": "--enable-prefix-caching",
        "extra_envs": {"VLLM_USE_V1": "1"},
        "optimization_stack": deepcopy(stack),
        "measurement": {"tput": 210.0, "launch_identity": "stack-identity"},
    }
    state.optimization_stack = deepcopy(stack)
    state.current_best = deepcopy(current_best)
    state.current_best_measurement = dict(current_best["measurement"])
    result = {
        **_full_measurement(total=26500.0, output=190.0, intvty=450.0),
        "materialized_config": str(tmp_path / "baseline.yaml"),
        "workspace": str(tmp_path / "baseline"),
    }

    await writer._promote_baseline(result, task=None, outcome=outcome)

    assert outcome.audit_decision == "promoted"
    assert outcome.changed
    assert state.baseline_tput == 190.0
    assert state.baseline_config_path == result["materialized_config"]
    assert state.baseline_perf == {
        "total_throughput": 26500.0,
        "intvty_p90": 450.0,
        "input_throughput": 26310.0,
        "output_throughput": 190.0,
    }
    assert state.optimization_stack == stack
    assert state.current_best == current_best
    assert state.current_best_measurement == current_best["measurement"]
    graded = resolve_graded_comparison(state, result, against_baseline=True)
    assert graded.objective == GRADED_TOTAL
    assert graded.reference == 26500.0


@pytest.mark.parametrize("stacked", [False, True])
async def test_rejected_baseline_preserves_the_entire_anchor(baseline_writer, tmp_path, stacked):
    writer, outcome = baseline_writer
    state = writer.shared_state
    state.baseline_tput = _ANCHOR["tput"]
    state.baseline_perf = dict(_ANCHOR)
    state.baseline_config_path = str(tmp_path / "old.yaml")
    state.baseline_accuracy = 0.9
    state.baseline_workload_extra = {"workload_mode": "old"}
    state.baseline_runtime_sec = 60.0
    state.baseline_warm_runtime_sec = 40.0
    state.baseline_post_ready_runtime_sec = 20.0
    state.optimization_stack = [{"action": "integrate_patch"}] if stacked else []
    state.current_best = dict(_ANCHOR)
    state.current_best_measurement = {"launch_identity": "old-identity"}
    prior_baseline = {key: deepcopy(value) for key, value in vars(state).items() if key.startswith("baseline_")}
    result = {
        **_full_measurement(total=30000.0, output=180.0, intvty=460.0),
        "materialized_config": str(tmp_path / "rejected.yaml"),
        "accuracy": 0.95,
        "subprocess_runtime_sec": 50.0,
        "measure_round_runtime_sec": 30.0,
        "post_ready_runtime_sec": 10.0,
    }

    await writer._promote_baseline(result, task=None, outcome=outcome)

    assert outcome.audit_decision == "no_promote"
    assert {key: getattr(state, key) for key in prior_baseline} == prior_baseline
    assert state.current_best == _ANCHOR
    assert state.current_best_measurement == {"launch_identity": "old-identity"}


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("missing_axes", [("intvty_p90",), ("input_throughput", "total_token_throughput")])
async def test_accepted_baseline_without_axes_clears_stale_perf(baseline_writer, tmp_path, stacked, missing_axes):
    writer, outcome = baseline_writer
    state = writer.shared_state
    old_config = tmp_path / "old.yaml"
    new_config = tmp_path / "new.yaml"
    old_config.write_text("benchmark:\n  envs:\n    EXTRA_VLLM_ARGS: --max-num-seqs 4\n", encoding="utf-8")
    new_config.write_text("benchmark:\n  envs:\n    EXTRA_VLLM_ARGS: --max-num-seqs 8\n", encoding="utf-8")
    state.baseline_tput = _ANCHOR["tput"]
    state.baseline_perf = dict(_ANCHOR)
    state.baseline_config_path = str(old_config)
    state.optimization_stack = [{"action": "integrate_patch"}] if stacked else []
    state.current_best = dict(_ANCHOR)
    writer._stamp_current_best_measurement({"workspace": str(tmp_path / "old-measurement")})
    prior_best = deepcopy(state.current_best)
    result = {
        **_full_measurement(total=26500.0, output=190.0, intvty=450.0),
        "materialized_config": str(new_config),
        "workspace": str(tmp_path / "new-measurement"),
    }
    for axis in missing_axes:
        result.pop(axis)

    await writer._promote_baseline(result, task=None, outcome=outcome)

    assert outcome.audit_decision == "promoted"
    assert state.baseline_tput == 190.0
    assert state.baseline_config_path == result["materialized_config"]
    assert state.baseline_perf == {}
    if stacked:
        assert state.optimization_stack == [{"action": "integrate_patch"}]
        assert state.current_best == prior_best
        assert state.current_best_measurement == prior_best["measurement"]
    else:
        assert state.current_best["tput"] == 190.0
        assert state.current_best_measurement["benchmark_workspace"] == result["workspace"]
        assert state.current_best_measurement["launch_identity"] != prior_best["measurement"]["launch_identity"]
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=27000.0, output=195.0, intvty=450.0),
        against_baseline=True,
    )
    assert graded.objective == GRADED_OUTPUT
    assert graded.reference == 190.0
    assert graded.degrade_reason == "baseline_axes_missing"
