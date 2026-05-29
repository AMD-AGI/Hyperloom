"""Unit tests for the local-health signal rule."""

from __future__ import annotations


from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import (
    SymptomSeverity,
    evaluate_local_health_signals,
)
from robustness_agent.signals.local_health import LocalHealthConfig
from robustness_agent.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# Server unreachable
# ---------------------------------------------------------------------------

def test_one_target_down_emits_medium_alert():
    data = SourceData(
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": True, "status": "ok"},
            {"url": "http://localhost:30001", "reachable": False, "status": "error", "error": "connect"},
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 1
    assert matched[0].severity is SymptomSeverity.MEDIUM
    assert matched[0].evidence["url"] == "http://localhost:30001"


def test_all_targets_down_promotes_severity_to_high():
    data = SourceData(
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": False, "status": "error"},
            {"url": "http://localhost:30001", "reachable": False, "status": "http_error"},
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 2
    assert all(s.severity is SymptomSeverity.HIGH for s in matched)


def test_no_unreachable_targets_is_silent():
    data = SourceData(
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": True, "status": "ok"},
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "local_server_unreachable" for s in out)


# ---------------------------------------------------------------------------
# Log error patterns
# ---------------------------------------------------------------------------

def test_oom_pattern_is_high_severity():
    data = SourceData(
        local_log_errors=[
            {"pattern": "CUDA out of memory", "line": "torch ... CUDA out of memory ..."}
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "log_error_pattern"]
    assert matched and matched[0].severity is SymptomSeverity.HIGH


def test_runtimeerror_pattern_is_medium_severity():
    data = SourceData(
        local_log_errors=[
            {"pattern": "RuntimeError", "line": "RuntimeError: ..."}
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "log_error_pattern"]
    assert matched and matched[0].severity is SymptomSeverity.MEDIUM


def test_log_error_groups_samples_by_pattern():
    data = SourceData(
        local_log_errors=[
            {"pattern": "RuntimeError", "line": f"err {i}"} for i in range(5)
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = next(s for s in out if s.name == "log_error_pattern")
    assert matched.evidence["count"] == 5
    assert len(matched.evidence["samples"]) == 3  # capped at 3


# ---------------------------------------------------------------------------
# GPU thermal
# ---------------------------------------------------------------------------

def test_gpu_at_warn_threshold_is_medium():
    data = SourceData(local_gpu={"gpus": [{"gpu_id": 0, "temperature_c": 92.0}]})
    out = evaluate_local_health_signals(_ctx(), data, config=LocalHealthConfig(gpu_temp_warn_c=90, gpu_temp_crit_c=100))
    matched = [s for s in out if s.name == "gpu_thermal_high"]
    assert matched and matched[0].severity is SymptomSeverity.MEDIUM


def test_gpu_at_crit_threshold_is_high():
    data = SourceData(local_gpu={"gpus": [{"gpu_id": 1, "temperature_c": 105.0}]})
    out = evaluate_local_health_signals(_ctx(), data, config=LocalHealthConfig(gpu_temp_warn_c=90, gpu_temp_crit_c=100))
    matched = [s for s in out if s.name == "gpu_thermal_high"]
    assert matched and matched[0].severity is SymptomSeverity.HIGH


def test_cool_gpu_emits_no_thermal_signal():
    data = SourceData(local_gpu={"gpus": [{"gpu_id": 0, "temperature_c": 60.0}]})
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "gpu_thermal_high" for s in out)


def test_no_local_gpu_data_silent():
    data = SourceData(local_gpu={})
    out = evaluate_local_health_signals(_ctx(), data)
    assert out == []


# ---------------------------------------------------------------------------
# Classifier integration
# ---------------------------------------------------------------------------

def test_classifier_includes_local_health_rule():
    from robustness_agent.signals import Classifier

    data = SourceData(
        local_log_errors=[{"pattern": "CUDA out of memory", "line": "..."}],
        local_server_health=[{"url": "u", "reachable": False, "status": "error"}],
        local_gpu={"gpus": [{"gpu_id": 0, "temperature_c": 99.5}]},
    )
    classifier = Classifier()
    out = classifier.classify(data, _ctx())
    names = {s.name for s in out}
    assert {"log_error_pattern", "local_server_unreachable", "gpu_thermal_high"} <= names
