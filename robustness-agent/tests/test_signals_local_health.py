# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the local-health signal rule (server/log/gpu/disk/shm/fd/ray
+ D1 log-pattern extensions)."""

from __future__ import annotations

import pytest

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
from robustness_agent.sources.local_probe import (
    _DEFAULT_LOG_ERROR_PATTERNS,
    _extract_log_errors,
)


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


# ===========================================================================
# D1 — log-pattern extension coverage
# ===========================================================================
# Patterns live in local_probe._DEFAULT_LOG_ERROR_PATTERNS and
# local_health._HIGH_SEVERITY_PATTERNS; verify each is detected + routed.

@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        ("Engine core EngineCore-1 died unexpectedly",
         r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed",
         r"RuntimeError: Engine core initialization failed"),
        ("OSError: [Errno 98] Address already in use",
         r"Address already in use"),
        ("sglang tokenizer worker tw-3 died (signal 9)",
         r"tokenizer worker .* died"),
        ("MLA-style attention not supported in this checkpoint",
         r"MLA.*not supported"),
        ("MTP draft model unavailable for spec decoding",
         r"MTP draft .* unavailable"),
        ("aiter rms_norm compilation failed: nvcc returned exit 1",
         r"aiter .* compilation failed"),
        ("hipcc died with signal 9 (SIGKILL)",
         r"hipcc .* signal"),
        ("accuracy MMLU gate failed; reverting integrate",
         r"accuracy .* gate failed"),
        ("Eval result: MMLU 67.3% below threshold (74%)",
         r"MMLU .* below threshold"),
        ("ROCblas internal error: rocblasStatus 2",
         r"ROCblas.*Status\s*\d+"),
        ("hipBLAS Error: handle is in invalid state",
         r"hipBLAS.*Error"),
        ("NCCL WARN [Worker 3] timeout after 600 seconds",
         r"NCCL WARN .* timeout"),
        ("Failed to load checkpoint /weights/dsr1/safetensors",
         r"Failed to load checkpoint"),
        ("runtime.cli prepare-review timed out after 30s",
         r"runtime\.cli .* timed out after \d+s"),
        ("cudaErrorOutOfDevice while allocating KV cache",
         r"cudaErrorOutOfDevice"),
        ("HSA_STATUS_ERROR_OUT_OF_RESOURCES at hipDeviceAlloc",
         r"HSA_STATUS_ERROR_OUT_OF_RESOURCES"),
    ],
)
def test_d1_new_pattern_matches(line, expected_pattern):
    hits = _extract_log_errors([line], _DEFAULT_LOG_ERROR_PATTERNS, window=10)
    matched_patterns = {h["pattern"] for h in hits}
    assert expected_pattern in matched_patterns


@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        ("CUDA out of memory at allocator.cc:42", "CUDA out of memory"),
        ("Engine core EngineCore-1 died unexpectedly",
         r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed",
         r"RuntimeError: Engine core initialization failed"),
        ("aiter fused_moe compilation failed: hipcc exit 1",
         r"aiter .* compilation failed"),
        ("Failed to load checkpoint /weights/dsr1",
         r"Failed to load checkpoint"),
        ("runtime.cli commit-review timed out after 30s",
         r"runtime\.cli .* timed out after \d+s"),
    ],
)
def test_d1_high_severity_pattern_emits_high(line, expected_pattern):
    data = SourceData(
        local_log_errors=[{"pattern": expected_pattern, "line": line}],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "log_error_pattern")
    assert sym.severity is SymptomSeverity.HIGH


@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        ("OSError: Address already in use", r"Address already in use"),
        ("tokenizer worker tw-2 died (signal 11)",
         r"tokenizer worker .* died"),
        ("NCCL WARN [Worker 0] timeout after 600s",
         r"NCCL WARN .* timeout"),
    ],
)
def test_d1_medium_severity_pattern_emits_medium(line, expected_pattern):
    data = SourceData(
        local_log_errors=[{"pattern": expected_pattern, "line": line}],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "log_error_pattern")
    assert sym.severity is SymptomSeverity.MEDIUM


# ===========================================================================
# disk / shm / fd / ray_head signals (A3 / A4 / A5 / A6)
# ===========================================================================

def test_disk_pressure_silent_below_warn():
    data = SourceData(local_disk={
        "/": {"used_pct": 50.0, "used_gb": 100, "free_gb": 100, "total_gb": 200},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "disk_pressure" for s in out)


def test_disk_pressure_medium_in_warn_zone():
    data = SourceData(local_disk={
        "/": {"used_pct": 88.0, "used_gb": 176, "free_gb": 24, "total_gb": 200},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.subject["mountpoint"] == "/"


def test_disk_pressure_high_in_crit_zone():
    data = SourceData(local_disk={
        "/": {"used_pct": 97.0, "used_gb": 194, "free_gb": 6, "total_gb": 200},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.HIGH


def test_disk_pressure_skips_shm_mountpoints():
    data = SourceData(local_disk={
        "/dev/shm": {"used_pct": 97.0, "used_gb": 31, "free_gb": 1, "total_gb": 32},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "disk_pressure" for s in out)
    assert any(s.name == "shm_pressure" for s in out)


def test_shm_pressure_silent_when_healthy():
    data = SourceData(local_disk={
        "/dev/shm": {"used_pct": 40.0, "used_gb": 12, "free_gb": 20, "total_gb": 32},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "shm_pressure" for s in out)


def test_shm_pressure_medium_at_warn():
    data = SourceData(local_disk={
        "/dev/shm": {"used_pct": 80.0, "used_gb": 25, "free_gb": 7, "total_gb": 32},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "shm_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_shm_pressure_high_at_crit():
    data = SourceData(local_disk={
        "/dev/shm": {"used_pct": 96.0, "used_gb": 31, "free_gb": 1, "total_gb": 32},
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "shm_pressure")
    assert sym.severity is SymptomSeverity.HIGH
    assert "SHM exhaustion" in sym.summary


def test_ray_head_dead_silent_when_no_data():
    data = SourceData()
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "ray_head_dead" for s in out)


def test_ray_head_dead_silent_when_healthy():
    data = SourceData(local_ray={"healthy": True, "stdout_head": "OK"})
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "ray_head_dead" for s in out)


def test_ray_head_dead_fires_high_when_unhealthy():
    data = SourceData(local_ray={
        "healthy": False,
        "reason": "ray status exit=1",
        "stderr": "Could not connect to GCS",
        "returncode": 1,
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "ray_head_dead")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["reason"] == "ray status exit=1"


def test_fd_pressure_silent_when_no_data():
    data = SourceData()
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "fd_pressure" for s in out)


def test_fd_pressure_silent_below_warn():
    data = SourceData(local_fd={
        "pid": 1234, "used": 200, "limit": 1024, "used_pct": 19.5,
    })
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "fd_pressure" for s in out)


def test_fd_pressure_medium_at_warn():
    data = SourceData(local_fd={
        "pid": 1234, "used": 850, "limit": 1024, "used_pct": 83.0,
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "fd_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_fd_pressure_high_at_crit():
    data = SourceData(local_fd={
        "pid": 1234, "used": 1000, "limit": 1024, "used_pct": 97.7,
    })
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "fd_pressure")
    assert sym.severity is SymptomSeverity.HIGH


def test_custom_disk_thresholds():
    cfg = LocalHealthConfig(disk_used_warn_pct=50.0, disk_used_crit_pct=70.0)
    data = SourceData(local_disk={
        "/": {"used_pct": 75.0, "used_gb": 150, "free_gb": 50, "total_gb": 200},
    })
    out = evaluate_local_health_signals(_ctx(), data, config=cfg)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.HIGH
