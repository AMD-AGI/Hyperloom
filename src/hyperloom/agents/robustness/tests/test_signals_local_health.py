# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the local-health signal rule (server/log/gpu/disk/shm/fd/ray
+ D1 log-pattern extensions)."""

from __future__ import annotations

import pytest

from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals import (
    SymptomSeverity,
    evaluate_local_health_signals,
)
from hyperloom.agents.robustness.signals.local_health import LocalHealthConfig
from hyperloom.agents.robustness.sources.base import SourceData
from hyperloom.agents.robustness.sources.local_probe import (
    _DEFAULT_LOG_ERROR_PATTERNS,
    _extract_log_errors,
)


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(),
        inbox=[],
        now_unix=1.0,
    )


def _live_server() -> list[dict]:
    """A server process the probe can hold accountable for answering."""
    return [{"pid": 4242, "rss_mb": 1024.0, "cmd": "python -m sglang.launch_server", "is_server": True}]


def test_one_target_down_emits_medium_alert():
    data = SourceData(
        local_processes=_live_server(),
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": True, "status": "ok"},
            {"url": "http://localhost:30001", "reachable": False, "status": "error", "error": "connect"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 1
    assert matched[0].severity is SymptomSeverity.MEDIUM
    assert matched[0].evidence["url"] == "http://localhost:30001"


def test_all_targets_down_promotes_severity_to_high():
    data = SourceData(
        local_processes=_live_server(),
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": False, "status": "error"},
            {"url": "http://localhost:30001", "reachable": False, "status": "http_error"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 2
    assert all(s.severity is SymptomSeverity.HIGH for s in matched)


def test_a_refused_port_with_no_server_behind_it_is_not_a_fault():
    """Preparation, analysis and the gap between variants all run with no server up."""
    data = SourceData(
        local_processes=[{"pid": 7, "rss_mb": 12.0, "cmd": "python -m Magpie.bench", "is_server": False}],
        local_server_health=[
            {"url": "http://localhost:8888/health", "reachable": False, "status": "error", "error": "connect"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "local_server_unreachable" for s in out)


def test_a_server_that_died_under_a_running_benchmark_is_still_a_fault():
    """A benchmark client hammering the port proves a server was meant to answer it.

    "Probed successfully, saw no server" is the gap between two variants, but it
    is also a server that crashed while its own client kept sending requests —
    the one snapshot where suppressing the alert hides the outage.

    No session directory is configured here, so the client check has nothing to
    attribute against and stays host-wide.
    """
    data = SourceData(
        local_processes=[
            {"pid": 7, "rss_mb": 12.0, "cmd": "python -m Magpie.bench", "is_server": False},
            {"pid": 8, "rss_mb": 96.0, "cmd": "python benchmark_serving.py --port 30000", "is_server": False},
        ],
        local_server_health=[
            {"url": "http://localhost:30000/health", "reachable": False, "status": "error", "error": "connect"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 1
    assert matched[0].severity is SymptomSeverity.HIGH
    assert matched[0].evidence["server_process_seen"] is False
    assert matched[0].evidence["benchmark_client_seen"] is True


def _client_snapshot(cmd: str, cwd: str) -> SourceData:
    """A refused port, no server process, and one benchmark client running."""
    return SourceData(
        local_processes=[{"pid": 8, "rss_mb": 96.0, "cmd": cmd, "cwd": cwd, "is_server": False}],
        local_server_health=[
            {"url": "http://localhost:30000/health", "reachable": False, "status": "error", "error": "connect"},
        ],
    )


_OUTSIDE_ANY_SESSION = "/tmp"  # nosec B108 - a path in a fixture, not a temp file.
_CLIENT = "python benchmark_serving.py --port 30000"
_CLIENT_RESULT_DIR = "python benchmark_serving.py --result-dir {dir}/runs/v1 --port 30000"
_CLIENT_RESULT_DIR_EQ = "python benchmark_serving.py --result-dir={dir}/runs/v1 --port 30000"


@pytest.mark.parametrize(
    ("client_dir", "client_cwd", "client_cmd", "vouches"),
    [
        pytest.param("ours", "{dir}/runs/v1", _CLIENT, True, id="ours_by_cwd"),
        pytest.param("ours", "{dir}", _CLIENT, True, id="ours_by_cwd_at_the_session_root"),
        pytest.param("ours", _OUTSIDE_ANY_SESSION, _CLIENT_RESULT_DIR, True, id="ours_by_result_dir"),
        pytest.param("ours", _OUTSIDE_ANY_SESSION, _CLIENT_RESULT_DIR_EQ, True, id="ours_by_result_dir_joined_by_="),
        pytest.param("ours", _OUTSIDE_ANY_SESSION, _CLIENT, False, id="no_anchor_at_all"),
        pytest.param("theirs", "{dir}/runs/v1", _CLIENT_RESULT_DIR, False, id="unrelated_co_tenant"),
        pytest.param("sibling", "{dir}/runs/v1", _CLIENT_RESULT_DIR, False, id="co_tenant_one_string_prefix_away"),
        pytest.param(
            "sibling",
            _OUTSIDE_ANY_SESSION,
            _CLIENT_RESULT_DIR_EQ,
            False,
            id="co_tenant_one_string_prefix_away_joined_by_=",
        ),
    ],
)
def test_only_this_sessions_benchmark_client_vouches_for_a_dead_server(
    tmp_path,
    client_dir,
    client_cwd,
    client_cmd,
    vouches,
):
    """A client vouches for a refused port only when it belongs to this session.

    The harness runs with its cwd inside the session and children inherit it, so
    the cwd is the anchor; a launch path that chdirs elsewhere still names a path
    under the session on its command line, which is the second anchor. Anything
    else is somebody else's traffic — including the sibling directory whose name
    merely starts with ours (``<session>-retry``), which a substring test reads
    as inside the session. The command line is held to that boundary in both
    spellings of the flag, since the two are read by different code.

    A client with neither anchor is indistinguishable from a co-tenant's and
    must not vouch either; every launch path in this repo carries one, which is
    what the grid-runner cwd test holds it to.
    """
    ours = tmp_path / "session-a"
    dirs = {"ours": ours, "theirs": tmp_path / "session-b", "sibling": tmp_path / "session-a-retry"}
    data = _client_snapshot(
        cmd=client_cmd.format(dir=dirs[client_dir]),
        cwd=client_cwd.format(dir=dirs[client_dir]),
    )
    matched = [
        s
        for s in evaluate_local_health_signals(_ctx(), data, config=LocalHealthConfig(session_dir=ours))
        if s.name == "local_server_unreachable"
    ]
    assert bool(matched) is vouches
    if vouches:
        assert matched[0].severity is SymptomSeverity.HIGH
        assert matched[0].evidence["benchmark_client_seen"] is True


def test_a_refused_port_is_still_a_fault_when_nobody_could_look_for_the_server():
    """A broken ``ps`` must not mute a finding that has nothing to do with it."""
    data = SourceData(
        local_processes=[],
        local_processes_known=False,
        local_server_health=[
            {"url": "http://localhost:8888/health", "reachable": False, "status": "error", "error": "connect"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "local_server_unreachable"]
    assert len(matched) == 1
    assert matched[0].evidence["server_process_seen"] is None
    assert matched[0].evidence["benchmark_client_seen"] is None


def test_a_seen_server_is_recorded_in_the_evidence():
    data = SourceData(
        local_processes=_live_server(),
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": False, "status": "error"},
        ],
    )
    matched = [s for s in evaluate_local_health_signals(_ctx(), data) if s.name == "local_server_unreachable"]
    assert matched and matched[0].evidence["server_process_seen"] is True


def test_all_targets_down_suggests_the_real_dispatchable_action():
    """The HIGH suggestion must not point at ``server_lifecycle`` -- PolicyGate
    rejects it as unknown_action; action_ladder routes this symptom to a real
    ``delegate(recover, force_gpu_cleanup=True)``, and the suggestion text
    reaching the orchestration prompt must say so instead.

    A live server process is included so the idle-server-detection added
    upstream (a refused port with no server and no benchmark client behind it
    is treated as an expected idle stretch, not a fault) doesn't suppress the
    symptom this test exists to check.
    """
    data = SourceData(
        local_processes=_live_server(),
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": False, "status": "error"},
        ],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    matched = next(s for s in out if s.name == "local_server_unreachable")
    assert "server_lifecycle" not in matched.suggestion
    assert "delegate(recover" in matched.suggestion


def test_no_unreachable_targets_is_silent():
    data = SourceData(
        local_server_health=[
            {"url": "http://localhost:30000", "reachable": True, "status": "ok"},
        ]
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "local_server_unreachable" for s in out)


def test_oom_pattern_is_high_severity():
    data = SourceData(local_log_errors=[{"pattern": "CUDA out of memory", "line": "torch ... CUDA out of memory ..."}])
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "log_error_pattern"]
    assert matched and matched[0].severity is SymptomSeverity.HIGH


def test_runtimeerror_pattern_is_medium_severity():
    data = SourceData(local_log_errors=[{"pattern": "RuntimeError", "line": "RuntimeError: ..."}])
    out = evaluate_local_health_signals(_ctx(), data)
    matched = [s for s in out if s.name == "log_error_pattern"]
    assert matched and matched[0].severity is SymptomSeverity.MEDIUM


def test_oom_pattern_suggests_the_real_dispatchable_action():
    data = SourceData(local_log_errors=[{"pattern": "CUDA out of memory", "line": "torch ... CUDA out of memory ..."}])
    out = evaluate_local_health_signals(_ctx(), data)
    matched = next(s for s in out if s.name == "log_error_pattern")
    assert "server_lifecycle" not in matched.suggestion
    assert "delegate(recover" in matched.suggestion


def test_log_error_groups_samples_by_pattern():
    data = SourceData(local_log_errors=[{"pattern": "RuntimeError", "line": f"err {i}"} for i in range(5)])
    out = evaluate_local_health_signals(_ctx(), data)
    matched = next(s for s in out if s.name == "log_error_pattern")
    assert matched.evidence["count"] == 5
    assert len(matched.evidence["samples"]) == 3


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


def test_classifier_includes_local_health_rule():
    from hyperloom.agents.robustness.signals import Classifier

    data = SourceData(
        local_log_errors=[{"pattern": "CUDA out of memory", "line": "..."}],
        local_processes=_live_server(),
        local_server_health=[{"url": "u", "reachable": False, "status": "error"}],
        local_gpu={"gpus": [{"gpu_id": 0, "temperature_c": 99.5}]},
    )
    classifier = Classifier()
    out = classifier.classify(data, _ctx())
    names = {s.name for s in out}
    assert {"log_error_pattern", "local_server_unreachable", "gpu_thermal_high"} <= names


# D1 — log-pattern extension coverage: verify each pattern is detected + routed.


@pytest.mark.parametrize(
    "line,expected_pattern",
    [
        ("Engine core EngineCore-1 died unexpectedly", r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed", r"RuntimeError: Engine core initialization failed"),
        ("OSError: [Errno 98] Address already in use", r"Address already in use"),
        ("sglang tokenizer worker tw-3 died (signal 9)", r"tokenizer worker .* died"),
        ("MLA-style attention not supported in this checkpoint", r"MLA.*not supported"),
        ("MTP draft model unavailable for spec decoding", r"MTP draft .* unavailable"),
        ("aiter rms_norm compilation failed: nvcc returned exit 1", r"aiter .* compilation failed"),
        ("hipcc died with signal 9 (SIGKILL)", r"hipcc .* signal"),
        ("accuracy MMLU gate failed; reverting integrate", r"accuracy .* gate failed"),
        ("Eval result: MMLU 67.3% below threshold (74%)", r"MMLU .* below threshold"),
        ("ROCblas internal error: rocblasStatus 2", r"ROCblas.*Status\s*\d+"),
        ("hipBLAS Error: handle is in invalid state", r"hipBLAS.*Error"),
        ("NCCL WARN [Worker 3] timeout after 600 seconds", r"NCCL WARN .* timeout"),
        ("Failed to load checkpoint /weights/dsr1/safetensors", r"Failed to load checkpoint"),
        ("runtime.cli prepare-review timed out after 30s", r"runtime\.cli .* timed out after \d+s"),
        ("cudaErrorOutOfDevice while allocating KV cache", r"cudaErrorOutOfDevice"),
        ("HSA_STATUS_ERROR_OUT_OF_RESOURCES at hipDeviceAlloc", r"HSA_STATUS_ERROR_OUT_OF_RESOURCES"),
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
        ("Engine core EngineCore-1 died unexpectedly", r"Engine core .* died"),
        ("RuntimeError: Engine core initialization failed", r"RuntimeError: Engine core initialization failed"),
        ("aiter fused_moe compilation failed: hipcc exit 1", r"aiter .* compilation failed"),
        ("Failed to load checkpoint /weights/dsr1", r"Failed to load checkpoint"),
        ("runtime.cli commit-review timed out after 30s", r"runtime\.cli .* timed out after \d+s"),
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
        ("tokenizer worker tw-2 died (signal 11)", r"tokenizer worker .* died"),
        ("NCCL WARN [Worker 0] timeout after 600s", r"NCCL WARN .* timeout"),
    ],
)
def test_d1_medium_severity_pattern_emits_medium(line, expected_pattern):
    data = SourceData(
        local_log_errors=[{"pattern": expected_pattern, "line": line}],
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "log_error_pattern")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_disk_pressure_silent_below_warn():
    data = SourceData(
        local_disk={
            "/": {"used_pct": 50.0, "used_gb": 100, "free_gb": 100, "total_gb": 200},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "disk_pressure" for s in out)


def test_disk_pressure_medium_in_warn_zone():
    data = SourceData(
        local_disk={
            "/": {"used_pct": 88.0, "used_gb": 176, "free_gb": 24, "total_gb": 200},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.subject["mountpoint"] == "/"


def test_disk_pressure_high_in_crit_zone():
    data = SourceData(
        local_disk={
            "/": {"used_pct": 97.0, "used_gb": 194, "free_gb": 6, "total_gb": 200},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.HIGH


def test_disk_pressure_skips_shm_mountpoints():
    data = SourceData(
        local_disk={
            "/dev/shm": {"used_pct": 97.0, "used_gb": 31, "free_gb": 1, "total_gb": 32},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "disk_pressure" for s in out)
    assert any(s.name == "shm_pressure" for s in out)


def test_shm_pressure_silent_when_healthy():
    data = SourceData(
        local_disk={
            "/dev/shm": {"used_pct": 40.0, "used_gb": 12, "free_gb": 20, "total_gb": 32},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "shm_pressure" for s in out)


def test_shm_pressure_medium_at_warn():
    data = SourceData(
        local_disk={
            "/dev/shm": {"used_pct": 80.0, "used_gb": 25, "free_gb": 7, "total_gb": 32},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "shm_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_shm_pressure_high_at_crit():
    data = SourceData(
        local_disk={
            "/dev/shm": {"used_pct": 96.0, "used_gb": 31, "free_gb": 1, "total_gb": 32},
        }
    )
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
    data = SourceData(
        local_ray={
            "healthy": False,
            "reason": "ray status exit=1",
            "stderr": "Could not connect to GCS",
            "returncode": 1,
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "ray_head_dead")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["reason"] == "ray status exit=1"


def test_fd_pressure_silent_when_no_data():
    data = SourceData()
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "fd_pressure" for s in out)


def test_fd_pressure_silent_below_warn():
    data = SourceData(
        local_fd={
            "pid": 1234,
            "used": 200,
            "limit": 1024,
            "used_pct": 19.5,
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    assert all(s.name != "fd_pressure" for s in out)


def test_fd_pressure_medium_at_warn():
    data = SourceData(
        local_fd={
            "pid": 1234,
            "used": 850,
            "limit": 1024,
            "used_pct": 83.0,
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "fd_pressure")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_fd_pressure_high_at_crit():
    data = SourceData(
        local_fd={
            "pid": 1234,
            "used": 1000,
            "limit": 1024,
            "used_pct": 97.7,
        }
    )
    out = evaluate_local_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "fd_pressure")
    assert sym.severity is SymptomSeverity.HIGH


def test_custom_disk_thresholds():
    cfg = LocalHealthConfig(disk_used_warn_pct=50.0, disk_used_crit_pct=70.0)
    data = SourceData(
        local_disk={
            "/": {"used_pct": 75.0, "used_gb": 150, "free_gb": 50, "total_gb": 200},
        }
    )
    out = evaluate_local_health_signals(_ctx(), data, config=cfg)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.HIGH
