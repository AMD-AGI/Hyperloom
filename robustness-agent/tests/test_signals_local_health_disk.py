"""Unit tests for disk / shm / fd / ray_head signals (A3 / A4 / A5 / A6)."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.local_health import (
    LocalHealthConfig,
    evaluate_local_health_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# disk_pressure
# ---------------------------------------------------------------------------

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
    # But shm_pressure fires.
    assert any(s.name == "shm_pressure" for s in out)


# ---------------------------------------------------------------------------
# shm_pressure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ray_head_dead
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# fd_pressure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------

def test_custom_disk_thresholds():
    cfg = LocalHealthConfig(disk_used_warn_pct=50.0, disk_used_crit_pct=70.0)
    data = SourceData(local_disk={
        "/": {"used_pct": 75.0, "used_gb": 150, "free_gb": 50, "total_gb": 200},
    })
    out = evaluate_local_health_signals(_ctx(), data, config=cfg)
    sym = next(s for s in out if s.name == "disk_pressure")
    assert sym.severity is SymptomSeverity.HIGH
