"""Unit tests for I1-I5 state-integrity signals."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.state_integrity import (
    StateIntegrityConfig,
    evaluate_state_integrity_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx(*, now_unix: float = 2_000_000.0) -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=now_unix,
    )


# ---------------------------------------------------------------------------
# I1 — state_json_corrupt
# ---------------------------------------------------------------------------

def test_i1_state_json_invalid_fires_high():
    data = SourceData(local_state_integrity={
        "state_json": {
            "valid": False, "error": "json_parse_failed",
            "path": "/p/state.json", "size_bytes": 23,
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "state_json_corrupt")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["error"] == "json_parse_failed"


def test_i1_silent_when_state_json_valid():
    data = SourceData(local_state_integrity={
        "state_json": {"valid": True, "path": "/p/state.json"},
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "state_json_corrupt" for s in out)


def test_i1_silent_on_missing_state_json():
    """``error='missing'`` is normal on tick 0 (Coordinator hasn't
    persisted yet). We only fire on corruption, not absence."""
    data = SourceData(local_state_integrity={
        "state_json": {"valid": False, "error": "missing",
                       "path": "/p/state.json"},
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "state_json_corrupt" for s in out)


def test_i1_fires_on_read_failed():
    """``error='read_failed: ...'`` (permission / I/O error) → HIGH."""
    data = SourceData(local_state_integrity={
        "state_json": {"valid": False, "error": "read_failed: EACCES",
                       "path": "/p/state.json"},
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert any(s.name == "state_json_corrupt" for s in out)


# ---------------------------------------------------------------------------
# I2 — coordinator_wal_bloat
# ---------------------------------------------------------------------------

def test_i2_wal_bloat_medium_at_warn():
    data = SourceData(local_state_integrity={
        "wal": {
            "wal_bytes": 2 * 1024 * 1024 * 1024,  # 2 GiB
            "db_bytes": 50 * 1024 * 1024,
            "wal_path": "/p/storage/coordinator.db-wal",
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "coordinator_wal_bloat")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_i2_wal_bloat_high_at_critical():
    data = SourceData(local_state_integrity={
        "wal": {
            "wal_bytes": 5 * 1024 * 1024 * 1024,  # 5 GiB
            "db_bytes": 50 * 1024 * 1024,
            "wal_path": "/p/storage/coordinator.db-wal",
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "coordinator_wal_bloat")
    assert sym.severity is SymptomSeverity.HIGH


def test_i2_silent_below_warn():
    data = SourceData(local_state_integrity={
        "wal": {"wal_bytes": 100 * 1024 * 1024, "db_bytes": 0},
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "coordinator_wal_bloat" for s in out)


# ---------------------------------------------------------------------------
# I3 — stale_lease
# ---------------------------------------------------------------------------

def test_i3_stale_lease_fires_when_holder_dead():
    """Lease held by dead PID for > stale_lease_min_age_s → HIGH."""
    data = SourceData(local_state_integrity={
        "leases": [
            {"task_id": "tsk-7", "holder_pid": 9999, "lane": "benchmark_lane",
             "alive": False, "acquired_at": 1_999_000.0},
        ],
    })
    out = evaluate_state_integrity_signals(_ctx(now_unix=2_000_000.0), data)
    sym = next(s for s in out if s.name == "stale_lease")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["task_id"] == "tsk-7"
    assert sym.evidence["holder_pid"] == 9999


def test_i3_silent_when_holder_alive():
    data = SourceData(local_state_integrity={
        "leases": [
            {"task_id": "tsk-8", "holder_pid": 1234, "lane": "lane-1",
             "alive": True, "acquired_at": 1_999_000.0},
        ],
    })
    out = evaluate_state_integrity_signals(_ctx(now_unix=2_000_000.0), data)
    assert all(s.name != "stale_lease" for s in out)


def test_i3_silent_when_lease_too_young():
    """Lease acquired 10s ago; reaper hasn't had a chance yet."""
    data = SourceData(local_state_integrity={
        "leases": [
            {"task_id": "tsk-9", "holder_pid": 9999, "lane": "lane-1",
             "alive": False, "acquired_at": 1_999_990.0},
        ],
    })
    out = evaluate_state_integrity_signals(_ctx(now_unix=2_000_000.0), data)
    # acquired 10s ago; default stale_lease_min_age_s=60.0 → silent.
    assert all(s.name != "stale_lease" for s in out)


def test_i3_handles_iso_timestamp_in_acquired_at():
    # Ancient ISO timestamp + ``now`` in 2026 → age_s >> threshold.
    now_2026 = 1_780_000_000.0  # ~2026-05
    data = SourceData(local_state_integrity={
        "leases": [
            {"task_id": "tsk-10", "holder_pid": 9999, "lane": "lane-1",
             "alive": False,
             "acquired_at": "2020-01-01T00:00:00+00:00"},
        ],
    })
    out = evaluate_state_integrity_signals(_ctx(now_unix=now_2026), data)
    sym = next(s for s in out if s.name == "stale_lease")
    assert sym.evidence["task_id"] == "tsk-10"


# ---------------------------------------------------------------------------
# I4 — inbox_bloat
# ---------------------------------------------------------------------------

def test_i4_inbox_bloat_low_at_warn():
    data = SourceData(local_state_integrity={
        "agents": {
            "orchestration": {
                "inbox_bytes": 150 * 1024 * 1024,  # 150 MiB > warn=100 MiB
                "outbox_bytes": 0,
                "inbox_path": "/p/agents/orchestration/inbox.jsonl",
            },
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "inbox_bloat")
    assert sym.severity is SymptomSeverity.LOW
    assert sym.evidence["role"] == "orchestration"
    assert sym.evidence["kind"] == "inbox"


def test_i4_inbox_bloat_medium_at_critical():
    data = SourceData(local_state_integrity={
        "agents": {
            "orchestration": {
                "inbox_bytes": 600 * 1024 * 1024,  # 600 MiB > crit=500 MiB
                "outbox_bytes": 0,
                "inbox_path": "/p/agents/orchestration/inbox.jsonl",
            },
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "inbox_bloat")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_i4_silent_below_threshold():
    data = SourceData(local_state_integrity={
        "agents": {
            "orchestration": {
                "inbox_bytes": 50 * 1024 * 1024,  # 50 MiB < 100 MiB
                "outbox_bytes": 0,
            },
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "inbox_bloat" for s in out)


def test_i4_separate_symptom_per_role_and_kind():
    data = SourceData(local_state_integrity={
        "agents": {
            "orchestration": {
                "inbox_bytes": 600 * 1024 * 1024,
                "outbox_bytes": 200 * 1024 * 1024,
            },
            "kernel": {
                "inbox_bytes": 150 * 1024 * 1024,
                "outbox_bytes": 0,
            },
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    syms = [s for s in out if s.name == "inbox_bloat"]
    assert len(syms) == 3
    keys = {(s.subject["role"], s.subject["kind"]) for s in syms}
    assert keys == {
        ("orchestration", "inbox"),
        ("orchestration", "outbox"),
        ("kernel", "inbox"),
    }


# ---------------------------------------------------------------------------
# I5 — coordinator_zombie
# ---------------------------------------------------------------------------

def test_i5_coordinator_zombie_fires():
    """PID dead + state.json valid + empty stop_reason → HIGH."""
    data = SourceData(local_state_integrity={
        "state_json": {"valid": True, "stop_reason": "", "path": "/p"},
        "coordinator": {
            "recorded_pid": 1234, "alive": False,
            "pid_file": "/p/optimizer_runs/run_x.pid",
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "coordinator_zombie")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["recorded_pid"] == 1234


def test_i5_silent_when_coordinator_alive():
    data = SourceData(local_state_integrity={
        "state_json": {"valid": True, "stop_reason": ""},
        "coordinator": {
            "recorded_pid": 1234, "alive": True,
            "pid_file": "/p/optimizer_runs/run_x.pid",
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "coordinator_zombie" for s in out)


def test_i5_silent_when_graceful_stop_reason():
    """PID dead but stop_reason set → clean wind-down."""
    data = SourceData(local_state_integrity={
        "state_json": {"valid": True, "stop_reason": "time_exhausted"},
        "coordinator": {
            "recorded_pid": 1234, "alive": False,
            "pid_file": "/p/optimizer_runs/run_x.pid",
        },
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "coordinator_zombie" for s in out)


def test_i5_silent_when_no_pid_recorded():
    data = SourceData(local_state_integrity={
        "state_json": {"valid": True, "stop_reason": ""},
        "coordinator": {"recorded_pid": None, "alive": None, "pid_file": ""},
    })
    out = evaluate_state_integrity_signals(_ctx(), data)
    assert all(s.name != "coordinator_zombie" for s in out)


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

def test_custom_thresholds_apply():
    cfg = StateIntegrityConfig(
        wal_bytes_warn_threshold=100 * 1024 * 1024,
        wal_bytes_critical_threshold=500 * 1024 * 1024,
        stale_lease_min_age_s=0.0,
    )
    data = SourceData(local_state_integrity={
        "wal": {"wal_bytes": 250 * 1024 * 1024, "db_bytes": 0},
        "leases": [
            {"task_id": "tsk-x", "holder_pid": 9999, "alive": False,
             "acquired_at": 1_999_999.5, "lane": "lane-1"},
        ],
    })
    out = evaluate_state_integrity_signals(_ctx(now_unix=2_000_000.0), data, config=cfg)
    assert any(s.name == "coordinator_wal_bloat" for s in out)
    assert any(s.name == "stale_lease" for s in out)


def test_evaluator_returns_empty_when_no_data():
    out = evaluate_state_integrity_signals(_ctx(), SourceData())
    assert out == []
