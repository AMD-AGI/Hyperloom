"""LocalProbe tests for the I + J sub-probes."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from hyperloom.agents.robustness.sources.local_probe import (
    LocalProbeConfig,
    LocalProbeSource,
    _is_pid_alive,
    _probe_agent_files,
    _probe_coordinator_pid,
    _probe_external_mounts,
    _probe_state_json,
    _probe_tracelens_cli,
    _probe_wal_size,
    _sample_state_integrity,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# _probe_state_json
# ---------------------------------------------------------------------------

def test_probe_state_json_valid(tmp_path):
    state = {"baseline_tput": 100.0, "stop_reason": ""}
    _write(tmp_path / "state.json", json.dumps(state))
    out = _probe_state_json(tmp_path)
    assert out["valid"] is True
    assert out["stop_reason"] == ""
    assert out["size_bytes"] > 0


def test_probe_state_json_missing(tmp_path):
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False
    assert out["error"] == "missing"


def test_probe_state_json_corrupt(tmp_path):
    _write(tmp_path / "state.json", "this is not json")
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False
    assert out["error"] == "json_parse_failed"


def test_probe_state_json_non_dict(tmp_path):
    _write(tmp_path / "state.json", json.dumps([1, 2, 3]))
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False


# ---------------------------------------------------------------------------
# _probe_wal_size
# ---------------------------------------------------------------------------

def test_probe_wal_size_reads_files(tmp_path):
    db = tmp_path / "storage" / "coordinator.db"
    wal = tmp_path / "storage" / "coordinator.db-wal"
    _write(db, "x" * 1000)
    _write(wal, "y" * 5000)
    out = _probe_wal_size(tmp_path)
    assert out["wal_bytes"] == 5000
    assert out["db_bytes"] == 1000


def test_probe_wal_size_silent_when_no_files(tmp_path):
    out = _probe_wal_size(tmp_path)
    assert out["wal_bytes"] == 0
    assert out["db_bytes"] == 0


# ---------------------------------------------------------------------------
# _probe_agent_files
# ---------------------------------------------------------------------------

def test_probe_agent_files_collects_inbox_outbox(tmp_path):
    for role in ("orchestration", "critic"):
        _write(tmp_path / "agents" / role / "inbox.jsonl", "a\nb\n")
        _write(tmp_path / "agents" / role / "outbox.jsonl", "c\n")
    out = _probe_agent_files(tmp_path)
    assert "orchestration" in out
    assert "critic" in out
    assert out["orchestration"]["inbox_bytes"] == 4
    assert out["orchestration"]["outbox_bytes"] == 2


def test_probe_agent_files_silent_when_no_agents(tmp_path):
    assert _probe_agent_files(tmp_path) == {}


# ---------------------------------------------------------------------------
# _probe_coordinator_pid
# ---------------------------------------------------------------------------

def test_probe_coordinator_pid_alive(tmp_path):
    """Current PID is always alive — used as the synthetic positive case."""
    pid = os.getpid()
    _write(tmp_path / "optimizer_runs" / "run_now.pid", f"{pid}\n")
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == pid
    assert out["alive"] is True


def test_probe_coordinator_pid_dead(tmp_path):
    """A PID we know is unused — pick a very high value unlikely to clash."""
    _write(tmp_path / "optimizer_runs" / "run_x.pid", "9999999\n")
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == 9999999
    assert out["alive"] is False


def test_probe_coordinator_pid_no_file(tmp_path):
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] is None
    assert out["alive"] is None


def test_probe_coordinator_pid_picks_newest(tmp_path):
    """Multiple pid files → newest mtime wins."""
    old = tmp_path / "optimizer_runs" / "run_old.pid"
    new = tmp_path / "optimizer_runs" / "run_new.pid"
    _write(old, "111\n")
    _write(new, f"{os.getpid()}\n")
    # Force ordered mtimes.
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (2000.0, 2000.0))
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == os.getpid()


def test_is_pid_alive_self():
    assert _is_pid_alive(os.getpid()) is True


def test_is_pid_alive_invalid():
    assert _is_pid_alive(9999999) is False


# ---------------------------------------------------------------------------
# _sample_state_integrity (aggregate)
# ---------------------------------------------------------------------------

def test_sample_state_integrity_aggregates_slots(tmp_path):
    _write(tmp_path / "state.json", json.dumps({"baseline_tput": 5.0}))
    _write(tmp_path / "storage" / "coordinator.db", "x")
    _write(tmp_path / "storage" / "coordinator.db-wal", "y" * 100)
    _write(tmp_path / "agents" / "kernel" / "inbox.jsonl", "data")
    _write(tmp_path / "optimizer_runs" / "run_x.pid", "9999999\n")
    out = _sample_state_integrity(tmp_path, "optimizer_runs")
    assert out["state_json"]["valid"] is True
    assert out["wal"]["wal_bytes"] == 100
    assert "kernel" in out["agents"]
    assert out["coordinator"]["recorded_pid"] == 9999999
    assert out["coordinator"]["alive"] is False


def test_sample_state_integrity_empty_session_dir():
    assert _sample_state_integrity(None, "optimizer_runs") == {}


# ---------------------------------------------------------------------------
# _probe_leases (sqlite)
# ---------------------------------------------------------------------------

def test_probe_leases_via_full_probe(tmp_path):
    """Use sqlite3 to write a fake leases table and verify probe reads."""
    db_path = tmp_path / "storage" / "coordinator.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE leases (
                task_id TEXT,
                holder_pid INTEGER,
                lane TEXT,
                acquired_at REAL
            )
        """)
        conn.execute(
            "INSERT INTO leases VALUES ('tsk-a', ?, 'lane-1', 1700000000.0)",
            (os.getpid(),),  # alive
        )
        conn.execute(
            "INSERT INTO leases VALUES ('tsk-b', 9999999, 'lane-2', 1700000000.0)",
        )
        conn.commit()
    finally:
        conn.close()
    out = _sample_state_integrity(tmp_path, "optimizer_runs")
    leases = out["leases"]
    assert len(leases) == 2
    by_task = {row["task_id"]: row for row in leases}
    assert by_task["tsk-a"]["alive"] is True
    assert by_task["tsk-b"]["alive"] is False


# ---------------------------------------------------------------------------
# _probe_external_mounts
# ---------------------------------------------------------------------------

def test_probe_external_mounts_records_latency(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    monkeypatch.setenv("INFERENCEX_PATH", "/nonexistent/path/zzz")
    monkeypatch.delenv("OOB_SRC", raising=False)
    out = _probe_external_mounts(timeout_s=5.0)
    by_env = {row["env_name"]: row for row in out}
    assert "TRACELENS_ROOT" in by_env
    assert by_env["TRACELENS_ROOT"]["ok"] is True
    assert "INFERENCEX_PATH" in by_env
    assert by_env["INFERENCEX_PATH"]["ok"] is False
    # OOB_SRC has no default — should be skipped silently.
    assert "OOB_SRC" not in by_env


# ---------------------------------------------------------------------------
# _probe_tracelens_cli
# ---------------------------------------------------------------------------

def test_probe_tracelens_cli_reports_absent(monkeypatch):
    """In CI env the CLI is not present → ``any_present=False``."""
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe.shutil.which",
        lambda _n: None,
    )
    out = _probe_tracelens_cli()
    assert out["any_present"] is False
    assert all(v is False for v in out["found"].values())


def test_probe_tracelens_cli_reports_present(monkeypatch):
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe.shutil.which",
        lambda name: "/usr/local/bin/" + name
        if name == "TraceLens_generate_perf_report_pytorch_inference"
        else None,
    )
    out = _probe_tracelens_cli()
    assert out["any_present"] is True
    assert out["found"]["TraceLens_generate_perf_report_pytorch_inference"] is True


# ---------------------------------------------------------------------------
# End-to-end through LocalProbeSource
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_populates_state_and_deps(tmp_path, monkeypatch):
    """Smoke: fetch() exposes both ``local_state_integrity`` and
    ``local_external_deps``."""
    _write(tmp_path / "state.json", json.dumps({"baseline_tput": 1.0}))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    cfg = LocalProbeConfig(
        session_dir=tmp_path,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False, preflight_enabled=False,
        critic_health_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_state_integrity["state_json"]["valid"] is True
    # External deps — only mounts populated (no gateway URL set, CLI absent).
    assert isinstance(data.local_external_deps.get("mounts"), list)
