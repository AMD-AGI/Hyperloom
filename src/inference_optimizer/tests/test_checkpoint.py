"""Tests for ``orchestrator.checkpoint`` — IMPL-CHECKLIST §6.1‒6.19."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.checkpoint import (
    Checkpoint,
    CheckpointHandle,
    ResumeState,
    TriggerReason,
    Verdict,
    evidence_check_matrix,
    resume_from_session_dir,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.task_registry import Task, TaskRegistry
from inference_optimizer.storage.connection import SqliteConnection


# ---------------------------------------------------------------------------
# Checkpoint.create / list_all / load_latest
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_writes_state_and_backup(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    state = SharedState(
        session_id="t",
        max_minutes=60.0,
        execution_mode=ExecutionMode.GUIDED_KERNEL_OPT,
    )
    handle = await Checkpoint.create(tmp_path, db, state)
    assert isinstance(handle, CheckpointHandle)
    assert (tmp_path / "state.json").is_file()
    bak = handle.path / "conductor.db.bak"
    assert bak.is_file()
    db.close()


@pytest.mark.asyncio
async def test_create_after_keep_tags_trigger(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    state = SharedState(session_id="t", max_minutes=60.0)
    handle = await Checkpoint.create_after_keep(tmp_path, db, state)
    assert handle.trigger == TriggerReason.AFTER_KEEP
    db.close()


@pytest.mark.asyncio
async def test_list_all_returns_chronological(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    state = SharedState(session_id="t", max_minutes=60.0)
    await Checkpoint.create(tmp_path, db, state, ts="20260101T000000Z")
    await Checkpoint.create(tmp_path, db, state, ts="20260102T000000Z")
    await Checkpoint.create(tmp_path, db, state, ts="20260103T000000Z")
    lst = Checkpoint.list_all(tmp_path)
    assert [h.ts for h in lst] == [
        "20260101T000000Z", "20260102T000000Z", "20260103T000000Z",
    ]
    latest = Checkpoint.load_latest(tmp_path)
    assert latest is not None and latest.ts == "20260103T000000Z"
    db.close()


def test_list_all_returns_empty_when_no_checkpoints(tmp_path: Path):
    assert Checkpoint.list_all(tmp_path) == []
    assert Checkpoint.load_latest(tmp_path) is None


@pytest.mark.asyncio
async def test_create_emits_checkpoint_event(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    state = SharedState(session_id="t", max_minutes=60.0)
    await Checkpoint.create(tmp_path, db, state)
    rows = db.fetchall_sync("SELECT topic FROM events WHERE topic=?",
                            ("checkpoint_taken",))
    assert rows
    db.close()


# ---------------------------------------------------------------------------
# resume_from_session_dir
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_collects_cursors_and_inflight(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    # seed cursor + inflight task
    from inference_optimizer.orchestrator.cursor_store import CursorStore
    cs = CursorStore(db)
    await cs.advance("executor", seq=42, msg_id="m1")
    await tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner"},
        idempotency_key="x-bench",
    )
    # also drop a persona file
    (tmp_path / "personas").mkdir(parents=True, exist_ok=True)
    (tmp_path / "personas" / "executor.md").write_text("hi", encoding="utf-8")

    state = await resume_from_session_dir(tmp_path, db, locks, tasks=tasks)
    assert state.cursors == {"executor": 42}
    assert len(state.found_inflight_tasks) == 1
    assert "executor" in state.persona_paths
    db.close()


@pytest.mark.asyncio
async def test_resume_reaps_expired_leases(tmp_path: Path, monkeypatch):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)

    async def fake_reap():
        return [{"lane": "benchmark_lane"}, {"lane": "server_lifecycle"}]

    monkeypatch.setattr(backend, "reap_expired", fake_reap)
    state = await resume_from_session_dir(tmp_path, db, locks)
    assert state.expired_leases_reaped == 2
    db.close()


@pytest.mark.asyncio
async def test_resume_handles_reap_failure(tmp_path: Path, monkeypatch):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    backend = SqliteLeaseBackend(db)
    locks = ResourceLockManager(backend)

    async def boom():
        raise RuntimeError("nope")

    monkeypatch.setattr(backend, "reap_expired", boom)
    state = await resume_from_session_dir(tmp_path, db, locks)
    assert state.expired_leases_reaped == 0
    db.close()


# ---------------------------------------------------------------------------
# evidence_check_matrix
# ---------------------------------------------------------------------------
def _bench_task(task_id: str = "t-bench") -> Task:
    return Task(
        task_id=task_id,
        kind="delegate",
        state="running",
        params={"action_name": "bench_runner"},
        idempotency_key=f"key-{task_id}",
    )


def _profile_task(task_id: str = "t-prof") -> Task:
    return Task(
        task_id=task_id,
        kind="delegate",
        state="running",
        params={"action_name": "profile"},
        idempotency_key=f"key-{task_id}",
    )


def _integrate_task(task_id: str = "t-int", fp: str | None = None) -> Task:
    params = {"action_name": "integrate"}
    if fp is not None:
        params["patch_fingerprint"] = fp
    return Task(
        task_id=task_id, kind="delegate", state="running",
        params=params, idempotency_key=f"key-{task_id}",
    )


def _server_task(task_id: str = "t-srv") -> Task:
    return Task(
        task_id=task_id, kind="delegate", state="running",
        params={"action_name": "server_restart"},
        idempotency_key=f"key-{task_id}",
    )


def _kernel_extract_task(task_id: str = "t-ke") -> Task:
    return Task(
        task_id=task_id, kind="delegate", state="running",
        params={"action_name": "kernel_extract"},
        idempotency_key=f"key-{task_id}",
    )


def _geak_task(task_id: str = "t-geak") -> Task:
    return Task(
        task_id=task_id, kind="delegate", state="running",
        params={"action_name": "kernel_opt"},
        idempotency_key=f"key-{task_id}",
    )


def _seed_metrics(session_dir: Path, task_id: str, body: str) -> Path:
    p = session_dir / "results" / task_id / "metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# bench_runner -------------------------------------------------------------
def test_check_bench_succeeded(tmp_path: Path):
    task = _bench_task()
    _seed_metrics(tmp_path, task.task_id, json.dumps({"tput": 5000}))
    assert evidence_check_matrix(task, tmp_path) == Verdict.SUCCEEDED


def test_check_bench_safely_failed_when_empty(tmp_path: Path):
    task = _bench_task()
    _seed_metrics(tmp_path, task.task_id, "")
    assert evidence_check_matrix(task, tmp_path) == Verdict.SAFELY_FAILED


def test_check_bench_evidence_insufficient_when_missing(tmp_path: Path):
    task = _bench_task()
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# profile ------------------------------------------------------------------
def test_check_profile_succeeded(tmp_path: Path):
    task = _profile_task()
    p = tmp_path / "results" / task.task_id / "trace" / "filtered-TP-0.trace.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 4096)
    assert evidence_check_matrix(task, tmp_path) == Verdict.SUCCEEDED


def test_check_profile_safely_failed_when_too_small(tmp_path: Path):
    task = _profile_task()
    p = tmp_path / "results" / task.task_id / "filtered-TP-0.trace.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert evidence_check_matrix(task, tmp_path) == Verdict.SAFELY_FAILED


def test_check_profile_evidence_insufficient_no_dir(tmp_path: Path):
    task = _profile_task()
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# integrate / patch --------------------------------------------------------
def test_check_integrate_succeeded_with_matching_fp(tmp_path: Path):
    expected = "abc123" * 8
    task = _integrate_task(fp=expected)
    fp_path = tmp_path / "results" / task.task_id / "patch_fingerprint.txt"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(expected, encoding="utf-8")
    assert evidence_check_matrix(task, tmp_path) == Verdict.SUCCEEDED


def test_check_integrate_evidence_insufficient_on_mismatch(tmp_path: Path):
    task = _integrate_task(fp="abc")
    fp_path = tmp_path / "results" / task.task_id / "patch_fingerprint.txt"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text("xyz", encoding="utf-8")
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


def test_check_integrate_evidence_insufficient_when_missing(tmp_path: Path):
    task = _integrate_task(fp="abc")
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# server_restart -----------------------------------------------------------
def test_check_server_restart_succeeded(tmp_path: Path):
    task = _server_task()
    pid_file = tmp_path / "results" / task.task_id / "server.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("4242", encoding="utf-8")
    verdict = evidence_check_matrix(
        task, tmp_path, server_health_fn=lambda pid: True
    )
    assert verdict == Verdict.SUCCEEDED


def test_check_server_restart_unhealthy(tmp_path: Path):
    task = _server_task()
    pid_file = tmp_path / "results" / task.task_id / "server.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("4242", encoding="utf-8")
    verdict = evidence_check_matrix(
        task, tmp_path, server_health_fn=lambda pid: False
    )
    assert verdict == Verdict.SAFELY_FAILED


def test_check_server_restart_no_pid(tmp_path: Path):
    task = _server_task()
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# kernel_extract -----------------------------------------------------------
def test_check_kernel_extract_succeeded(tmp_path: Path):
    task = _kernel_extract_task()
    art = tmp_path / "results" / task.task_id / "out.kernel.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text("{}", encoding="utf-8")
    assert evidence_check_matrix(task, tmp_path) == Verdict.SUCCEEDED


def test_check_kernel_extract_evidence_insufficient(tmp_path: Path):
    task = _kernel_extract_task()
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# GEAK submit --------------------------------------------------------------
def test_check_geak_succeeded(tmp_path: Path):
    task = _geak_task()
    p = tmp_path / "results" / task.task_id / "geak_request_id.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("req-123", encoding="utf-8")
    verdict = evidence_check_matrix(
        task, tmp_path, geak_status_fn=lambda req_id: "succeeded"
    )
    assert verdict == Verdict.SUCCEEDED


def test_check_geak_failed_status(tmp_path: Path):
    task = _geak_task()
    p = tmp_path / "results" / task.task_id / "geak_request_id.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("req-123", encoding="utf-8")
    verdict = evidence_check_matrix(
        task, tmp_path, geak_status_fn=lambda req_id: "failed"
    )
    assert verdict == Verdict.SAFELY_FAILED


def test_check_geak_no_id(tmp_path: Path):
    task = _geak_task()
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT


# unknown action -----------------------------------------------------------
def test_unknown_action_evidence_insufficient(tmp_path: Path):
    task = Task(
        task_id="x", kind="weird", state="running",
        params={"action_name": "totally_new"}, idempotency_key="k",
    )
    assert evidence_check_matrix(task, tmp_path) == Verdict.EVIDENCE_INSUFFICIENT
