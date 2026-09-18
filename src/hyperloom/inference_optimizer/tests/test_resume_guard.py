# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resume admission for retained execution ownership."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from contextlib import closing
from unittest.mock import Mock

import pytest

from hyperloom.inference_optimizer.session.paths import db_path_for
from hyperloom.inference_optimizer.session.resume_guard import ResumeBlocked, ensure_resume_safe


def test_real_coordinator_database_path_is_checked_before_resume(tmp_path):
    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE leases (lane TEXT, holder_id TEXT, task_id TEXT, pid INTEGER)")
        db.execute("INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123)")
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked, match="holder-1"):
        ensure_resume_safe(tmp_path, owner_scope="local")

    assert path.read_bytes() == before


def _database(tmp_path, *, legacy=False):
    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        scope = "" if legacy else ", owner_scope TEXT NOT NULL DEFAULT ''"
        db.executescript(
            "CREATE TABLE leases (lane TEXT, holder_id TEXT, task_id TEXT, pid INTEGER" + scope + ");"
            "CREATE TABLE tasks (task_id TEXT, state TEXT, requires_lanes TEXT, history TEXT, params TEXT);"
            "CREATE TABLE bringup_rounds (round_id TEXT, state TEXT, holder_task_id TEXT);"
            "CREATE TABLE gpu_leases (gpu_id INTEGER, holder_id TEXT, task_id TEXT);"
        )
    return path


def _execute(path, sql, values=()):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(sql, values)


def _task(path, *, state="running", evidence=None, task_id="task-1"):
    history = [] if evidence is None else [{"from": "running", "to": state, "evidence": evidence}]
    _execute(
        path,
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        (task_id, state, '["benchmark_lane"]', json.dumps(history), '{"api_key": "never-print-this-secret"}'),
    )


@pytest.mark.parametrize("scope", ["", "foreign-boot:pidns"])
def test_unknown_lease_scope_blocks_without_mutating_database(tmp_path, scope):
    path = _database(tmp_path)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, ?)", (scope,))
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="this-boot:pidns")

    message = str(exc.value)
    assert "task-1" in message and "benchmark_lane" in message and "holder-1" in message
    assert "never-print-this-secret" not in message
    assert "force" not in message.lower()
    assert path.read_bytes() == before


def test_legacy_lease_without_scope_column_is_not_migrated(tmp_path):
    path = _database(tmp_path, legacy=True)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123)")
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked, match="task-1"):
        ensure_resume_safe(tmp_path, owner_scope="this-boot:pidns")

    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as db:
        assert "owner_scope" not in {row[1] for row in db.execute("PRAGMA table_info(leases)")}


def test_matching_scope_is_left_for_existing_reaper(tmp_path):
    path = _database(tmp_path)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, 'local')")
    before = path.read_bytes()

    ensure_resume_safe(tmp_path, owner_scope="local")

    assert path.read_bytes() == before


def test_absent_database_is_not_created(tmp_path):
    ensure_resume_safe(tmp_path, owner_scope="local")
    assert not db_path_for(tmp_path).exists()
    assert not db_path_for(tmp_path).parent.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_database_has_no_unknown_execution(tmp_path, legacy):
    path = _database(tmp_path, legacy=legacy)
    before = path.read_bytes()
    ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


@pytest.mark.parametrize("table", ["leases", "tasks", "bringup_rounds", "gpu_leases"])
def test_missing_unused_table_is_not_created(tmp_path, table):
    path = _database(tmp_path)
    _execute(path, f"DROP TABLE {table}")
    before = path.read_bytes()
    ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


@pytest.mark.parametrize("missing_table", [False, True])
def test_running_task_without_lease_is_not_proof_of_exit(tmp_path, missing_table):
    path = _database(tmp_path)
    _task(path)
    if missing_table:
        _execute(path, "DROP TABLE leases")
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked, match="task-1") as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")

    assert "benchmark_lane" in str(exc.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize("evidence", [{"reason": "cancelled_in_flight"}, {"lease_ttl_sec": 10}])
@pytest.mark.parametrize("with_lease", [False, True])
def test_unobserved_terminal_holder_keeps_open_round_blocked(tmp_path, evidence, with_lease):
    path = _database(tmp_path)
    _task(path, state="cancelled", evidence=evidence)
    _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    _execute(path, "INSERT INTO leases VALUES ('bringup_round', 'round-1', 'task-1', 0, '')")
    if with_lease:
        _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, 'local')")
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")

    message = str(exc.value)
    assert "task-1" in message and "bringup_round" in message and "round-1" in message
    assert path.read_bytes() == before


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled", "queued"])
def test_completed_or_unstarted_holder_without_resources_can_resume(tmp_path, state):
    path = _database(tmp_path)
    _task(path, state=state, evidence={"reason": "completed"})
    _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    _execute(path, "INSERT INTO leases VALUES ('bringup_round', 'round-1', 'task-1', 0, '')")

    ensure_resume_safe(tmp_path, owner_scope="local")


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("appended", [False, True], ids=["transition", "terminal-race"])
@pytest.mark.parametrize("confirmed", [False, True], ids=["unconfirmed", "confirmed"])
def test_open_round_uses_latest_cleanup_outcome(tmp_path, state, appended, confirmed):
    path = _database(tmp_path)
    evidence = {
        "outcome": {"state": "succeeded", "result": {"status": "ok"}},
        "cleanup_confirmed": confirmed,
    }
    prior_evidence = {"reason": "cancelled_in_flight" if confirmed else "completed"}
    _task(path, state=state, evidence=prior_evidence if appended else evidence)
    if appended:
        with closing(sqlite3.connect(path)) as db, db:
            history = json.loads(db.execute("SELECT history FROM tasks").fetchone()[0])
            history.append({"ts": "2026-09-18T00:00:00Z", "evidence": evidence})
            history.append({"progress": {"message": "completion recorded"}})
            db.execute("UPDATE tasks SET history=?", (json.dumps(history),))
    _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    _execute(path, "INSERT INTO leases VALUES ('bringup_round', 'round-1', 'task-1', 0, '')")
    before = path.read_bytes()

    if confirmed:
        ensure_resume_safe(tmp_path, owner_scope="local")
    else:
        with pytest.raises(ResumeBlocked, match="cleanup"):
            ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


def test_open_round_without_holder_record_is_blocked(tmp_path):
    path = _database(tmp_path)
    _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'missing-task')")
    with pytest.raises(ResumeBlocked, match="missing-task"):
        ensure_resume_safe(tmp_path, owner_scope="local")


def test_gpu_lease_does_not_infer_worker_exit_from_coordinator_scope(tmp_path):
    path = _database(tmp_path)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('gpu_research_lane', 'coord-1', 'task-1', 123, 'local')")
    _execute(path, "INSERT INTO gpu_leases VALUES (7, 'worker-1', 'task-1')")
    before = path.read_bytes()

    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")

    assert "gpu_id='7'" in str(exc.value) and "worker-1" in str(exc.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize("pid", [0, -1, None, "invalid"])
def test_unobservable_pid_cannot_use_matching_scope(tmp_path, pid):
    path = _database(tmp_path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', ?, 'local')", (pid,))
    with pytest.raises(ResumeBlocked, match="holder-1"):
        ensure_resume_safe(tmp_path, owner_scope="local")


def test_missing_local_scope_does_not_match_unknown_lease(tmp_path):
    path = _database(tmp_path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, '')")
    with pytest.raises(ResumeBlocked):
        ensure_resume_safe(tmp_path, owner_scope="")


def test_diagnostics_are_bounded_and_escape_control_characters(tmp_path):
    path = _database(tmp_path)
    for index in range(50):
        _execute(
            path,
            "INSERT INTO leases VALUES ('benchmark_lane', ?, ?, 123, 'foreign')",
            (f"holder-{index}\n" + "x" * 2000, f"task-{index}"),
        )
    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")
    message = str(exc.value)
    assert len(message) < 6000
    assert "additional" in message
    assert "x" * 100 not in message


def test_corrupt_database_fails_closed_without_sqlite_details(tmp_path):
    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a database")
    with pytest.raises(ResumeBlocked, match="cannot inspect"):
        ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == b"not a database"


@pytest.mark.parametrize("state", ["queued", "succeeded", "failed"])
def test_current_schema_round_holder_without_unknown_execution_is_allowed(tmp_path, state):
    from hyperloom.orchestrator.bus.storage.schema import ensure_schema

    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        ensure_schema(db)
        db.execute(
            "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, history, created_at, updated_at) "
            "VALUES ('task-1', 'baseline', ?, '{}', 'key-1', ?, '2026-01-01', '2026-01-01')",
            (state, json.dumps([{"to": state, "evidence": {"reason": "completed"}}])),
        )
        db.execute(
            "INSERT INTO bringup_rounds (round_id, state, holder_task_id, opened_unix, renewed_unix, expires_unix) "
            "VALUES ('round-1', 'open', 'task-1', 1, 1, 2)"
        )
        db.execute(
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at) "
            "VALUES ('bringup_round', 'round-1', 'task-1', 'bringup_round', 0, 'old', 'old', 'old')"
        )
    before = path.read_bytes()
    ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_matching_scope_dead_holder_reaches_existing_reaper(tmp_path, monkeypatch):
    from hyperloom.orchestrator.bus import resource_lock
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.bus.storage.schema import ensure_schema
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    monkeypatch.setattr(resource_lock, "local_owner_scope", lambda: "local")
    db = SqliteConnection(db_path_for(tmp_path))
    try:
        ensure_schema(db.raw)
        tasks = TaskRegistry(db)
        leases = resource_lock.SqliteLeaseBackend(db)
        await tasks.create(kind="baseline", params={}, idempotency_key="task-1", task_id="task-1")
        await tasks.transition("task-1", "running")
        await leases.acquire_many(
            ["benchmark_lane"], holder_id="holder-1", task_id="task-1", action="baseline", ttl_sec=1
        )
        monkeypatch.setattr(resource_lock.SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: False))

        ensure_resume_safe(tmp_path, owner_scope="local")

        assert (await tasks.get("task-1")).state == "running"
        assert await tasks.reclaim_dead_running() == ["task-1"]
        assert await leases.reap_dead_holders()
        assert (await tasks.get("task-1")).state == "failed"
        assert await leases.lane_holders() == {}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_current_handoff_to_unstarted_holder_remains_resumable(tmp_path):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.bus.storage.schema import ensure_schema
    from hyperloom.orchestrator.state.round_store import RoundStore
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteConnection(path)
    try:
        ensure_schema(db.raw)
        tasks = TaskRegistry(db)
        rounds = RoundStore(db)
        for task_id in ("first", "next"):
            await tasks.create(kind="baseline", params={}, idempotency_key=task_id, task_id=task_id)
        opened = await rounds.open("round-1", holder_task_id="first", lease_sec=1, now_unix=1, request_id="open")
        assert opened.ok
        await tasks.transition("first", "running")
        await tasks.transition("first", "succeeded", evidence={"reason": "completed"})
        handed_off = await rounds.handoff(
            "round-1",
            holder_task_id="first",
            fence=opened.fence,
            new_holder_task_id="next",
            lease_sec=1,
            now_unix=2,
            request_id="handoff",
        )
        assert handed_off.ok
    finally:
        db.close()
    before = path.read_bytes()
    ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


def test_gpu_zero_diagnostic_keeps_actual_resource_id(tmp_path):
    path = _database(tmp_path)
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker-0', 'task-0')")
    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")
    assert "gpu_id='0'" in str(exc.value)


def test_unicode_diagnostics_stay_bounded(tmp_path):
    path = _database(tmp_path)
    for index in range(30):
        _execute(path, "INSERT INTO leases VALUES (?, ?, ?, 123, '')", ("☃" * 200, "☃" * 200, f"task-{index}"))
    with pytest.raises(ResumeBlocked) as exc:
        ensure_resume_safe(tmp_path, owner_scope="local")
    assert len(str(exc.value)) < 6000


@pytest.mark.parametrize("with_round", [False, True])
def test_terminal_without_history_is_not_completion_proof_for_open_round(tmp_path, with_round):
    path = _database(tmp_path)
    _task(path, state="cancelled")
    if with_round:
        _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
        with pytest.raises(ResumeBlocked, match="task-1"):
            ensure_resume_safe(tmp_path, owner_scope="local")
    else:
        ensure_resume_safe(tmp_path, owner_scope="local")


@pytest.mark.parametrize("evidence_kind", ["missing", "null", "malformed", "empty"])
@pytest.mark.parametrize("with_round", [False, True])
def test_terminal_evidence_must_be_recorded_for_open_round(tmp_path, evidence_kind, with_round):
    path = _database(tmp_path)
    _task(path, state="cancelled")
    transition = {"from": "running", "to": "cancelled"}
    if evidence_kind != "missing":
        transition["evidence"] = {"null": None, "malformed": [], "empty": {}}[evidence_kind]
    _execute(path, "UPDATE tasks SET history=?", (json.dumps([transition]),))
    if with_round:
        _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    before = path.read_bytes()

    if with_round and evidence_kind != "empty":
        with pytest.raises(ResumeBlocked, match="task-1"):
            ensure_resume_safe(tmp_path, owner_scope="local")
    else:
        ensure_resume_safe(tmp_path, owner_scope="local")
    assert path.read_bytes() == before


def test_wal_ownership_is_seen_without_changing_database_or_history(tmp_path):
    path = _database(tmp_path)
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, 'foreign')")
        writer.commit()
        before = path.read_bytes()
        wal_path = path.with_name(path.name + "-wal")
        before_wal = wal_path.read_bytes()
        with pytest.raises(ResumeBlocked, match="holder-1"):
            ensure_resume_safe(tmp_path, owner_scope="local")
        assert path.read_bytes() == before
        assert wal_path.read_bytes() == before_wal


@pytest.mark.skipif(sys.platform == "win32", reason="CLI imports require POSIX fcntl")
@pytest.mark.parametrize("residual", ["foreign", "legacy", "cancelled", "matching"])
def test_cli_checks_ownership_before_state_changes_or_execution(tmp_path, monkeypatch, capsys, residual):
    import hyperloom.inference_optimizer.cli as cli
    from hyperloom.orchestrator.bus import resource_lock

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    path = _database(session_dir, legacy=residual == "legacy")
    if residual == "cancelled":
        _task(path, state="cancelled", evidence={"reason": "cancelled_in_flight"})
        _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    elif residual == "legacy":
        _task(path)
        _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123)")
    else:
        _task(path)
        scope = "local" if residual == "matching" else "foreign"
        _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder-1', 'task-1', 123, ?)", (scope,))
    state_path = session_dir / "state.json"
    state_path.write_text('{"stop_reason":"signal","crash_count":2}', encoding="utf-8")
    before_db = path.read_bytes()
    before_state = state_path.read_bytes()

    monkeypatch.setattr(cli, "clean_stale_aiter_locks", lambda: {"dir": ""})
    monkeypatch.setattr(cli, "_claude_model_should_follow_codex", lambda: False)
    monkeypatch.setattr(cli, "_codex_model_should_follow_claude", lambda: False)
    monkeypatch.setattr(cli, "_preflight", Mock(return_value={}))
    monkeypatch.setattr(cli, "_resolve_models_for_run", Mock())
    monkeypatch.setattr(cli, "_preflight_agentx_backend", Mock())
    monkeypatch.setattr(cli, "_apply_agentx_budget_profile", Mock())
    monkeypatch.setattr(resource_lock, "local_owner_scope", lambda: "local")
    lock = Mock()
    monkeypatch.setattr(cli, "_acquire_session_lock_or_exit", Mock(return_value=lock))
    install = Mock()
    monkeypatch.setattr(cli, "_persist_install_event", install)
    coordinator = Mock(side_effect=AssertionError("execution must not start"))
    monkeypatch.setattr(cli, "Coordinator", coordinator)
    resume_leg = Mock(side_effect=AssertionError("resume history must not change"))
    monkeypatch.setattr(cli, "_begin_resume_leg", resume_leg)

    class AdmissionPassed(Exception):
        pass

    monkeypatch.setattr(cli, "load_manifest", Mock(side_effect=AdmissionPassed))
    args = cli._build_parser().parse_args(["optimize", "--resume-from", str(session_dir)])
    if residual == "matching":
        with pytest.raises(AdmissionPassed):
            asyncio.run(cli._run_optimize(args))
        install.assert_called_once()
    else:
        with pytest.raises(SystemExit) as exc:
            asyncio.run(cli._run_optimize(args))
        assert exc.value.code == 2
        lock.release.assert_called_once()
        install.assert_not_called()
        assert "task-1" in capsys.readouterr().err
    coordinator.assert_not_called()
    resume_leg.assert_not_called()
    assert path.read_bytes() == before_db
    assert state_path.read_bytes() == before_state
