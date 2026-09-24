# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resume admission for retained execution ownership."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
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


def _recover_module():
    # Load the real offline handler without the POSIX-only CLI package on Windows.
    path = Path(__file__).parents[1] / "cli" / "recover.py"
    spec = importlib.util.spec_from_file_location("hyperloom.inference_optimizer.cli.recover", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recover_args(tmp_path, **changes):
    values = dict(
        session_dir=tmp_path,
        force=False,
        backfill_trace=False,
        confirm_stopped="task-1",
        confirmation_reason="Verified process tree and Ray actors stopped; secret-ticket",
    )
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("status", ["confirmed", "already_confirmed", "nothing_to_confirm"])
def test_confirmation_cli_is_offline_and_does_not_resume(tmp_path, monkeypatch, capsys, status):
    from hyperloom.inference_optimizer.session import resume_guard

    recover = _recover_module()
    status_probe = Mock(side_effect=AssertionError("reports and network must not be reached"))
    monkeypatch.setattr(recover, "_session_recovery_status", status_probe)
    confirm = Mock(return_value=dict(status=status, task_id="task-1", released_leases=1, released_gpu_leases=1))
    monkeypatch.setattr(resume_guard, "confirm_task_stopped", confirm, raising=False)
    args = _recover_args(tmp_path)

    result = recover._run_recover_session(args)

    assert result == 0
    confirm.assert_called_once_with(tmp_path.resolve(), task_id="task-1", reason=args.confirmation_reason)
    status_probe.assert_not_called()
    output = capsys.readouterr()
    assert status in output.out and "task-1" in output.out
    assert "secret-ticket" not in output.out + output.err


@pytest.mark.parametrize(
    "changes",
    [
        {"confirm_stopped": None},
        {"confirmation_reason": None},
        {"confirm_stopped": ""},
        {"confirmation_reason": " \t"},
        {"force": True},
        {"backfill_trace": True},
    ],
)
def test_confirmation_cli_rejects_bad_argument_pairs_before_reports(tmp_path, monkeypatch, changes):
    from hyperloom.inference_optimizer.session import resume_guard

    recover = _recover_module()
    status_probe = Mock(side_effect=AssertionError("reports must not be read"))
    monkeypatch.setattr(recover, "_session_recovery_status", status_probe)
    confirm = Mock(side_effect=AssertionError("invalid arguments must not reach confirmation"))
    monkeypatch.setattr(resume_guard, "confirm_task_stopped", confirm, raising=False)

    result = recover._run_recover_session(_recover_args(tmp_path, **changes))

    assert result == 2
    status_probe.assert_not_called()
    confirm.assert_not_called()


def test_confirmation_cli_keeps_old_namespace_compatible(tmp_path, monkeypatch):
    recover = _recover_module()
    status_probe = Mock(
        return_value=dict(
            close_done=True, breakdown_exists=True, breakdown_recorded=True, counts_final=True, looks_complete=True
        )
    )
    monkeypatch.setattr(recover, "_session_recovery_status", status_probe)
    args = argparse.Namespace(session_dir=tmp_path, force=False, backfill_trace=False)

    result = recover._run_recover_session(args)

    assert result == 0
    status_probe.assert_called_once_with(tmp_path.resolve())


@pytest.mark.skipif(sys.platform == "win32", reason="CLI parser imports require POSIX fcntl")
def test_confirmation_parser_accepts_only_explicit_task_arguments(tmp_path):
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "recover-session",
            "--session-dir",
            str(tmp_path),
            "--confirm-stopped",
            "task-1",
            "--confirmation-reason",
            "Verified all workers stopped",
        ]
    )
    assert args.confirm_stopped == "task-1"
    assert args.confirmation_reason == "Verified all workers stopped"
    old = parser.parse_args(["recover-session", "--session-dir", str(tmp_path)])
    assert old.confirm_stopped is None and old.confirmation_reason is None


_CONFIRMED_AT = "2026-09-18T12:00:00+00:00"
_CONFIRMATION_REASON = "Verified the entire process tree and every Ray actor stopped"


def _confirm_in_sqlite(path, *, task_id="task-1", reason=_CONFIRMATION_REASON):
    from hyperloom.inference_optimizer.session.resume_guard import _confirm_task_stopped_in_transaction

    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        return _confirm_task_stopped_in_transaction(
            db, task_id=task_id, reason=reason, operator="test-operator", confirmed_at=_CONFIRMED_AT
        )


def _rows(path, table):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"SELECT * FROM {table}")]


@pytest.mark.parametrize("legacy", [False, True])
def test_confirmation_clears_only_target_execution_with_durable_audit(tmp_path, legacy):
    path = _database(tmp_path, legacy=legacy)
    _task(path)
    _task(path, state="queued", task_id="other-task")
    scope = "" if legacy else ", ''"
    for lane, holder, task in [
        ("benchmark_lane", "execution-1", "task-1"),
        ("bringup_round", "round-1", "task-1"),
        ("other_lane", "execution-2", "other-task"),
    ]:
        _execute(path, f"INSERT INTO leases VALUES (?, ?, ?, 123{scope})", (lane, holder, task))
    _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker-0', 'task-1')")
    _execute(path, "INSERT INTO gpu_leases VALUES (1, 'worker-1', 'other-task')")
    before_rounds = _rows(path, "bringup_rounds")
    before_round_leases = [row for row in _rows(path, "leases") if row["lane"] == "bringup_round"]
    before_other = _rows(path, "tasks")[1]
    with pytest.raises(ResumeBlocked):
        ensure_resume_safe(tmp_path, owner_scope="local")

    result = _confirm_in_sqlite(path)

    assert result == dict(status="confirmed", task_id="task-1", released_leases=1, released_gpu_leases=1)
    task, other = _rows(path, "tasks")
    assert task["state"] == "cancelled"
    assert other == before_other
    entry = json.loads(task["history"])[-1]
    assert entry["from"] == "running" and entry["to"] == "cancelled"
    assert entry["ts"] == _CONFIRMED_AT and "progress" not in entry
    evidence = entry["evidence"]
    assert evidence["reason"] == "operator_confirmed_stopped"
    assert evidence["cleanup_confirmed"] is True
    assert evidence["operator"] == "test-operator"
    assert evidence["confirmation_reason"] == _CONFIRMATION_REASON
    assert evidence["released_leases"] == [{"lane": "benchmark_lane", "holder_id": "execution-1"}]
    assert evidence["released_gpu_leases"] == [{"gpu_id": 0, "holder_id": "worker-0"}]
    outcome = evidence["outcome"]
    assert set(outcome) == {"task_id", "state", "result", "error", "error_class"}
    assert outcome["task_id"] == "task-1" and outcome["state"] == "cancelled" and outcome["result"] == {}
    assert outcome["error"] and outcome["error_class"] == "operator_confirmed_stopped"
    assert _rows(path, "bringup_rounds") == before_rounds
    assert [row for row in _rows(path, "leases") if row["lane"] == "bringup_round"] == before_round_leases
    assert _rows(path, "gpu_leases") == [{"gpu_id": 1, "holder_id": "worker-1", "task_id": "other-task"}]
    with pytest.raises(ResumeBlocked, match="other-task"):
        ensure_resume_safe(tmp_path, owner_scope="local")
    _confirm_in_sqlite(path, task_id="other-task")
    ensure_resume_safe(tmp_path, owner_scope="local")
    if legacy:
        assert "owner_scope" not in _rows(path, "leases")[0]


@pytest.mark.parametrize("state", ["queued", "running", "succeeded", "failed", "cancelled"])
def test_confirmation_preserves_terminal_history_payload_and_timestamp(tmp_path, state):
    path = _database(tmp_path)
    old_evidence = {"cleanup_confirmed": False, "outcome": {"state": "succeeded", "result": {"secret": "old-result"}}}
    _task(path, state=state, evidence=old_evidence)
    _execute(path, "ALTER TABLE tasks ADD COLUMN updated_at TEXT DEFAULT '2026-01-01T00:00:00+00:00'")
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker-0', 'task-1')")
    old_history = json.loads(_rows(path, "tasks")[0]["history"])

    _confirm_in_sqlite(path)

    task = _rows(path, "tasks")[0]
    history = json.loads(task["history"])
    assert history[:-1] == old_history
    assert history[-1]["evidence"]["outcome"]["result"] == {}
    assert "old-result" not in json.dumps(history[-1])
    if state in {"queued", "running"}:
        assert task["state"] == "cancelled" and task["updated_at"] == _CONFIRMED_AT
    else:
        assert task["state"] == state and task["updated_at"] == "2026-01-01T00:00:00+00:00"
        assert set(history[-1]) == {"ts", "evidence"}


@pytest.mark.parametrize("state", ["running", "cancelled"])
def test_confirmation_without_execution_leases_can_resolve_uncertain_owner(tmp_path, state):
    path = _database(tmp_path)
    _task(path, state=state, evidence={"reason": "cancelled_in_flight"})
    if state == "cancelled":
        _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
    for table in ("leases", "gpu_leases"):
        _execute(path, f"DROP TABLE {table}")
    result = _confirm_in_sqlite(path)
    assert result == dict(status="confirmed", task_id="task-1", released_leases=0, released_gpu_leases=0)
    ensure_resume_safe(tmp_path, owner_scope="local")
    with closing(sqlite3.connect(path)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "leases" not in tables and "gpu_leases" not in tables


@pytest.mark.parametrize("state", ["queued", "succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("open_round", [False, True])
def test_confirmation_does_not_cancel_clean_or_unstarted_task(tmp_path, state, open_round):
    path = _database(tmp_path)
    _task(path, state=state, evidence={"reason": "completed"})
    if open_round:
        _execute(path, "INSERT INTO bringup_rounds VALUES ('round-1', 'open', 'task-1')")
        _execute(path, "INSERT INTO leases VALUES ('bringup_round', 'round-1', 'task-1', 0, '')")
    before = path.read_bytes()
    result = _confirm_in_sqlite(path)
    assert result == dict(status="nothing_to_confirm", task_id="task-1", released_leases=0, released_gpu_leases=0)
    assert path.read_bytes() == before


@pytest.mark.parametrize("scope", ["local", "foreign", " ", None, 0])
@pytest.mark.parametrize("lane", ["benchmark_lane", "bringup_round"])
def test_confirmation_rejects_any_nonempty_or_invalid_target_scope(tmp_path, scope, lane):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError

    path = _database(tmp_path, legacy=True)
    _task(path)
    _execute(path, "ALTER TABLE leases ADD COLUMN owner_scope")
    _execute(path, "INSERT INTO leases VALUES ('other_lane', 'empty-owner', 'task-1', 123, '')")
    _execute(path, "INSERT INTO leases VALUES (?, 'owner', 'task-1', 123, ?)", (lane, scope))
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker-0', 'task-1')")
    before = path.read_bytes()
    with pytest.raises(CleanupConfirmationError, match="scope"):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("later_work", ["gpu", "lease", "running"])
def test_confirmation_is_idempotent_but_cannot_authorize_later_work(tmp_path, later_work):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError

    path = _database(tmp_path)
    _task(path)
    first = _confirm_in_sqlite(path)
    assert first["status"] == "confirmed"
    before = path.read_bytes()
    second = _confirm_in_sqlite(path)
    assert second == dict(status="already_confirmed", task_id="task-1", released_leases=0, released_gpu_leases=0)
    assert path.read_bytes() == before
    if later_work == "gpu":
        _execute(path, "INSERT INTO gpu_leases VALUES (0, 'new-worker', 'task-1')")
    elif later_work == "lease":
        _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'new-holder', 'task-1', 123, '')")
    else:
        _execute(path, "UPDATE tasks SET state='running'")
    before = path.read_bytes()
    with pytest.raises(CleanupConfirmationError, match="after|later|new"):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "later",
    [
        {"evidence": {"cleanup_confirmed": False, "outcome": {"state": "cancelled"}}},
        {"to": "cancelled", "evidence": {"reason": "cancelled_in_flight"}},
        {"to": "cancelled", "evidence": {"lease_ttl_sec": 5}},
        {"to": "running", "evidence": {}},
        {"to": [], "evidence": {}},
        {"evidence": {"cleanup_confirmed": True, "outcome": {"state": "succeeded", "result": {}}}},
    ],
)
def test_confirmation_rejects_evidence_after_prior_manual_confirmation(tmp_path, later):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError

    path = _database(tmp_path)
    _task(path)
    _confirm_in_sqlite(path)
    history = json.loads(_rows(path, "tasks")[0]["history"]) + [later]
    _execute(path, "UPDATE tasks SET history=?", (json.dumps(history),))
    before = path.read_bytes()
    with pytest.raises(CleanupConfirmationError):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad_history", [None, "{broken", "{}", "null", '"text"'])
def test_confirmation_rejects_unreadable_history_without_changes(tmp_path, bad_history):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError

    path = _database(tmp_path)
    _task(path)
    _execute(path, "UPDATE tasks SET history=?", (bad_history,))
    before = path.read_bytes()
    with pytest.raises(CleanupConfirmationError, match="history"):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("missing", ["tasks", "row", "state"])
def test_confirmation_requires_existing_valid_task(tmp_path, missing):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError

    path = _database(tmp_path)
    if missing == "tasks":
        _execute(path, "DROP TABLE tasks")
    elif missing == "state":
        _task(path, state="invalid")
    before = path.read_bytes()
    with pytest.raises(CleanupConfirmationError, match="task|state"):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failed_table", ["leases", "gpu_leases"])
def test_confirmation_sql_failure_rolls_back_history_and_all_releases(tmp_path, failed_table):
    path = _database(tmp_path)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'owner', 'task-1', 123, '')")
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker', 'task-1')")
    _execute(
        path,
        f"CREATE TRIGGER reject_delete BEFORE DELETE ON {failed_table} "
        "BEGIN SELECT RAISE(ABORT, 'injected deletion failure'); END",
    )
    before = path.read_bytes()
    with pytest.raises(sqlite3.IntegrityError, match="injected deletion failure"):
        _confirm_in_sqlite(path)
    assert path.read_bytes() == before


def test_confirmation_requires_fcntl_before_acquiring_session_lock(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.session import lock, resume_guard

    path = _database(tmp_path)
    _task(path)
    before = path.read_bytes()
    monkeypatch.setattr(lock, "fcntl", None)
    acquire = Mock(side_effect=AssertionError("non-exclusive fallback must not be used"))
    monkeypatch.setattr(lock.SessionLock, "acquire", acquire)
    with pytest.raises(resume_guard.CleanupConfirmationError, match="POSIX|fcntl"):
        resume_guard.confirm_task_stopped(tmp_path, task_id="task-1", reason=_CONFIRMATION_REASON)
    acquire.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.skipif(sys.platform == "win32", reason="Confirmation requires a real POSIX flock")
def test_confirmation_public_uses_real_session_lock_and_existing_database(tmp_path):
    from hyperloom.inference_optimizer.session.lock import SessionLock
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError, confirm_task_stopped

    path = _database(tmp_path, legacy=True)
    _task(path)
    before = path.read_bytes()
    with SessionLock(tmp_path):
        with pytest.raises(CleanupConfirmationError, match="lock|running"):
            confirm_task_stopped(tmp_path, task_id="task-1", reason=_CONFIRMATION_REASON)
    assert path.read_bytes() == before
    result = confirm_task_stopped(tmp_path, task_id="task-1", reason=_CONFIRMATION_REASON)
    assert result["status"] == "confirmed"
    ensure_resume_safe(tmp_path, owner_scope="local")


@pytest.mark.skipif(sys.platform == "win32", reason="Confirmation requires a real POSIX flock")
@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_confirmation_public_missing_or_corrupt_database_is_not_created_or_repaired(tmp_path, kind):
    from hyperloom.inference_optimizer.session.resume_guard import CleanupConfirmationError, confirm_task_stopped

    path = db_path_for(tmp_path)
    if kind == "corrupt":
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not sqlite")
    with pytest.raises(CleanupConfirmationError):
        confirm_task_stopped(tmp_path, task_id="task-1", reason=_CONFIRMATION_REASON)
    if kind == "corrupt":
        assert path.read_bytes() == b"not sqlite"
    else:
        assert not path.exists()


@pytest.mark.parametrize(
    "task_id,reason", [("", "valid"), (" ", "valid"), (" task-1", "valid"), ("task-1", ""), ("task-1", " \t")]
)
def test_confirmation_public_validates_arguments_before_lock(tmp_path, monkeypatch, task_id, reason):
    from hyperloom.inference_optimizer.session import lock, resume_guard

    acquire = Mock(side_effect=AssertionError("invalid arguments must not acquire the lock"))
    monkeypatch.setattr(lock.SessionLock, "acquire", acquire)
    with pytest.raises(resume_guard.CleanupConfirmationError, match="task ID|reason"):
        resume_guard.confirm_task_stopped(tmp_path, task_id=task_id, reason=reason)
    acquire.assert_not_called()
    assert not db_path_for(tmp_path).exists()


def test_confirmation_cli_reports_confirmation_failure_without_reports(tmp_path, monkeypatch, capsys):
    from hyperloom.inference_optimizer.session import resume_guard

    recover = _recover_module()
    confirm = Mock(side_effect=resume_guard.CleanupConfirmationError("scope is not empty"))
    monkeypatch.setattr(resume_guard, "confirm_task_stopped", confirm)
    status = Mock(side_effect=AssertionError("reports must not run after a rejected confirmation"))
    monkeypatch.setattr(recover, "_session_recovery_status", status)
    result = recover._run_recover_session(_recover_args(tmp_path))
    assert result == 2
    assert "scope" in capsys.readouterr().err
    status.assert_not_called()


@pytest.mark.asyncio
async def test_confirmation_leaves_round_fence_and_events_for_existing_reconciler(tmp_path):
    from hyperloom.orchestrator.bringup.reconcile import Reconciler
    from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.state.round_store import EXPIRED_REAPED, RoundStore
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    path = db_path_for(tmp_path)
    db = SqliteConnection(path)
    try:
        tasks = TaskRegistry(db)
        rounds = RoundStore(db)
        await tasks.create(kind="baseline", params={}, idempotency_key="task-1", task_id="task-1")
        await tasks.transition("task-1", "running")
        await tasks.transition("task-1", "cancelled", evidence={"reason": "cancelled_in_flight"})
        opened = await rounds.open("round-1", holder_task_id="task-1", lease_sec=1, now_unix=1, request_id="open")
        assert opened.ok
        reconciler = Reconciler(
            rounds=rounds,
            tasks=tasks,
            locks=ResourceLockManager(SqliteLeaseBackend(db)),
            shared_state=None,
            terminal_holder_cap_sec=0,
        )
        report = await reconciler.run(2_000_000_000)
        assert not report.settled and not report.failures
        before = {table: _rows(path, table) for table in ("bringup_rounds", "leases", "events", "round_events")}
        old_updated_at = (await tasks.get("task-1")).updated_at

        result = _confirm_in_sqlite(path)

        assert result["status"] == "confirmed"
        for table, rows in before.items():
            assert _rows(path, table) == rows
        assert (await tasks.get("task-1")).updated_at == old_updated_at
        ensure_resume_safe(tmp_path, owner_scope="local")
        report = await reconciler.run(2_000_000_000)
        assert report.settled == [("round-1", EXPIRED_REAPED)] and not report.failures
        assert (await rounds.get("round-1")).state == "settled"
        assert _rows(path, "leases") == []
        assert _rows(path, "events") == before["events"]
    finally:
        db.close()


@pytest.mark.skipif(sys.platform == "win32", reason="End-to-end CLI confirmation requires a real POSIX flock")
def test_confirmation_cli_real_legacy_ledger_then_normal_resume_admission(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    path = _database(tmp_path, legacy=True)
    _task(path)
    _execute(path, "INSERT INTO leases VALUES ('benchmark_lane', 'holder', 'task-1', 123)")
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker', 'task-1')")
    with pytest.raises(ResumeBlocked):
        ensure_resume_safe(tmp_path, owner_scope="local")
    recover = _recover_module()
    monkeypatch.setattr(recover, "_session_recovery_status", Mock(side_effect=AssertionError("not a report command")))
    args = _build_parser().parse_args(
        [
            "recover-session",
            "--session-dir",
            str(tmp_path),
            "--confirm-stopped",
            "task-1",
            "--confirmation-reason",
            _CONFIRMATION_REASON,
        ]
    )
    result = recover._run_recover_session(args)
    assert result == 0
    ensure_resume_safe(tmp_path, owner_scope="local")
    from datetime import datetime, timedelta
    import pwd

    entry = json.loads(_rows(path, "tasks")[0]["history"])[-1]
    assert entry["evidence"]["operator"] == pwd.getpwuid(os.getuid()).pw_name
    assert datetime.fromisoformat(entry["ts"]).utcoffset() == timedelta(0)
