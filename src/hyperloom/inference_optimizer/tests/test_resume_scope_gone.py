# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resuming a session whose previous execution environment vanished mid-task."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone

import pytest

from hyperloom.inference_optimizer.session import session_paths
from hyperloom.inference_optimizer.session.paths import db_path_for
from hyperloom.inference_optimizer.session.resume_guard import (
    CleanupConfirmationError,
    ResumeBlocked,
    confirm_scope_gone,
    confirm_task_stopped,
    ensure_resume_safe,
)
from hyperloom.orchestrator.bus.storage.schema import ensure_schema

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Confirmation requires a real POSIX flock")

LOST = "0a0a0a0a-1111-4111-8111-000000000001:4026531001"
HERE = "0b0b0b0b-2222-4222-8222-000000000002:4026531002"
OTHER = "0c0c0c0c-3333-4333-8333-000000000003:4026531003"
REASON = "machine of the previous execution environment was lost"
PAST = "2026-10-12T19:58:19.558779+00:00"
FUTURE = "2999-01-01T00:00:00.000000+00:00"

# Shapes left behind by real environments lost mid-task: one coordinator PID held
# every lane of an in-flight measurement (and, in some, a concurrent specialist),
# with queued work and finished history around it.
LOST_SHAPES = {
    "explore": {
        "explore-1": ["benchmark_lane", "gpu_research_lane", "profile_lane", "server_lifecycle"],
    },
    "integrate_and_specialist": {
        "integrate-1": [
            "benchmark_lane",
            "gpu_research_lane",
            "profile_lane",
            "server_lifecycle",
            "workspace_mutation",
        ],
        "specialist-1": ["research_lane"],
    },
}


def _session(tmp_path, running: dict[str, list[str]], *, scope=LOST, expires=PAST):
    path = db_path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        ensure_schema(db)
        rows = [(task_id, "running") for task_id in running]
        rows += [("queued-1", "queued"), ("done-1", "succeeded")]
        for task_id, state in rows:
            db.execute(
                "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, history,"
                " created_at, updated_at) VALUES (?, 'explore', ?, '{}', ?, '[]', ?, ?, ?)",
                (task_id, state, task_id, json.dumps([{"ts": PAST, "from": "queued", "to": state}]), PAST, PAST),
            )
        for task_id, lanes in running.items():
            for lane in lanes:
                _lease(db, lane, task_id, scope, expires)
    return path


def _lease(db, lane, task_id, scope, expires):
    db.execute(
        "INSERT INTO leases VALUES (?, ?, ?, 'explore', 4519, ?, ?, ?)",
        (lane, task_id, task_id, scope, PAST, expires),
    )


def _execute(path, sql, values=()):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(sql, values)


def _rows(path, sql):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(sql)]


def _ledger(tmp_path):
    path = session_paths.pod_history_path(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [record for record in map(json.loads, lines) if record.get("event") == "owner_scope_gone_confirmed"]


@pytest.mark.parametrize("shape", sorted(LOST_SHAPES))
@pytest.mark.parametrize("expires", [PAST, FUTURE])
def test_lost_scope_blocks_resume_until_confirmed_then_work_is_redone(tmp_path, shape, expires):
    running = LOST_SHAPES[shape]
    path = _session(tmp_path, running, expires=expires)
    with pytest.raises(ResumeBlocked, match=LOST):
        ensure_resume_safe(tmp_path, owner_scope=HERE)

    result = confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    ensure_resume_safe(tmp_path, owner_scope=HERE)
    assert result["status"] == "confirmed"
    assert [task["task_id"] for task in result["tasks"]] == sorted(running)
    assert result["released_leases"] == sum(len(lanes) for lanes in running.values())
    assert _rows(path, "SELECT * FROM leases") == []
    states = {row["task_id"]: row["state"] for row in _rows(path, "SELECT task_id, state FROM tasks")}
    assert states == {**dict.fromkeys(running, "cancelled"), "queued-1": "queued", "done-1": "succeeded"}
    for row in _rows(path, "SELECT task_id, history FROM tasks WHERE state='cancelled'"):
        entry = json.loads(row["history"])[-1]
        assert (entry["from"], entry["to"]) == ("running", "cancelled")
        evidence = entry["evidence"]
        assert evidence["reason"] == "operator_confirmed_stopped"
        assert evidence["confirmed_scope_gone"] == LOST
        assert evidence["confirmation_reason"] == REASON
        assert evidence["cleanup_confirmed"] is True
        assert evidence["outcome"]["state"] == "cancelled" and evidence["outcome"]["result"] == {}
    (record,) = _ledger(tmp_path)
    assert record["owner_scope"] == LOST and record["confirmed_by_scope"] == HERE
    assert record["confirmation_reason"] == REASON and record["operator"]
    assert record["cancelled_tasks"] == sorted(running)


def test_lane_less_close_task_needs_confirm_stopped_after_the_scope_is_confirmed(tmp_path):
    """A pod lost during CLOSE: the close step ran without a lane, so it holds no scoped lease."""
    path = _session(tmp_path, LOST_SHAPES["explore"])
    _execute(
        path,
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, history,"
        " created_at, updated_at) VALUES ('report-1', 'report', 'running', '{}', 'report-1', '[]', ?, ?, ?)",
        (json.dumps([{"ts": PAST, "from": "queued", "to": "running"}]), PAST, PAST),
    )
    confirmed = confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)
    assert [task["task_id"] for task in confirmed["tasks"]] == ["explore-1"]

    with pytest.raises(ResumeBlocked) as blocked:
        ensure_resume_safe(tmp_path, owner_scope=HERE)
    message = str(blocked.value)
    assert "task='report-1' lane='unknown' holder='unknown'" in message
    assert "other owner scopes:" not in message
    assert "legacy" not in message
    assert "For a listed task that names no owner scope" in message
    assert "--confirm-stopped <task-id>" in message

    confirm_task_stopped(tmp_path, task_id="report-1", reason=REASON)

    ensure_resume_safe(tmp_path, owner_scope=HERE)
    states = {row["task_id"]: row["state"] for row in _rows(path, "SELECT task_id, state FROM tasks")}
    assert states["report-1"] == "cancelled"


@pytest.mark.parametrize("round_expires_unix", [1.0, 32503680000.0])
def test_open_bringup_round_held_by_the_lost_task_stays_with_the_round_reconciler(tmp_path, round_expires_unix):
    from hyperloom.orchestrator.bus.resource_lock import hold_round_lane

    path = _session(tmp_path, LOST_SHAPES["integrate_and_specialist"])
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "INSERT INTO bringup_rounds (round_id, state, holder_task_id, opened_unix, renewed_unix, expires_unix)"
            " VALUES ('round-1', 'open', 'integrate-1', 0, 0, ?)",
            (round_expires_unix,),
        )
        hold_round_lane(
            db.cursor(), round_id="round-1", holder_task_id="integrate-1", expires_unix=round_expires_unix, now_unix=0
        )
    rounds = _rows(path, "SELECT * FROM bringup_rounds")
    with pytest.raises(ResumeBlocked):
        ensure_resume_safe(tmp_path, owner_scope=HERE)

    result = confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert result["status"] == "confirmed" and result["released_leases"] == 6
    assert [row["lane"] for row in _rows(path, "SELECT lane FROM leases")] == ["bringup_round"]
    assert _rows(path, "SELECT * FROM bringup_rounds") == rounds
    ensure_resume_safe(tmp_path, owner_scope=HERE)


def test_confirmation_is_idempotent(tmp_path):
    path = _session(tmp_path, LOST_SHAPES["explore"])
    confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)
    before = path.read_bytes()

    again = confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert again["status"] == "nothing_to_confirm" and again["tasks"] == []
    assert path.read_bytes() == before
    ensure_resume_safe(tmp_path, owner_scope=HERE)


def test_current_scope_is_refused_without_change(tmp_path):
    path = _session(tmp_path, LOST_SHAPES["explore"], scope=HERE)
    before = path.read_bytes()

    with pytest.raises(CleanupConfirmationError, match="current execution environment"):
        confirm_scope_gone(tmp_path, owner_scope=HERE, reason=REASON, current_scope=HERE)
    with pytest.raises(CleanupConfirmationError, match="unobservable"):
        confirm_scope_gone(tmp_path, owner_scope=HERE, reason=REASON, current_scope="")

    assert path.read_bytes() == before
    assert _ledger(tmp_path) == []


def test_live_lease_of_another_scope_refuses_without_change(tmp_path):
    path = _session(tmp_path, LOST_SHAPES["explore"])
    _execute(
        path,
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, history,"
        " created_at, updated_at) VALUES ('live-1', 'explore', 'running', '{}', 'live-1', '[]', '[]', ?, ?)",
        (PAST, PAST),
    )
    with closing(sqlite3.connect(path)) as db, db:
        _lease(db, "research_lane", "live-1", OTHER, FUTURE)
    before = path.read_bytes()

    with pytest.raises(CleanupConfirmationError, match="live owner scope") as exc:
        confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert OTHER in str(exc.value)
    assert path.read_bytes() == before
    assert _ledger(tmp_path) == []


@pytest.mark.parametrize("expires", [PAST, "not-a-time", "2999-01-01T00:00:00"])
def test_expired_lease_of_another_scope_is_kept_and_still_blocks(tmp_path, expires):
    path = _session(tmp_path, LOST_SHAPES["explore"])
    _execute(
        path,
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, history,"
        " created_at, updated_at) VALUES ('other-1', 'explore', 'running', '{}', 'other-1', '[]', '[]', ?, ?)",
        (PAST, PAST),
    )
    with closing(sqlite3.connect(path)) as db, db:
        _lease(db, "research_lane", "other-1", OTHER, expires)

    if expires != PAST:
        with pytest.raises(CleanupConfirmationError, match="live owner scope"):
            confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)
        return
    confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert [row["owner_scope"] for row in _rows(path, "SELECT owner_scope FROM leases")] == [OTHER]
    with pytest.raises(ResumeBlocked, match="other-1"):
        ensure_resume_safe(tmp_path, owner_scope=HERE)


def test_task_holding_leases_in_two_scopes_is_refused_atomically(tmp_path):
    path = _session(tmp_path, {"a-task": ["benchmark_lane"], "b-task": ["profile_lane"]})
    with closing(sqlite3.connect(path)) as db, db:
        _lease(db, "research_lane", "b-task", OTHER, PAST)
    before = path.read_bytes()

    with pytest.raises(CleanupConfirmationError, match="outside the confirmed owner scope"):
        confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert path.read_bytes() == before


def test_gpu_leases_of_the_lost_scope_are_released_and_others_kept(tmp_path):
    path = _session(tmp_path, LOST_SHAPES["explore"])
    _execute(path, "INSERT INTO gpu_leases VALUES (0, 'worker-0', 'explore-1')")
    _execute(path, "INSERT INTO gpu_leases VALUES (1, 'worker-1', 'unrelated')")

    result = confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)

    assert result["released_gpu_leases"] == 1
    assert [row["gpu_id"] for row in _rows(path, "SELECT gpu_id FROM gpu_leases")] == [1]


def test_held_session_lock_refuses(tmp_path):
    from hyperloom.inference_optimizer.session.lock import SessionLock

    path = _session(tmp_path, LOST_SHAPES["explore"])
    before = path.read_bytes()
    with SessionLock(tmp_path):
        with pytest.raises(CleanupConfirmationError, match="session lock"):
            confirm_scope_gone(tmp_path, owner_scope=LOST, reason=REASON, current_scope=HERE)
    assert path.read_bytes() == before


@pytest.mark.parametrize("scope,reason", [("", REASON), (" " + LOST, REASON), (LOST, ""), (LOST, " ")])
def test_arguments_are_validated_before_the_lock(tmp_path, scope, reason):
    with pytest.raises(CleanupConfirmationError, match="owner scope|reason"):
        confirm_scope_gone(tmp_path, owner_scope=scope, reason=reason, current_scope=HERE)
    assert not session_paths.optimizer_lock_path(tmp_path).exists()


def _recover(tmp_path, monkeypatch, *argv):
    from hyperloom.common import owner_scope
    from hyperloom.inference_optimizer.cli import main

    monkeypatch.setattr(owner_scope, "local_owner_scope", lambda: HERE)
    return main(["recover", "--session-dir", str(tmp_path), *argv])


def test_recover_cli_confirms_lost_scope_end_to_end(tmp_path, monkeypatch, capsys):
    _session(tmp_path, LOST_SHAPES["integrate_and_specialist"])

    code = _recover(tmp_path, monkeypatch, "--confirm-scope-gone", LOST, "--confirmation-reason", REASON)

    assert code == 0
    out = capsys.readouterr().out
    assert "status=confirmed cancelled_tasks=2 released_leases=6" in out
    ensure_resume_safe(tmp_path, owner_scope=HERE)


def test_recover_cli_refuses_current_scope(tmp_path, monkeypatch, capsys):
    path = _session(tmp_path, LOST_SHAPES["explore"], scope=HERE)
    before = path.read_bytes()

    code = _recover(tmp_path, monkeypatch, "--confirm-scope-gone", HERE, "--confirmation-reason", REASON)

    assert code == 2 and "current execution environment" in capsys.readouterr().err
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "argv",
    [
        ["--confirm-scope-gone", LOST],
        ["--confirm-scope-gone", LOST, "--confirm-stopped", "explore-1", "--confirmation-reason", REASON],
        ["--confirm-scope-gone", LOST, "--confirmation-reason", REASON, "--force"],
        ["--confirm-scope-gone", " ", "--confirmation-reason", REASON],
    ],
)
def test_recover_cli_rejects_bad_combinations_without_change(tmp_path, monkeypatch, argv):
    path = _session(tmp_path, LOST_SHAPES["explore"])
    before = path.read_bytes()

    assert _recover(tmp_path, monkeypatch, *argv) == 2

    assert path.read_bytes() == before
    with pytest.raises(ResumeBlocked):
        ensure_resume_safe(tmp_path, owner_scope=HERE)


# The two-part scope earlier versions recorded, and this container's three-part scope of
# the same boot and PID namespace: a later container on the node repeated the inode.
LEGACY = "0d0d0d0d-4444-4444-8444-000000000004:4026552241"
HERE_NOW = LEGACY + ":123456"


def _legacy_session(tmp_path, *, pid, acquired):
    path = _session(tmp_path, {"explore-1": ["benchmark_lane"]}, scope=LEGACY, expires=FUTURE)
    _execute(path, "UPDATE leases SET pid=?, acquired_at=?", (pid, acquired))
    return path


def test_an_earlier_containers_two_part_scope_with_no_holder_here_is_confirmed(tmp_path):
    """Production 10-14: the lost container's two-part scope equaled the new container's
    boot and namespace; recover refused it as current. Its holder does not run here."""
    path = _legacy_session(tmp_path, pid=4519, acquired=PAST)
    result = confirm_scope_gone(tmp_path, owner_scope=LEGACY, reason=REASON, current_scope=HERE_NOW)
    assert result["status"] == "confirmed"
    assert _rows(path, "SELECT * FROM leases") == []


def test_a_two_part_scope_whose_holder_runs_in_this_container_is_refused(tmp_path):
    """An optimizer of an earlier version still running here took the lease: not gone."""
    started = datetime.now(timezone.utc).isoformat()
    path = _legacy_session(tmp_path, pid=os.getpid(), acquired=started)
    with pytest.raises(CleanupConfirmationError, match="still runs in this execution environment"):
        confirm_scope_gone(tmp_path, owner_scope=LEGACY, reason=REASON, current_scope=HERE_NOW)
    assert len(_rows(path, "SELECT * FROM leases")) == 1
    assert _ledger(tmp_path) == []


def test_a_reused_pid_that_started_after_the_lease_is_not_its_holder(tmp_path):
    path = _legacy_session(tmp_path, pid=os.getpid(), acquired=PAST)
    result = confirm_scope_gone(tmp_path, owner_scope=LEGACY, reason=REASON, current_scope=HERE_NOW)
    assert result["status"] == "confirmed"
    assert _rows(path, "SELECT * FROM leases") == []


def test_resume_after_an_in_place_upgrade_still_sees_its_own_two_part_lease(tmp_path):
    """A session continued in the container that recorded the two-part scope: its rows are
    local, for the dead-holder reaper to judge, not a foreign scope that blocks."""
    _legacy_session(tmp_path, pid=os.getpid(), acquired=datetime.now(timezone.utc).isoformat())
    ensure_resume_safe(tmp_path, owner_scope=HERE_NOW)
