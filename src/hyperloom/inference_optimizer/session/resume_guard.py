# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resume admission and explicit operator confirmation of cleanup the ledger cannot prove."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from pathlib import Path

from hyperloom.common.timeutil import now_iso

from . import lock as session_lock
from . import session_paths
from .paths import db_path_for


class ResumeBlocked(RuntimeError):
    """Persisted ownership cannot be reconciled in this execution scope."""


class CleanupConfirmationError(RuntimeError):
    """An operator confirmation cannot safely update the retained ownership ledger."""


def _label(value: object) -> str:
    text = "unknown" if value is None or value == "" else str(value)
    label = ascii(text[:80])
    return label if len(label) <= 82 else label[:78] + "...'"


def _terminal_uncertainty(state: str, history_json: str) -> str:
    if state not in {"succeeded", "failed", "cancelled"}:
        return ""
    history = json.loads(history_json)
    if not isinstance(history, list):
        return "unreadable terminal execution evidence"
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("outcome"), dict):
            if evidence.get("cleanup_confirmed") is False:
                return "physical cleanup is unconfirmed"
            if evidence.get("cleanup_confirmed") is True:
                return ""
        if entry.get("to") != state:
            continue
        if not isinstance(evidence, dict):
            return "unreadable terminal execution evidence"
        if evidence.get("reason") == "cancelled_in_flight":
            return "cancelled_in_flight is not proof of execution exit"
        if "lease_ttl_sec" in evidence and "dead_pid" not in evidence:
            return "lease timeout is not proof of execution exit"
        return ""
    return "no recorded terminal execution transition"


def _inspect(db: sqlite3.Connection, owner_scope: str) -> list[str]:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    diagnostics: list[str] = []
    total = 0

    def blocked(task: object, lane: object, holder: object, reason: str) -> None:
        nonlocal total
        total += 1
        if len(diagnostics) < 20:
            diagnostics.append(f"task={_label(task)} lane={_label(lane)} holder={_label(holder)}: {reason}")

    local_tasks: set[str] = set()
    round_leases: list[sqlite3.Row] = []
    if "leases" in tables:
        columns = {row[1] for row in db.execute("PRAGMA table_info(leases)")}
        query = (
            "SELECT lane, holder_id, task_id, pid, owner_scope FROM leases"
            if "owner_scope" in columns
            else "SELECT lane, holder_id, task_id, pid, '' AS owner_scope FROM leases"
        )
        for row in db.execute(query):
            if row["lane"] == "bringup_round":
                round_leases.append(row)
                continue
            try:
                pid = int(row["pid"] or 0)
            except (TypeError, ValueError):
                pid = 0
            if owner_scope and row["owner_scope"] == owner_scope and pid > 0:
                local_tasks.add(row["task_id"])
                continue
            blocked(row["task_id"], row["lane"], row["holder_id"], "execution owner scope or PID is unobservable")

    if "gpu_leases" in tables:
        for row in db.execute("SELECT gpu_id, holder_id, task_id FROM gpu_leases"):
            blocked(
                row["task_id"],
                "unknown",
                row["holder_id"],
                f"gpu_leases gpu_id={_label(row['gpu_id'])}: worker exit is unproven",
            )

    tasks = {}
    uncertainties = {}
    if "tasks" in tables:
        for row in db.execute("SELECT task_id, state, requires_lanes, history FROM tasks"):
            tasks[row["task_id"]] = row
            if row["state"] == "running" and row["task_id"] not in local_tasks:
                uncertainties[row["task_id"]] = "running execution has no observable local owner"

    open_rounds = {}
    if "bringup_rounds" in tables:
        for row in db.execute("SELECT round_id, holder_task_id FROM bringup_rounds WHERE state='open'"):
            open_rounds[row["round_id"]] = row["holder_task_id"]
            task_id = row["holder_task_id"]
            if task_id not in tasks:
                blocked(task_id, "bringup_round", row["round_id"], "OPEN round has no holder execution record")
            else:
                task = tasks[task_id]
                reason = uncertainties.get(task_id) or _terminal_uncertainty(task["state"], task["history"])
                if reason:
                    blocked(task_id, "bringup_round", row["round_id"], reason)

    for row in round_leases:
        if open_rounds.get(row["holder_id"]) != row["task_id"]:
            blocked(row["task_id"], row["lane"], row["holder_id"], "round ownership has no matching OPEN round")

    for task_id, reason in uncertainties.items():
        lanes = json.loads(tasks[task_id]["requires_lanes"])
        if not isinstance(lanes, list):
            lanes = []
        blocked(task_id, ",".join(str(lane)[:80] for lane in lanes[:3]) or "unknown", "unknown", reason)

    if total > len(diagnostics):
        diagnostics.append(f"{total - len(diagnostics)} additional ownership records omitted")
    return diagnostics


def _other_scopes(db: sqlite3.Connection, owner_scope: str) -> list[str]:
    """Name up to five recorded execution scopes other than ``owner_scope``, for a blocked resume."""
    if "owner_scope" not in {row[1] for row in db.execute("PRAGMA table_info(leases)")}:
        return []
    rows = db.execute(
        "SELECT DISTINCT owner_scope FROM leases WHERE owner_scope NOT IN ('', ?) ORDER BY owner_scope LIMIT 5",
        (owner_scope,),
    )
    scopes = [_label(row[0]) for row in rows]
    return ["other owner scopes: " + " ".join(scopes)] if scopes else []


def ensure_resume_safe(session_dir: Path, *, owner_scope: str) -> None:
    """Reject unobservable residual work without migrating or repairing the DB.

    Observable local owners are left to the coordinator's existing reaper. An
    empty ledger does not prove a recorded running or cancelled execution exited.
    """
    path = db_path_for(session_dir)
    try:
        if not path.exists():
            return
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            diagnostics = _inspect(db, owner_scope)
            if diagnostics:
                diagnostics += _other_scopes(db, owner_scope)
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise ResumeBlocked(
            "cannot inspect persisted execution ownership read-only; resume is blocked. "
            "Keep the session unchanged and inspect it in its original execution environment."
        ) from exc
    if diagnostics:
        raise ResumeBlocked(
            "resume blocked: prior execution exit cannot be established in this environment.\n  "
            + "\n  ".join(diagnostics)
            + "\nNo ownership was cleared. Inspect and finish cleanup in the original execution environment "
            "before retrying; do not start replacement work on the same resources. "
            "For legacy empty-scope ownership only, after verifying the task's entire process tree and any "
            "remote workers/Ray actors have stopped, record that confirmation with hyperloom recover "
            "--session-dir <session> --confirm-stopped <task-id> --confirmation-reason <reason>. "
            "If the environment of another owner scope is gone, record that with hyperloom recover "
            "--session-dir <session> --confirm-scope-gone <owner-scope> --confirmation-reason <reason>."
        )


def _task_leases(db: sqlite3.Connection, tables: set[str], task_id: str, owner_scope: str) -> list[dict]:
    """Return the execution leases a confirmation releases; every one must carry ``owner_scope``.

    The empty scope is the legacy unattributed ownership confirmed task by task. A
    bring-up round's lane never records a scope and stays with the round reconciler.
    """
    if "leases" not in tables:
        return []
    columns = {row[1] for row in db.execute("PRAGMA table_info(leases)")}
    query = (
        "SELECT lane, holder_id, owner_scope FROM leases WHERE task_id=?"
        if "owner_scope" in columns
        else "SELECT lane, holder_id, '' AS owner_scope FROM leases WHERE task_id=?"
    )
    rows = list(db.execute(query, (task_id,)))
    scoped = [row for row in rows if not owner_scope or row["lane"] != "bringup_round"]
    if any(row["owner_scope"] != owner_scope for row in scoped):
        raise CleanupConfirmationError(
            "confirmation is limited to strictly empty owner scope on every target lease"
            if not owner_scope
            else f"task={_label(task_id)} holds a lease outside the confirmed owner scope"
        )
    return [{"lane": row["lane"], "holder_id": row["holder_id"]} for row in rows if row["lane"] != "bringup_round"]


def _confirm_task_stopped_in_transaction(
    db: sqlite3.Connection,
    *,
    task_id: str,
    reason: str,
    operator: str,
    confirmed_at: str,
    owner_scope: str = "",
) -> dict:
    """Update one task using the caller's existing IMMEDIATE transaction and Row factory."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "tasks" not in tables:
        raise CleanupConfirmationError("confirmation requires an existing tasks table")
    task = db.execute("SELECT state, history FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if task is None:
        raise CleanupConfirmationError(f"task={_label(task_id)} does not exist")
    state = task["state"]
    terminal = state in {"succeeded", "failed", "cancelled"}
    if not terminal and state not in {"queued", "running"}:
        raise CleanupConfirmationError("task has an unsupported state")
    try:
        history = json.loads(task["history"])
    except (TypeError, ValueError) as exc:
        raise CleanupConfirmationError("task history is unreadable") from exc
    if not isinstance(history, list):
        raise CleanupConfirmationError("task history must be a list")

    leases = _task_leases(db, tables, task_id, owner_scope)
    gpu_leases = []
    if "gpu_leases" in tables:
        gpu_leases = [
            dict(row) for row in db.execute("SELECT gpu_id, holder_id FROM gpu_leases WHERE task_id=?", (task_id,))
        ]

    result = {"status": "nothing_to_confirm", "task_id": task_id, "released_leases": 0, "released_gpu_leases": 0}
    manual_index = next(
        (
            index
            for index, entry in enumerate(history)
            if isinstance(entry, dict)
            and isinstance(entry.get("evidence"), dict)
            and entry["evidence"].get("reason") == "operator_confirmed_stopped"
        ),
        None,
    )
    if manual_index is not None:
        if not terminal or leases or gpu_leases:
            raise CleanupConfirmationError("new execution or resources exist after the prior operator confirmation")
        for index, entry in enumerate(history[manual_index:], start=manual_index):
            if not isinstance(entry, dict):
                continue
            evidence = entry.get("evidence")
            if isinstance(evidence, dict) and (
                (index > manual_index and "outcome" in evidence)
                or evidence.get("cleanup_confirmed") is False
                or evidence.get("reason") == "cancelled_in_flight"
                or ("lease_ttl_sec" in evidence and "dead_pid" not in evidence)
            ):
                raise CleanupConfirmationError("later evidence contradicts the prior operator confirmation")
            transition = entry.get("to")
            if transition is not None and not isinstance(transition, str):
                raise CleanupConfirmationError("task history contains an invalid transition")
            if transition in {"queued", "running"}:
                raise CleanupConfirmationError("later execution contradicts the prior operator confirmation")
        if _terminal_uncertainty(state, task["history"]):
            raise CleanupConfirmationError("prior operator confirmation no longer establishes cleanup")
        result["status"] = "already_confirmed"
        return result

    uncertain_round = False
    if terminal and "bringup_rounds" in tables:
        open_round = db.execute(
            "SELECT 1 FROM bringup_rounds WHERE state='open' AND holder_task_id=? LIMIT 1", (task_id,)
        ).fetchone()
        uncertain_round = open_round is not None and bool(_terminal_uncertainty(state, task["history"]))
    if state != "running" and not leases and not gpu_leases and not uncertain_round:
        return result

    evidence = {
        "reason": "operator_confirmed_stopped",
        "cleanup_confirmed": True,
        "operator": operator,
        "confirmation_reason": reason,
        "released_leases": leases,
        "released_gpu_leases": gpu_leases,
        **({"confirmed_scope_gone": owner_scope} if owner_scope else {}),
        "outcome": {
            "task_id": task_id,
            "state": "cancelled",
            "result": {},
            "error": "Operator confirmed the entire task process tree and remote workers/Ray actors stopped.",
            "error_class": "operator_confirmed_stopped",
        },
    }
    entry = {"ts": confirmed_at, "evidence": evidence}
    if not terminal:
        entry.update({"from": state, "to": "cancelled"})
    history.append(entry)
    if terminal:
        db.execute("UPDATE tasks SET history=? WHERE task_id=?", (json.dumps(history), task_id))
    else:
        task_columns = {row[1] for row in db.execute("PRAGMA table_info(tasks)")}
        if "updated_at" in task_columns:
            db.execute(
                "UPDATE tasks SET state='cancelled', history=?, updated_at=? WHERE task_id=?",
                (json.dumps(history), confirmed_at, task_id),
            )
        else:
            db.execute("UPDATE tasks SET state='cancelled', history=? WHERE task_id=?", (json.dumps(history), task_id))
    if leases:
        db.execute("DELETE FROM leases WHERE task_id=? AND lane IS NOT 'bringup_round'", (task_id,))
    if gpu_leases:
        db.execute("DELETE FROM gpu_leases WHERE task_id=?", (task_id,))
    result.update(status="confirmed", released_leases=len(leases), released_gpu_leases=len(gpu_leases))
    return result


def _lease_is_live(expires_at: object, now: datetime) -> bool:
    """A lease counts as live until its recorded expiry; an unreadable expiry counts as live."""
    try:
        expires = datetime.fromisoformat(str(expires_at))
    except ValueError:
        return True
    if expires.tzinfo is None:
        return True
    return expires > now


def _confirm_scope_gone_in_transaction(
    db: sqlite3.Connection,
    *,
    owner_scope: str,
    reason: str,
    operator: str,
    confirmed_at: str,
) -> dict:
    """Cancel and release every execution recorded under one vanished owner scope.

    Uses the caller's IMMEDIATE transaction. Refuses while any lease outside that
    scope is unexpired: another execution environment may still be writing.
    """
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = {row[1] for row in db.execute("PRAGMA table_info(leases)")} if "leases" in tables else set()
    if "owner_scope" not in columns:
        raise CleanupConfirmationError("the ownership ledger records no owner scopes; use --confirm-stopped")
    now = datetime.fromisoformat(confirmed_at)
    target_tasks: set[str] = set()
    for row in db.execute("SELECT lane, holder_id, task_id, owner_scope, expires_at FROM leases"):
        if row["lane"] == "bringup_round":
            continue
        if row["owner_scope"] == owner_scope:
            target_tasks.add(row["task_id"])
        elif _lease_is_live(row["expires_at"], now):
            raise CleanupConfirmationError(
                f"lane={_label(row['lane'])} holder={_label(row['holder_id'])} is held by live owner scope "
                f"{_label(row['owner_scope'])} until {_label(row['expires_at'])}; another execution may be "
                "writing this session"
            )
    tasks = [
        _confirm_task_stopped_in_transaction(
            db,
            task_id=task_id,
            reason=reason,
            operator=operator,
            confirmed_at=confirmed_at,
            owner_scope=owner_scope,
        )
        for task_id in sorted(target_tasks)
    ]
    return {
        "status": "confirmed" if any(task["status"] == "confirmed" for task in tasks) else "nothing_to_confirm",
        "owner_scope": owner_scope,
        "tasks": tasks,
        "released_leases": sum(task["released_leases"] for task in tasks),
        "released_gpu_leases": sum(task["released_gpu_leases"] for task in tasks),
    }


def _in_locked_ledger(session_dir: Path, update: Callable[[sqlite3.Connection, str, str], dict]) -> dict:
    """Run ``update(db, operator, confirmed_at)`` in one IMMEDIATE transaction under the session lock.

    The session lock writes its own metadata; the ownership DB changes atomically
    or not at all.
    """
    if session_lock.fcntl is None:
        raise CleanupConfirmationError("confirmation requires POSIX fcntl session locking")
    try:
        if not session_dir.is_dir():
            raise CleanupConfirmationError("confirmation requires an existing session directory")
        lock = session_lock.SessionLock(session_dir)
        try:
            lock.acquire()
            path = db_path_for(session_dir)
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db, db:
                import pwd

                db.row_factory = sqlite3.Row
                db.execute("BEGIN IMMEDIATE")
                return update(db, pwd.getpwuid(os.getuid()).pw_name, now_iso())
        finally:
            lock.release()
    except (session_lock.SessionAlreadyRunning, session_lock.SessionLockPathError) as exc:
        raise CleanupConfirmationError("cannot confirm cleanup while the session lock is unavailable") from exc
    except (OSError, sqlite3.Error, KeyError) as exc:
        raise CleanupConfirmationError(
            "cannot confirm cleanup in the existing ownership database; no changes committed"
        ) from exc


def confirm_task_stopped(session_dir: Path, *, task_id: str, reason: str) -> dict:
    """Record an operator's explicit physical-cleanup confirmation for one legacy task.

    This does not stop or inspect processes or Ray actors. The operator must first
    verify their exit. Only empty-scope execution ownership is eligible.
    """
    if not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip():
        raise CleanupConfirmationError("confirmation requires one exact nonempty task ID")
    if not isinstance(reason, str) or not reason.strip():
        raise CleanupConfirmationError("confirmation requires a nonempty reason")
    return _in_locked_ledger(
        session_dir,
        lambda db, operator, confirmed_at: _confirm_task_stopped_in_transaction(
            db, task_id=task_id, reason=reason, operator=operator, confirmed_at=confirmed_at
        ),
    )


def confirm_scope_gone(session_dir: Path, *, owner_scope: str, reason: str, current_scope: str) -> dict:
    """Record a caller's assertion that a whole execution environment no longer exists.

    Cancels the unfinished tasks recorded under ``owner_scope`` so their work is
    redone rather than counted, and releases their leases and GPU leases. Nothing
    is stopped or inspected: the caller vouches that the machine or container that
    owned the scope is gone. Refuses the current scope, and refuses while another
    scope holds an unexpired lease. The confirmation is appended to the session's
    owner ledger once the database change has committed.
    """
    if not isinstance(owner_scope, str) or not owner_scope.strip() or owner_scope != owner_scope.strip():
        raise CleanupConfirmationError("confirmation requires one exact nonempty owner scope")
    if not isinstance(reason, str) or not reason.strip():
        raise CleanupConfirmationError("confirmation requires a nonempty reason")
    if not current_scope:
        raise CleanupConfirmationError("the current execution scope is unobservable; cannot tell it from the named one")
    if owner_scope == current_scope:
        raise CleanupConfirmationError("the named owner scope is the current execution environment")
    result = _in_locked_ledger(
        session_dir,
        lambda db, operator, confirmed_at: {
            **_confirm_scope_gone_in_transaction(
                db, owner_scope=owner_scope, reason=reason, operator=operator, confirmed_at=confirmed_at
            ),
            "operator": operator,
            "confirmed_at": confirmed_at,
        },
    )
    record = {
        "event": "owner_scope_gone_confirmed",
        "owner_scope": owner_scope,
        "confirmed_by_scope": current_scope,
        "operator": result["operator"],
        "confirmed_at": result["confirmed_at"],
        "confirmation_reason": reason,
        "cancelled_tasks": [task["task_id"] for task in result["tasks"] if task["status"] == "confirmed"],
        "released_leases": result["released_leases"],
        "released_gpu_leases": result["released_gpu_leases"],
    }
    try:
        with session_paths.pod_history_path(session_dir).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        result["ledger_recorded"] = False
    else:
        result["ledger_recorded"] = True
    return result
