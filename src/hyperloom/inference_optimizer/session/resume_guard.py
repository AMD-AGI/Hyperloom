# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resume admission and explicit operator confirmation of legacy cleanup."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from hyperloom.common.timeutil import now_iso

from . import lock as session_lock
from .paths import db_path_for


class ResumeBlocked(RuntimeError):
    """Persisted ownership cannot be reconciled in this execution scope."""


class CleanupConfirmationError(RuntimeError):
    """An operator confirmation cannot safely update the retained ownership ledger."""


# The states a task cannot leave. A confirmation acts on one of these, or on a
# task still in flight -- never on anything else.
TERMINAL_TASK_STATES = frozenset({"succeeded", "failed", "cancelled"})
CONFIRMABLE_TASK_STATES = TERMINAL_TASK_STATES | {"queued", "running"}


def _label(value: object) -> str:
    text = "unknown" if value is None or value == "" else str(value)
    label = ascii(text[:80])
    return label if len(label) <= 82 else label[:78] + "...'"


def _terminal_uncertainty(state: str, history_json: str) -> str:
    if state not in TERMINAL_TASK_STATES:
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


class _Diagnostics:
    """The ownership records that block a resume, reported up to the first 20."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.total = 0

    def blocked(self, task: object, lane: object, holder: object, reason: str) -> None:
        """Book one record whose owner cannot be shown to have exited."""
        self.total += 1
        if len(self.messages) < 20:
            self.messages.append(f"task={_label(task)} lane={_label(lane)} holder={_label(holder)}: {reason}")

    def rendered(self) -> list[str]:
        """The messages, with a count of any the cap left out."""
        if self.total > len(self.messages):
            return [*self.messages, f"{self.total - len(self.messages)} additional ownership records omitted"]
        return list(self.messages)


def _scan_leases(
    db: sqlite3.Connection,
    tables: set[str],
    owner_scope: str,
    found: _Diagnostics,
) -> tuple[set[str], list[sqlite3.Row]]:
    """Judge each lane lease; return the locally owned task ids and the round leases.

    A round lease is checked later, against the OPEN rounds it must belong to.
    """
    if "leases" not in tables:
        return set(), []
    local_tasks: set[str] = set()
    round_leases: list[sqlite3.Row] = []
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
        found.blocked(row["task_id"], row["lane"], row["holder_id"], "execution owner scope or PID is unobservable")
    return local_tasks, round_leases


def _scan_gpu_leases(db: sqlite3.Connection, tables: set[str], found: _Diagnostics) -> None:
    """A held GPU lease is a worker whose exit is unproven, whatever the task says."""
    if "gpu_leases" not in tables:
        return
    for row in db.execute("SELECT gpu_id, holder_id, task_id FROM gpu_leases"):
        found.blocked(
            row["task_id"],
            "unknown",
            row["holder_id"],
            f"gpu_leases gpu_id={_label(row['gpu_id'])}: worker exit is unproven",
        )


def _scan_tasks(db: sqlite3.Connection, tables: set[str], local_tasks: set[str]) -> tuple[dict, dict]:
    """The task records, and the uncertainty each running one with no local owner carries."""
    tasks: dict = {}
    uncertainties: dict = {}
    if "tasks" not in tables:
        return tasks, uncertainties
    for row in db.execute("SELECT task_id, state, requires_lanes, history FROM tasks"):
        tasks[row["task_id"]] = row
        if row["state"] == "running" and row["task_id"] not in local_tasks:
            uncertainties[row["task_id"]] = "running execution has no observable local owner"
    return tasks, uncertainties


def _scan_open_rounds(
    db: sqlite3.Connection,
    tables: set[str],
    tasks: dict,
    uncertainties: dict,
    found: _Diagnostics,
) -> dict:
    """Judge every OPEN bring-up round against the execution that holds it."""
    open_rounds: dict = {}
    if "bringup_rounds" not in tables:
        return open_rounds
    for row in db.execute("SELECT round_id, holder_task_id FROM bringup_rounds WHERE state='open'"):
        open_rounds[row["round_id"]] = row["holder_task_id"]
        task_id = row["holder_task_id"]
        if task_id not in tasks:
            found.blocked(task_id, "bringup_round", row["round_id"], "OPEN round has no holder execution record")
            continue
        task = tasks[task_id]
        reason = uncertainties.get(task_id) or _terminal_uncertainty(task["state"], task["history"])
        if reason:
            found.blocked(task_id, "bringup_round", row["round_id"], reason)
    return open_rounds


def _inspect(db: sqlite3.Connection, owner_scope: str) -> list[str]:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    found = _Diagnostics()
    local_tasks, round_leases = _scan_leases(db, tables, owner_scope, found)
    _scan_gpu_leases(db, tables, found)
    tasks, uncertainties = _scan_tasks(db, tables, local_tasks)
    open_rounds = _scan_open_rounds(db, tables, tasks, uncertainties, found)

    for row in round_leases:
        if open_rounds.get(row["holder_id"]) != row["task_id"]:
            found.blocked(row["task_id"], row["lane"], row["holder_id"], "round ownership has no matching OPEN round")

    for task_id, reason in uncertainties.items():
        lanes = json.loads(tasks[task_id]["requires_lanes"])
        if not isinstance(lanes, list):
            lanes = []
        found.blocked(task_id, ",".join(str(lane)[:80] for lane in lanes[:3]) or "unknown", "unknown", reason)
    return found.rendered()


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
            "--session-dir <session> --confirm-stopped <task-id> --confirmation-reason <reason>."
        )


def _confirmable_history(task: sqlite3.Row) -> list:
    """The task's history, once its state and shape allow a confirmation at all."""
    state = task["state"]
    if state not in CONFIRMABLE_TASK_STATES:
        raise CleanupConfirmationError("task has an unsupported state")
    try:
        history = json.loads(task["history"])
    except (TypeError, ValueError) as exc:
        raise CleanupConfirmationError("task history is unreadable") from exc
    if not isinstance(history, list):
        raise CleanupConfirmationError("task history must be a list")
    return history


def _target_leases(db: sqlite3.Connection, tables: set[str], task_id: str) -> tuple[list[dict], list[dict]]:
    """The lane and GPU leases this confirmation would release.

    Confirmation is limited to strictly empty owner scope: a lease naming a
    scope has an owner the operator cannot have verified from here.
    """
    leases: list[dict] = []
    if "leases" in tables:
        columns = {row[1] for row in db.execute("PRAGMA table_info(leases)")}
        query = (
            "SELECT lane, holder_id, owner_scope FROM leases WHERE task_id=?"
            if "owner_scope" in columns
            else "SELECT lane, holder_id, '' AS owner_scope FROM leases WHERE task_id=?"
        )
        rows = list(db.execute(query, (task_id,)))
        if any(row["owner_scope"] != "" for row in rows):
            raise CleanupConfirmationError(
                "confirmation is limited to strictly empty owner scope on every target lease"
            )
        leases = [
            {"lane": row["lane"], "holder_id": row["holder_id"]} for row in rows if row["lane"] != "bringup_round"
        ]
    gpu_leases: list[dict] = []
    if "gpu_leases" in tables:
        gpu_leases = [
            dict(row) for row in db.execute("SELECT gpu_id, holder_id FROM gpu_leases WHERE task_id=?", (task_id,))
        ]
    return leases, gpu_leases


def _prior_confirmation_index(history: list) -> int | None:
    """Where an earlier operator confirmation sits in this history, if one does."""
    return next(
        (
            index
            for index, entry in enumerate(history)
            if isinstance(entry, dict)
            and isinstance(entry.get("evidence"), dict)
            and entry["evidence"].get("reason") == "operator_confirmed_stopped"
        ),
        None,
    )


def _check_prior_confirmation_still_holds(
    history: list,
    manual_index: int,
    task: sqlite3.Row,
    *,
    terminal: bool,
    leases: list,
    gpu_leases: list,
) -> None:
    """Refuse when anything after an earlier confirmation takes back what it established."""
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
    if _terminal_uncertainty(task["state"], task["history"]):
        raise CleanupConfirmationError("prior operator confirmation no longer establishes cleanup")


def _write_confirmation(
    db: sqlite3.Connection,
    *,
    task_id: str,
    history: list,
    entry: dict,
    terminal: bool,
    confirmed_at: str,
    leases: list,
    gpu_leases: list,
) -> None:
    """Record the confirmation and release what it covers, inside the caller's transaction."""
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


def _confirm_task_stopped_in_transaction(
    db: sqlite3.Connection,
    *,
    task_id: str,
    reason: str,
    operator: str,
    confirmed_at: str,
) -> dict:
    """Update one task using the caller's existing IMMEDIATE transaction and Row factory."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "tasks" not in tables:
        raise CleanupConfirmationError("confirmation requires an existing tasks table")
    task = db.execute("SELECT state, history FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if task is None:
        raise CleanupConfirmationError(f"task={_label(task_id)} does not exist")
    state = task["state"]
    terminal = state in TERMINAL_TASK_STATES
    history = _confirmable_history(task)
    leases, gpu_leases = _target_leases(db, tables, task_id)

    result = {"status": "nothing_to_confirm", "task_id": task_id, "released_leases": 0, "released_gpu_leases": 0}
    manual_index = _prior_confirmation_index(history)
    if manual_index is not None:
        _check_prior_confirmation_still_holds(
            history,
            manual_index,
            task,
            terminal=terminal,
            leases=leases,
            gpu_leases=gpu_leases,
        )
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
    _write_confirmation(
        db,
        task_id=task_id,
        history=history,
        entry=entry,
        terminal=terminal,
        confirmed_at=confirmed_at,
        leases=leases,
        gpu_leases=gpu_leases,
    )
    result.update(status="confirmed", released_leases=len(leases), released_gpu_leases=len(gpu_leases))
    return result


def confirm_task_stopped(session_dir: Path, *, task_id: str, reason: str) -> dict:
    """Record an operator's explicit physical-cleanup confirmation for one legacy task.

    This does not stop or inspect processes or Ray actors. The operator must first
    verify their exit. Only empty-scope execution ownership is eligible. The session
    lock writes its own metadata; the ownership DB changes atomically or not at all.
    """
    if not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip():
        raise CleanupConfirmationError("confirmation requires one exact nonempty task ID")
    if not isinstance(reason, str) or not reason.strip():
        raise CleanupConfirmationError("confirmation requires a nonempty reason")
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
                return _confirm_task_stopped_in_transaction(
                    db,
                    task_id=task_id,
                    reason=reason,
                    operator=pwd.getpwuid(os.getuid()).pw_name,
                    confirmed_at=now_iso(),
                )
        finally:
            lock.release()
    except (session_lock.SessionAlreadyRunning, session_lock.SessionLockPathError) as exc:
        raise CleanupConfirmationError("cannot confirm cleanup while the session lock is unavailable") from exc
    except (OSError, sqlite3.Error, KeyError) as exc:
        raise CleanupConfirmationError(
            "cannot confirm cleanup in the existing ownership database; no changes committed"
        ) from exc
