# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resume admission and explicit operator confirmation of cleanup the ledger cannot prove."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from pathlib import Path

from hyperloom.common.owner_scope import process_started_at, same_environment
from hyperloom.common.timeutil import now_iso

from . import lock as session_lock
from . import session_paths
from .coordinator_db import OlderCoordinatorDB, refuse_older_schema
from .paths import db_path_for


class ResumeBlocked(RuntimeError):
    """Persisted ownership cannot be reconciled in this execution scope."""


class CleanupConfirmationError(RuntimeError):
    """An operator confirmation cannot safely update the retained ownership ledger."""


_CONFIRMED = "confirmed"


def _tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _lease_columns(db: sqlite3.Connection) -> set[str]:
    return {row[1] for row in db.execute("PRAGMA table_info(leases)")}


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
        return ""
    return "no recorded terminal execution transition"


@dataclass
class _Diagnostics:
    """The records that block a resume, capped at the first 20.

    The cap is the concept: an operator needs to see what is holding the
    session, not every row of it, and the tail is reported as a count.
    """

    lines: list[str] = field(default_factory=list)
    total: int = 0

    def blocked(self, task: object, lane: object, holder: object, reason: str) -> None:
        self.total += 1
        if len(self.lines) < 20:
            self.lines.append(f"task={_label(task)} lane={_label(lane)} holder={_label(holder)}: {reason}")

    def report(self) -> list[str]:
        if self.total > len(self.lines):
            self.lines.append(f"{self.total - len(self.lines)} additional ownership records omitted")
        return self.lines


def _scan_leases(db: sqlite3.Connection, owner_scope: str, found: _Diagnostics) -> tuple[set[str], list[sqlite3.Row]]:
    """Judge the execution leases; return the locally owned tasks and the round leases.

    A round lease is held by a bring-up round rather than an execution, so it
    is judged later, against the rounds that are actually open.
    """
    columns = _lease_columns(db)
    query = (
        "SELECT lane, holder_id, task_id, pid, owner_scope FROM leases"
        if "owner_scope" in columns
        else "SELECT lane, holder_id, task_id, pid, '' AS owner_scope FROM leases"
    )
    local_tasks: set[str] = set()
    round_leases: list[sqlite3.Row] = []
    for row in db.execute(query):
        if row["lane"] == "bringup_round":
            round_leases.append(row)
            continue
        try:
            pid = int(row["pid"] or 0)
        except (TypeError, ValueError):
            pid = 0
        if same_environment(row["owner_scope"], owner_scope) and pid > 0:
            local_tasks.add(row["task_id"])
            continue
        found.blocked(row["task_id"], row["lane"], row["holder_id"], "execution owner scope or PID is unobservable")
    return local_tasks, round_leases


def _scan_gpu_leases(db: sqlite3.Connection, found: _Diagnostics) -> None:
    """A held GPU lease means a worker's exit was never proven."""
    for row in db.execute("SELECT gpu_id, holder_id, task_id FROM gpu_leases"):
        found.blocked(
            row["task_id"],
            "unknown",
            row["holder_id"],
            f"gpu_leases gpu_id={_label(row['gpu_id'])}: worker exit is unproven",
        )


def _scan_tasks(db: sqlite3.Connection, local_tasks: set[str]) -> tuple[dict, dict]:
    """Every task record, and the ones whose running execution nothing here owns."""
    tasks: dict = {}
    uncertainties: dict = {}
    for row in db.execute("SELECT task_id, state, requires_lanes, history FROM tasks"):
        tasks[row["task_id"]] = row
        if row["state"] == "running" and row["task_id"] not in local_tasks:
            uncertainties[row["task_id"]] = "running execution has no observable local owner"
    return tasks, uncertainties


def _scan_open_rounds(db: sqlite3.Connection, tasks: dict, uncertainties: dict, found: _Diagnostics) -> dict:
    """Judge each OPEN bring-up round against its holder; return round -> holder."""
    open_rounds: dict = {}
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
    tables = _tables(db)
    found = _Diagnostics()
    local_tasks: set[str] = set()
    round_leases: list[sqlite3.Row] = []
    if "leases" in tables:
        local_tasks, round_leases = _scan_leases(db, owner_scope, found)
    if "gpu_leases" in tables:
        _scan_gpu_leases(db, found)
    tasks, uncertainties = _scan_tasks(db, local_tasks) if "tasks" in tables else ({}, {})
    open_rounds = _scan_open_rounds(db, tasks, uncertainties, found) if "bringup_rounds" in tables else {}

    for row in round_leases:
        if open_rounds.get(row["holder_id"]) != row["task_id"]:
            found.blocked(row["task_id"], row["lane"], row["holder_id"], "round ownership has no matching OPEN round")

    for task_id, reason in uncertainties.items():
        lanes = json.loads(tasks[task_id]["requires_lanes"])
        if not isinstance(lanes, list):
            lanes = []
        found.blocked(task_id, ",".join(str(lane)[:80] for lane in lanes[:3]) or "unknown", "unknown", reason)
    return found.report()


def _other_scopes(db: sqlite3.Connection, owner_scope: str) -> list[str]:
    """Name up to five recorded execution scopes other than ``owner_scope``, for a blocked resume."""
    if "owner_scope" not in _lease_columns(db):
        return []
    rows = db.execute(
        "SELECT DISTINCT owner_scope FROM leases WHERE owner_scope NOT IN ('', ?) ORDER BY owner_scope LIMIT 5",
        (owner_scope,),
    )
    scopes = [_label(row[0]) for row in rows]
    return ["other owner scopes: " + " ".join(scopes)] if scopes else []


def foreign_owner_scopes(session_dir: Path, owner_scope: str) -> list[str]:
    """The exact owner scopes, other than ``owner_scope``, that hold execution leases in the session's ledger.

    Empty for a session without a readable ledger: ``ensure_resume_safe`` reports what cannot be read.
    """
    try:
        with closing(sqlite3.connect(db_path_for(session_dir).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if "owner_scope" not in _lease_columns(db):
                return []
            rows = db.execute("SELECT DISTINCT owner_scope FROM leases WHERE owner_scope != '' ORDER BY owner_scope")
            return [row[0] for row in rows if not same_environment(row[0], owner_scope)]
    except (OSError, sqlite3.Error):
        return []


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
            refuse_older_schema(db)
            diagnostics = _inspect(db, owner_scope)
            if diagnostics:
                diagnostics += _other_scopes(db, owner_scope)
    except OlderCoordinatorDB as exc:
        raise ResumeBlocked(str(exc)) from exc
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
            "If the environment of another owner scope is gone, record that with hyperloom recover "
            "--session-dir <session> --confirm-scope-gone <owner-scope> --confirmation-reason <reason>. "
            "For a listed task that names no owner scope (such as a lane-less CLOSE-phase step), after "
            "verifying its entire process tree and any remote workers/Ray actors have stopped, record that "
            "confirmation with hyperloom recover --session-dir <session> --confirm-stopped <task-id> "
            "--confirmation-reason <reason>; it refuses, changing nothing, a task recorded under an owner scope."
        )


def _task_leases(db: sqlite3.Connection, tables: set[str], task_id: str, owner_scope: str) -> list[dict]:
    """Return the execution leases a confirmation releases; every one must carry ``owner_scope``.

    The empty scope is the legacy unattributed ownership confirmed task by task. A
    bring-up round's lane never records a scope and stays with the round reconciler.
    """
    if "leases" not in tables:
        return []
    columns = _lease_columns(db)
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


# The states a task cannot leave. A confirmation acts on one of these, or on a
# task still in flight -- never on anything else.
TERMINAL_TASK_STATES = frozenset({"succeeded", "failed", "cancelled"})
IN_FLIGHT_TASK_STATES = frozenset({"queued", "running"})


def _confirmable_history(task: sqlite3.Row) -> list:
    """The task's history, once its state and shape allow a confirmation at all."""
    state = task["state"]
    if state not in TERMINAL_TASK_STATES | IN_FLIGHT_TASK_STATES:
        raise CleanupConfirmationError("task has an unsupported state")
    try:
        history = json.loads(task["history"])
    except (TypeError, ValueError) as exc:
        raise CleanupConfirmationError("task history is unreadable") from exc
    if not isinstance(history, list):
        raise CleanupConfirmationError("task history must be a list")
    return history


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


def _entry_contradicts_confirmation(entry: object, *, index: int, manual_index: int) -> str:
    """Why this history entry outlives the prior confirmation, or \"\".

    Anything recorded after the confirmation that reports an outcome, denies
    cleanup, or moves the task back into flight means the confirmation no
    longer describes the task.
    """
    if not isinstance(entry, dict):
        return ""
    evidence = entry.get("evidence")
    if isinstance(evidence, dict) and (
        (index > manual_index and "outcome" in evidence)
        or evidence.get("cleanup_confirmed") is False
        or evidence.get("reason") == "cancelled_in_flight"
    ):
        return "later evidence contradicts the prior operator confirmation"
    transition = entry.get("to")
    if transition is not None and not isinstance(transition, str):
        return "task history contains an invalid transition"
    if transition in IN_FLIGHT_TASK_STATES:
        return "later execution contradicts the prior operator confirmation"
    return ""


@dataclass(frozen=True)
class _Confirmation:
    """Who confirms cleanup, why and when, and the owner scope vouched gone (\"\" for one task's own confirmation)."""

    reason: str
    operator: str
    confirmed_at: str
    owner_scope: str = ""


@dataclass(frozen=True)
class _ConfirmTarget:
    """The task a confirmation acts on as read in its transaction: row, parsed history and the leases it holds."""

    task_id: str
    row: sqlite3.Row
    history: list
    leases: list
    gpu_leases: list

    @property
    def state(self) -> str:
        return self.row["state"]

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_TASK_STATES


def _check_prior_confirmation_still_holds(target: _ConfirmTarget, *, manual_index: int) -> None:
    """Refuse when anything since the prior confirmation outlived it."""
    if not target.terminal or target.leases or target.gpu_leases:
        raise CleanupConfirmationError("new execution or resources exist after the prior operator confirmation")
    for index, entry in enumerate(target.history[manual_index:], start=manual_index):
        complaint = _entry_contradicts_confirmation(entry, index=index, manual_index=manual_index)
        if complaint:
            raise CleanupConfirmationError(complaint)
    if _terminal_uncertainty(target.state, target.row["history"]):
        raise CleanupConfirmationError("prior operator confirmation no longer establishes cleanup")


def _round_is_uncertain(db: sqlite3.Connection, tables: set, task: sqlite3.Row, task_id: str, terminal: bool) -> bool:
    """Whether a still-open bring-up round leaves this terminal task's cleanup unproven."""
    if not terminal or "bringup_rounds" not in tables:
        return False
    open_round = db.execute(
        "SELECT 1 FROM bringup_rounds WHERE state='open' AND holder_task_id=? LIMIT 1", (task_id,)
    ).fetchone()
    return open_round is not None and bool(_terminal_uncertainty(task["state"], task["history"]))


def _confirmation_entry(target: _ConfirmTarget, confirmation: _Confirmation) -> dict:
    """The history entry this confirmation records, and the transition it carries."""
    evidence = {
        "reason": "operator_confirmed_stopped",
        "cleanup_confirmed": True,
        "operator": confirmation.operator,
        "confirmation_reason": confirmation.reason,
        "released_leases": target.leases,
        "released_gpu_leases": target.gpu_leases,
        **({"confirmed_scope_gone": confirmation.owner_scope} if confirmation.owner_scope else {}),
        "outcome": {
            "task_id": target.task_id,
            "state": "cancelled",
            "result": {},
            "error": "Operator confirmed the entire task process tree and remote workers/Ray actors stopped.",
            "error_class": "operator_confirmed_stopped",
        },
    }
    entry = {"ts": confirmation.confirmed_at, "evidence": evidence}
    if not target.terminal:
        entry.update({"from": target.state, "to": "cancelled"})
    return entry


def confirmed_stopped_scope(history: object) -> str | None:
    """Read back the confirmation :func:`_confirmation_entry` wrote, when it is the task's last outcome.

    Returns:
        ``None`` unless the latest recorded outcome is an operator confirmation
        that the task's execution stopped and its cleanup is complete; otherwise
        the owner scope confirmed gone with it, or ``""`` when the confirmation
        named this one task rather than a whole vanished execution environment.
    """
    for entry in reversed(history if isinstance(history, list) else []):
        evidence = entry.get("evidence") if isinstance(entry, dict) else None
        if not isinstance(evidence, dict) or not isinstance(evidence.get("outcome"), dict):
            continue
        if evidence.get("reason") != "operator_confirmed_stopped" or evidence.get("cleanup_confirmed") is not True:
            return None
        return str(evidence.get("confirmed_scope_gone") or "")
    return None


def _write_confirmation(db: sqlite3.Connection, target: _ConfirmTarget, *, confirmed_at: str) -> None:
    """Persist the confirmed history, the cancellation it implies, and release what it freed."""
    task_id, history = target.task_id, target.history
    if target.terminal:
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
    if target.leases:
        db.execute("DELETE FROM leases WHERE task_id=? AND lane IS NOT 'bringup_round'", (task_id,))
    if target.gpu_leases:
        db.execute("DELETE FROM gpu_leases WHERE task_id=?", (task_id,))


def _confirm_task_stopped_in_transaction(db: sqlite3.Connection, task_id: str, confirmation: _Confirmation) -> dict:
    """Update one task using the caller's existing IMMEDIATE transaction and Row factory."""
    tables = _tables(db)
    if "tasks" not in tables:
        raise CleanupConfirmationError("confirmation requires an existing tasks table")
    task = db.execute("SELECT state, history FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if task is None:
        raise CleanupConfirmationError(f"task={_label(task_id)} does not exist")
    history = _confirmable_history(task)

    leases = _task_leases(db, tables, task_id, confirmation.owner_scope)
    gpu_leases = []
    if "gpu_leases" in tables:
        gpu_leases = [
            dict(row) for row in db.execute("SELECT gpu_id, holder_id FROM gpu_leases WHERE task_id=?", (task_id,))
        ]
    target = _ConfirmTarget(task_id=task_id, row=task, history=history, leases=leases, gpu_leases=gpu_leases)

    result = {"status": "nothing_to_confirm", "task_id": task_id, "released_leases": 0, "released_gpu_leases": 0}
    manual_index = _prior_confirmation_index(history)
    if manual_index is not None:
        _check_prior_confirmation_still_holds(target, manual_index=manual_index)
        result["status"] = "already_confirmed"
        return result

    uncertain_round = _round_is_uncertain(db, tables, task, task_id, target.terminal)
    if target.state != "running" and not leases and not gpu_leases and not uncertain_round:
        return result

    history.append(_confirmation_entry(target, confirmation))
    _write_confirmation(db, target, confirmed_at=confirmation.confirmed_at)
    result.update(status=_CONFIRMED, released_leases=len(leases), released_gpu_leases=len(gpu_leases))
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
    db: sqlite3.Connection, confirmation: _Confirmation, *, check_holders: bool = False
) -> dict:
    """Cancel and release every execution recorded under one vanished owner scope.

    Uses the caller's IMMEDIATE transaction. Refuses while any lease outside that
    scope is unexpired: another execution environment may still be writing. With
    ``check_holders`` (an older two-part scope of this boot and PID namespace, which an
    earlier container may have left), refuses while a holder process that started before
    its lease still runs here.
    """
    tables = _tables(db)
    columns = _lease_columns(db) if "leases" in tables else set()
    if "owner_scope" not in columns:
        raise CleanupConfirmationError("the ownership ledger records no owner scopes; use --confirm-stopped")
    owner_scope = confirmation.owner_scope
    now = datetime.fromisoformat(confirmation.confirmed_at)
    target_tasks: set[str] = set()
    for row in db.execute("SELECT lane, holder_id, task_id, pid, owner_scope, acquired_at, expires_at FROM leases"):
        if row["lane"] == "bringup_round":
            continue
        if row["owner_scope"] == owner_scope:
            if check_holders and _holder_runs_here(row["pid"], row["acquired_at"]):
                raise CleanupConfirmationError(
                    f"lane={_label(row['lane'])} holder={_label(row['holder_id'])} pid={_label(row['pid'])} still "
                    "runs in this execution environment; the named owner scope is not gone"
                )
            target_tasks.add(row["task_id"])
        elif _lease_is_live(row["expires_at"], now):
            raise CleanupConfirmationError(
                f"lane={_label(row['lane'])} holder={_label(row['holder_id'])} is held by live owner scope "
                f"{_label(row['owner_scope'])} until {_label(row['expires_at'])}; another execution may be "
                "writing this session"
            )
    tasks = [_confirm_task_stopped_in_transaction(db, task_id, confirmation) for task_id in sorted(target_tasks)]
    cancelled = [task["task_id"] for task in tasks if task["status"] == _CONFIRMED]
    return {
        "status": _CONFIRMED if cancelled else "nothing_to_confirm",
        "owner_scope": owner_scope,
        "tasks": tasks,
        "cancelled_tasks": cancelled,
        "released_leases": sum(task["released_leases"] for task in tasks),
        "released_gpu_leases": sum(task["released_gpu_leases"] for task in tasks),
    }


def _holder_runs_here(pid: object, acquired_at: object) -> bool:
    """Whether the lease's recorded holder runs here: its PID is live and started no later
    than the lease was taken. A later start is a reused PID; an unreadable record counts as
    running."""
    try:
        holder = int(pid or 0)
        taken = datetime.fromisoformat(str(acquired_at))
    except (TypeError, ValueError):
        return True
    if holder <= 0:
        return False
    started = process_started_at(holder)
    if started is None:
        return False
    if taken.tzinfo is None:
        return True
    return (started - taken).total_seconds() <= 1.0


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
    """Record an operator's explicit physical-cleanup confirmation for one task that names no owner scope.

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
            db, task_id, _Confirmation(reason=reason, operator=operator, confirmed_at=confirmed_at)
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
    here = same_environment(owner_scope, current_scope)
    result = _in_locked_ledger(
        session_dir,
        lambda db, operator, confirmed_at: {
            **_confirm_scope_gone_in_transaction(
                db,
                _Confirmation(reason=reason, operator=operator, confirmed_at=confirmed_at, owner_scope=owner_scope),
                check_holders=here,
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
        "cancelled_tasks": result["cancelled_tasks"],
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
