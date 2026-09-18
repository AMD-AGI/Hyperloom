# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only admission for resuming sessions with retained execution ownership."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .paths import db_path_for


class ResumeBlocked(RuntimeError):
    """Persisted ownership cannot be reconciled in this execution scope."""


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
            "before retrying; do not start replacement work on the same resources."
        )
