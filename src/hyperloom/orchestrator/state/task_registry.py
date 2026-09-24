# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TaskRegistry — DelegatedTask state machine, persisted in the ``tasks`` table."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.state.task_states import TERMINAL_STATES, TRANSITIONS


TASK_STATES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)

# Re-exported: the state machine moved to ``task_states`` so ``bus`` can read it
# without importing this module back. Existing callers keep their import site.
_TRANSITIONS = TRANSITIONS

# Progress notes a task's ``history`` retains, oldest dropped first.
_MAX_PROGRESS_NOTES = 120


@dataclass
class Task:
    """A delegated task row persisted in the ``tasks`` table."""

    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    requires_lanes: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    lease_ttl_sec: int = 0
    history: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def from_row(cls, row) -> "Task":
        """Build a :class:`Task` from a ``tasks`` table row."""
        return cls(
            task_id=row["task_id"],
            kind=row["kind"],
            state=row["state"],
            params=json.loads(row["params"]),
            idempotency_key=row["idempotency_key"],
            requires_lanes=json.loads(row["requires_lanes"]),
            side_effects=json.loads(row["side_effects"]),
            lease_ttl_sec=row["lease_ttl_sec"],
            history=json.loads(row["history"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class IllegalTransition(RuntimeError):
    """Raised when a requested task state transition is not allowed."""

    pass


class TaskNotFound(RuntimeError):
    """Raised when a task lookup by ``task_id`` finds no row."""

    pass


class TerminalTaskReuse(RuntimeError):
    """An idempotency key already names a task in a terminal state."""


def _insert_queued_task(
    cur: Any,
    *,
    kind: str,
    params: dict,
    idempotency_key: str,
    requires_lanes: list[str] | None,
    side_effects: list[str] | None,
    lease_ttl_sec: int,
    task_id: str | None,
) -> Task:
    """INSERT one ``queued`` row on ``cur`` and return the task it holds.

    The row and the returned :class:`Task` are built from the same values, so
    an in-memory task never describes a row that was written differently.
    ``cur`` belongs to the caller's write transaction.
    """
    now = now_iso()
    task = Task(
        task_id=task_id or uuid.uuid4().hex,
        kind=kind,
        state="queued",
        params=params,
        idempotency_key=idempotency_key,
        requires_lanes=[] if requires_lanes is None else list(requires_lanes),
        side_effects=[] if side_effects is None else list(side_effects),
        lease_ttl_sec=lease_ttl_sec,
        history=[],
        created_at=now,
        updated_at=now,
    )
    cur.execute(
        "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, "
        "requires_lanes, side_effects, lease_ttl_sec, "
        "history, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            task.task_id,
            task.kind,
            task.state,
            json.dumps(task.params),
            task.idempotency_key,
            json.dumps(task.requires_lanes),
            json.dumps(task.side_effects),
            task.lease_ttl_sec,
            "[]",
            task.created_at,
            task.updated_at,
        ),
    )
    return task


def create_in_cursor(
    cur: Any,
    *,
    kind: str,
    params: dict,
    idempotency_key: str,
    requires_lanes: list[str] | None = None,
    side_effects: list[str] | None = None,
    lease_ttl_sec: int = 0,
    task_id: str | None = None,
) -> tuple[Task, bool]:
    """Create (or adopt) a task row on a cursor the caller already owns.

    Unlike :meth:`TaskRegistry.create_or_return_existing`, which opens its own
    transaction, the row commits with the caller's work or not at all.

    Args:
        cur: Open cursor inside the caller's write transaction.
        kind: Task kind tag.
        params: Task parameters serialised into the row.
        idempotency_key: UNIQUE key used to detect an existing task.
        requires_lanes: Lanes the task must hold while running.
        side_effects: Declared side effects of the task.
        lease_ttl_sec: Lease time-to-live in seconds.
        task_id: Optional explicit task id; generated when omitted.

    Returns:
        tuple[Task, bool]: ``(task, was_existing)``.

    Raises:
        TerminalTaskReuse: When the key already names a task in a terminal
            state.
    """
    cur.execute("SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,))
    existing = cur.fetchone()
    if existing is not None:
        task = Task.from_row(existing)
        if task.state in TERMINAL_STATES:
            raise TerminalTaskReuse(f"idempotency key {idempotency_key!r} already names a {task.state} task")
        return task, True

    return (
        _insert_queued_task(
            cur,
            kind=kind,
            params=params,
            idempotency_key=idempotency_key,
            requires_lanes=requires_lanes,
            side_effects=side_effects,
            lease_ttl_sec=lease_ttl_sec,
            task_id=task_id,
        ),
        False,
    )


def _is_progress_note(entry: Any) -> bool:
    """Report whether a ``history`` entry is a progress note."""
    return isinstance(entry, dict) and "progress" in entry


def _drop_oldest_progress_notes(history: list[Any], keep: int) -> list[Any]:
    """Retain the newest ``keep`` progress notes and every other entry."""
    surplus = sum(1 for entry in history if _is_progress_note(entry)) - keep
    if surplus <= 0:
        return history
    kept: list[Any] = []
    for entry in history:
        if surplus > 0 and _is_progress_note(entry):
            surplus -= 1
            continue
        kept.append(entry)
    return kept


class TaskRegistry:
    """State machine + persistence layer for delegated tasks."""

    def __init__(self, db: SqliteConnection):
        """Initialise the registry."""
        self.db = db

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list[str] | None = None,
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> tuple[Task, bool]:
        """Insert a new task row OR return the existing one keyed by idempotency_key. Returns ``(task, was_existing)``."""
        existing = await self.db.fetchone("SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,))
        if existing is not None:
            return Task.from_row(existing), True

        async with self.db.transaction() as cur:
            task = _insert_queued_task(
                cur,
                kind=kind,
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=requires_lanes,
                side_effects=side_effects,
                lease_ttl_sec=lease_ttl_sec,
                task_id=task_id,
            )
        return task, False

    async def create(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list[str] | None = None,
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> Task:
        """Thin wrapper around :meth:`create_or_return_existing` for callers that don't need ``was_existing``."""
        task, _was_existing = await self.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=idempotency_key,
            requires_lanes=requires_lanes,
            side_effects=side_effects,
            lease_ttl_sec=lease_ttl_sec,
            task_id=task_id,
        )
        return task

    async def get(self, task_id: str) -> Task:
        """Fetch a single task by id."""
        row = await self.db.fetchone("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        if row is None:
            raise TaskNotFound(task_id)
        return Task.from_row(row)

    async def transition(
        self,
        task_id: str,
        new_state: str,
        evidence: dict[str, Any] | None = None,
    ) -> Task:
        """Transition a task to a new state, recording history."""
        if new_state not in TASK_STATES:
            raise ValueError(f"unknown state: {new_state!r}")
        async with self.db.transaction() as cur:
            cur.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
            row = cur.fetchone()
            if row is None:
                raise TaskNotFound(task_id)
            current_state = row["state"]
            allowed = _TRANSITIONS.get(current_state, frozenset())
            if new_state not in allowed:
                raise IllegalTransition(f"cannot transition {task_id!r} from {current_state!r} to {new_state!r}")
            now = now_iso()
            history = json.loads(row["history"])
            history.append(
                {
                    "from": current_state,
                    "to": new_state,
                    "ts": now,
                    "evidence": evidence or {},
                }
            )
            cur.execute(
                "UPDATE tasks SET state=?, history=?, updated_at=? WHERE task_id=?",
                (new_state, json.dumps(history), now, task_id),
            )
        return await self.get(task_id)

    async def record_progress(
        self,
        task_id: str,
        note: dict[str, Any] | None = None,
    ) -> None:
        """Record that a running task made progress, without changing its state."""
        async with self.db.transaction() as cur:
            cur.execute("SELECT history FROM tasks WHERE task_id=?", (task_id,))
            row = cur.fetchone()
            if row is None:
                return
            history = json.loads(row["history"])
            history.append({"progress": note or {}, "ts": now_iso()})
            history = _drop_oldest_progress_notes(history, _MAX_PROGRESS_NOTES)
            cur.execute(
                "UPDATE tasks SET history=? WHERE task_id=?",
                (json.dumps(history), task_id),
            )

    async def append_completion_evidence(self, task_id: str, evidence: dict[str, Any] | None = None) -> None:
        """Append durable completion evidence without changing state or updated_at.

        This is not progress and must survive progress-history pruning. A
        repeated completion appends again; a missing row remains an error.
        """
        async with self.db.transaction() as cur:
            cur.execute("SELECT history FROM tasks WHERE task_id=?", (task_id,))
            history = json.loads(cur.fetchone()["history"])
            history.append({"ts": now_iso(), "evidence": evidence or {}})
            cur.execute("UPDATE tasks SET history=? WHERE task_id=?", (json.dumps(history), task_id))

    async def integrate_reconcile_child_exists(self, base_key: str, *, states: tuple[str, ...]) -> bool:
        """Return whether an integrate-patch reconcile child exists in these states."""
        if not states:
            return False
        prefix = f"{base_key}-reconcile%"
        placeholders = ",".join("?" for _ in states)
        row = await self.db.fetchone(
            "SELECT 1 FROM tasks WHERE kind='integrate_patch' "
            f"AND idempotency_key LIKE ? AND state IN ({placeholders}) LIMIT 1",
            (prefix, *states),
        )
        return row is not None

    async def find_by_idempotency_key(self, idempotency_key: str) -> Task | None:
        """Return the task registered under ``idempotency_key``, or None."""
        row = await self.db.fetchone(
            "SELECT * FROM tasks WHERE idempotency_key=?",
            (idempotency_key,),
        )
        return None if row is None else Task.from_row(row)

    async def queued(self) -> list[Task]:
        """Return all queued tasks ordered oldest-first."""
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state='queued' ORDER BY created_at ASC")
        return [Task.from_row(r) for r in rows]

    async def running(self) -> list[Task]:
        """Return all running tasks ordered least-recently-updated-first."""
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC")
        return [Task.from_row(r) for r in rows]

    def running_context_sync(self) -> Iterator[tuple[Task, list[str], str, list[int]]]:
        """Read running tasks, then project their lanes, soonest expiry and GPUs.

        Only the task query runs eagerly. Iteration reads resources separately
        before decoding tasks; these reads do not form a transactional snapshot.
        """
        rows = self.db.fetchall_sync(
            "SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC",
            (),
        )

        def project() -> Iterator[tuple[Task, list[str], str, list[int]]]:
            if not rows:
                return
            lanes_by_task: dict[str, list[str]] = {}
            expiry_by_task: dict[str, str] = {}
            for row in self.db.fetchall_sync("SELECT lane, task_id, expires_at FROM leases", ()):
                tid = str(row["task_id"])
                lanes_by_task.setdefault(tid, []).append(str(row["lane"]))
                expires = str(row["expires_at"])
                prev = expiry_by_task.get(tid)
                if prev is None or expires < prev:
                    expiry_by_task[tid] = expires
            gpus_by_task: dict[str, list[int]] = {}
            for row in self.db.fetchall_sync("SELECT gpu_id, task_id FROM gpu_leases", ()):
                gpus_by_task.setdefault(str(row["task_id"]), []).append(int(row["gpu_id"]))
            for row in rows:
                task = Task.from_row(row)
                yield (
                    task,
                    lanes_by_task.get(task.task_id, []),
                    expiry_by_task.get(task.task_id, ""),
                    gpus_by_task.get(task.task_id, []),
                )

        return project()

    async def extend_lease(self, task_id: str, extra_sec: int) -> int:
        """Grow a running task's ``lease_ttl_sec`` by ``extra_sec``."""
        async with self.db.transaction() as cur:
            cur.execute("SELECT state, lease_ttl_sec FROM tasks WHERE task_id=?", (task_id,))
            row = cur.fetchone()
            if row is None:
                raise TaskNotFound(task_id)
            if row["state"] != "running":
                raise IllegalTransition(f"cannot extend lease of {task_id!r} in state {row['state']!r}")
            new_ttl = int(row["lease_ttl_sec"] or 0) + max(0, int(extra_sec))
            cur.execute(
                "UPDATE tasks SET lease_ttl_sec=? WHERE task_id=?",
                (new_ttl, task_id),
            )
        return new_ttl

    async def by_state(self, state: str) -> list[Task]:
        """Return all tasks in the given state."""
        if state not in TASK_STATES:
            raise ValueError(f"unknown state: {state!r}")
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state=? ORDER BY updated_at ASC", (state,))
        return [Task.from_row(r) for r in rows]

    async def reclaim_dead_running(
        self,
        *,
        reason: str = "dead_holder",
    ) -> list[str]:
        """Fail running tasks whose lease-holder process is provably dead."""
        reclaimed: list[str] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT t.task_id, t.history, l.pid, l.owner_scope "
                "FROM tasks t JOIN leases l ON l.task_id = t.task_id "
                "WHERE t.state='running' AND l.pid > 0"
            )
            holders: dict[str, list] = {}
            for row in cur.fetchall():
                holders.setdefault(row["task_id"], []).append(row)
            ts_now = now_iso()
            for task_id, rows in holders.items():
                if not all(SqliteLeaseBackend.holder_is_dead(row) for row in rows):
                    continue
                pid = int(rows[0]["pid"])
                history_json = rows[0]["history"]
                history = json.loads(history_json)
                history.append(
                    {
                        "from": "running",
                        "to": "failed",
                        "ts": ts_now,
                        "evidence": {"reason": reason, "dead_pid": pid},
                    }
                )
                cur.execute(
                    "UPDATE tasks SET state='failed', history=?, updated_at=? WHERE task_id=?",
                    (json.dumps(history), ts_now, task_id),
                )
                reclaimed.append(task_id)
        return reclaimed

    async def cancel_family(
        self,
        family_kinds: list[str],
        *,
        reason: str = "prune_branch",
        exclude_task_ids: Iterable[str] = (),
    ) -> list[str]:
        """Bulk-cancel queued tasks of the given kinds; returns cancelled task_ids."""
        if not family_kinds:
            return []
        spared = {str(t or "").strip() for t in exclude_task_ids if str(t or "").strip()}
        cancelled: list[str] = []
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(family_kinds))
            cur.execute(
                f"SELECT task_id, history FROM tasks WHERE state='queued' AND kind IN ({placeholders})",  # nosec B608 - generated placeholders only.
                family_kinds,
            )
            rows = [(r["task_id"], r["history"]) for r in cur.fetchall()]
            now = now_iso()
            for task_id, history_json in rows:
                if str(task_id or "").strip() in spared:
                    continue
                history = json.loads(history_json)
                history.append(
                    {
                        "from": "queued",
                        "to": "cancelled",
                        "ts": now,
                        "evidence": {"reason": reason},
                    }
                )
                cur.execute(
                    "UPDATE tasks SET state='cancelled', history=?, updated_at=? WHERE task_id=?",
                    (json.dumps(history), now, task_id),
                )
                cancelled.append(task_id)
        return cancelled

    async def cancel_queued(
        self,
        *,
        allowed_kinds: set[str] | frozenset[str],
        reason: str,
    ) -> list[str]:
        """Bulk-cancel queued tasks whose kind is not in ``allowed_kinds``."""
        allowed = {str(kind or "").strip() for kind in allowed_kinds if str(kind or "").strip()}
        cancelled: list[str] = []
        async with self.db.transaction() as cur:
            cur.execute("SELECT task_id, kind, history FROM tasks WHERE state='queued'")
            rows = [(r["task_id"], r["kind"], r["history"]) for r in cur.fetchall()]
            now = now_iso()
            for task_id, kind, history_json in rows:
                if str(kind or "").strip() in allowed:
                    continue
                history = json.loads(history_json)
                history.append(
                    {
                        "from": "queued",
                        "to": "cancelled",
                        "ts": now,
                        "evidence": {"reason": reason},
                    }
                )
                cur.execute(
                    "UPDATE tasks SET state='cancelled', history=?, updated_at=? WHERE task_id=?",
                    (json.dumps(history), now, task_id),
                )
                cancelled.append(task_id)
        return cancelled


__all__ = [
    "IllegalTransition",
    "TASK_STATES",
    "TERMINAL_STATES",
    "Task",
    "TaskNotFound",
    "TaskRegistry",
    "TerminalTaskReuse",
    "create_in_cursor",
]
