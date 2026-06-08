# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""TaskRegistry

DelegatedTask state machine, persisted in the ``tasks`` table.

Allowed transitions::

    queued       -> running, cancelled
    running      -> succeeded, failed, cancelled, needs_manual_review
    failed       -> running                    (retry, only when allowed)
    succeeded    -> (terminal)
    cancelled    -> (terminal)
    needs_manual_review -> (terminal blocking)

``idempotency_key`` is a UNIQUE column so re-creating the same logical
task (e.g. after a partial-write crash) returns the existing row rather
than producing a duplicate.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..storage.connection import SqliteConnection


TASK_STATES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "needs_manual_review",
)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled", "needs_manual_review"}),
    "failed": frozenset({"running"}),
    "succeeded": frozenset(),
    "cancelled": frozenset(),
    "needs_manual_review": frozenset(),
}

TERMINAL_STATES = frozenset({"succeeded", "cancelled", "needs_manual_review"})


def _now_iso() -> str:
    """Return the current UTC time as a microsecond-precision ISO 8601 string.

    Returns:
        str: The current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class Task:
    """A delegated task row persisted in the ``tasks`` table.

    Attributes:
        task_id (str): Unique task identifier.
        kind (str): The task kind/action name.
        state (str): Current lifecycle state (one of :data:`TASK_STATES`).
        params (dict): Action parameters.
        idempotency_key (str): UNIQUE key used to de-duplicate re-creations.
        requires_lanes (list[str]): Resource lanes the task needs.
        allowed_tools (list[str]): Tool whitelist for the task.
        side_effects (list[str]): Declared side effects.
        lease_ttl_sec (int): Lease TTL in seconds.
        attempts (int): Number of run attempts so far.
        history (list[dict]): Recorded state-transition history.
        created_at (str): ISO creation timestamp.
        updated_at (str): ISO last-update timestamp.
    """

    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    requires_lanes: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    lease_ttl_sec: int = 0
    attempts: int = 0
    history: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_row(cls, row) -> "Task":
        """Build a :class:`Task` from a ``tasks`` table row.

        Args:
            row: A mapping-like DB row with the ``tasks`` columns; JSON columns
                are decoded.

        Returns:
            Task: The reconstructed task instance.
        """
        return cls(
            task_id=row["task_id"],
            kind=row["kind"],
            state=row["state"],
            params=json.loads(row["params"]),
            idempotency_key=row["idempotency_key"],
            requires_lanes=json.loads(row["requires_lanes"]),
            allowed_tools=json.loads(row["allowed_tools"]),
            side_effects=json.loads(row["side_effects"]),
            lease_ttl_sec=row["lease_ttl_sec"],
            attempts=row["attempts"],
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


# ---------------------------------------------------------------------------
class TaskRegistry:
    """State machine + persistence layer for delegated tasks.

    Wraps the ``tasks`` SQLite table and enforces the allowed lifecycle
    transitions documented at module level.

    Attributes:
        db (SqliteConnection): The backing SQLite connection.
    """

    def __init__(self, db: SqliteConnection):
        """Initialise the registry.

        Args:
            db (SqliteConnection): The backing SQLite connection.
        """
        self.db = db

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> tuple[Task, bool]:
        """Insert a new task row OR return the existing one keyed by idempotency_key.

        Returns ``(task, was_existing)``. ``was_existing=True`` means the row
        was already in the DB (any state, including terminal); callers should
        treat this as a duplicate-emission signal instead of silently
        proceeding as if a fresh task was queued.

        Args:
            kind (str): The task kind/action name.
            params (dict): Action parameters.
            idempotency_key (str): UNIQUE de-duplication key.
            requires_lanes (list[str] | None): Resource lanes needed.
            allowed_tools (list[str] | None): Tool whitelist for the task.
            side_effects (list[str] | None): Declared side effects.
            lease_ttl_sec (int): Lease TTL in seconds. Defaults to ``0``.
            task_id (str | None): Explicit id; a random hex id when ``None``.

        Returns:
            tuple[Task, bool]: ``(task, was_existing)`` where ``was_existing``
                is ``True`` when an existing row was returned.
        """
        existing = await self.db.fetchone(
            "SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)
        )
        if existing is not None:
            return Task.from_row(existing), True

        task_id = task_id or uuid.uuid4().hex
        now = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, "
                "requires_lanes, allowed_tools, side_effects, lease_ttl_sec, "
                "attempts, history, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    kind,
                    "queued",
                    json.dumps(params),
                    idempotency_key,
                    json.dumps(requires_lanes or []),
                    json.dumps(allowed_tools or []),
                    json.dumps(side_effects or []),
                    lease_ttl_sec,
                    0,
                    "[]",
                    now,
                    now,
                ),
            )
        return (
            Task(
                task_id=task_id,
                kind=kind,
                state="queued",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=requires_lanes or [],
                allowed_tools=allowed_tools or [],
                side_effects=side_effects or [],
                lease_ttl_sec=lease_ttl_sec,
                attempts=0,
                history=[],
                created_at=now,
                updated_at=now,
            ),
            False,
        )

    async def create(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        requires_lanes: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> Task:
        """Thin wrapper around :meth:`create_or_return_existing` for callers
        that don't need the ``was_existing`` signal (most legacy callers).

        Args:
            kind (str): The task kind/action name.
            params (dict): Action parameters.
            idempotency_key (str): UNIQUE de-duplication key.
            requires_lanes (list[str] | None): Resource lanes needed.
            allowed_tools (list[str] | None): Tool whitelist for the task.
            side_effects (list[str] | None): Declared side effects.
            lease_ttl_sec (int): Lease TTL in seconds. Defaults to ``0``.
            task_id (str | None): Explicit id; a random hex id when ``None``.

        Returns:
            Task: The created or pre-existing task.
        """
        task, _was_existing = await self.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=idempotency_key,
            requires_lanes=requires_lanes,
            allowed_tools=allowed_tools,
            side_effects=side_effects,
            lease_ttl_sec=lease_ttl_sec,
            task_id=task_id,
        )
        return task

    async def get(self, task_id: str) -> Task:
        """Fetch a single task by id.

        Args:
            task_id (str): The task identifier.

        Returns:
            Task: The matching task.

        Raises:
            TaskNotFound: If no row matches ``task_id``.
        """
        row = await self.db.fetchone(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        )
        if row is None:
            raise TaskNotFound(task_id)
        return Task.from_row(row)

    async def transition(
        self,
        task_id: str,
        new_state: str,
        evidence: dict[str, Any] | None = None,
    ) -> Task:
        """Transition a task to a new state, recording history.

        Increments ``attempts`` when entering ``running`` from ``queued`` or
        ``failed``.

        Args:
            task_id (str): The task identifier.
            new_state (str): The target state (must be in :data:`TASK_STATES`).
            evidence (dict[str, Any] | None): Optional evidence recorded in the
                transition history entry.

        Returns:
            Task: The task after the transition.

        Raises:
            ValueError: If ``new_state`` is not a known state.
            TaskNotFound: If no row matches ``task_id``.
            IllegalTransition: If the transition is not permitted.
        """
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
                raise IllegalTransition(
                    f"cannot transition {task_id!r} from "
                    f"{current_state!r} to {new_state!r}"
                )
            now = _now_iso()
            history = json.loads(row["history"])
            history.append({
                "from": current_state,
                "to": new_state,
                "ts": now,
                "evidence": evidence or {},
            })
            attempts = row["attempts"]
            if new_state == "running" and current_state in ("queued", "failed"):
                attempts += 1
            cur.execute(
                "UPDATE tasks SET state=?, history=?, attempts=?, updated_at=? "
                "WHERE task_id=?",
                (new_state, json.dumps(history), attempts, now, task_id),
            )
        return await self.get(task_id)

    async def queued(self) -> list[Task]:
        """Return all queued tasks ordered oldest-first.

        Returns:
            list[Task]: Queued tasks sorted by creation time.
        """
        rows = await self.db.fetchall(
            "SELECT * FROM tasks WHERE state='queued' ORDER BY created_at ASC"
        )
        return [Task.from_row(r) for r in rows]

    async def running(self) -> list[Task]:
        """Return all running tasks ordered least-recently-updated-first.

        Returns:
            list[Task]: Running tasks sorted by update time.
        """
        rows = await self.db.fetchall(
            "SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC"
        )
        return [Task.from_row(r) for r in rows]

    async def by_state(self, state: str) -> list[Task]:
        """Return all tasks in the given state.

        Args:
            state (str): The state to filter on (must be in
                :data:`TASK_STATES`).

        Returns:
            list[Task]: Matching tasks ordered by update time.

        Raises:
            ValueError: If ``state`` is not a known state.
        """
        if state not in TASK_STATES:
            raise ValueError(f"unknown state: {state!r}")
        rows = await self.db.fetchall(
            "SELECT * FROM tasks WHERE state=? ORDER BY updated_at ASC", (state,)
        )
        return [Task.from_row(r) for r in rows]

    async def cancel_family(self, family_kinds: list[str]) -> list[str]:
        """Bulk-cancel queued tasks of the given kinds (Robustness prune_branch).

        Args:
            family_kinds (list[str]): Task kinds whose queued rows to cancel.

        Returns:
            list[str]: The cancelled task ids.
        """
        if not family_kinds:
            return []
        cancelled: list[str] = []
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(family_kinds))
            cur.execute(
                f"SELECT task_id, history FROM tasks WHERE state='queued' "
                f"AND kind IN ({placeholders})",
                family_kinds,
            )
            rows = [(r["task_id"], r["history"]) for r in cur.fetchall()]
            now = _now_iso()
            for task_id, history_json in rows:
                history = json.loads(history_json)
                history.append({
                    "from": "queued",
                    "to": "cancelled",
                    "ts": now,
                    "evidence": {"reason": "prune_branch"},
                })
                cur.execute(
                    "UPDATE tasks SET state='cancelled', history=?, updated_at=? "
                    "WHERE task_id=?",
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
]
