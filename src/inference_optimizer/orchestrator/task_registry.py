"""TaskRegistry — DESIGN §13.4.

DelegatedTask state machine, persisted in the ``tasks`` table.

Allowed transitions:

    queued       -> running, cancelled
    running      -> succeeded, failed, cancelled, needs_manual_review
    failed       -> running                    (retry, only when allowed)
    succeeded    -> (terminal)
    cancelled    -> (terminal)
    needs_manual_review -> (terminal blocking — Conductor never auto-retries)

``idempotency_key`` is a UNIQUE column so that re-creating the same
logical task (e.g. after a partial-write crash) returns the existing
row rather than producing a duplicate.
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

# state -> set of allowed next states
_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled", "needs_manual_review"}),
    "failed": frozenset({"running"}),  # retry path
    "succeeded": frozenset(),
    "cancelled": frozenset(),
    "needs_manual_review": frozenset(),
}

TERMINAL_STATES = frozenset({"succeeded", "cancelled", "needs_manual_review"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
@dataclass
class Task:
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
    pass


class TaskNotFound(RuntimeError):
    pass


# ---------------------------------------------------------------------------
class TaskRegistry:
    def __init__(self, db: SqliteConnection):
        self.db = db

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
        # Honour idempotency: if a row with this key exists, return it.
        existing = await self.db.fetchone(
            "SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)
        )
        if existing is not None:
            return Task.from_row(existing)

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
        return Task(
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
        )

    async def get(self, task_id: str) -> Task:
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
                    f"cannot transition {task_id!r} from {current_state!r} "
                    f"to {new_state!r}"
                )
            now = _now_iso()
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
                "UPDATE tasks SET state=?, history=?, updated_at=? "
                "WHERE task_id=?",
                (new_state, json.dumps(history), now, task_id),
            )
        return await self.get(task_id)

    async def bump_attempts(self, task_id: str) -> int:
        async with self.db.transaction() as cur:
            cur.execute(
                "UPDATE tasks SET attempts = attempts + 1, updated_at=? "
                "WHERE task_id=?",
                (_now_iso(), task_id),
            )
            cur.execute(
                "SELECT attempts FROM tasks WHERE task_id=?", (task_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise TaskNotFound(task_id)
            return int(row["attempts"])

    async def list_by_state(self, state: str) -> list[Task]:
        rows = await self.db.fetchall(
            "SELECT * FROM tasks WHERE state=? ORDER BY updated_at",
            (state,),
        )
        return [Task.from_row(r) for r in rows]

    async def inflight(self) -> list[Task]:
        return await self.list_by_state("running")

    async def all(self) -> list[Task]:
        rows = await self.db.fetchall(
            "SELECT * FROM tasks ORDER BY created_at"
        )
        return [Task.from_row(r) for r in rows]
