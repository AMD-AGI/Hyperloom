# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TaskRegistry — DelegatedTask state machine, persisted in the ``tasks`` table."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.state.task_states import TERMINAL_STATES, TRANSITIONS
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_TASK,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_STARTED,
    EventSpan,
    record_event,
    scalar_attributes,
)


TASK_STATES = tuple(TRANSITIONS)

_TRAJECTORY_STATUS: dict[str, str] = {
    "queued": STATUS_QUEUED,
    "running": STATUS_STARTED,
    "succeeded": STATUS_COMPLETED,
    "failed": STATUS_FAILED,
    "cancelled": STATUS_CANCELLED,
}

# Progress notes a task's ``history`` retains, oldest dropped first.
_MAX_PROGRESS_NOTES = 120
_DISPATCH_CLASSES = frozenset({"llm", "coordinator", "inline"})
_QUEUED_SQL = "SELECT * FROM tasks WHERE state='queued' ORDER BY created_at ASC"
_RUNNING_SQL = "SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC"


@dataclass(frozen=True)
class TaskSpec:
    """What a new ``queued`` task row is created from."""

    kind: str
    params: dict
    idempotency_key: str
    requires_lanes: list[str] | None = None
    lease_ttl_sec: int = 0
    task_id: str | None = None
    dispatch_class: str | None = None
    dispatch_origin: Mapping[str, Any] | None = None


@dataclass
class Task:
    """A delegated task row persisted in the ``tasks`` table."""

    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    requires_lanes: list[str] = field(default_factory=list)
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
            lease_ttl_sec=row["lease_ttl_sec"],
            history=json.loads(row["history"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _validate_dispatch_class(dispatch_class: str | None) -> None:
    if dispatch_class is not None and dispatch_class not in _DISPATCH_CLASSES:
        raise ValueError(f"unknown dispatch_class: {dispatch_class!r}")


def task_dispatch_record(task: Task) -> dict[str, Any] | None:
    """Return complete author-time dispatch evidence; legacy tasks have none."""
    for entry in getattr(task, "history", ()) or ():
        if not isinstance(entry, dict) or "dispatch_class" not in entry:
            continue
        value = str(entry.get("dispatch_class") or "")
        _validate_dispatch_class(value)
        phase = str(entry.get("phase") or "").strip().upper()
        if not phase or "macro_cycle" not in entry or "tick" not in entry:
            return None
        return {
            "dispatch_class": value,
            "allowed": bool(entry.get("allowed")),
            "denial_rule": str(entry.get("denial_rule") or "") or None,
            "phase": phase,
            "macro_cycle": int(entry["macro_cycle"]),
            "tick": int(entry["tick"]),
        }
    return None


def record_task_dispatch(task: Task) -> None:
    """Write a task's author-time dispatch evidence to the phase timeline.

    Fresh tasks carry the phase coordinates frozen when their row was authored;
    legacy rows have no record and remain legacy on resume, so they dispatch
    without evidence.

    Args:
        task: The task row being dispatched.

    Raises:
        RuntimeError: The evidence was not durably recorded.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import ActionDispatch, PhaseCycle

    record = task_dispatch_record(task)
    if record is None:
        return
    recorded = phase_event.record_dispatch(
        ActionDispatch(
            at=PhaseCycle(str(record["phase"]), int(record["macro_cycle"])),
            action=str(task.kind or ""),
            task_id=str(task.task_id or ""),
            tick=int(record["tick"]),
            dispatch_class=str(record["dispatch_class"]),
            allowed=bool(record["allowed"]),
            denial_rule=record["denial_rule"],
            dispatched_unix=time.time(),
        )
    )
    if not recorded:
        raise RuntimeError("dispatch evidence was not durably recorded")


def task_dispatch_evidence(task: Task) -> tuple[str, bool, str | None] | None:
    """Return persisted dispatch admission evidence; legacy tasks have none."""
    for entry in getattr(task, "history", ()) or ():
        if not isinstance(entry, dict) or "dispatch_class" not in entry:
            continue
        value = str(entry.get("dispatch_class") or "")
        _validate_dispatch_class(value)
        return value, bool(entry.get("allowed")), str(entry.get("denial_rule") or "") or None
    return None


def task_dispatch_origin(state: Any) -> dict[str, Any]:
    """Freeze the phase coordinates that own a task when its row is created."""
    return {
        "phase": str(getattr(state, "phase", "") or "").strip().upper(),
        "macro_cycle": int(getattr(state, "macro_cycle", 0) or 0),
        "tick": int(getattr(state, "tick", 0) or 0),
    }


def _validated_dispatch_origin(origin: Mapping[str, Any] | None) -> dict[str, Any]:
    if origin is None:
        raise ValueError("dispatch_origin is required when dispatch_class is set")
    phase = str(origin.get("phase") or "").strip().upper()
    macro_cycle = int(origin.get("macro_cycle", 0) or 0)
    tick = int(origin.get("tick", 0) or 0)
    if not phase:
        raise ValueError("dispatch_origin.phase is required")
    if macro_cycle < 0 or tick < 0:
        raise ValueError("dispatch_origin macro_cycle and tick must be non-negative")
    return {"phase": phase, "macro_cycle": macro_cycle, "tick": tick}


def task_dispatch_class(task: Task) -> str | None:
    """Return explicit dispatch provenance, never a guess for legacy rows."""
    evidence = task_dispatch_evidence(task)
    return evidence[0] if evidence is not None else None


def _validate_dispatch_reuse(task: Task, requested: str | None) -> None:
    _validate_dispatch_class(requested)
    if requested is None:
        return
    existing = task_dispatch_class(task)
    if existing is None:
        return
    if existing != requested:
        raise ValueError(f"idempotent task dispatch_class mismatch: existing={existing!r}, requested={requested!r}")


class IllegalTransition(RuntimeError):
    """Raised when a requested task state transition is not allowed."""


class TaskNotFound(RuntimeError):
    """Raised when a task lookup by ``task_id`` finds no row."""


class TerminalTaskReuse(RuntimeError):
    """An idempotency key already names a task in a terminal state."""


def _record_task_state(
    task_id: str,
    kind: str,
    state: str,
    *,
    evidence: dict[str, Any] | None = None,
    **attributes: Any,
) -> None:
    """Put one task state change on the trajectory; ``queued`` is parented to the scope that created the task."""
    context: dict[str, Any] = {} if state == "queued" else {"parent_span_id": None}
    record_event(
        EVENT_TASK,
        span=EventSpan(span_id=task_id, status=_TRAJECTORY_STATUS[state]),
        task_id=task_id,
        attributes={**scalar_attributes(evidence), **attributes, "name": kind, "kind": kind},
        **context,
    )


def _insert_queued_task(cur: Any, spec: TaskSpec) -> Task:
    """INSERT one ``queued`` row on ``cur`` and return the task it holds.

    The row and the returned :class:`Task` are built from the same values, so
    an in-memory task never describes a row that was written differently.
    ``cur`` belongs to the caller's write transaction.
    """
    now = now_iso()
    dispatch_class = spec.dispatch_class
    _validate_dispatch_class(dispatch_class)
    initial_history: list[dict[str, Any]] = []
    if dispatch_class:
        initial_history.append(
            {
                "dispatch_class": dispatch_class,
                "allowed": True,
                "denial_rule": None,
                **_validated_dispatch_origin(spec.dispatch_origin),
                "ts": now,
            }
        )
    task = Task(
        task_id=spec.task_id or uuid.uuid4().hex,
        kind=spec.kind,
        state="queued",
        params=spec.params,
        idempotency_key=spec.idempotency_key,
        requires_lanes=[] if spec.requires_lanes is None else list(spec.requires_lanes),
        lease_ttl_sec=spec.lease_ttl_sec,
        history=initial_history,
        created_at=now,
        updated_at=now,
    )
    cur.execute(
        "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, "
        "requires_lanes, lease_ttl_sec, "
        "history, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            task.task_id,
            task.kind,
            task.state,
            json.dumps(task.params),
            task.idempotency_key,
            json.dumps(task.requires_lanes),
            task.lease_ttl_sec,
            json.dumps(task.history),
            task.created_at,
            task.updated_at,
        ),
    )
    # A rolled-back insert leaves an open row that never closes, and open rows never project.
    _record_task_state(
        task.task_id,
        task.kind,
        "queued",
        requires_lanes=task.requires_lanes,
        **({"dispatch_class": dispatch_class} if dispatch_class else {}),
    )
    return task


def create_in_cursor(cur: Any, spec: TaskSpec) -> tuple[Task, bool]:
    """Create (or adopt) a task row on a cursor the caller already owns.

    Unlike :meth:`TaskRegistry.create_or_return_existing`, which opens its own
    transaction, the row commits with the caller's work or not at all.

    Args:
        cur: Open cursor inside the caller's write transaction.
        spec: The task to create; ``idempotency_key`` is the UNIQUE key used to
            detect an existing task.

    Returns:
        tuple[Task, bool]: ``(task, was_existing)``.

    Raises:
        TerminalTaskReuse: When the key already names a task in a terminal
            state.
    """
    _validate_dispatch_class(spec.dispatch_class)
    cur.execute("SELECT * FROM tasks WHERE idempotency_key=?", (spec.idempotency_key,))
    existing = cur.fetchone()
    if existing is not None:
        task = Task.from_row(existing)
        _validate_dispatch_reuse(task, spec.dispatch_class)
        if task.state in TERMINAL_STATES:
            raise TerminalTaskReuse(f"idempotency key {spec.idempotency_key!r} already names a {task.state} task")
        return task, True

    return _insert_queued_task(cur, spec), False


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


def _cancel_queued_row(cur: Any, task_id: str, history_json: str, *, now: str, reason: str) -> None:
    """Move one queued row to ``cancelled`` inside the caller's transaction."""
    history = json.loads(history_json)
    history.append({"from": "queued", "to": "cancelled", "ts": now, "evidence": {"reason": reason}})
    cur.execute(
        "UPDATE tasks SET state='cancelled', history=?, updated_at=? WHERE task_id=?",
        (json.dumps(history), now, task_id),
    )


class TaskRegistry:
    """State machine + persistence layer for delegated tasks."""

    def __init__(
        self,
        db: SqliteConnection,
        *,
        dispatch_origin_provider: Callable[[], Mapping[str, Any]] | None = None,
    ):
        """Initialise the registry.

        ``dispatch_origin_provider`` supplies the author-time evidence of a fresh
        dispatched task whose caller passed no ``dispatch_origin``.
        """
        self.db = db
        self._dispatch_origin_provider = dispatch_origin_provider

    async def create_or_return_existing(self, spec: TaskSpec) -> tuple[Task, bool]:
        """Insert a new task row OR return the existing one keyed by idempotency_key. Returns ``(task, was_existing)``."""
        _validate_dispatch_class(spec.dispatch_class)
        existing = await self.db.fetchone("SELECT * FROM tasks WHERE idempotency_key=?", (spec.idempotency_key,))
        if existing is not None:
            task = Task.from_row(existing)
            _validate_dispatch_reuse(task, spec.dispatch_class)
            return task, True

        if (
            spec.dispatch_class is not None
            and spec.dispatch_origin is None
            and self._dispatch_origin_provider is not None
        ):
            spec = replace(spec, dispatch_origin=self._dispatch_origin_provider())
        async with self.db.transaction() as cur:
            task = _insert_queued_task(cur, spec)
        return task, False

    async def get(self, task_id: str) -> Task:
        """Fetch a single task by id."""
        row = await self.db.fetchone("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        if row is None:
            raise TaskNotFound(task_id)
        return Task.from_row(row)

    async def find(self, task_id: str, *, kind: str) -> Task | None:
        """Fetch the task of ``kind`` with this id, or ``None`` when there is no such row."""
        row = await self.db.fetchone("SELECT * FROM tasks WHERE task_id=? AND kind=?", (task_id, kind))
        return Task.from_row(row) if row is not None else None

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
            allowed = TRANSITIONS.get(current_state, frozenset())
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
        _record_task_state(task_id, row["kind"], new_state, evidence=evidence)
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

    async def exists_with_key_prefix(self, kind: str, key_prefix: str, *, states: tuple[str, ...]) -> bool:
        """Return whether a task of ``kind`` in one of ``states`` has this key prefix."""
        placeholders = ",".join("?" for _ in states)
        row = await self.db.fetchone(
            "SELECT 1 FROM tasks WHERE kind=? AND substr(idempotency_key, 1, ?)=? "
            f"AND state IN ({placeholders}) LIMIT 1",  # nosec B608 - generated placeholders only.
            (kind, len(key_prefix), key_prefix, *states),
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
        return [Task.from_row(r) for r in await self.db.fetchall(_QUEUED_SQL)]

    async def running(self) -> list[Task]:
        """Return all running tasks ordered least-recently-updated-first."""
        return [Task.from_row(r) for r in await self.db.fetchall(_RUNNING_SQL)]

    def running_context_sync(self) -> list[Task]:
        """:meth:`running` off the sync path."""
        return [Task.from_row(row) for row in self.db.fetchall_sync(_RUNNING_SQL)]

    def queued_context_sync(self) -> list[Task]:
        """:meth:`queued` off the sync path."""
        return [Task.from_row(row) for row in self.db.fetchall_sync(_QUEUED_SQL)]

    def queued_kind_counts_sync(self) -> dict[str, int]:
        """Return {kind: count} for queued tasks, synchronously."""
        rows = self.db.fetchall_sync(
            "SELECT kind, COUNT(*) AS n FROM tasks WHERE state='queued' GROUP BY kind",
            (),
        )
        return {str(row["kind"]): int(row["n"]) for row in rows}

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
        reclaimed_rows: list[tuple[str, str, int]] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT t.task_id, t.kind, t.history, l.pid, l.owner_scope "
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
                reclaimed_rows.append((task_id, rows[0]["kind"], pid))
        for task_id, kind, pid in reclaimed_rows:
            _record_task_state(task_id, kind, "failed", reason=reason, dead_pid=pid)
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
        kinds: dict[str, str] = {}
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(family_kinds))
            cur.execute(
                f"SELECT task_id, kind, history FROM tasks WHERE state='queued' AND kind IN ({placeholders})",  # nosec B608 - generated placeholders only.
                family_kinds,
            )
            rows = [(r["task_id"], r["kind"], r["history"]) for r in cur.fetchall()]
            now = now_iso()
            for task_id, kind, history_json in rows:
                if str(task_id or "").strip() in spared:
                    continue
                _cancel_queued_row(cur, task_id, history_json, now=now, reason=reason)
                cancelled.append(task_id)
                kinds[task_id] = kind
        for task_id in cancelled:
            _record_task_state(task_id, kinds[task_id], "cancelled", reason=reason)
        return cancelled

    async def cancel_if_queued(self, task_id: str, *, kind: str, reason: str) -> bool:
        """Cancel ``task_id`` if it is a ``kind`` row that has not started; returns whether it was cancelled."""
        async with self.db.transaction() as cur:
            cur.execute("SELECT history FROM tasks WHERE task_id=? AND kind=? AND state='queued'", (task_id, kind))
            row = cur.fetchone()
            if row is None:
                return False
            _cancel_queued_row(cur, task_id, row["history"], now=now_iso(), reason=reason)
        _record_task_state(task_id, kind, "cancelled", reason=reason)
        return True

    async def cancel_queued(
        self,
        *,
        allowed_kinds: set[str] | frozenset[str],
        reason: str,
    ) -> list[str]:
        """Bulk-cancel queued tasks whose kind is not in ``allowed_kinds``."""
        allowed = {str(kind or "").strip() for kind in allowed_kinds if str(kind or "").strip()}
        cancelled: list[str] = []
        kinds: dict[str, str] = {}
        async with self.db.transaction() as cur:
            cur.execute("SELECT task_id, kind, history FROM tasks WHERE state='queued'")
            rows = [(r["task_id"], r["kind"], r["history"]) for r in cur.fetchall()]
            now = now_iso()
            for task_id, kind, history_json in rows:
                if str(kind or "").strip() in allowed:
                    continue
                _cancel_queued_row(cur, task_id, history_json, now=now, reason=reason)
                cancelled.append(task_id)
                kinds[task_id] = kind
        for task_id in cancelled:
            _record_task_state(task_id, kinds[task_id], "cancelled", reason=reason)
        return cancelled


__all__ = [
    "IllegalTransition",
    "TASK_STATES",
    "TERMINAL_STATES",
    "Task",
    "TaskNotFound",
    "TaskRegistry",
    "TaskSpec",
    "TerminalTaskReuse",
    "create_in_cursor",
]
