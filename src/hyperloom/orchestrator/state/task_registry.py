# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TaskRegistry — DelegatedTask state machine, persisted in the ``tasks`` table.

Allowed transitions::

    queued       -> running, cancelled
    running      -> succeeded, failed, cancelled
    succeeded / failed / cancelled -> (terminal)

A retry creates a new row under a fresh ``idempotency_key`` rather than
re-entering ``running``.

``idempotency_key`` is UNIQUE so re-creating a logical task returns the existing row.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection


TASK_STATES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "failed": frozenset(),
    "succeeded": frozenset(),
    "cancelled": frozenset(),
}

TERMINAL_STATES = frozenset(state for state, outgoing in _TRANSITIONS.items() if not outgoing)

# Progress notes a task's ``history`` retains, oldest dropped first.
# ``record_progress`` re-reads and rewrites the whole blob inside a
# ``BEGIN IMMEDIATE`` on the one connection every other writer serialises
# behind, and the robustness probe re-parses it for every running task on every
# tick, so an uncapped trail makes both costs grow with the session: 12 hours at
# the 60s heartbeat is 720 notes, a blob measured between 100 and 160 KB
# depending on the note, and tens of MB of cumulative row rewrites. 120 notes
# hold two hours of trail — longer than the longest measured single work unit, a
# 3941s warmup — for a blob under 20 KB, which turns the growth from quadratic
# in the session's length into linear at a bounded rate. The only consumer that
# reads the notes wants the newest one.
_MAX_PROGRESS_NOTES = 120


# microseconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = now_iso


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
        side_effects (list[str]): Declared side effects.
        lease_ttl_sec (int): Lease TTL in seconds.
        history (list[dict]): Recorded state transitions, plus the newest
            progress notes a running task reported (bounded by
            :data:`_MAX_PROGRESS_NOTES`).
        created_at (str): ISO creation timestamp.
        updated_at (str): ISO last-update timestamp.
    """

    task_id: str
    kind: str
    state: str
    params: dict
    idempotency_key: str
    requires_lanes: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    lease_ttl_sec: int = 0
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


def _is_progress_note(entry: Any) -> bool:
    """Report whether a ``history`` entry is a progress note.

    A note carries a ``progress`` payload where a transition carries
    ``from``/``to``; the robustness probe keys on the same field.

    Args:
        entry (Any): One decoded ``history`` entry.

    Returns:
        bool: ``True`` for a progress note.
    """
    return isinstance(entry, dict) and "progress" in entry


def _drop_oldest_progress_notes(history: list[Any], keep: int) -> list[Any]:
    """Retain the newest ``keep`` progress notes and every other entry.

    Transitions are never dropped: consumers read them positionally — the last
    entry for a failure class, the newest ``queued -> cancelled`` for a policy
    denial — and losing one would make a task's state history lie. Notes are
    only ever read newest-first, so the oldest are the ones that can go.

    Args:
        history (list[Any]): The task's decoded ``history``.
        keep (int): Progress notes to retain.

    Returns:
        list[Any]: ``history`` itself when it is already within the bound,
        otherwise a copy with the oldest surplus notes removed.
    """
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
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> tuple[Task, bool]:
        """Insert a new task row OR return the existing one keyed by idempotency_key. Returns ``(task, was_existing)``.

        Args:
            kind: Task kind tag.
            params: Task parameters serialised into the row.
            idempotency_key: Key used to detect and return an existing task.
            requires_lanes: Lanes the task must hold while running.
            side_effects: Declared side effects of the task.
            lease_ttl_sec: Lease time-to-live in seconds.
            task_id: Optional explicit task id; generated when omitted.

        Returns:
            A tuple ``(task, was_existing)`` where ``was_existing`` is ``True``
            when a task with the same idempotency key already existed.
        """
        existing = await self.db.fetchone("SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,))
        if existing is not None:
            return Task.from_row(existing), True

        task_id = task_id or uuid.uuid4().hex
        now = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, "
                "requires_lanes, side_effects, lease_ttl_sec, "
                "history, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    kind,
                    "queued",
                    json.dumps(params),
                    idempotency_key,
                    json.dumps(requires_lanes or []),
                    json.dumps(side_effects or []),
                    lease_ttl_sec,
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
                side_effects=side_effects or [],
                lease_ttl_sec=lease_ttl_sec,
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
        side_effects: list[str] | None = None,
        lease_ttl_sec: int = 0,
        task_id: str | None = None,
    ) -> Task:
        """Thin wrapper around :meth:`create_or_return_existing` for callers that don't need ``was_existing``.

        Args:
            kind: Task kind tag.
            params: Task parameters serialised into the row.
            idempotency_key: Key used to detect and return an existing task.
            requires_lanes: Lanes the task must hold while running.
            side_effects: Declared side effects of the task.
            lease_ttl_sec: Lease time-to-live in seconds.
            task_id: Optional explicit task id; generated when omitted.

        Returns:
            The created or pre-existing ``Task``.
        """
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
        """Fetch a single task by id.

        Args:
            task_id (str): The task identifier.

        Returns:
            Task: The matching task.

        Raises:
            TaskNotFound: If no row matches ``task_id``.
        """
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
        """Transition a task to a new state, recording history.

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
                raise IllegalTransition(f"cannot transition {task_id!r} from {current_state!r} to {new_state!r}")
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
                "UPDATE tasks SET state=?, history=?, updated_at=? WHERE task_id=?",
                (new_state, json.dumps(history), now, task_id),
            )
        return await self.get(task_id)

    async def record_progress(
        self,
        task_id: str,
        note: dict[str, Any] | None = None,
    ) -> None:
        """Record that a running task made progress, without changing its state.

        A composite action — an explore grid, a baseline double-run, a profile
        and its analysis — is one task that internally completes many units of
        work over hours. Until it returns, its row looks identical to a task
        that hung at second one, which is why a healthy 80-minute analysis and
        a wedged Coordinator produce the same stall evidence. The note carries
        the difference: it lands on ``history`` with its own timestamp, and a
        consumer that wants freshness reads the notes.

        ``updated_at`` is deliberately left alone. It marks when the task
        entered ``running``, and the R6 lease watchdog, the ``extend_lease``
        remaining-budget math and the "Tasks in flight" projection all measure
        elapsed runtime from it; moving it would turn a cumulative budget into
        an inactivity timeout and make an 80-minute task render as seconds old.

        Only the newest :data:`_MAX_PROGRESS_NOTES` notes are retained. Each
        note costs a rewrite of the whole ``history`` blob, so an unbounded
        trail charges a session for its own length in exactly the runs this
        feature exists for; the notes are read newest-first, and transitions are
        kept whatever the bound.

        Best-effort: a task that vanished under a reaper must not take its
        executor down over a progress note.

        Args:
            task_id (str): The running task reporting progress.
            note (dict[str, Any] | None): Structured detail (unit name, index,
                outcome) recorded on the task's history.
        """
        async with self.db.transaction() as cur:
            cur.execute("SELECT history FROM tasks WHERE task_id=?", (task_id,))
            row = cur.fetchone()
            if row is None:
                return
            history = json.loads(row["history"])
            history.append({"progress": note or {}, "ts": _now_iso()})
            history = _drop_oldest_progress_notes(history, _MAX_PROGRESS_NOTES)
            cur.execute(
                "UPDATE tasks SET history=? WHERE task_id=?",
                (json.dumps(history), task_id),
            )

    async def queued(self) -> list[Task]:
        """Return all queued tasks ordered oldest-first.

        Returns:
            list[Task]: Queued tasks sorted by creation time.
        """
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state='queued' ORDER BY created_at ASC")
        return [Task.from_row(r) for r in rows]

    async def running(self) -> list[Task]:
        """Return all running tasks ordered least-recently-updated-first.

        Returns:
            list[Task]: Running tasks sorted by update time.
        """
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC")
        return [Task.from_row(r) for r in rows]

    async def extend_lease(self, task_id: str, extra_sec: int) -> int:
        """Grow a running task's ``lease_ttl_sec`` by ``extra_sec``.

        ``updated_at`` is left alone: it marks when the task started running,
        and both the TTL watchdog and the elapsed-time projections measure from
        it.

        Args:
            task_id: The running task to extend.
            extra_sec: Seconds to add; non-positive values are a no-op.

        Returns:
            The task's new ``lease_ttl_sec``.

        Raises:
            TaskNotFound: If no row matches ``task_id``.
            IllegalTransition: If the task is not ``running``.
        """
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
        rows = await self.db.fetchall("SELECT * FROM tasks WHERE state=? ORDER BY updated_at ASC", (state,))
        return [Task.from_row(r) for r in rows]

    async def reclaim_expired_running(
        self,
        *,
        now_unix: float | None = None,
        reason: str = "lease_expired",
    ) -> list[str]:
        """Fail running tasks whose execution lease (``lease_ttl_sec`` since
        ``updated_at``) has expired (R6 watchdog / cycle soft-restart cleanup).

        A ``running`` row that has not advanced for longer than its own
        ``lease_ttl_sec`` is orphaned — the worker died or was reaped — so we
        transition it ``running -> failed`` (retry-eligible, lanes freed). Tasks
        with ``lease_ttl_sec <= 0`` are left untouched (no lease to expire).

        Idempotent: a second call finds the rows already ``failed`` and is a
        no-op. Returns the reclaimed task_ids.

        Args:
            now_unix: Reference unix time for lease-age comparison; defaults to
                the current time.
            reason: Reason label recorded in the transition history evidence.

        Returns:
            The task ids whose expired running lease was reclaimed to
            ``failed`` (empty when none expired).
        """
        import time as _time

        now = float(now_unix if now_unix is not None else _time.time())
        reclaimed: list[str] = []
        async with self.db.transaction() as cur:
            cur.execute("SELECT task_id, lease_ttl_sec, updated_at, history FROM tasks WHERE state='running'")
            rows = [(r["task_id"], r["lease_ttl_sec"], r["updated_at"], r["history"]) for r in cur.fetchall()]
            now_iso = _now_iso()
            for task_id, ttl, updated_at, history_json in rows:
                try:
                    ttl_sec = float(ttl or 0)
                except (TypeError, ValueError):
                    ttl_sec = 0.0
                if ttl_sec <= 0:
                    continue
                try:
                    updated = datetime.fromisoformat(str(updated_at))
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    age = now - updated.timestamp()
                except (TypeError, ValueError):
                    continue
                if age < ttl_sec:
                    continue
                history = json.loads(history_json)
                history.append(
                    {
                        "from": "running",
                        "to": "failed",
                        "ts": now_iso,
                        "evidence": {
                            "reason": reason,
                            "age_sec": round(age, 1),
                            "lease_ttl_sec": ttl_sec,
                        },
                    }
                )
                cur.execute(
                    "UPDATE tasks SET state='failed', history=?, updated_at=? WHERE task_id=?",
                    (json.dumps(history), now_iso, task_id),
                )
                reclaimed.append(task_id)
        return reclaimed

    async def reclaim_dead_running(
        self,
        *,
        reason: str = "dead_holder",
    ) -> list[str]:
        """Fail running tasks whose lease-holder process is provably dead.

        Joins ``running`` tasks to the ``leases`` table on ``task_id`` and
        transitions ``running -> failed`` for any whose recorded ``pid`` is no
        longer alive (and not the current process). Tasks with no lease row or a
        null/non-positive pid are left untouched (cannot prove dead). Idempotent.

        Args:
            reason: Reason label recorded in the transition history evidence.

        Returns:
            The reclaimed task ids (empty when none had a dead holder).
        """
        import os as _os

        def _alive(pid: int) -> bool:
            if pid <= 0:
                return True
            try:
                _os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except OSError:
                return True
            return True

        self_pid = _os.getpid()
        reclaimed: list[str] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT t.task_id AS task_id, t.history AS history, "
                "MAX(l.pid) AS pid "
                "FROM tasks t JOIN leases l ON l.task_id = t.task_id "
                "WHERE t.state='running' GROUP BY t.task_id"
            )
            rows = [(r["task_id"], r["history"], r["pid"]) for r in cur.fetchall()]
            now_iso = _now_iso()
            for task_id, history_json, pid_raw in rows:
                try:
                    pid = int(pid_raw) if pid_raw is not None else 0
                except (TypeError, ValueError):
                    pid = 0
                if pid <= 0 or pid == self_pid or _alive(pid):
                    continue
                history = json.loads(history_json)
                history.append(
                    {
                        "from": "running",
                        "to": "failed",
                        "ts": now_iso,
                        "evidence": {"reason": reason, "dead_pid": pid},
                    }
                )
                cur.execute(
                    "UPDATE tasks SET state='failed', history=?, updated_at=? WHERE task_id=?",
                    (json.dumps(history), now_iso, task_id),
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
        """Bulk-cancel queued tasks of the given kinds; returns cancelled task_ids.

        Args:
            family_kinds: Task kinds whose queued tasks should be cancelled.
            reason: Stamped onto each cancellation's history evidence.
            exclude_task_ids: Task ids to leave queued.

        Returns:
            The task ids that were cancelled (empty when none matched).
        """
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
            now = _now_iso()
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

    async def cancel_queued_not_allowed(
        self,
        *,
        allowed_kinds: set[str] | frozenset[str],
        reason: str,
    ) -> list[str]:
        """Bulk-cancel queued tasks whose kind is not allowed at a phase boundary."""
        allowed = {str(kind or "").strip() for kind in allowed_kinds if str(kind or "").strip()}
        cancelled: list[str] = []
        async with self.db.transaction() as cur:
            cur.execute("SELECT task_id, kind, history FROM tasks WHERE state='queued'")
            rows = [(r["task_id"], r["kind"], r["history"]) for r in cur.fetchall()]
            now = _now_iso()
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
]
