# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repair first, admit second: the tick's opening act, run with nothing in its way."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hyperloom.common.timeutil import now_iso

from ..bus.resource_lock import BRINGUP_ROUND_LANE, ResourceLockManager
from ..state.round_store import (
    EXPIRED_REAPED,
    Round,
    RoundStore,
)
from ..state.task_registry import TERMINAL_STATES, Task, TaskNotFound, TaskRegistry

log = logging.getLogger(__name__)

#: How long a round may stay open after its holder went terminal with nothing
#: following it. A cap rather than an instant, because the successor is created
#: by the tick after the one that finished the holder.
TERMINAL_HOLDER_CAP_SEC: float = 300.0

#: How long a proposal may sit undecided before the coordinator denies it.
REVIEW_TTL_SEC: float = 1800.0

#: The verdict a timeout writes.
TIMEOUT_VERDICT = "reject"

#: Evidence key ``TaskRegistry.reclaim_dead_running`` writes when it proved a pid
#: dead, and the key its lease watchdog writes when it only timed a lease out.
_EVIDENCE_DEAD_PID = "dead_pid"
_EVIDENCE_LEASE_TTL = "lease_ttl_sec"

#: Proposals with no verdict against them. The inner SELECT must exclude NULL and
#: empty targets: ``msg_id NOT IN (.., NULL)`` is NULL for every row, so one
#: malformed verdict would make every proposal look decided.
_UNDECIDED_PROPOSALS_SQL = """
    SELECT msg_id, from_agent, ts FROM events
    WHERE topic = 'proposal'
      AND msg_id NOT IN (
        SELECT json_extract(payload, '$.target_proposal_msg_id')
        FROM events
        WHERE topic = 'review_verdict'
          AND json_extract(payload, '$.target_proposal_msg_id') IS NOT NULL
          AND json_extract(payload, '$.target_proposal_msg_id') != ''
      )
    ORDER BY seq ASC
"""

#: The timeout deny, written only if no verdict targets the proposal at the
#: instant of the write. The guard is in the statement rather than in a read
#: before it, so a verdict arriving in between wins.
_TIMEOUT_DENY_SQL = """
    INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts)
    SELECT ?, ?, ?, 'review_verdict', NULL, ?, ?
    WHERE NOT EXISTS (
        SELECT 1 FROM events
        WHERE topic = 'review_verdict'
          AND json_extract(payload, '$.target_proposal_msg_id') = ?
    )
"""

__all__ = [
    "TIMEOUT_VERDICT",
    "ReconcileReport",
    "Reconciler",
]


@dataclass
class ReconcileReport:
    """What one pass did, for the log and for tests.

    Attributes:
        leases_reaped: Lease rows the pass swept.
        leases_unverifiable: Lane rows still held after the sweep by a holder
            that ended without confirming its cleanup. Nothing decides these:
            no probe here can tell a lane in use from one merely abandoned, so
            they are retained by contract rather than left over by accident.
            Not a failure count -- but the number an operator needs to see
            climb, and the maintenance summary carries it out of here.
        settled: ``(round_id, outcome)`` for every round this pass ended.
        handed_off: Rounds moved onto the successor that owes their result.
        failed_tasks: Task ids marked failed on proof their process is gone.
        denied_reviews: Proposal ids denied on the review TTL.
        closed_windows: Revalidation task ids whose window the pass closed.
        failures: Rules that raised, by name.
    """

    leases_reaped: int = 0
    leases_unverifiable: int = 0
    settled: list[tuple[str, str]] = field(default_factory=list)
    handed_off: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    denied_reviews: list[str] = field(default_factory=list)
    closed_windows: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        """bool: Whether the pass changed anything worth logging."""
        return bool(self.settled or self.handed_off or self.failed_tasks or self.denied_reviews or self.closed_windows)


class Reconciler:
    """The unconditional repair pass, and the projection rebuild that ends it.

    Attributes:
        terminal_holder_cap_sec (float): How long a round waits after its
            holder went terminal with no successor.
        review_ttl_sec (float): How long a proposal may sit undecided.
        last_report (ReconcileReport): What the most recent pass did; read by
            the maintenance tick, which no longer sweeps leases itself.
    """

    def __init__(
        self,
        *,
        rounds: RoundStore,
        tasks: TaskRegistry,
        locks: ResourceLockManager,
        shared_state: Any,
        resources: Any = None,
        proposals: Callable[[], Mapping[str, Any]] | None = None,
        session_dir: Any = None,
        terminal_holder_cap_sec: float = TERMINAL_HOLDER_CAP_SEC,
        review_ttl_sec: float = REVIEW_TTL_SEC,
    ):
        """Initialise the reconciler.

        Args:
            rounds: The durable round store.
            tasks: The task registry.
            locks: The lease manager; this pass is its only sweeper.
            shared_state: SharedState, read for the projection and written when
                an unreaped round stops the session.
            resources: The gate's resource facts; ``None`` skips the update.
            proposals: Returns the in-memory pending proposals, so a durable
                timeout deny also reaches the copy the loop consults.
            session_dir: Where SharedState is saved after a terminal is set.
            terminal_holder_cap_sec: Cap for a round whose holder went terminal.
            review_ttl_sec: TTL for an undecided review.
        """
        self._rounds = rounds
        # One connection serves the whole session; a second would have to be
        # arbitrated against this one by the database's busy timeout.
        self._db = rounds.db
        self._tasks = tasks
        self._locks = locks
        self._shared_state = shared_state
        self._resources = resources
        self._proposals = proposals
        self._session_dir = session_dir
        self.terminal_holder_cap_sec = max(0.0, float(terminal_holder_cap_sec))
        self.review_ttl_sec = max(0.0, float(review_ttl_sec))
        self.last_report = ReconcileReport()

    async def run(self, now_unix: float) -> ReconcileReport:
        """Run every rule in order, the projection rebuild last.

        Args:
            now_unix: Current wall time.

        Returns:
            ReconcileReport: What the pass did.
        """
        report = ReconcileReport()
        for rule in (
            self._fail_dead_tasks,
            self._reap_leases,
            self._deny_timed_out_reviews,
            self._resolve_open_rounds,
            self._close_stale_validation_window,
            self._rebuild_projection,
        ):
            try:
                await rule(now_unix, report)
            except Exception:  # noqa: BLE001 — independent repairs; one failing must not skip the rest
                log.exception("reconcile: rule %s raised", rule.__name__)
                report.failures.append(rule.__name__)
        self.last_report = report
        if report.acted:
            log.info(
                "RECONCILE: settled=%s handed_off=%s failed_tasks=%d denied_reviews=%d leases_reaped=%d",
                report.settled,
                report.handed_off,
                len(report.failed_tasks),
                len(report.denied_reviews),
                report.leases_reaped,
            )
        return report

    async def _fail_dead_tasks(self, now_unix: float, report: ReconcileReport) -> None:
        """Fail every running task whose process is provably gone."""
        report.failed_tasks.extend(await self._tasks.reclaim_dead_running(reason="reconciler_dead_holder"))

    async def _deny_timed_out_reviews(self, now_unix: float, report: ReconcileReport) -> None:
        """Deny every proposal that has waited longer than the review TTL."""
        rows = await self._db.fetchall(_UNDECIDED_PROPOSALS_SQL, ())
        for row in rows:
            age = float(now_unix) - _unix_of(row["ts"])
            if age < self.review_ttl_sec:
                continue
            msg_id = str(row["msg_id"])
            if await self._author_timeout_deny(msg_id, str(row["from_agent"]), age=age, now_unix=now_unix):
                report.denied_reviews.append(msg_id)
            # Marked either way: a verdict that beat this write to the log is
            # still one the copy the loop reads has to carry.
            self._mark_decided(msg_id)

    async def _author_timeout_deny(self, msg_id: str, to_agent: str, *, age: float, now_unix: float) -> bool:
        """Write the coordinator's timeout deny, unless a verdict beat it there.

        Returns:
            bool: Whether this call wrote the verdict.
        """
        payload = json.dumps(
            {
                "target_proposal_msg_id": msg_id,
                "verdict": TIMEOUT_VERDICT,
                "reasoning": (
                    f"no review verdict arrived within {self.review_ttl_sec:.0f}s "
                    f"(waited {age:.0f}s); denied by the coordinator so the round "
                    "it holds can end. A patch nobody reviewed is not an accepted layer."
                ),
                "authored_by": "coordinator_review_timeout",
            },
            sort_keys=True,
        )
        async with self._db.transaction() as cur:
            cur.execute(
                _TIMEOUT_DENY_SQL,
                (uuid.uuid4().hex, "coordinator", to_agent, payload, now_iso(), msg_id),
            )
            applied = cur.rowcount == 1
        if applied:
            log.warning("RECONCILE: review timeout denied proposal %s after %.0fs", msg_id, age)
        return applied

    def _mark_decided(self, msg_id: str) -> None:
        """Record the deny on the in-memory proposal the loop consults."""
        if self._proposals is None:
            return
        pending = self._proposals().get(msg_id)
        if pending is None:
            return
        pending.decided = True
        pending.verdict = TIMEOUT_VERDICT

    async def _resolve_open_rounds(self, now_unix: float, report: ReconcileReport) -> None:
        """Advance completed owners without timing out active ownership."""
        for round_row in await self._rounds.open_rounds():
            await self._advance_or_expire(round_row, now_unix, report)

    async def _close_stale_validation_window(self, now_unix: float, report: ReconcileReport) -> None:
        """Close a revalidation window whose task will never report."""
        state = self._shared_state
        if state is None or not bool(state.enablement.validation_pending):
            return
        tracked = str(state.enablement.revalidation_task_id or "").strip()
        if not tracked:
            return
        row = await self._task(tracked)
        if row is not None and row.state not in TERMINAL_STATES:
            return
        state.enablement.validation_pending = False
        state.enablement.revalidation_task_id = ""
        self._save_state()
        report.closed_windows.append(tracked)
        log.info("RECONCILE: closed revalidation window held by terminal task %s", tracked)

    async def _reap_leases(self, now_unix: float, report: ReconcileReport) -> None:
        """Release confirmed-dead local owners, then lanes nothing is using.

        Liveness alone cannot refute a coordinator-side holder that dropped its
        work without releasing -- the pid on the lane row is this process.
        2026-09-21: six lanes were held that way for two hours, starving 19
        queued tasks, by holders that had already ended and whose processes had
        gone with them. The same thing would happen under this code, and is
        meant to: nothing here decides that a lane is free. What changed is that
        it announces itself within a tick, naming the lane and the statement
        that clears it, instead of taking an hour of py-spy and sqlite to find.

        A holder that ended with its cleanup unconfirmed keeps its lane, and
        nothing here takes it back. Seven rounds of review each proposed a
        cheaper proof that the lane was free -- the holder is terminal, its
        recorded process group is empty, no pidfile names a live server -- and
        each was shown by probe to be a proxy a real process can slip out of. A
        served process is setsid'd by design, its pidfile appears only after it
        answers, and a cmdline is a guess. Releasing a lane wrongly puts two
        rounds on the same cards, which corrupts quietly; holding one wrongly
        stalls a queue until an operator spends 90 seconds. The asymmetry
        decides it.

        So this pass reclaims only what liveness alone settles
        (:meth:`reap_dead_holders`), and everything else is counted and handed
        to an operator by :meth:`diagnose_unverifiable_holders`, once per
        ``(lane, holder)``, with the statement that releases it. Closing that
        gap for real needs an identity a descendant cannot escape -- a
        per-execution cgroup -- which is its own project.
        """
        report.leases_reaped = len(await self._locks.reap_dead_holders())
        # After the sweep, so a row it just took back is not also reported stuck.
        report.leases_unverifiable = len(await self._locks.diagnose_unverifiable_holders())

    async def _advance_or_expire(self, round_row: Round, now_unix: float, report: ReconcileReport) -> None:
        """Move a terminal-holder round forward, or end it once its cap passes."""
        holder = await self._task(round_row.holder_task_id)
        if holder is None or not _terminal_by_observation(holder):
            return
        if await self._holder_has_resources(holder.task_id):
            return
        successor = await self._successor(round_row.holder_task_id)
        if successor is not None:
            moved = await self._rounds.handoff(
                round_row.round_id,
                holder_task_id=round_row.holder_task_id,
                fence=round_row.fence,
                new_holder_task_id=successor.task_id,
                # A successor that declares no TTL inherits what the round has left.
                lease_sec=float(successor.lease_ttl_sec) or (round_row.expires_unix - round_row.renewed_unix),
                now_unix=now_unix,
                request_id=f"reconcile:handoff:{successor.task_id}",
                evidence={"reason": "holder_terminal_with_successor"},
            )
            if moved.ok:
                report.handed_off.append(round_row.round_id)
            return
        if self._review_owes_a_verdict(round_row.holder_task_id):
            return
        if float(now_unix) - _unix_of(holder.updated_at) < self.terminal_holder_cap_sec:
            return
        await self._expire(round_row, now_unix, report, why="holder_terminal_without_result")

    async def _expire(self, round_row: Round, now_unix: float, report: ReconcileReport, *, why: str) -> None:
        """Settle a completed owner after its successor/review window ends."""
        outcome = EXPIRED_REAPED
        result = await self._rounds.settle(
            round_row.round_id,
            holder_task_id=round_row.holder_task_id,
            fence=round_row.fence,
            outcome=outcome,
            now_unix=now_unix,
            request_id=f"reconcile:{why}:{round_row.round_id}:{round_row.fence}",
            evidence={"reason": why, "cleanup": "holder_terminal_without_resources"},
        )
        if not result.ok:
            return
        report.settled.append((round_row.round_id, outcome))

    async def _holder_has_resources(self, task_id: str) -> bool:
        """Keep round ownership while execution or GPU cleanup still owns rows."""
        row = await self._db.fetchone(
            "SELECT 1 FROM leases WHERE task_id=? AND lane != ? "
            "UNION ALL SELECT 1 FROM gpu_leases WHERE task_id=? LIMIT 1",
            (task_id, BRINGUP_ROUND_LANE, task_id),
        )
        return row is not None

    def _save_state(self) -> None:
        """Persist SharedState after a terminal, if there is somewhere to put it."""
        if self._session_dir is None or self._shared_state is None:
            return
        self._shared_state.save(self._session_dir)

    async def _rebuild_projection(self, now_unix: float, report: ReconcileReport) -> None:
        """Re-read the facts the gate's resource rules judge against."""
        if self._resources is None:
            return
        self._resources.update(
            self._shared_state,
            rounds=await self._rounds.excluding(now_unix),
            live_task_ids=[t.task_id for t in await self._tasks.running()],
        )

    async def _task(self, task_id: str) -> Task | None:
        """Read one task, or ``None`` when the registry has no such row."""
        try:
            return await self._tasks.get(task_id)
        except TaskNotFound:
            return None

    async def _successor(self, holder_task_id: str) -> Task | None:
        """Return the queued or running task created from the holder's deliverable."""
        for task in (await self._tasks.queued()) + (await self._tasks.running()):
            if str(task.params.get("specialist_task_id", "")) == holder_task_id:
                return task
        return None

    def _review_owes_a_verdict(self, holder_task_id: str) -> bool:
        """Whether an undecided proposal still stands on the holder's work."""
        if self._proposals is None:
            return False
        for pending in self._proposals().values():
            if pending.decided:
                continue
            params = pending.payload.get("params", {})
            if str(params.get("specialist_task_id", "")) == holder_task_id:
                return True
        return False


def _terminal_by_observation(task: Task) -> bool:
    """Whether the task's terminal transition was written because work ended."""
    if task.state not in TERMINAL_STATES:
        return False
    for entry in reversed(task.history):
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("outcome"), dict):
            if isinstance(evidence.get("cleanup_confirmed"), bool):
                return evidence["cleanup_confirmed"]
        if entry.get("to") != task.state:
            continue
        if not isinstance(evidence, dict):
            return True
        if _EVIDENCE_DEAD_PID in evidence:
            return True
        return _EVIDENCE_LEASE_TTL not in evidence and evidence.get("reason") != "cancelled_in_flight"
    return True


def _unix_of(stamp: str) -> float:
    """Read an ISO-8601 timestamp from a task or event row as unix seconds."""
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
