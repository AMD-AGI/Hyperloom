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

from ..bus.db_maintenance import DECIDED_PROPOSAL_IDS_SQL
from ..bus.resource_lock import BRINGUP_ROUND_LANE, SqliteLeaseBackend
from ..state.round_store import (
    EXPIRED_REAPED,
    Round,
    RoundRequest,
    RoundStore,
)
from ..state.task_registry import TERMINAL_STATES, Task, TaskNotFound, TaskRegistry
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_PROPOSAL,
    STATUS_CANCELLED,
    EventSpan,
    record_event,
)
from hyperloom.inference_optimizer.breakdown.recorder import enablement_event

log = logging.getLogger(__name__)

#: How long a round may stay open after its holder went terminal with nothing
#: following it. A cap rather than an instant, because the successor is created
#: by the tick after the one that finished the holder.
TERMINAL_HOLDER_CAP_SEC: float = 300.0

#: How long a proposal may sit undecided before the coordinator denies it.
REVIEW_TTL_SEC: float = 1800.0

#: The verdict a timeout writes.
TIMEOUT_VERDICT = "reject"

#: Proposals with no verdict against them.
_UNDECIDED_PROPOSALS_SQL = f"""
    SELECT msg_id, from_agent, ts FROM events
    WHERE topic = 'proposal'
      AND msg_id NOT IN ({DECIDED_PROPOSAL_IDS_SQL})
    ORDER BY seq ASC
"""  # nosec B608 - interpolates a constant SQL fragment only.

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
    "ReconcileLinks",
    "ReconcileReport",
    "ReconcileStores",
    "Reconciler",
]


@dataclass(frozen=True)
class ReconcileStores:
    """The durable stores the pass repairs.

    Attributes:
        rounds: The durable round store.
        tasks: The task registry.
        locks: The lane lease store; this pass is its only sweeper.
    """

    rounds: RoundStore
    tasks: TaskRegistry
    locks: SqliteLeaseBackend


@dataclass(frozen=True)
class ReconcileLinks:
    """The running loop's state the pass reads and writes.

    Attributes:
        resources: The gate's resource facts.
        proposals: Returns the in-memory pending proposals, so a durable
            timeout deny also reaches the copy the loop consults.
        drop_proposal: Settles a timed-out pending proposal as dropped on the
            framework timeline, as a Critic reject would.
        session_dir: Where SharedState is saved after a terminal is set.
        result_pending: Whether a terminal task's result is still to be
            booked by the dispatcher.
    """

    resources: Any
    proposals: Callable[[], Mapping[str, Any]]
    drop_proposal: Callable[[Any, str], None]
    session_dir: Any
    result_pending: Callable[[str], bool]


@dataclass
class ReconcileReport:
    """What one pass did, for the log and for tests.

    Attributes:
        leases_reaped: Lease rows the pass swept.
        leases_unverifiable: Lane rows still held after the sweep by a holder
            that ended. A holder releases its lanes where it writes its
            terminal row, so each row here is one that release never reached:
            the coordinator died in between, or the task was failed from the
            outside. Nothing decides these: no probe here can tell a lane in
            use from one merely abandoned. Not a failure count -- but the
            number an operator needs to see climb, and the maintenance summary
            carries it out of here.
        settled: ``(round_id, outcome)`` for every round this pass ended.
        handed_off: Rounds moved onto the successor that owes their result.
        failed_tasks: Task ids marked failed on proof their process is gone.
        closed_windows: Revalidation task ids whose window the pass closed.
        failures: Rules that raised, by name.
    """

    leases_reaped: int = 0
    leases_unverifiable: int = 0
    settled: list[tuple[str, str]] = field(default_factory=list)
    handed_off: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    closed_windows: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        """bool: Whether the pass changed a round, task or window; a review deny logs itself."""
        return bool(self.settled or self.handed_off or self.failed_tasks or self.closed_windows)


class Reconciler:
    """The unconditional repair pass, and the projection rebuild that ends it.

    Attributes:
        last_report (ReconcileReport): What the most recent pass did; read by
            the maintenance tick, which leaves lease sweeping to this pass.
    """

    def __init__(
        self,
        stores: ReconcileStores,
        shared_state: Any,
        *,
        links: ReconcileLinks,
    ):
        """Initialise the reconciler.

        Args:
            stores: The durable stores the pass repairs.
            shared_state: SharedState, read for the projection and written when
                an unreaped round stops the session.
            links: The running loop's state the pass reads and writes.
        """
        self._rounds = stores.rounds
        # One connection serves the whole session; a second would have to be
        # arbitrated against this one by the database's busy timeout.
        self._db = stores.rounds.db
        self._tasks = stores.tasks
        self._locks = stores.locks
        self._shared_state = shared_state
        self._resources = links.resources
        self._proposals = links.proposals
        self._drop_proposal = links.drop_proposal
        self._session_dir = links.session_dir
        self._result_pending = links.result_pending
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
            except Exception:  # isolation boundary: one reconcile rule of the pass
                log.exception("reconcile: rule %s raised", rule.__name__)
                report.failures.append(rule.__name__)
        self.last_report = report
        if report.acted:
            log.info(
                "RECONCILE: settled=%s handed_off=%s failed_tasks=%d leases_reaped=%d",
                report.settled,
                report.handed_off,
                len(report.failed_tasks),
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
            if age < REVIEW_TTL_SEC:
                continue
            await self._author_timeout_deny(str(row["msg_id"]), str(row["from_agent"]), age=age, now_unix=now_unix)

    async def _author_timeout_deny(self, msg_id: str, to_agent: str, *, age: float, now_unix: float) -> None:
        """Write the coordinator's timeout deny, unless a verdict beat it there."""
        payload = json.dumps(
            {
                "target_proposal_msg_id": msg_id,
                "verdict": TIMEOUT_VERDICT,
                "reasoning": (
                    f"no review verdict arrived within {REVIEW_TTL_SEC:.0f}s "
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
        if not applied:
            return
        log.warning("RECONCILE: review timeout denied proposal %s after %.0fs", msg_id, age)
        pending = self._proposals().pop(msg_id, None)
        self._record_timeout_terminal(msg_id, pending, age=age)
        if pending is not None:
            self._drop_proposal(pending, "review_timeout")

    def _record_timeout_terminal(self, msg_id: str, pending: Any, *, age: float) -> None:
        """Close the proposal's trajectory span: a timeout deny never passes through the verdict handler."""
        action_name = str(getattr(pending, "action_name", "") or "") or None
        record_event(
            EVENT_PROPOSAL,
            span=EventSpan(span_id=msg_id, status=STATUS_CANCELLED),
            attributes={
                "name": action_name,
                "action_name": action_name,
                "verdict": TIMEOUT_VERDICT,
                "reason": "review_timeout",
                "waited_sec": round(age, 1),
            },
        )

    async def _resolve_open_rounds(self, now_unix: float, report: ReconcileReport) -> None:
        """Advance completed owners without timing out active ownership."""
        for round_row in await self._rounds.open_rounds():
            await self._advance_or_expire(round_row, now_unix, report)

    async def _close_stale_validation_window(self, now_unix: float, report: ReconcileReport) -> None:
        """Close a revalidation window whose task will never report.

        A terminal row is not yet a task that will never report: the row goes
        terminal inside the execution, and the result is booked by the pump that
        runs after this pass. Closing the window in between leaves the writeback
        a promoted revalidation with no window to credit it to, so the lane never
        succeeds although the baseline it was waiting for passed.
        """
        state = self._shared_state
        if not bool(state.enablement.validation_pending):
            return
        tracked = str(state.enablement.revalidation_task_id or "").strip()
        if not tracked:
            return
        row = await self._task(tracked)
        if row is not None and row.state not in TERMINAL_STATES:
            return
        if self._result_pending(tracked):
            return
        enablement_event.record_revalidation_cleared(
            generation=int(state.enablement.revalidation_generation or 0), task_id=tracked
        )
        state.enablement.validation_pending = False
        state.enablement.revalidation_task_id = ""
        state.save(self._session_dir)
        report.closed_windows.append(tracked)
        log.info("RECONCILE: closed revalidation window held by terminal task %s", tracked)

    async def _reap_leases(self, now_unix: float, report: ReconcileReport) -> None:
        """Release confirmed-dead local owners, then lanes nothing is using.

        Liveness alone cannot refute a coordinator-side holder that dropped its
        work without releasing -- the pid on the lane row is this process.
        2026-09-21: six lanes were held that way for two hours, starving 19
        queued tasks, by holders that had already ended and whose processes had
        gone with them. Nothing here would decide those lanes were free either;
        what keeps them from getting this far is that a holder now releases its
        lanes where it writes its terminal row. Whatever still arrives announces
        itself within a tick, naming the lane and the statement that clears it,
        instead of taking an hour of py-spy and sqlite to find.

        What arrives is the residue: a lane whose release never ran at all, the
        coordinator having died between the row and the release, or the task
        having been failed from the outside. Nothing here takes those back.
        Every cheaper proof that such a lane is free -- the holder is terminal,
        its recorded process group is empty, no pidfile names a live server --
        was shown by probe to be a proxy a real process can slip out of. A
        served process is setsid'd by design, its pidfile appears only after it
        answers, and a cmdline is a guess. Releasing a lane wrongly puts two
        rounds on the same cards, which corrupts quietly, so these are counted
        and handed to an operator instead.

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
                round_row,
                RoundRequest(
                    now_unix=now_unix,
                    request_id=f"reconcile:handoff:{successor.task_id}",
                    evidence={"reason": "holder_terminal_with_successor"},
                ),
                new_holder_task_id=successor.task_id,
                # A successor that declares no TTL inherits what the round has left.
                lease_sec=float(successor.lease_ttl_sec) or (round_row.expires_unix - round_row.renewed_unix),
            )
            if moved.ok:
                report.handed_off.append(round_row.round_id)
            return
        if self._review_owes_a_verdict(round_row.holder_task_id):
            return
        if float(now_unix) - _unix_of(holder.updated_at) < TERMINAL_HOLDER_CAP_SEC:
            return
        await self._expire(round_row, now_unix, report, why="holder_terminal_without_result")

    async def _expire(self, round_row: Round, now_unix: float, report: ReconcileReport, *, why: str) -> None:
        """Settle a completed owner after its successor/review window ends."""
        outcome = EXPIRED_REAPED
        result = await self._rounds.settle(
            round_row,
            RoundRequest(
                now_unix=now_unix,
                request_id=f"reconcile:{why}:{round_row.round_id}:{round_row.fence}",
                evidence={"reason": why, "cleanup": "holder_terminal_without_resources"},
            ),
            outcome=outcome,
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

    async def _rebuild_projection(self, now_unix: float, report: ReconcileReport) -> None:
        """Re-read the facts the gate's resource rules judge against."""
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
        for pending in self._proposals().values():
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
        return evidence.get("reason") != "cancelled_in_flight"
    return True


def _unix_of(stamp: str) -> float:
    """Read an ISO-8601 timestamp from a task or event row as unix seconds."""
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
