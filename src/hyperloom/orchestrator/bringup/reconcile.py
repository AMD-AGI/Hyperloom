# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repair first, admit second: the tick's opening act, run with nothing in its way.

Runs at the top of every tick with no condition on phase, budget, mode or state,
because the state it repairs is exactly what stops the dispatcher from
dispatching. Each rule is an independent repair -- an expired round, a terminal
holder, a task whose process is provably gone, an undecided review past its TTL,
a settle the store rejected -- and the resource facts the gate reads are
re-read from whatever they leave. A round ended here is also charged to the
round ledger, so a round that died without reporting still costs the session one.

This pass is also the loop's only sweep of the lease table. An open round holds
a lane row for as long as it is open, so a round's expiry and a lease's expiry
are the same fact, and the sweep has to run where the round can be settled in
the same pass.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hyperloom.common.timeutil import now_iso

from ..bus.resource_lock import ResourceLockManager
from .budget import STALLED_STOP_REASON, session_budget
from .reap import (
    CLAIM_REACHABLE,
    REAP_HOLDER_ALIVE,
    REAP_HOLDER_REPORTED,
    REAP_UNOBSERVABLE,
    Reap,
    ReapBackend,
    holder_target,
    select_reaper,
)
from ..state.round_store import (
    EXPIRED_REAPED,
    EXPIRED_UNREAPED,
    EXPIRY_OUTCOMES,
    OPEN,
    UNREAPED_STOP_REASON,
    Round,
    RoundStore,
)
from ..state.task_registry import TERMINAL_STATES, Task, TaskNotFound, TaskRegistry
from ..supervisor import store as supervisor_store

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
    INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, priority, ts)
    SELECT ?, ?, ?, 'review_verdict', NULL, ?, 0, ?
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
        settled: ``(round_id, outcome)`` for every round this pass ended.
        handed_off: Rounds moved onto the successor that owes their result.
        redriven: Rounds whose rejected settle was re-driven.
        failed_tasks: Task ids marked failed on proof their process is gone.
        denied_reviews: Proposal ids denied on the review TTL.
        closed_windows: Revalidation task ids whose window the pass closed.
        failures: Rules that raised, by name.
    """

    leases_reaped: int = 0
    settled: list[tuple[str, str]] = field(default_factory=list)
    handed_off: list[str] = field(default_factory=list)
    redriven: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    denied_reviews: list[str] = field(default_factory=list)
    closed_windows: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        """bool: Whether the pass changed anything worth logging."""
        return bool(
            self.settled
            or self.handed_off
            or self.redriven
            or self.failed_tasks
            or self.denied_reviews
            or self.closed_windows
        )


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
        reaper: Any = None,
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
            reaper: The reap unit; defaults to :func:`.reap.select_reaper`.
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
        self._reaper: ReapBackend = reaper if reaper is not None else select_reaper()
        self.terminal_holder_cap_sec = max(0.0, float(terminal_holder_cap_sec))
        self.review_ttl_sec = max(0.0, float(review_ttl_sec))
        self.last_report = ReconcileReport()

    async def run(self, now_unix: float) -> ReconcileReport:
        """Run every rule in order, the projection rebuild last.

        A rule that raises is recorded on the report and does not stop the ones
        after it.

        Args:
            now_unix: Current wall time.

        Returns:
            ReconcileReport: What the pass did.
        """
        report = ReconcileReport()
        for rule in (
            self._stamp_tick,
            self._fail_dead_tasks,
            self._deny_timed_out_reviews,
            self._redrive_rejected_settles,
            self._resolve_open_rounds,
            self._close_stale_validation_window,
            self._reap_leases,
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
                "RECONCILE: settled=%s handed_off=%s redriven=%s failed_tasks=%d denied_reviews=%d leases_reaped=%d",
                report.settled,
                report.handed_off,
                report.redriven,
                len(report.failed_tasks),
                len(report.denied_reviews),
                report.leases_reaped,
            )
        return report

    async def _stamp_tick(self, now_unix: float, report: ReconcileReport) -> None:
        """Record that a tick has started, for the process watching from outside.

        Runs before any rule that can block, so the supervisor can tell a tick
        that started and is still going from one that finished. It is the only
        thing the supervisor reads: a coordinator that has stopped ticking is
        past reading anything this pass could have been left.
        """
        if self._session_dir is None:
            return
        supervisor_store.stamp_tick(
            self._session_dir,
            tick=int(getattr(self._shared_state, "tick", 0)),
            now_unix=now_unix,
        )

    async def _fail_dead_tasks(self, now_unix: float, report: ReconcileReport) -> None:
        """Fail every running task whose process is provably gone.

        A row with no recorded pid, or one whose pid still answers, is left
        running: inability to observe is not terminality.
        """
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

    async def _redrive_rejected_settles(self, now_unix: float, report: ReconcileReport) -> None:
        """Re-drive settles the store refused while the round is still open.

        Expiry outcomes are skipped: replaying one would date a reap that never
        happened, so :meth:`_expire` re-derives them from a fresh reap instead.
        """
        for event in await self._rounds.redrivable_settles():
            if event.outcome in EXPIRY_OUTCOMES or not event.outcome:
                continue
            round_row = await self._rounds.get(event.round_id)
            if round_row is None or round_row.state != OPEN:
                continue
            result = await self._rounds.settle(
                round_row.round_id,
                holder_task_id=round_row.holder_task_id,
                fence=round_row.fence,
                outcome=event.outcome,
                now_unix=now_unix,
                request_id=event.request_id,
                evidence={**event.evidence, "redriven_from_event": event.event_id},
            )
            if result.ok:
                report.redriven.append(round_row.round_id)

    async def _resolve_open_rounds(self, now_unix: float, report: ReconcileReport) -> None:
        """Expire, advance or leave each open round, oldest first.

        A round has run out when it no longer holds its lane. The lane row is
        the round's only clock, so whichever pass swept it -- and whether the
        row is already deleted or merely past its TTL -- the answer is the same.
        """
        holding = await self._locks.bringup_round_holders(now_unix)
        for round_row in await self._rounds.open_rounds():
            if round_row.round_id not in holding:
                await self._expire(round_row, now_unix, report, why="lease_expired")
                continue
            await self._advance_or_expire(round_row, now_unix, report)

    async def _close_stale_validation_window(self, now_unix: float, report: ReconcileReport) -> None:
        """Close a revalidation window whose task will never report.

        ``validation_pending`` holds ``enablement_close_guard_active()`` true in
        every phase, and that guard drops ``skip_to_close``. A revalidation task
        that ends without a result leaves the flag set, so the session keeps the
        one exit an unpromotable run has left. Only a terminal tracked task
        closes the window; an unobservable one is left alone.
        """
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
        """Sweep every lease past its TTL, this loop's only sweep of the table.

        An open round's lane row is one of these leases, which is why the sweep
        belongs to this pass rather than to the maintenance tick. It runs after
        the rounds are resolved because :func:`.reap.holder_target` reads a
        holder's processes out of the same rows: deleting them first would throw
        away the evidence an expiry outcome is decided on.
        """
        report.leases_reaped = len(await self._locks.reap_expired())

    async def _advance_or_expire(self, round_row: Round, now_unix: float, report: ReconcileReport) -> None:
        """Move a terminal-holder round forward, or end it once its cap passes.

        A terminal task never settles the round by itself: the round covers the
        specialist and the integrate that consumes its deliverable.
        """
        holder = await self._task(round_row.holder_task_id)
        if holder is None or holder.state not in TERMINAL_STATES:
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
        """End a round, reaped on proof and unreaped without it.

        ``why`` names the rule that ended it and is recorded on the outbox row.
        """
        reap = await self._confirm_gone(round_row.holder_task_id, now_unix)
        outcome = EXPIRED_REAPED if reap.confirmed_unix is not None else EXPIRED_UNREAPED
        result = await self._rounds.settle(
            round_row.round_id,
            holder_task_id=round_row.holder_task_id,
            fence=round_row.fence,
            outcome=outcome,
            now_unix=now_unix,
            request_id=f"reconcile:{why}:{round_row.round_id}:{round_row.fence}",
            kill_confirmed_unix=reap.confirmed_unix,
            reap_backend=reap.backend,
            evidence={"reason": why, "reap": reap.outcome, "claim": reap.claim},
        )
        if not result.ok:
            return
        report.settled.append((round_row.round_id, outcome))
        await self._charge_the_round(round_row, now_unix, why=why)
        if outcome == EXPIRED_UNREAPED:
            self._stop_loudly(round_row, reap)

    async def _charge_the_round(self, round_row: Round, now_unix: float, *, why: str) -> None:
        """Charge the ledger for a round that ended without reporting anything.

        Charged at stage zero with no digest, which spends an evidence-stall
        credit, and stops the session once the budget is spent.
        """
        await self._rounds.observe(
            round_row.round_id,
            actor_task_id=round_row.holder_task_id,
            stage=0,
            failure_digest="",
            now_unix=now_unix,
            request_id=f"reconcile:observe:{round_row.round_id}:{round_row.fence}",
            evidence={"status": "expired", "reason": why},
        )
        budget = await session_budget(self._rounds)
        state = self._shared_state
        if not budget.exhausted or state is None or state.stop_reason:
            return
        log.warning("RECONCILE: progress budget spent -- %s", budget.reason)
        state.set_stop_reason(STALLED_STOP_REASON)
        self._save_state()

    async def _confirm_gone(self, holder_task_id: str, now_unix: float) -> Reap:
        """Establish whether the holder is gone, and say so only if it is.

        Two things can establish it: the reap unit reporting every process it
        addressed gone, or -- when nothing was recorded -- the holder's own task
        row carrying a transition written because its work ended.

        Returns:
            Reap: The confirmation, carrying the claim of whichever unit ran, or
            the reason there is none.
        """
        target = await holder_target(self._db, holder_task_id)
        reap = await self._reaper.reap(target, now_unix=now_unix)
        if reap.confirmed_unix is not None or reap.outcome == REAP_HOLDER_ALIVE:
            return reap
        holder = await self._task(holder_task_id)
        if holder is not None and _terminal_by_observation(holder):
            # A row the worker wrote as its work ended is a report, not an
            # enumeration: no unit looked at a process to produce it.
            return Reap(float(now_unix), REAP_HOLDER_REPORTED, self._reaper.name, CLAIM_REACHABLE)
        return Reap(None, REAP_UNOBSERVABLE, self._reaper.name, reap.claim)

    def _stop_loudly(self, round_row: Round, reap: Reap) -> None:
        """Stop the session on a round whose holder was never confirmed dead."""
        log.error(
            "RECONCILE: round %s expired with nothing confirming holder %s dead (%s); "
            "the machine stays excluded and the session stops",
            round_row.round_id,
            round_row.holder_task_id,
            reap.outcome,
        )
        state = self._shared_state
        if state is None or state.stop_reason:
            return
        state.set_stop_reason(UNREAPED_STOP_REASON)
        self._save_state()

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
    """Whether the task's terminal transition was written because work ended.

    Returns True when the row is terminal and its last transition observed the
    work stop -- the worker reported it, or a reclaimer proved the pid dead. A
    lease watchdog's transition returns False: it timed a lease, not a process.
    """
    if task.state not in TERMINAL_STATES:
        return False
    for entry in reversed(task.history):
        if not isinstance(entry, dict) or entry.get("to") != task.state:
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            return True
        if _EVIDENCE_DEAD_PID in evidence:
            return True
        return _EVIDENCE_LEASE_TTL not in evidence
    return True


def _unix_of(stamp: str) -> float:
    """Read an ISO-8601 timestamp from a task or event row as unix seconds.

    A naive stamp is read as UTC, which is what ``now_iso`` wrote it in.
    """
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
