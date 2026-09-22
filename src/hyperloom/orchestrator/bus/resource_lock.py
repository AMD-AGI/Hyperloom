# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ResourceLockManager + SqliteLeaseBackend."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.state.task_states import TERMINAL_STATES

from .storage.connection import SqliteConnection
from .storage.schema import DEFAULT_LANE_CAPACITIES


log = logging.getLogger(__name__)


KNOWN_LANES = (
    "server_lifecycle",
    "workspace_mutation",
    "benchmark_lane",
    "profile_lane",
    # research_lane carries LLM specialist sub-agents; no serving-lane conflict and capacity may exceed 1.
    "research_lane",
    # gpu_research_lane carries GPU-holding specialists; mutually exclusive with the serving lanes and capacity-1 /
    # strictly serial (one GPU specialist holds the machine at a time; the GPU pool partitions cards within it).
    "gpu_research_lane",
    # build_lane serializes off-loop compile tasks; capacity-1 with no serving-lane conflict (the compile step needs
    # no GPU/server).
    "build_lane",
)

# Lane → lanes that must *also* be free or co-acquired.
LANE_CONFLICTS: dict[str, frozenset[str]] = {
    "benchmark_lane": frozenset({"profile_lane", "server_lifecycle", "gpu_research_lane"}),
    "profile_lane": frozenset({"benchmark_lane", "server_lifecycle", "gpu_research_lane"}),
    "server_lifecycle": frozenset({"benchmark_lane", "profile_lane", "gpu_research_lane"}),
    "workspace_mutation": frozenset(),
    # research_lane does not conflict with any serving-side lane.
    "research_lane": frozenset(),
    # gpu_research_lane ⊥ serving lanes; capacity-1 so GPU specialists serialize.
    "gpu_research_lane": frozenset({"benchmark_lane", "profile_lane", "server_lifecycle"}),
    # build_lane is a serialization/observability primitive only; no conflicts.
    "build_lane": frozenset(),
}


#: The lane an open bring-up round holds for as long as it is open.
#:
#: Deliberately absent from :data:`KNOWN_LANES`, because no task may request it.
#: A round is held by a task that still has to be dispatchable while the round
#: stands, and every serving lane mutexes against ``gpu_research_lane``, so a
#: round holding a serving lane under a holder id of its own would deny its own
#: holder the dispatch the round exists to cover. ``RoundStore`` writes and
#: drops this row inside the transaction that opens, renews and settles the
#: round; capacity readers and the breakdown lane timeline read it like any
#: other ownership row.
BRINGUP_ROUND_LANE = "bringup_round"

#: Recorded on a round's lane row in place of a pid. The round's holder is a
#: task, not this process, and only the task registry can prove a task's process
#: dead. The dead-holder pass skips non-positive pids, leaving these rows to
#: explicit round settlement.
ROUND_LEASE_PID = 0

#: Recorded in the lane row's ``action`` column.
_ROUND_LEASE_ACTION = "bringup_round"


_now_iso = now_iso


def local_owner_scope() -> str:
    """Identify this boot and PID namespace, or leave ownership unobservable."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        namespace = Path("/proc/self/ns/pid").stat().st_ino
    except OSError:
        return ""
    return f"{boot}:{namespace}" if boot else ""


def _lease_iso(unix_ts: float) -> str:
    """Render unix seconds the way the lease table's timestamps compare.

    Budget timestamps use the same offset and precision as :func:`now_iso`.

    Args:
        unix_ts: The instant to render.

    Returns:
        str: A fixed-width UTC ISO-8601 timestamp.
    """
    return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).isoformat(timespec="microseconds")


def hold_round_lane(
    cur: sqlite3.Cursor,
    *,
    round_id: str,
    holder_task_id: str,
    expires_unix: float,
    now_unix: float,
) -> None:
    """Write the lane row an open round holds, inside the caller's transaction.

    Keyed on ``(BRINGUP_ROUND_LANE, round_id)`` and idempotent, so the one call
    serves the acquire, every renewal, and a handoff that moves the round to a
    new holder. ``acquired_at`` survives a renewal; only the holder and the two
    clock columns move.

    Args:
        cur: Cursor of the transaction writing the round row, so the round and
            its lane land together or not at all.
        round_id: The round holding the lane; also the lease's holder id.
        holder_task_id: The task the round is held by right now.
        expires_unix: When the round's lease runs out.
        now_unix: Current wall time.
    """
    stamp = _lease_iso(now_unix)
    cur.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid,"
        "  acquired_at, expires_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(lane, holder_id) DO UPDATE SET"
        "  task_id = excluded.task_id,"
        "  expires_at = excluded.expires_at,"
        "  heartbeat_at = excluded.heartbeat_at",
        (
            BRINGUP_ROUND_LANE,
            round_id,
            holder_task_id,
            _ROUND_LEASE_ACTION,
            ROUND_LEASE_PID,
            stamp,
            _lease_iso(expires_unix),
            stamp,
        ),
    )


def drop_round_lane(cur: sqlite3.Cursor, *, round_id: str) -> None:
    """Release the lane row a round held, inside the caller's transaction.

    Args:
        cur: Cursor of the transaction settling the round.
        round_id: The round whose lane is released.
    """
    cur.execute(
        "DELETE FROM leases WHERE lane = ? AND holder_id = ?",
        (BRINGUP_ROUND_LANE, round_id),
    )


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Expand requested lanes by transitive conflicts; sorted deterministically."""
    out: set[str] = set()
    for lane in lanes:
        if lane not in KNOWN_LANES:
            raise ValueError(f"unknown lane: {lane!r}")
        out.add(lane)
        out.update(LANE_CONFLICTS.get(lane, frozenset()))
    return sorted(out)


#: Evidence keys :meth:`SubAgentRunner.run_task` builds and hands to
#: :meth:`SubAgentRunner._write_terminal` on the terminal transition of a task
#: it ran (``loop/sub_agent_runner``). ``cleanup_confirmed`` rides every
#: terminal transition; the process-group id is recorded only on the unconfirmed
#: path, and only when the raise site knew which group it had failed to confirm.
#:
#: Neither is probed. Nothing reclaims a lane from them: a served process is
#: setsid'd by design and leaves the group its spawn created, so three attempts
#: to decide from such an identity whether a lane was free were refuted in
#: review. They are read only to tell an operator WHY a lane is retained and
#: where to start looking -- see :func:`_report_unverifiable`.
#:
#: Writer and reader still take both names from here so they cannot drift apart:
#: a reader looking for a key the writer stopped writing would print a diagnostic
#: that silently lost its only lead.
CLEANUP_CONFIRMED_KEY = "cleanup_confirmed"
CLEANUP_TREE_PGID_KEY = "cleanup_tree_pgid"

#: Why a retained row could not be verified. Logged verbatim to the operator,
#: so each one reads as a cause rather than a code.
UNVERIFIABLE_FOREIGN_SCOPE = "the row was acquired in another boot or PID namespace"
#: The ordinary case: the holder ended without confirming its cleanup, so the
#: release was skipped on purpose and its work may still be running. Nothing
#: this process can observe settles it -- every cheaper proof tried during
#: review (terminal state, an empty spawn process group, no pidfile naming a
#: live server) turned out to be a proxy a real process slips out of, because a
#: served process is setsid'd by design and its pidfile appears only after it
#: answers. The lane stays held and an operator decides.
UNVERIFIABLE_CLEANUP_UNCONFIRMED = "the holder ended without confirming its cleanup"
#: The odd one: cleanup WAS confirmed, which means the release ran -- so this
#: row should not exist. It is reported rather than reclaimed because nothing
#: here can tell a release that half-happened from one whose row simply
#: outlived it, and because a lane nobody is coming back for is exactly what an
#: operator needs to hear about.
UNVERIFIABLE_CONFIRMED_BUT_HELD = "the holder confirmed its cleanup yet its lane row is still here"
#: Distinct from the above: the row may well be ours, but this process cannot
#: name its own ownership domain (an unreadable /proc), so nothing here may be
#: judged at all. Labelling this "another namespace" would tell an operator to
#: go look at a machine that is not involved.
UNVERIFIABLE_NO_LOCAL_SCOPE = "this process cannot read its own boot/PID namespace, so no row here can be judged"
#: Retained by the gpu_leases exemption. Not a defect -- a lane records its
#: coordinator, not the specialist's GPU worker -- but the operator still has to
#: be told, because this is the very path that wedged 2026-09-21 and the
#: exemption is a reason not to RECLAIM, never a reason not to REPORT.
UNVERIFIABLE_HOLDS_GPU = "the holder still owns GPU cards, so its lane is exempt from reclamation"
#: The holder's ``tasks`` row was pruned out from under its lease. Nothing can
#: judge it any more: the state that said it ended is gone.
UNVERIFIABLE_HOLDER_PRUNED = "the holder's task row no longer exists, so nothing can attest it ended"

#: ``(lane, holder_id)`` pairs whose remedy has already been printed. A sweep
#: runs every maintenance tick and an unverifiable row stays unverifiable by
#: design -- possibly for the life of the session -- so without this the same
#: warning would be reprinted every tick and bury everything else in the log.
_DIAGNOSED: set[tuple[str, str]] = set()


def _last_cleanup_evidence(history_json: str) -> dict:
    """Return the newest history entry's cleanup evidence, or ``{}``.

    Args:
        history_json: The holder task's ``history`` column, as stored.

    Returns:
        dict: The ``evidence`` of the newest entry carrying
        :data:`CLEANUP_CONFIRMED_KEY`; empty when no entry carries one, which
        includes a history this process cannot parse.
    """
    try:
        history = json.loads(history_json or "[]")
    except (TypeError, ValueError):
        return {}
    if not isinstance(history, list):
        return {}
    for entry in reversed(history):
        evidence = entry.get("evidence") if isinstance(entry, dict) else None
        if isinstance(evidence, dict) and CLEANUP_CONFIRMED_KEY in evidence:
            return evidence
    return {}


def cleanup_confirmation_rate(cur: sqlite3.Cursor) -> tuple[int, int]:
    """How often an ended task confirmed its teardown, this session.

    ``leases_unverifiable`` says how many lanes are held right now. It cannot
    say whether that is a rare accident or the ordinary outcome, and that is the
    number which decides whether an automatic release mechanism is worth its
    risk at all: every portable one was refuted, and the only candidate left
    (a sealed descriptor behind a seccomp filter) is safety-critical to get
    right. Measure before building it.

    Reads what :meth:`SubAgentRunner._write_terminal` already records, so it
    costs one query and writes nothing.

    Args:
        cur: Cursor of the caller's transaction.

    Returns:
        tuple[int, int]: Ended tasks whose cleanup was NOT confirmed, and ended
        tasks in total. A task whose history carries no cleanup evidence at all
        counts as unconfirmed: it is exactly the shape that strands a lane.
    """
    unconfirmed = total = 0
    for (history,) in cur.execute("SELECT history FROM tasks WHERE state IN (?,?,?)", tuple(sorted(TERMINAL_STATES))):
        total += 1
        if _last_cleanup_evidence(history).get(CLEANUP_CONFIRMED_KEY) is not True:
            unconfirmed += 1
    return unconfirmed, total


def _unverifiable_holders(cur: sqlite3.Cursor, *, scope: str) -> list[dict]:
    """Retained rows whose holder ended and which nothing can show unused.

    This is the whole operator story for a leaked lane: there is no cleanup
    command, so :func:`_report_unverifiable` logging the remedy IS the
    interface. Nothing reclaims these rows any more, so everything it
    so everything it returns is a row that survived reclamation.

    Unlike that pass it does NOT filter on ``owner_scope``. A row this process
    may not judge is precisely the kind an operator has to be told about, since
    no reaper here will ever take it back -- including the inherited case of a
    row acquired while ``/proc`` was unreadable, which is stored with
    ``owner_scope=''`` and is unreclaimable for good.

    Args:
        cur: Cursor of the caller's transaction.
        scope: The boot/PID namespace whose rows this process may judge; empty
            when the kernel would not name one, in which case no row is ours.

    Returns:
        list[dict]: One entry per unverifiable row, each with a ``reason``.
    """
    # A LEFT JOIN, and no gpu_leases exemption: both of those filters belong to
    # reclamation, and either would silence exactly the rows an operator most
    # needs to see. An inner join drops a row whose holder was pruned
    # (bus/db_maintenance.py prune_tasks does not spare a task that still holds
    # a lease), and the exemption drops the GPU path -- the one
    # dispatcher.py's release_resources() fails on, i.e. the shape of the
    # 2026-09-21 incident.
    rows = [
        dict(r)
        for r in cur.execute(
            "SELECT leases.lane AS lane, leases.holder_id AS holder_id, leases.task_id AS task_id, "
            "leases.owner_scope AS owner_scope, tasks.state AS holder_state, tasks.history AS holder_history, "
            "(SELECT 1 FROM gpu_leases WHERE gpu_leases.task_id = leases.task_id LIMIT 1) AS holds_gpu "
            "FROM leases LEFT JOIN tasks ON tasks.task_id = leases.task_id "
            "WHERE leases.lane != ?",
            (BRINGUP_ROUND_LANE,),
        )
    ]
    stuck: list[dict] = []
    for row in rows:
        state = row.pop("holder_state", None)
        holds_gpu = row.pop("holds_gpu", None)
        history = row.pop("holder_history", None)
        if state is None:
            # The task row is gone; its lease outlived it and no pass can judge it.
            row["reason"] = UNVERIFIABLE_HOLDER_PRUNED
            stuck.append(row)
            continue
        if str(state) not in TERMINAL_STATES:
            continue
        evidence = _last_cleanup_evidence(history)
        # Carried for the operator log only; see _report_unverifiable.
        row["pgid"] = evidence.get(CLEANUP_TREE_PGID_KEY)
        if evidence.get(CLEANUP_CONFIRMED_KEY) is True:
            # Nothing reclaims rows any more, so a confirmed holder whose row
            # survived is no longer quietly cleaned up behind the scenes: it
            # sits there like any other. Say so instead of skipping it.
            row["reason"] = UNVERIFIABLE_CONFIRMED_BUT_HELD
            stuck.append(row)
            continue
        if holds_gpu:
            row["reason"] = UNVERIFIABLE_HOLDS_GPU
            stuck.append(row)
            continue
        if not scope or row["owner_scope"] != scope:
            # Never probed, but for two different reasons that owe the operator
            # two different messages: with no local scope nothing here can be
            # judged at all, whereas a mismatching scope means the recorded id
            # names a process on a machine this one cannot see.
            if evidence.get(CLEANUP_CONFIRMED_KEY) is True:
                continue
            row["reason"] = UNVERIFIABLE_NO_LOCAL_SCOPE if not scope else UNVERIFIABLE_FOREIGN_SCOPE
            stuck.append(row)
            continue
        # Nothing here tries to decide whether the lane is free. Seven rounds of
        # review established that this process cannot know: a served process is
        # setsid'd by design (:mod:`actions.executors._server_lifecycle` says so
        # where it reads a pidfile), so it leaves the group recorded at spawn;
        # its pidfile is written only after the server answers, so the whole
        # model-load window has a live server nothing names; and matching a
        # cmdline is the same kind of guess as matching a group. Each candidate
        # proof turned out to be a proxy with a way around it, and releasing a
        # lane wrongly lets two rounds onto the same cards -- silent corruption,
        # against a deadlock an operator clears in 90 seconds. So the lane is
        # kept and the operator is told; that asymmetry is the design.
        row["reason"] = UNVERIFIABLE_CLEANUP_UNCONFIRMED
        stuck.append(row)
    return stuck


def _report_unverifiable(rows: list[dict], *, db_path: str = "") -> None:
    """Log what happened and what to do about it, once per ``(lane, holder)``.

    One warning per row, carrying the lane, the holder task, why it could not be
    verified, and a statement the operator can paste. It took an hour of
    ``py-spy`` and sqlite spelunking to work that out on 2026-09-21; it should
    take reading one log line now.

    The caution is not decoration. Reclamation is deliberately refused here
    exactly because nothing could prove the task's processes stopped, so an
    operator deleting the row without checking is doing the one thing the
    retention exists to prevent.

    Args:
        rows: What :func:`_unverifiable_holders` returned.
        db_path: Real path of this session's database, so the remedy is a
            statement the operator can paste rather than one they must first
            resolve ``$SESSION_DIR`` for themselves.
    """
    for row in rows:
        key = (str(row["lane"]), str(row["holder_id"]))
        if key in _DIAGNOSED:
            continue
        _DIAGNOSED.add(key)
        log.warning(
            "resource_lock: lane %s is held by ended task %s (holder %s) and cannot be verified free: %s. "
            "It is retained on purpose -- nothing observable proves that task's processes stopped, and "
            "releasing it would let conflicting work onto the machine. To release it by hand, FIRST confirm "
            "no process of task %s is still running%s, then run: "
            'sqlite3 "%s" '
            "\"DELETE FROM leases WHERE lane='%s' AND holder_id='%s';\"",
            row["lane"],
            row["task_id"],
            row["holder_id"],
            row["reason"],
            row["task_id"],
            # The spawn process group, when one was recorded. It is no longer
            # evidence -- a served process setsid's out of it -- but it is still
            # the best starting point a human has for "what did this task leave".
            f" (its spawn process group was {row['pgid']}; a server it started may have left it)"
            if row.get("pgid")
            else "",
            db_path or "$SESSION_DIR/storage/coordinator.db",
            row["lane"],
            row["holder_id"],
        )


@dataclass
class Lease:
    """Lease handle returned by ``acquire_many``."""

    holder_id: str
    task_id: str
    action: str
    lanes: tuple[str, ...]
    acquired_at: str
    expires_at: str
    pid: int = field(default_factory=os.getpid)


class LaneBusy(RuntimeError):
    """Raised by ``acquire_many`` on a cross-lane conflict (Inv-7.1); kept distinct from capacity."""

    def __init__(self, busy_lanes: list[str]):
        """Initialise with the lanes that triggered the cross-lane conflict."""
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class LaneFull(RuntimeError):
    """Raised by ``acquire_many`` when a lane hits its ``capacity`` cap (pure capacity decision, distinct from :class:`LaneBusy`)."""

    def __init__(self, full_lanes: list[str]):
        """Initialise with the lanes that were at capacity."""
        super().__init__(f"lanes full: {full_lanes!r}")
        self.full_lanes = full_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


class SqliteLeaseBackend:
    """Lease backend behind :class:`ResourceLockManager`; ``BEGIN IMMEDIATE`` + PK uniqueness gives atomic acquire-many."""

    def __init__(self, db: SqliteConnection):
        """Bind the backend to a SQLite connection."""
        self.db = db

    async def acquire_many(
        self,
        lanes: list[str],
        *,
        holder_id: str,
        task_id: str,
        action: str,
        ttl_sec: int,
    ) -> Lease:
        """Acquire ``lanes`` + transitive conflicts as one atomic batch."""
        if not lanes:
            raise ValueError("acquire_many called with no lanes")
        expanded = _expand_lanes(lanes)
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + ttl_sec
        expires_iso = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()

        async with self.db.transaction() as cur:
            # Resolve capacity per lane (fallback for unseeded DBs).
            capacity_by_lane: dict[str, int] = {}
            placeholders = ",".join("?" * len(expanded))
            cur.execute(
                f"SELECT lane, capacity FROM lane_capacity WHERE lane IN ({placeholders})",  # nosec B608 - generated placeholders only.
                expanded,
            )
            for row in cur.fetchall():
                capacity_by_lane[row["lane"]] = int(row["capacity"])
            for lane in expanded:
                capacity_by_lane.setdefault(
                    lane,
                    int(DEFAULT_LANE_CAPACITIES.get(lane, 1)),
                )

            cur.execute(
                f"SELECT lane, holder_id FROM leases WHERE lane IN ({placeholders})",  # nosec B608 - generated placeholders only.
                expanded,
            )
            holders_per_lane: dict[str, set[str]] = {lane: set() for lane in expanded}
            for row in cur.fetchall():
                holders_per_lane[row["lane"]].add(row["holder_id"])

            # Distinguish capacity (LaneFull) from cross-lane mutex (LaneBusy).
            full: list[str] = []
            busy: list[str] = []
            for lane in expanded:
                live = holders_per_lane.get(lane, set())
                # Same-holder attempt is idempotent (acts as TTL refresh).
                if holder_id in live:
                    continue
                cap = capacity_by_lane.get(lane, 1)
                if cap <= 0:
                    # capacity=0 → lane disabled; LaneFull so dispatcher drops.
                    full.append(lane)
                    continue
                if len(live) >= cap:
                    # cap>1 full → LaneFull; cap==1 full → LaneBusy.
                    if cap > 1:
                        full.append(lane)
                    else:
                        busy.append(lane)

            if busy:
                raise LaneBusy(busy)
            if full:
                raise LaneFull(full)

            for lane in expanded:
                # INSERT OR REPLACE lets the same holder refresh its row.
                cur.execute(
                    "INSERT OR REPLACE INTO leases(lane, holder_id, "
                    "task_id, action, pid, acquired_at, expires_at, "
                    "heartbeat_at, owner_scope) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        lane,
                        holder_id,
                        task_id,
                        action,
                        os.getpid(),
                        now_iso,
                        expires_iso,
                        now_iso,
                        local_owner_scope(),
                    ),
                )

        return Lease(
            holder_id=holder_id,
            task_id=task_id,
            action=action,
            lanes=tuple(expanded),
            acquired_at=now_iso,
            expires_at=expires_iso,
        )

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        """Refresh ``expires_at`` for every lane this holder owns (keyed on ``(lane, holder_id)`` PK)."""
        new_expires_iso = datetime.fromtimestamp(time.time() + ttl_sec, tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"UPDATE leases SET expires_at=?, heartbeat_at=? WHERE lane IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                (new_expires_iso, now_iso, *lease.lanes, lease.holder_id),
            )
            if cur.rowcount != len(lease.lanes):
                raise StaleLeaseError(f"heartbeat mismatch: expected {len(lease.lanes)} rows, got {cur.rowcount}")

    async def heartbeat_by_task(self, task_id: str, *, ttl_sec: int) -> list[str]:
        """Refresh every lane row a task holds, whoever the holder is."""
        new_expires_iso = datetime.fromtimestamp(time.time() + ttl_sec, tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute("SELECT lane FROM leases WHERE task_id=?", (task_id,))
            lanes = sorted(str(r["lane"]) for r in cur.fetchall())
            if lanes:
                cur.execute(
                    "UPDATE leases SET expires_at=?, heartbeat_at=? WHERE task_id=?",
                    (new_expires_iso, now_iso, task_id),
                )
        return lanes

    async def release(self, lease: Lease) -> int:
        """Drop every (lane, holder_id) row this lease owns."""
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"DELETE FROM leases WHERE lane IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                (*lease.lanes, lease.holder_id),
            )
            return cur.rowcount

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Best-effort liveness probe for a lease-holder PID."""
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    @classmethod
    def holder_is_dead(cls, row) -> bool:
        """Accept only an absent PID observed in its recorded local PID domain."""
        scope = local_owner_scope()
        if not scope or row["owner_scope"] != scope:
            return False
        try:
            pid = int(row["pid"] or 0)
        except (TypeError, ValueError):
            return False
        return pid > 0 and not cls._pid_alive(pid)

    async def reap_dead_holders(self) -> list[dict]:
        """Release leases whose local holder process is confirmed absent."""
        reaped: list[dict] = []
        async with self.db.transaction() as cur:
            cur.execute("SELECT * FROM leases")
            live_rows = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT task_id FROM gpu_leases")
            gpu_tasks = {row["task_id"] for row in cur.fetchall()}
            for row in live_rows:
                # A lane records its coordinator, not the specialist's GPU worker.
                if row["task_id"] in gpu_tasks or not self.holder_is_dead(row):
                    continue
                cur.execute(
                    "DELETE FROM leases WHERE lane=? AND holder_id=?",
                    (row["lane"], row["holder_id"]),
                )
                reaped.append(row)
        if reaped:
            log.warning(
                "resource_lock: reaped %d lease(s) from dead holders: %s",
                len(reaped),
                ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(pid={r.get('pid')})" for r in reaped),
            )
        return reaped

    async def cleanup_confirmation_rate(self) -> tuple[int, int]:
        """Unconfirmed-cleanup count and total ended tasks, for this session.

        Returns:
            tuple[int, int]: ``(unconfirmed, total)``.
        """
        async with self.db.transaction() as cur:
            return cleanup_confirmation_rate(cur)

    async def diagnose_unverifiable_holders(self) -> list[dict]:
        """Report every retained row nothing can prove free, and how to free it.

        Run after :meth:`reap_dead_holders` on the same tick. There is no
        cleanup command and there will not be one -- a row is released by an
        operator who has checked what the reaper could not -- so this warning is
        the entire remedy path.

        Returns:
            list[dict]: The rows currently held and unverifiable, whether or not
            this tick was the one that logged their remedy.
        """
        async with self.db.transaction() as cur:
            stuck = _unverifiable_holders(cur, scope=local_owner_scope())
        _report_unverifiable(stuck, db_path=str(getattr(self.db, "db_path", "") or ""))
        return stuck

    async def bringup_round_holders(self, now_unix: float) -> set[str]:
        """Return round ids with a retained ownership row, irrespective of age."""
        rows = await self.db.fetchall(
            "SELECT holder_id FROM leases WHERE lane = ?",
            (BRINGUP_ROUND_LANE,),
        )
        return {str(r["holder_id"]) for r in rows}

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: holder_count}`` for every retained ownership row."""
        rows = await self.db.fetchall("SELECT lane, COUNT(*) AS n FROM leases GROUP BY lane")
        return {r["lane"]: int(r["n"]) for r in rows}

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` for every row in ``lane_capacity``."""
        try:
            rows = await self.db.fetchall("SELECT lane, capacity FROM lane_capacity")
        except sqlite3.OperationalError as exc:
            # Legacy DB never opened with v0.8 lacks the table; fall back to defaults.
            log.debug("lane_capacities: lane_capacity table unavailable: %s", exc)
            return dict(DEFAULT_LANE_CAPACITIES)
        out: dict[str, int] = dict(DEFAULT_LANE_CAPACITIES)
        for r in rows:
            out[r["lane"]] = int(r["capacity"])
        return out


class ResourceLockManager:
    """Coordinator-facing wrapper."""

    def __init__(self, backend: SqliteLeaseBackend):
        """Wrap a lease backend and initialise the per-process counters."""
        self.backend = backend
        # Per-process cumulative acquire / lane-full / lane-busy counters.
        self._counters: dict[str, dict[str, int]] = {}

    async def acquire_many(self, lanes: list[str], **kwargs) -> Lease:
        """Acquire lanes via the backend, updating lifetime counters."""
        try:
            lease = await self.backend.acquire_many(lanes, **kwargs)
        except LaneFull as exc:
            for lane in exc.full_lanes:
                self._bump_counter(lane, "lane_full_count")
            raise
        except LaneBusy as exc:
            for lane in exc.busy_lanes:
                self._bump_counter(lane, "lane_busy_count")
            raise
        for lane in lease.lanes:
            self._bump_counter(lane, "acquire_count")
        return lease

    async def try_acquire_many(self, lanes: list[str], **kwargs) -> Lease | None:
        """Non-blocking variant of :meth:`acquire_many`."""
        try:
            return await self.acquire_many(lanes, **kwargs)
        except (LaneBusy, LaneFull):
            return None

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        """Refresh a lease's TTL via the backend."""
        return await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def heartbeat_by_task(self, task_id: str, *, ttl_sec: int) -> list[str]:
        """Refresh every lane row a task holds."""
        return await self.backend.heartbeat_by_task(task_id, ttl_sec=ttl_sec)

    async def release(self, lease: Lease) -> int:
        """Release a lease and bump each lane's release counter."""
        n = await self.backend.release(lease)
        for lane in lease.lanes:
            self._bump_counter(lane, "release_count")
        return n

    async def reap_dead_holders(self) -> list[dict]:
        """Release leases whose holder process is dead via the backend."""
        fn = getattr(self.backend, "reap_dead_holders", None)
        if not callable(fn):
            return []
        return await fn()

    async def diagnose_unverifiable_holders(self) -> list[dict]:
        """Report retained lanes nothing can prove free, via the backend."""
        fn = getattr(self.backend, "diagnose_unverifiable_holders", None)
        if not callable(fn):
            return []
        return await fn()

    async def cleanup_confirmation_rate(self) -> tuple[int, int]:
        """Unconfirmed-cleanup count and total ended tasks, via the backend."""
        fn = getattr(self.backend, "cleanup_confirmation_rate", None)
        if not callable(fn):
            return 0, 0
        return await fn()

    async def bringup_round_holders(self, now_unix: float) -> set[str]:
        """Return the ids of rounds still holding their lane, via the backend.

        Args:
            now_unix: The instant to test.

        Returns:
            set[str]: Round ids holding :data:`BRINGUP_ROUND_LANE` then.
        """
        return await self.backend.bringup_round_holders(now_unix)

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: live_holder_count}`` via the backend."""
        return await self.backend.lane_holders()

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` via the backend."""
        return await self.backend.lane_capacities()

    def _bump_counter(self, lane: str, field: str) -> None:
        """Increment one per-lane lifetime counter by 1."""
        d = self._counters.setdefault(lane, {})
        d[field] = int(d.get(field, 0)) + 1


__all__ = [
    "BRINGUP_ROUND_LANE",
    "CLEANUP_CONFIRMED_KEY",
    "CLEANUP_TREE_PGID_KEY",
    "KNOWN_LANES",
    "LANE_CONFLICTS",
    "ROUND_LEASE_PID",
    "UNVERIFIABLE_FOREIGN_SCOPE",
    "UNVERIFIABLE_CONFIRMED_BUT_HELD",
    "UNVERIFIABLE_CLEANUP_UNCONFIRMED",
    "LaneBusy",
    "LaneFull",
    "Lease",
    "ResourceLockManager",
    "SqliteLeaseBackend",
    "StaleLeaseError",
    "drop_round_lane",
    "hold_round_lane",
]
