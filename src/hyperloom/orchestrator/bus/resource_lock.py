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

from hyperloom.common.proctree import collect_tree, group_alive, tree_alive
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
#: it ran (``loop/sub_agent_runner``). They are the only durable trace a later
#: reaper has of what became of that task's processes. ``cleanup_confirmed``
#: rides every terminal transition; the process-group id is recorded only on the
#: unconfirmed path, and only when the raise site knew which group it had failed
#: to confirm.
#:
#: Writer and reader take both names from here so they cannot drift apart: a
#: reader looking for a key the writer stopped writing would find no identity on
#: any row, judge every one of them unverifiable, and never free a lane again.
CLEANUP_CONFIRMED_KEY = "cleanup_confirmed"
CLEANUP_TREE_PGID_KEY = "cleanup_tree_pgid"

#: Why a retained row could not be verified. Logged verbatim to the operator,
#: so each one reads as a cause rather than a code.
UNVERIFIABLE_NO_IDENTITY = "the holder's terminal row records no process-group id"
UNVERIFIABLE_UNREADABLE = "the recorded process group could not be read from /proc"
UNVERIFIABLE_FOREIGN_SCOPE = "the row was acquired in another boot or PID namespace"
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


def _process_group_is_gone(pgid: int) -> bool | None:
    """Whether anything from the recorded process group can still be observed.

    The recorded number is a process-GROUP id, and that is the whole reason this
    probe is worth running. Both launch sites spawn with
    ``start_new_session=True`` (``specialists/subprocess_.py`` and
    ``enablement/runtime/targeted_build.py``), so the root starts out the leader
    of a brand-new group and session: pid == pgid == sid. A pid stops naming
    anything the instant its process exits, but the group id keeps naming the
    group for as long as any member lives, and a plainly forked child stays in
    it. That is the common survivor shape -- root exits, child keeps running in
    the root's group -- and ``group_alive`` catches it.

    WHAT THIS PROBE CANNOT SEE, stated plainly because it is a real hole rather
    than an oversight: a descendant that calls ``setsid`` itself leaves the
    recorded group for one of its own. It is no longer under the (dead) root, so
    walking the tree misses it; it is no longer in the group, so ``killpg``
    misses it too. Nothing else in ``/proc`` ties it back to the recorded id --
    its ppid has been rewritten to the reaper and its session is its own. Such a
    survivor is reported gone here and its lane is handed on while it runs. A
    per-spawn cgroup WOULD name it, since cgroup membership is inherited and
    ``setsid`` does not change it, and that is deliberately a separate project:
    it needs the spawn sites, the cgroup lifetime and the delegation designed
    together, which is more than a lane reaper may settle on its own.
    ``test_a_survivor_that_calls_setsid_is_past_what_this_probe_can_see`` pins
    the gap so a reader finds it known rather than missed.

    So the honest reading of a True answer is "nothing we can observe remains",
    not "the tree is gone".

    This only ever asks. A reaper that killed what it found would be ending work
    on behalf of a holder that never asked it to -- and from the maintenance
    loop, outside any action's cancellation path -- so
    :func:`~hyperloom.common.proctree.kill_tree` has no place here, even though
    the raise site that recorded this id reached it through exactly that call.

    :func:`~hyperloom.common.proctree.collect_tree` enumerates the descendants
    while the root still holds them, and ``tree_alive`` answers for every pid it
    pinned; the group probe then covers the case where the root has already gone
    and its children have re-parented away from it.

    Args:
        pgid: The process group recorded when cleanup was left unconfirmed,
            already established by the caller to belong to this boot and PID
            namespace.

    Returns:
        bool | None: True when nothing from that group answers, False when
        something does, and None when the question could not be put at all --
        an ``OSError`` listing ``/proc``. Only True releases a lane; None hands
        the row to the operator diagnostic instead.
    """
    if pgid <= 0:
        return False
    try:
        tree = collect_tree([pgid])
    except OSError:
        return None
    if tree_alive(tree):
        return False
    # ``group_alive`` also answers True for an indeterminate reading -- a
    # sandbox refusing ``killpg`` -- which keeps the lane held. It is not told
    # apart from a real member here because either way somebody may still be
    # there, and only an affirmative empty reading may free a lane.
    return not group_alive(pgid)


def _holder_stopped_using_the_lane(evidence: dict) -> tuple[bool, str]:
    """Whether a terminal holder's own record proves its lane is free.

    Being terminal does not prove it. :meth:`SubAgentRunner.run_task`
    (``loop/sub_agent_runner``) writes the terminal row on the
    :class:`ExecutionCleanupUnconfirmed` path too, and on that path it
    deliberately SKIPS the release: the processes may still be running, and
    holding the lane is what keeps conflicting work off the machine. Terminal
    state is therefore evidence about the task, not about the lane.

    So a holder that has ended is asked for one of two proofs:

    * ``cleanup_confirmed`` -- the release ran and physical teardown was
      acknowledged. Nothing is left to probe.
    * a recorded process group that no longer answers. A raise site that failed
      to confirm a LOCAL group names it (``specialists/subprocess_.py``,
      ``actions/executors/targeted_build_executor.py``), and
      :class:`ExecutionCleanupUnconfirmed` carries it to the terminal row.

    With neither, the lane stays held indefinitely, which is the intended answer
    rather than a gap: most of those raise sites hold no GPU lease at all, so
    the ``gpu_leases`` exemption does not speak for them, and a holder whose
    group cannot be probed reads exactly like one whose processes are still
    working. No age or TTL is consulted to break the tie -- a row that cannot be
    verified is held, for the life of the session if need be, and
    :func:`_unverifiable_holders` tells the operator how to end it by hand.

    The one raise site that names no group is the Ray-actor cleanup, whose pid
    belongs to another node; it holds GPU cards for its whole run, so the
    ``gpu_leases`` exemption is what covers its lane instead.

    Args:
        evidence: Cleanup evidence from the holder's terminal transition.

    Returns:
        tuple[bool, str]: Whether the lane may be reclaimed, and why it could
        not be verified when it may not. That reason is empty when the holder's
        processes were positively observed alive: such a row is not stuck, it is
        in use, and the operator has nothing to do about it.
    """
    if evidence.get(CLEANUP_CONFIRMED_KEY) is True:
        return True, ""
    pgid = evidence.get(CLEANUP_TREE_PGID_KEY)
    # ``True`` is an ``int`` in Python and survives a JSON round-trip as one.
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 0:
        return False, UNVERIFIABLE_NO_IDENTITY
    gone = _process_group_is_gone(pgid)
    if gone is None:
        return False, UNVERIFIABLE_UNREADABLE
    return gone, ""


def _ended_holder_rows(cur: sqlite3.Cursor, lanes: list[str], *, extra_where: str, extra_params: tuple) -> list[dict]:
    """Select lane rows whose holder task the registry already wrote terminal.

    The join is an inner one on purpose: a holder with no ``tasks`` row is left
    alone. Absence is silence, not a finished holder, and
    :meth:`SqliteLeaseBackend.reap_dead_holders` already covers the rows a
    crashed process leaves behind.

    A lane records its coordinator, not the specialist's GPU worker: a holder
    still on the cards is still working, whatever the coordinator task row says.
    Same exemption ``reap_dead_holders`` makes.

    Args:
        cur: Cursor of the caller's transaction.
        lanes: Lanes to look at; empty means every lane but the round lane.
        extra_where: An additional SQL predicate, already parameterised.
        extra_params: Its parameters.

    Returns:
        list[dict]: The matching rows, each carrying the holder's ``history``
        as ``holder_history``.
    """
    terminal = sorted(TERMINAL_STATES)
    states = ",".join("?" * len(terminal))
    where = (
        f"{extra_where} "
        "AND leases.task_id NOT IN (SELECT task_id FROM gpu_leases) "
        f"AND tasks.state IN ({states})"  # nosec B608 - generated placeholders only.
    )
    params: tuple = (*extra_params, *terminal)
    if lanes:
        # Callers pass a ``_expand_lanes`` result, which is drawn from
        # KNOWN_LANES and so never names BRINGUP_ROUND_LANE.
        placeholders = ",".join("?" * len(lanes))
        where = f"leases.lane IN ({placeholders}) AND {where}"  # nosec B608 - generated placeholders only.
        params = (*lanes, *params)
    else:
        # A round outlives its holder on purpose: only RoundStore may say a
        # round is over, and its holder going terminal is the very input
        # ``Reconciler._advance_or_expire`` weighs before handing off.
        where = f"leases.lane != ? AND {where}"  # nosec B608 - generated placeholders only.
        params = (BRINGUP_ROUND_LANE, *params)
    cur.execute(
        "SELECT leases.*, tasks.history AS holder_history FROM leases "
        f"JOIN tasks ON tasks.task_id = leases.task_id WHERE {where}",  # nosec B608 - generated placeholders only.
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def _unverifiable_holders(cur: sqlite3.Cursor, *, scope: str) -> list[dict]:
    """Retained rows whose holder ended and which nothing can show unused.

    This is the whole operator story for a leaked lane: there is no cleanup
    command, so :func:`_report_unverifiable` logging the remedy IS the
    interface. It runs after :func:`_reclaim_finished_holders` in the same tick,
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
    # Deliberately NOT _ended_holder_rows: that query is shaped for reclamation,
    # and two of its filters would silence exactly the rows an operator most
    # needs to see. Its INNER JOIN on ``tasks`` drops a row whose holder was
    # pruned (bus/db_maintenance.py prune_tasks does not spare a task that still
    # holds a lease), and its gpu_leases exemption drops the GPU path -- which
    # is the one dispatcher.py's release_resources() fails on, i.e. the shape of
    # the 2026-09-21 incident. Reclamation must respect both filters; reporting
    # must not.
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
        if evidence.get(CLEANUP_CONFIRMED_KEY) is True:
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
        reclaimable, reason = _holder_stopped_using_the_lane(evidence)
        if reclaimable or not reason:
            # An empty reason means the group answered, i.e. something is still
            # alive under it. Deliberately NOT reported: that is the normal
            # shape of a holder still winding down, and reporting it would put
            # a remedy in the log for every orderly teardown.
            #
            # Known limitation, accepted rather than overlooked: an unrelated
            # process that recycled the recorded group id answers exactly the
            # same way, so a lane whose real survivors are long gone can sit
            # held and unreported. Distinguishing the two needs an identity
            # that cannot be recycled (a per-spawn cgroup), which is a separate
            # project -- the same boundary the probe's own docstring names.
            continue
        row["reason"] = reason
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
            "no process of task %s is still running, then run: "
            'sqlite3 "%s" '
            "\"DELETE FROM leases WHERE lane='%s' AND holder_id='%s';\"",
            row["lane"],
            row["task_id"],
            row["holder_id"],
            row["reason"],
            row["task_id"],
            db_path or "$SESSION_DIR/storage/coordinator.db",
            row["lane"],
            row["holder_id"],
        )


def _reclaim_finished_holders(cur: sqlite3.Cursor, lanes: list[str], *, scope: str) -> list[dict]:
    """Drop lane rows whose holder ended and left the lane unused, in the caller's transaction.

    The rows leaked on 2026-09-21 belonged to holders already recorded terminal,
    and reclaiming on that alone is what this pass must not do: see
    :func:`_holder_stopped_using_the_lane` for the proof each candidate owes.
    Terminal state is only the cheap prefilter that makes the probe worth
    running -- a task that is still queued or running has not stopped using
    anything, and a task that reached a terminal state never resumes.

    ``expires_at`` is not consulted at all, and neither is any other clock. A
    TTL is a static per-action budget that nothing enforces, so a lapse says
    only that a legitimately long run outlived its estimate (``explore`` budgets
    7200s and holds ``server_lifecycle`` plus ``benchmark_lane`` across a
    benchmark that can exceed it). Age is not evidence either: a row that cannot
    be verified is held however old it gets, and :func:`_unverifiable_holders`
    hands it to the operator instead.

    Sharing the caller's cursor keeps the sweep and the read that follows it in
    one ``BEGIN IMMEDIATE``, so no second acquirer can see a row this one has
    already reclaimed.

    Args:
        cur: Cursor of the transaction that acts on the surviving rows.
        lanes: Lanes to sweep; empty sweeps every lane but the round lane.
        scope: The boot/PID namespace whose rows this process may judge.

    Returns:
        list[dict]: The rows reclaimed.
    """
    # Same first act as :meth:`SqliteLeaseBackend.holder_is_dead`: with no
    # observable ownership domain, no row here is ours to judge. It is also what
    # makes the recorded process-group id mean anything -- a pgid is an identity
    # only inside one boot and PID namespace, and the row's ``owner_scope`` is
    # the namespace the holder ran in, because the process that acquired the
    # lane is the same process that later wrote the terminal evidence read here.
    #
    # Pre-existing, inherited here, and deliberately NOT fixed by this change: a
    # row acquired while /proc is unreadable is stored with ``owner_scope=''``
    # (:func:`local_owner_scope` returns "" on OSError), and no reaper can ever
    # take it back. ``holder_is_dead`` rejects it on this same guard while /proc
    # stays unreadable and on ``row["owner_scope"] != scope`` once /proc
    # recovers; this pass rejects it here and again on the ``owner_scope = ?``
    # predicate below. :meth:`release` still drops such a row -- it keys on
    # ``(lane, holder_id)`` with no scope predicate -- so a holder that ends by
    # releasing is fine; what no reaper can do is take the row back for a holder
    # that ends without releasing, which is precisely the case this pass exists
    # for. Closing that means giving ownership a fallback identity for the case
    # where the kernel will not name one, which changes what a scope asserts; it
    # is its own change, not a widening smuggled into this one, which only
    # reuses the guard already in force. What IS new is that such a row no
    # longer goes unmentioned: :func:`_unverifiable_holders` ignores scope and
    # logs it with its remedy.
    if not scope:
        return []
    candidates = _ended_holder_rows(cur, lanes, extra_where="leases.owner_scope = ?", extra_params=(scope,))
    reclaimed: list[dict] = []
    for row in candidates:
        reclaimable, _reason = _holder_stopped_using_the_lane(_last_cleanup_evidence(row.pop("holder_history")))
        if not reclaimable:
            continue
        cur.execute(
            "DELETE FROM leases WHERE lane=? AND holder_id=?",
            (row["lane"], row["holder_id"]),
        )
        _DIAGNOSED.discard((str(row["lane"]), str(row["holder_id"])))
        reclaimed.append(row)
    return reclaimed


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
            # 2026-09-21: a dispatcher that deliberately retained capacity on
            # an unconfirmed cleanup left six lanes held by holders already
            # recorded terminal, whose process trees had in fact gone. The rows
            # still occupied the lanes two hours later, with 19 queued tasks
            # starved behind them. Sweeping here as well as in the standalone
            # pass keeps the capacity read below from being decided against a
            # holder that ended since that pass ran. Only a holder that can
            # prove its lane is free is taken; see
            # :func:`_holder_stopped_using_the_lane`.
            reclaimed = _reclaim_finished_holders(cur, expanded, scope=local_owner_scope())
            if reclaimed:
                # A lane that vanishes on the acquire path leaves no other trace
                # to attribute it to; the standalone sweep logs its own.
                log.warning(
                    "resource_lock: acquire reclaimed %d lease(s) from ended holders: %s",
                    len(reclaimed),
                    ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(task={r['task_id'][:12]})" for r in reclaimed),
                )

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

    async def reap_finished_holders(self) -> list[dict]:
        """Release every lane whose holder ended and proved the lane unused.

        A pass of its own because the dispatcher's lane gate reads
        :meth:`lane_holders` before it attempts an acquire: a leaked row starves
        the queue without any acquire ever running to reclaim it.

        A holder that ended without that proof keeps its lane, by design -- see
        :func:`_holder_stopped_using_the_lane`, and
        :meth:`diagnose_unverifiable_holders` for what the operator is told
        about it.

        Returns:
            list[dict]: The rows reclaimed.
        """
        async with self.db.transaction() as cur:
            reaped = _reclaim_finished_holders(cur, [], scope=local_owner_scope())
        if reaped:
            log.warning(
                "resource_lock: reclaimed %d lease(s) from ended holders: %s",
                len(reaped),
                ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(task={r['task_id'][:12]})" for r in reaped),
            )
        return reaped

    async def diagnose_unverifiable_holders(self) -> list[dict]:
        """Report every retained row nothing can prove free, and how to free it.

        Run after :meth:`reap_finished_holders` on the same tick. There is no
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

    async def reap_finished_holders(self) -> list[dict]:
        """Release lanes whose holder ended and left them unused, via the backend."""
        fn = getattr(self.backend, "reap_finished_holders", None)
        if not callable(fn):
            return []
        return await fn()

    async def diagnose_unverifiable_holders(self) -> list[dict]:
        """Report retained lanes nothing can prove free, via the backend."""
        fn = getattr(self.backend, "diagnose_unverifiable_holders", None)
        if not callable(fn):
            return []
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
    "UNVERIFIABLE_NO_IDENTITY",
    "UNVERIFIABLE_UNREADABLE",
    "LaneBusy",
    "LaneFull",
    "Lease",
    "ResourceLockManager",
    "SqliteLeaseBackend",
    "StaleLeaseError",
    "drop_round_lane",
    "hold_round_lane",
]
