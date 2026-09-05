# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The repair pass, exercised against the states it exists to get a session out of.

Every case here is a session that cannot move: a round nobody will settle, a
task row with no process, a review nobody answered. The point of each test is
not that the repair is possible but that it happens without anything having to
admit it first.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - starts and kills a sleep, to exercise the reaper
import sys
import time

import pytest

from hyperloom.orchestrator.bringup.reap import (
    BACKEND_PROCESS_GROUP,
    REAP_HOLDER_REPORTED,
    REAP_KILLED,
    REAP_UNOBSERVABLE,
    ProcessGroupReaper,
    Reap,
)
from hyperloom.orchestrator.bringup.reconcile import Reconciler, TIMEOUT_VERDICT
from hyperloom.orchestrator.supervisor import store as supervisor_store
from hyperloom.orchestrator.bus.resource_lock import (
    BRINGUP_ROUND_LANE,
    ResourceLockManager,
    SqliteLeaseBackend,
    drop_round_lane,
)
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.policy.projection import ResourceFacts
from hyperloom.orchestrator.state.round_store import (
    EXPIRED_REAPED,
    EXPIRED_UNREAPED,
    FAILED,
    OPEN,
    SETTLED,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry

_LEASE = 600.0

#: The tests share one wall-clock origin with the task registry, which stamps
#: its own transitions with real time. A fabricated origin would date every
#: task row hours away from the instant the rules are asked about.
_NOW = time.time()


class _Enablement:
    """The enablement fields the pass reads and writes."""

    def __init__(self) -> None:
        self.validation_pending = False
        self.revalidation_task_id = ""


class _State:
    """The slice of SharedState the pass reads and writes."""

    def __init__(self) -> None:
        self.stop_reason = ""
        self.saved = 0
        self.baseline_tput = 0.0
        self.tp = 0
        self.enablement = _Enablement()

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def save(self, _session_dir) -> None:
        self.saved += 1


class _Pending:
    """A proposal the loop is still waiting on."""

    def __init__(self, specialist: str) -> None:
        self.payload = {"params": {"specialist_task_id": specialist}}
        self.decided = False
        self.verdict = None


class _Reaper:
    """A reap whose answer the test decides."""

    name = BACKEND_PROCESS_GROUP
    claim = "reachable"

    def __init__(self, answer: Reap) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def available(self) -> str:
        return ""

    async def reap(self, target, *, now_unix: float) -> Reap:
        self.calls.append(target.label)
        if self.answer.confirmed_unix is None:
            return self.answer
        return Reap(float(now_unix), self.answer.outcome, self.name, self.claim)


@pytest.fixture
def db(tmp_path):
    """A real session database."""
    conn = SqliteConnection(tmp_path / "coordinator.db")
    yield conn
    conn.close()


def _build(db, *, reaper=None, proposals=None, state=None, **kw) -> tuple[Reconciler, RoundStore, TaskRegistry, _State]:
    """A reconciler over ``db``, with the pieces a test needs to inspect."""
    rounds = RoundStore(db)
    tasks = TaskRegistry(db)
    shared = state or _State()
    rec = Reconciler(
        rounds=rounds,
        tasks=tasks,
        locks=ResourceLockManager(SqliteLeaseBackend(db)),
        shared_state=shared,
        resources=ResourceFacts(),
        proposals=(lambda: proposals) if proposals is not None else None,
        # The session dir the database lives in: the pass writes its tick stamp
        # and its directive cursor there.
        session_dir=db.db_path.parent,
        reaper=reaper if reaper is not None else _Reaper(Reap(None, REAP_UNOBSERVABLE)),
        **kw,
    )
    return rec, rounds, tasks, shared


async def _open_round(rounds: RoundStore, tasks: TaskRegistry, *, holder: str, lease: float = _LEASE) -> None:
    """Open a round held by a real task row."""
    await tasks.create(kind="specialist", params={}, idempotency_key=f"k:{holder}", task_id=holder)
    result = await rounds.open(
        f"round-{holder}",
        holder_task_id=holder,
        lease_sec=lease,
        now_unix=_NOW,
        request_id=f"open:{holder}",
    )
    assert result.ok


@pytest.mark.asyncio
async def test_an_expired_round_is_settled_though_every_other_path_is_shut(db):
    """The pass is the one thing that runs when the session is already stopping.

    A stopped session dispatches nothing, so every repair reached through a
    dispatch is unreachable exactly when it is needed. Nothing here consults
    the stop reason, the phase or the mode.
    """
    state = _State()
    state.stop_reason = "enablement_stalled"
    rec, rounds, tasks, _ = _build(db, state=state)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    report = await rec.run(_NOW + 10.0)

    assert report.settled == [("round-spec-1", EXPIRED_UNREAPED)]
    assert (await rounds.get("round-spec-1")).state == SETTLED


@pytest.mark.asyncio
async def test_a_round_whose_holder_cannot_be_confirmed_dead_still_releases(db):
    """Recorded, not acted on.

    A process-group reap can never prove a tree gone, so "unconfirmed" is the
    ordinary answer rather than an emergency. The lease has run out either way,
    and a round that held the machine on the strength of what nobody could
    observe is the shape that trapped a session before.
    """
    rec, rounds, tasks, state = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    await rec.run(_NOW + 10.0)

    settled = await rounds.get("round-spec-1")
    assert settled.outcome == EXPIRED_UNREAPED, "the outcome still records that nothing confirmed it"
    assert settled.excludes_at(_NOW + 10.0) is False
    assert state.stop_reason == "", "an unobservable reap does not end the session"


@pytest.mark.asyncio
async def test_a_confirmed_reap_gives_the_machine_back_at_once(db):
    """With proof the round expires reaped, and a settled round holds nothing."""
    rec, rounds, tasks, state = _build(db, reaper=_Reaper(Reap(_NOW, REAP_KILLED)))
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    await rec.run(_NOW + 10.0)

    settled = await rounds.get("round-spec-1")
    assert settled.outcome == EXPIRED_REAPED
    assert settled.excludes_at(_NOW + 10.0) is False
    assert state.stop_reason == ""


@pytest.mark.asyncio
async def test_the_reaper_kills_the_holders_recorded_process_and_confirms_it(db):
    """The default reaper is a real kill, confirmed by looking afterwards."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])  # nosec B603
    try:
        rec, rounds, tasks, _ = _build(db, reaper=ProcessGroupReaper())
        await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
        async with db.transaction() as cur:
            cur.execute(
                "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at)"
                " VALUES ('server_lifecycle', 'h1', 'spec-1', 'bench', ?, '', '', '')",
                (child.pid,),
            )

        await rec.run(_NOW + 10.0)

        settled = await rounds.get("round-spec-1")
        assert settled.outcome == EXPIRED_REAPED
        assert settled.reap_backend == BACKEND_PROCESS_GROUP
    finally:
        child.kill()
        child.wait()


@pytest.mark.asyncio
async def test_a_terminal_holder_alone_never_settles_the_round(db):
    """A round spans the specialist and the integrate that consumes its patch."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")

    await rec.run(_NOW + 1.0)

    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_terminal_holder_hands_the_round_to_the_integrate_that_follows_it(db):
    """The successor takes the round, and the fence moves with it."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")
    integrate = await tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-1"},
        idempotency_key="integrate:1",
        lease_ttl_sec=900,
    )

    report = await rec.run(_NOW + 1.0)

    round_row = await rounds.get("round-spec-1")
    assert report.handed_off == ["round-spec-1"]
    assert round_row.state == OPEN
    assert round_row.holder_task_id == integrate.task_id
    assert round_row.fence == 2


@pytest.mark.asyncio
async def test_an_undecided_review_holds_the_round_open_until_its_ttl(db):
    """The gap between the specialist and the integrate is a proposal, not a task."""
    pending = {"m1": _Pending("spec-1")}
    rec, rounds, tasks, _ = _build(db, proposals=pending)
    # A lease long enough that the deadline rule is not what answers here.
    await _open_round(rounds, tasks, holder="spec-1", lease=100_000.0)
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")

    await rec.run(_NOW + 10_000.0)

    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_terminal_holder_with_nothing_following_it_expires_on_its_cap(db):
    """Not on the tick it went terminal -- the successor is created by a later one."""
    rec, rounds, tasks, _ = _build(db, reaper=_Reaper(Reap(_NOW, REAP_HOLDER_REPORTED)), terminal_holder_cap_sec=300.0)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")

    await rec.run(_NOW + 10.0)
    assert (await rounds.get("round-spec-1")).state == OPEN

    await rec.run(_NOW + 600.0)
    assert (await rounds.get("round-spec-1")).state == SETTLED


@pytest.mark.asyncio
async def test_a_holder_that_reported_its_own_end_is_proof_but_a_lease_watchdog_is_not(db):
    """A watchdog times a lease; it never looks at a process."""
    rec, rounds, tasks, state = _build(db, terminal_holder_cap_sec=0.0)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "failed", {"reason": "lease_expired", "lease_ttl_sec": 60.0, "age_sec": 99.0})

    await rec.run(_NOW + 10.0)

    assert (await rounds.get("round-spec-1")).outcome == EXPIRED_UNREAPED


@pytest.mark.asyncio
async def test_a_running_task_whose_process_is_gone_is_failed_and_one_unobservable_is_not(db):
    """Inability to observe is UNKNOWN. Nothing is manufactured from it."""
    rec, _rounds, tasks, _ = _build(db)
    dead = await tasks.create(kind="specialist", params={}, idempotency_key="dead")
    blind = await tasks.create(kind="specialist", params={}, idempotency_key="blind")
    await tasks.transition(dead.task_id, "running")
    await tasks.transition(blind.task_id, "running")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    gone.wait()
    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at)"
            " VALUES ('server_lifecycle', 'h1', ?, 'bench', ?, '', '', '')",
            (dead.task_id, gone.pid),
        )

    report = await rec.run(_NOW)

    assert report.failed_tasks == [dead.task_id]
    assert (await tasks.get(dead.task_id)).state == "failed"
    assert (await tasks.get(blind.task_id)).state == "running"


@pytest.mark.asyncio
async def test_an_unanswered_review_is_denied_and_a_verdict_that_arrived_is_not_overwritten(db):
    """The timeout is a deny, and it is written compare-and-set."""
    pending = {"m-late": _Pending("spec-1"), "m-answered": _Pending("spec-2")}
    rec, _rounds, _tasks, _ = _build(db, proposals=pending, review_ttl_sec=1.0)
    async with db.transaction() as cur:
        for msg_id in ("m-late", "m-answered"):
            cur.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, priority, ts)"
                " VALUES (?, 'orchestration', '*', 'proposal', NULL, '{}', 1, '2020-01-01T00:00:00+00:00')",
                (msg_id,),
            )
        cur.execute(
            "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, priority, ts)"
            " VALUES ('v1', 'critic', '*', 'review_verdict', NULL, ?, 1, '2020-01-01T00:01:00+00:00')",
            (json.dumps({"target_proposal_msg_id": "m-answered", "verdict": "approve"}),),
        )

    report = await rec.run(_NOW)

    verdicts = await db.fetchall("SELECT payload FROM events WHERE topic = 'review_verdict' ORDER BY seq")
    decoded = [json.loads(row["payload"]) for row in verdicts]
    assert report.denied_reviews == ["m-late"]
    assert [(d["target_proposal_msg_id"], d["verdict"]) for d in decoded] == [
        ("m-answered", "approve"),
        ("m-late", TIMEOUT_VERDICT),
    ]
    assert pending["m-late"].verdict == TIMEOUT_VERDICT
    assert pending["m-late"].decided is True


@pytest.mark.asyncio
async def test_a_second_pass_does_not_deny_a_proposal_twice(db):
    """The compare-and-set is the guard, so the pass is safe to run every tick."""
    rec, _rounds, _tasks, _ = _build(db, review_ttl_sec=1.0)
    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, priority, ts)"
            " VALUES ('m1', 'orchestration', '*', 'proposal', NULL, '{}', 1, '2020-01-01T00:00:00+00:00')"
        )

    await rec.run(_NOW)
    await rec.run(_NOW)

    rows = await db.fetchall("SELECT 1 FROM events WHERE topic = 'review_verdict'")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_a_settle_the_store_rejected_is_re_driven_from_the_outbox(db):
    """The round still owes the outcome somebody asked for."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1")
    refused = await rounds.settle(
        "round-spec-1",
        holder_task_id="spec-1",
        fence=99,
        outcome=FAILED,
        now_unix=_NOW,
        request_id="settle:stale",
    )
    assert not refused.ok

    report = await rec.run(_NOW + 1.0)

    settled = await rounds.get("round-spec-1")
    assert report.redriven == ["round-spec-1"]
    assert settled.state == SETTLED
    assert settled.outcome == FAILED


@pytest.mark.asyncio
async def test_the_resource_facts_are_reread_from_what_the_rules_left(db):
    """The facts the gate reads reflect the repair, not the state before it."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
    await rec.run(_NOW - 100.0)
    assert rec._resources.excluding_round_id == "round-spec-1"

    await rec.run(_NOW + 10_000.0)

    # The pass settled the round, so the facts now say the machine is free:
    # the repair, not the state before it.
    assert rec._resources.excluding_round_id == ""


@pytest.mark.asyncio
async def test_a_rule_that_raises_does_not_stop_the_rules_after_it(db):
    """Isolation, not admission: every rule is attempted on every tick."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    async def _boom(*_a, **_k):
        raise RuntimeError("rule failed")

    rec._deny_timed_out_reviews = _boom

    report = await rec.run(_NOW + 10.0)

    assert report.failures == ["_boom"]
    assert report.settled == [("round-spec-1", EXPIRED_UNREAPED)]
    # The re-read runs last, so it saw the settle the raising rule did not stop.
    assert rec._resources.excluding_round_id == ""


@pytest.mark.asyncio
async def test_every_expired_round_is_charged_to_the_ledger(db):
    """A round that died without reporting still cost the session a round."""
    rec, rounds, tasks, _ = _build(db, reaper=_Reaper(Reap(_NOW, REAP_KILLED)))
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    await rec.run(_NOW + 10.0)

    observations = await rounds.observations()
    assert [o.round_id for o in observations] == ["round-spec-1"]
    assert observations[0].evidence["stage"] == 0


@pytest.mark.asyncio
async def test_the_tick_is_stamped_before_any_rule_can_block(db):
    """The stamp is what tells a watcher outside the process that a tick began."""
    rec, _, _, _ = _build(db)

    await rec.run(_NOW)

    stamp = supervisor_store.read_tick(db.db_path.parent)
    assert stamp is not None
    assert stamp.pid == os.getpid()
    assert stamp.stamped_unix == _NOW


@pytest.mark.asyncio
async def test_a_round_whose_lane_another_pass_took_is_settled_here(db):
    """Whoever swept the lease, the round it belonged to still ends.

    The round's row says it has hours of lease left. Its lane row is gone, which
    is what a sweep running outside this pass leaves behind, and that -- not the
    round's own column -- is what decides the round has run out.
    """
    rec, rounds, tasks, _ = _build(db, reaper=_Reaper(Reap(_NOW, REAP_KILLED)))
    await _open_round(rounds, tasks, holder="spec-1", lease=_LEASE)
    async with db.transaction() as cur:
        drop_round_lane(cur, round_id="round-spec-1")

    report = await rec.run(_NOW + 1.0)

    assert report.settled == [("round-spec-1", EXPIRED_REAPED)]
    settled = await rounds.get("round-spec-1")
    assert settled.state == SETTLED
    assert settled.expires_unix > _NOW + 1.0, "the round's own column never said it had run out"


@pytest.mark.asyncio
async def test_a_revalidation_window_whose_task_is_terminal_is_closed(db):
    """A window nobody will close holds the guard that drops ``skip_to_close``.

    ``validation_pending`` is true in every phase, so a revalidation task that
    ended without reporting takes away the only exit an unpromotable run has.
    """
    state = _State()
    state.enablement.validation_pending = True
    state.enablement.revalidation_task_id = "reval-1"
    rec, _, tasks, _ = _build(db, state=state)
    await tasks.create(kind="baseline", params={}, idempotency_key="k:reval-1", task_id="reval-1")
    await tasks.transition("reval-1", "running")
    await tasks.transition("reval-1", "failed")

    report = await rec.run(_NOW + 1.0)

    assert report.closed_windows == ["reval-1"]
    assert state.enablement.validation_pending is False
    assert state.enablement.revalidation_task_id == ""
    assert state.saved == 1, "the close has to outlive the process that made it"


@pytest.mark.asyncio
async def test_a_revalidation_window_whose_task_still_runs_is_left_alone(db):
    """The window is the run's own, until the task holding it ends."""
    state = _State()
    state.enablement.validation_pending = True
    state.enablement.revalidation_task_id = "reval-1"
    rec, _, tasks, _ = _build(db, state=state)
    await tasks.create(kind="baseline", params={}, idempotency_key="k:reval-1", task_id="reval-1")

    report = await rec.run(_NOW + 1.0)

    assert report.closed_windows == []
    assert state.enablement.validation_pending is True


@pytest.mark.asyncio
async def test_the_pass_sweeps_the_leases_and_reports_what_it_swept(db):
    """This pass is the lease sweeper, and the maintenance tick reads its count."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=_LEASE)
    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at)"
            " VALUES ('server_lifecycle', 'h1', 'other', 'bench', 0, '', '', '')",
        )

    report = await rec.run(_NOW + 1.0)

    assert (report.leases_reaped, report.settled) == (1, [])
    assert rec.last_report is report
    # The round is still inside its lease, so it still holds its lane and the
    # sweep left it alone.
    lanes = await db.fetchall("SELECT lane, holder_id FROM leases")
    assert [(r["lane"], r["holder_id"]) for r in lanes] == [(BRINGUP_ROUND_LANE, "round-spec-1")]
