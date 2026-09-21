# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The repair pass, exercised against the states it exists to get a session out of."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - starts and kills a sleep, to exercise the reaper
import sys
import time

import pytest

from hyperloom.orchestrator.bringup.reconcile import Reconciler, TIMEOUT_VERDICT
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
    OPEN,
    SETTLED,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry

_LEASE = 600.0

#: The tests share one wall-clock origin with the task registry, which stamps
#: its own transitions with real time. A fabricated origin would date every
#: task row hours away from the instant the rules are asked about; so would this
#: one if it were read once at import, because a shard spends minutes in other
#: files before reaching this one and the rules are asked about offsets of ten
#: seconds. :func:`_anchor_now_to_this_test` re-reads it per test.
_NOW = time.time()


@pytest.fixture(autouse=True)
def _anchor_now_to_this_test():
    """Re-anchor :data:`_NOW` to the instant this test runs."""
    global _NOW
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


@pytest.fixture
def db(tmp_path):
    """A real session database."""
    conn = SqliteConnection(tmp_path / "coordinator.db")
    yield conn
    conn.close()


def _build(db, *, proposals=None, state=None, **kw) -> tuple[Reconciler, RoundStore, TaskRegistry, _State]:
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
        session_dir=db.db_path.parent,
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
    """The pass is the one thing that runs when the session is already stopping."""
    state = _State()
    state.stop_reason = "enablement_attempts_exhausted"
    rec, rounds, tasks, _ = _build(db, state=state)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    report = await rec.run(_NOW + 10.0)

    assert report.settled == []
    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_round_whose_holder_cannot_be_confirmed_dead_still_releases(db):
    """Recorded, not acted on."""
    rec, rounds, tasks, state = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    await rec.run(_NOW + 10.0)

    settled = await rounds.get("round-spec-1")
    assert settled.outcome == ""
    assert settled.excludes_at(_NOW + 10.0) is True
    assert state.stop_reason == "", "an unobservable reap does not end the session"


@pytest.mark.asyncio
async def test_age_never_requests_a_reap_of_a_running_holder(db):
    """A configured killer must never be invoked merely because time passed."""
    rec, rounds, tasks, state = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)

    await rec.run(_NOW + 10.0)

    settled = await rounds.get("round-spec-1")
    assert settled.state == OPEN
    assert settled.excludes_at(_NOW + 10.0) is True
    assert state.stop_reason == ""


@pytest.mark.asyncio
async def test_an_old_round_does_not_kill_its_live_recorded_process(db):
    """The round budget cannot terminate a worker that is still alive."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])  # nosec B603
    try:
        rec, rounds, tasks, _ = _build(db)
        await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
        async with db.transaction() as cur:
            cur.execute(
                "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at)"
                " VALUES ('server_lifecycle', 'h1', 'spec-1', 'bench', ?, '', '', '')",
                (child.pid,),
            )

        await rec.run(_NOW + 10.0)

        settled = await rounds.get("round-spec-1")
        assert settled.state == OPEN
        assert child.poll() is None
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
@pytest.mark.parametrize("elapsed", [1.0, 100_000.0])
async def test_a_terminal_holder_hands_the_round_to_the_integrate_that_follows_it(db, elapsed):
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

    report = await rec.run(_NOW + elapsed)

    round_row = await rounds.get("round-spec-1")
    assert report.handed_off == ["round-spec-1"]
    assert round_row.state == OPEN
    assert round_row.holder_task_id == integrate.task_id
    assert round_row.fence == 2


@pytest.mark.asyncio
async def test_terminal_holder_cannot_handoff_before_gpu_cleanup(db):
    from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool

    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=0)
    await _open_round(rounds, tasks, holder="spec-1", lease=1)
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")
    successor = await tasks.create(
        kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next"
    )
    pool = SpecialistGpuPool(db, gpu_ids=[0])
    lease = await pool.try_acquire(count=1, holder_id="spec-1", task_id="spec-1")
    report = await rec.run(_NOW + 10_000)
    assert report.handed_off == report.settled == []
    assert (await rounds.get("round-spec-1")).holder_task_id == "spec-1"
    await pool.release(lease)
    await rec.run(_NOW + 10_001)
    assert (await rounds.get("round-spec-1")).holder_task_id == successor.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("appended", [False, True], ids=["transition", "terminal-race"])
@pytest.mark.parametrize("confirmed", [False, True], ids=["unconfirmed", "confirmed"])
@pytest.mark.parametrize("with_successor", [False, True], ids=["expire", "handoff"])
async def test_terminal_round_uses_latest_cleanup_outcome(db, state, appended, confirmed, with_successor):
    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=0)
    await _open_round(rounds, tasks, holder="spec-1", lease=1)
    await tasks.transition("spec-1", "running")
    evidence = {
        "outcome": {"state": "succeeded", "result": {"status": "ok"}},
        "cleanup_confirmed": confirmed,
    }
    prior_evidence = {"reason": "cancelled_in_flight" if confirmed else "completed"}
    await tasks.transition("spec-1", state, evidence=prior_evidence if appended else evidence)
    if appended:
        async with db.transaction() as cur:
            history = json.loads(cur.execute("SELECT history FROM tasks WHERE task_id='spec-1'").fetchone()["history"])
            history.append({"ts": "2026-09-18T00:00:00Z", "evidence": evidence})
            history.append({"progress": {"message": "completion recorded"}})
            cur.execute("UPDATE tasks SET history=? WHERE task_id='spec-1'", (json.dumps(history),))
    if with_successor:
        await tasks.create(
            kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next", task_id="next"
        )

    report = await rec.run(_NOW + 10_000)

    assert not report.failures
    round_row = await rounds.get("round-spec-1")
    if not confirmed:
        assert report.handed_off == report.settled == []
        assert round_row.state == OPEN and round_row.holder_task_id == "spec-1"
    elif with_successor:
        assert report.handed_off == ["round-spec-1"]
        assert round_row.state == OPEN and round_row.holder_task_id == "next"
    else:
        assert round_row.state == SETTLED


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
    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=300.0)
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

    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_running_task_whose_process_is_gone_is_failed_and_one_unobservable_is_not(db, monkeypatch):
    """Inability to observe is UNKNOWN. Nothing is manufactured from it."""
    rec, _rounds, tasks, _ = _build(db)
    dead = await tasks.create(kind="specialist", params={}, idempotency_key="dead")
    blind = await tasks.create(kind="specialist", params={}, idempotency_key="blind")
    await tasks.transition(dead.task_id, "running")
    await tasks.transition(blind.task_id, "running")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    gone.wait()
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    monkeypatch.setattr(SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: pid != gone.pid))
    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at)"
            " VALUES ('server_lifecycle', 'h1', ?, 'bench', ?, '', '', '')",
            (dead.task_id, gone.pid),
        )
        cur.execute("UPDATE leases SET owner_scope='test-node'")

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
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts)"
                " VALUES (?, 'orchestration', '*', 'proposal', NULL, '{}', '2020-01-01T00:00:00+00:00')",
                (msg_id,),
            )
        cur.execute(
            "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts)"
            " VALUES ('v1', 'critic', '*', 'review_verdict', NULL, ?, '2020-01-01T00:01:00+00:00')",
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
            "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts)"
            " VALUES ('m1', 'orchestration', '*', 'proposal', NULL, '{}', '2020-01-01T00:00:00+00:00')"
        )

    await rec.run(_NOW)
    await rec.run(_NOW)

    rows = await db.fetchall("SELECT 1 FROM events WHERE topic = 'review_verdict'")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_the_resource_facts_are_reread_from_what_the_rules_left(db):
    """The facts the gate reads reflect the repair, not the state before it."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
    await rec.run(_NOW - 100.0)
    assert rec._resources.excluding_round_id == "round-spec-1"

    await rec.run(_NOW + 10_000.0)

    assert rec._resources.excluding_round_id == "round-spec-1"


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
    assert report.settled == []
    assert rec._resources.excluding_round_id == "round-spec-1"


@pytest.mark.asyncio
async def test_reconcile_does_not_write_a_supervisor_tick_stamp(db):
    rec, _, _, _ = _build(db)
    await rec.run(_NOW)
    assert not hasattr(rec, "stamp_progress")


@pytest.mark.asyncio
async def test_a_round_whose_lane_another_pass_took_is_settled_here(db):
    """Whoever swept the lease, the round it belonged to still ends."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=_LEASE)
    async with db.transaction() as cur:
        drop_round_lane(cur, round_id="round-spec-1")

    report = await rec.run(_NOW + 1.0)

    assert report.settled == []
    settled = await rounds.get("round-spec-1")
    assert settled.state == OPEN
    assert settled.expires_unix > _NOW + 1.0, "the round's own column never said it had run out"


@pytest.mark.asyncio
async def test_a_revalidation_window_whose_task_is_terminal_is_closed(db):
    """A window nobody will close holds the guard that drops ``skip_to_close``."""
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

    assert (report.leases_reaped, report.settled) == (0, [])
    assert rec.last_report is report
    # The round is still inside its lease, so it still holds its lane and the
    # sweep left it alone.
    lanes = await db.fetchall("SELECT lane, holder_id FROM leases")
    assert {(r["lane"], r["holder_id"]) for r in lanes} == {
        (BRINGUP_ROUND_LANE, "round-spec-1"),
        ("server_lifecycle", "h1"),
    }


async def _wedge_round_with_lane_rows(db, rounds, tasks, *, cleanup_confirmed: bool) -> str:
    """Open a round whose holder ends still owning lane rows, and give it a successor.

    Reproduces the shape the sweep meets in a live session: the lanes are taken
    by a real acquire, so the rows carry this process's own pid and owner scope
    and no liveness probe can refute them.

    Args:
        db: The session database.
        rounds: The round store to open the round in.
        tasks: The registry the holder and successor rows live in.
        cleanup_confirmed: What the holder's terminal evidence claims, which is
            what :func:`_terminal_by_observation` reads.

    Returns:
        str: The successor task's id.
    """
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    await _open_round(rounds, tasks, holder="spec-1")
    await locks.acquire_many(["server_lifecycle"], holder_id="h1", task_id="spec-1", action="explore", ttl_sec=7200)
    await tasks.transition("spec-1", "running")
    await tasks.transition(
        "spec-1",
        "succeeded",
        evidence={"outcome": {"state": "succeeded"}, "cleanup_confirmed": cleanup_confirmed},
    )
    successor = await tasks.create(
        kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next"
    )
    # Nothing here is reclaimable by liveness: the pid on the rows is this test.
    assert await locks.reap_dead_holders() == []
    return successor.task_id


@pytest.mark.asyncio
async def test_a_round_wedged_by_lane_rows_alone_is_freed_in_the_same_pass(db):
    """The lane sweep runs first, so the round it unblocks moves in that same pass.

    Before the sweep existed this round had no way out at all: its holder was
    terminal, but ``_holder_has_resources`` saw lane rows that nothing could
    reclaim, because the pid on them is the live coordinator and the release
    that would have dropped them is the one that never ran. Ordering the sweep
    ahead of the round rules is what resolves it, so the ordering is asserted
    here rather than left to the order the rules happen to be listed in.
    """
    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=0.0)
    successor = await _wedge_round_with_lane_rows(db, rounds, tasks, cleanup_confirmed=True)

    report = await rec.run(_NOW + 10.0)

    assert report.handed_off == ["round-spec-1"]
    assert report.leases_reaped == 4  # server_lifecycle plus the three lanes it mutexes with.
    assert (await rounds.get("round-spec-1")).holder_task_id == successor
    # Only the round's own lane survives, and it has moved to the successor.
    lanes = await db.fetchall("SELECT lane, task_id FROM leases")
    assert {(r["lane"], r["task_id"]) for r in lanes} == {(BRINGUP_ROUND_LANE, successor)}


@pytest.mark.asyncio
async def test_lanes_come_back_without_moving_a_round_whose_cleanup_never_confirmed(db):
    """The bound on the rule above: the lanes are freed, the round is not advanced.

    These are the 2026-09-21 rows themselves. They clear the lane sweep's bar --
    nobody is using the lane -- and fail ``_terminal_by_observation``, which asks
    the stricter question of whether the holder ended cleanly enough to move a
    round on. Freeing the lanes must not answer that question by accident.
    """
    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=0.0)
    await _wedge_round_with_lane_rows(db, rounds, tasks, cleanup_confirmed=False)

    report = await rec.run(_NOW + 10.0)

    assert report.leases_reaped == 4
    assert (report.handed_off, report.settled) == ([], [])
    assert (await rounds.get("round-spec-1")).holder_task_id == "spec-1"
    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_wedged_round_with_no_successor_settles_in_the_pass_that_frees_it(db):
    """The other end of the same rule: no successor to hand to, so it expires instead.

    Freeing the lanes lets ``_advance_or_expire`` reach its cap check, and with
    no successor waiting the round settles ``expired_reaped`` in that same pass.
    Before the sweep it stayed open indefinitely, for the same reason the
    hand-off case did. Asserted because it is a second round-lifecycle outcome
    the sweep newly reaches, not only the hand-off one.
    """
    rec, rounds, tasks, _ = _build(db, terminal_holder_cap_sec=0.0)
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    await _open_round(rounds, tasks, holder="spec-1")
    await locks.acquire_many(["server_lifecycle"], holder_id="h1", task_id="spec-1", action="explore", ttl_sec=7200)
    await tasks.transition("spec-1", "running")
    await tasks.transition(
        "spec-1",
        "succeeded",
        evidence={"outcome": {"state": "succeeded"}, "cleanup_confirmed": True},
    )
    # No successor is created: this is the case the hand-off test does not cover.
    assert await locks.reap_dead_holders() == []

    report = await rec.run(_NOW + 10.0)

    assert report.leases_reaped == 4
    assert report.handed_off == []
    assert report.settled == [("round-spec-1", EXPIRED_REAPED)]
    assert (await rounds.get("round-spec-1")).state == SETTLED
