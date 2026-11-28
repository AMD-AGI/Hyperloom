# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The repair pass, exercised against the states it exists to get a session out of."""

from __future__ import annotations

import json
import sqlite3
import subprocess  # nosec B404 - starts and kills a sleep, to exercise the reaper
import sys
import time

import pytest

from hyperloom.orchestrator.bringup import reconcile
from hyperloom.orchestrator.bringup.reconcile import (
    ReconcileLinks,
    Reconciler,
    ReconcileStores,
)
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.policy.bringup_round import bringup_round_denial
from hyperloom.orchestrator.state.round_store import (
    OPEN,
    SETTLED,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec

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
        self.revalidation_generation = 0


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


@pytest.fixture
def db(tmp_path):
    """A real session database."""
    conn = SqliteConnection(tmp_path / "coordinator.db")
    yield conn
    conn.close()


def _build(db, *, state=None, result_pending=None) -> tuple[Reconciler, RoundStore, TaskRegistry, _State]:
    """A reconciler over ``db``, with the pieces a test needs to inspect."""
    rounds = RoundStore(db)
    tasks = TaskRegistry(db)
    shared = state or _State()
    rec = Reconciler(
        ReconcileStores(rounds=rounds, tasks=tasks, locks=SqliteLeaseBackend(db)),
        shared,
        links=ReconcileLinks(
            session_dir=db.db_path.parent,
            result_pending=result_pending or (lambda _task_id: False),
        ),
    )
    return rec, rounds, tasks, shared


async def _open_round(rounds: RoundStore, tasks: TaskRegistry, *, holder: str, lease: float = _LEASE) -> None:
    """Open a round held by a real task row."""
    await tasks.create_or_return_existing(
        TaskSpec(kind="specialist", params={}, idempotency_key=f"k:{holder}", task_id=holder)
    )
    result = await rounds.open(
        f"round-{holder}",
        now_unix=_NOW,
        holder_task_id=holder,
        lease_sec=lease,
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
    assert settled.state == OPEN
    assert state.stop_reason == "", "an unobservable reap does not end the session"


@pytest.mark.asyncio
async def test_an_old_round_does_not_kill_its_live_recorded_process(db):
    """The round budget cannot terminate a worker that is still alive."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])  # nosec B603
    try:
        rec, rounds, tasks, _ = _build(db)
        await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
        async with db.transaction() as cur:
            cur.execute(
                "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at)"
                " VALUES ('server_lifecycle', 'h1', 'spec-1', 'bench', ?, '', '')",
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
    integrate, _ = await tasks.create_or_return_existing(
        TaskSpec(
            kind="integrate_patch",
            params={"specialist_task_id": "spec-1"},
            idempotency_key="integrate:1",
            lease_ttl_sec=900,
        ),
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

    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1)
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")
    successor, _ = await tasks.create_or_return_existing(
        TaskSpec(kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next")
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
    rec, rounds, tasks, _ = _build(db)
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
        await tasks.create_or_return_existing(
            TaskSpec(
                kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next", task_id="next"
            )
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
async def test_a_terminal_holder_with_nothing_following_it_expires_on_its_cap(db):
    """Not on the tick it went terminal -- the successor is created by a later one."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")

    await rec.run(_NOW + 10.0)
    assert (await rounds.get("round-spec-1")).state == OPEN

    await rec.run(_NOW + 600.0)
    assert (await rounds.get("round-spec-1")).state == SETTLED


@pytest.mark.asyncio
async def test_a_running_task_whose_process_is_gone_is_failed_and_one_unobservable_is_not(db, monkeypatch):
    """Inability to observe is UNKNOWN. Nothing is manufactured from it."""
    rec, _rounds, tasks, _ = _build(db)
    dead, _ = await tasks.create_or_return_existing(TaskSpec(kind="specialist", params={}, idempotency_key="dead"))
    blind, _ = await tasks.create_or_return_existing(TaskSpec(kind="specialist", params={}, idempotency_key="blind"))
    await tasks.transition(dead.task_id, "running")
    await tasks.transition(blind.task_id, "running")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    gone.wait()
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    monkeypatch.setattr(SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: pid != gone.pid))
    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at)"
            " VALUES ('server_lifecycle', 'h1', ?, 'bench', ?, '', '')",
            (dead.task_id, gone.pid),
        )
        cur.execute("UPDATE leases SET owner_scope='test-node'")

    report = await rec.run(_NOW)

    assert report.failed_tasks == [dead.task_id]
    assert (await tasks.get(dead.task_id)).state == "failed"
    assert (await tasks.get(blind.task_id)).state == "running"


@pytest.mark.asyncio
async def test_a_baseline_is_admitted_once_the_pass_has_ended_an_abandoned_round(db):
    """The baseline rule reads the round the repair left, so the pass that ends a round is what admits the baseline."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1")
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")
    await rec.run(_NOW + 10.0)
    assert await bringup_round_denial("baseline", rounds) is not None

    await rec.run(_NOW + 600.0)

    assert (await rounds.get("round-spec-1")).state == SETTLED
    assert await bringup_round_denial("baseline", rounds) is None


@pytest.mark.asyncio
async def test_a_rule_that_raises_does_not_stop_the_rules_after_it(db):
    """Isolation, not admission: every rule is attempted on every tick."""
    rec, rounds, tasks, _ = _build(db)
    await _open_round(rounds, tasks, holder="spec-1", lease=1.0)
    await tasks.transition("spec-1", "running")
    await tasks.transition("spec-1", "succeeded")

    async def _boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    rec._reap_leases = _boom

    report = await rec.run(_NOW + 600.0)

    assert report.failures == ["_boom"]
    assert [round_id for round_id, _outcome in report.settled] == ["round-spec-1"]


@pytest.mark.asyncio
async def test_a_rule_that_raises_a_bug_fails_the_pass(db):
    """Only storage errors are isolated; anything else reaches the tick boundary."""
    rec, _, _, _ = _build(db)

    async def _boom(*_a, **_k):
        raise RuntimeError("rule failed")

    rec._reap_leases = _boom

    with pytest.raises(RuntimeError, match="^rule failed$"):
        await rec.run(_NOW)


@pytest.mark.asyncio
async def test_a_revalidation_window_whose_task_is_terminal_is_closed(db):
    """A window nobody will close holds the guard on the phase exit."""
    state = _State()
    state.enablement.validation_pending = True
    state.enablement.revalidation_task_id = "reval-1"
    rec, _, tasks, _ = _build(db, state=state)
    await tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="k:reval-1", task_id="reval-1")
    )
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
    await tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="k:reval-1", task_id="reval-1")
    )

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
            "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at)"
            " VALUES ('server_lifecycle', 'h1', 'other', 'bench', 0, '', '')",
        )

    report = await rec.run(_NOW + 1.0)

    assert (report.leases_reaped, report.settled) == (0, [])
    assert rec.last_report is report
    lanes = await db.fetchall("SELECT lane, holder_id FROM leases")
    assert {(r["lane"], r["holder_id"]) for r in lanes} == {("server_lifecycle", "h1")}
    # The round is still inside its lease, so the sweep left it alone.
    assert (await rounds.get("round-spec-1")).state == OPEN


async def _wedge_round_with_lane_rows(db, rounds, tasks, *, cleanup_confirmed: bool) -> str:
    """Open a round whose holder ends still owning lane rows, and give it a successor.

    Reproduces the shape the sweep meets in a live session: the lanes are taken
    by a real acquire, so the rows carry this process's own pid and owner scope
    and no liveness probe can refute them.

    Args:
        db: The session database.
        rounds: The round store to open the round in.
        tasks: The registry the holder and successor rows live in.
        cleanup_confirmed: What the holder's terminal evidence claims. Both the
            lane sweep and :func:`_terminal_by_observation` read it, for their
            two different questions.

    Returns:
        str: The successor task's id.
    """
    locks = SqliteLeaseBackend(db)
    await _open_round(rounds, tasks, holder="spec-1")
    await locks.acquire_many(["server_lifecycle"], holder_id="h1", task_id="spec-1", action="explore", ttl_sec=7200)
    await tasks.transition("spec-1", "running")
    await tasks.transition(
        "spec-1",
        "succeeded",
        evidence={"outcome": {"state": "succeeded"}, "cleanup_confirmed": cleanup_confirmed},
    )
    successor, _ = await tasks.create_or_return_existing(
        TaskSpec(kind="integrate_patch", params={"specialist_task_id": "spec-1"}, idempotency_key="next")
    )
    # Nothing here is reclaimable by liveness: the pid on the rows is this test.
    assert await locks.reap_dead_holders() == []
    return successor.task_id


@pytest.mark.asyncio
async def test_a_round_whose_holder_left_cleanup_unconfirmed_keeps_everything(db, monkeypatch):
    """The bound on the rule above: nothing is freed and nothing is advanced.

    This is the shape of the 2026-09-21 rows. Nothing resolves them, by design:
    no probe decides that a lane is free, because every identity available here
    is one a served process can leave. The holder also fails
    ``_terminal_by_observation``, which asks the stricter question of whether it
    ended cleanly enough to move a round on. A pass that freed these lanes would
    be freeing a lane whose work may still be running.
    """
    monkeypatch.setattr(reconcile, "TERMINAL_HOLDER_CAP_SEC", 0.0)
    rec, rounds, tasks, _ = _build(db)
    await _wedge_round_with_lane_rows(db, rounds, tasks, cleanup_confirmed=False)

    report = await rec.run(_NOW + 10.0)

    assert report.leases_reaped == 0
    assert (report.handed_off, report.settled) == ([], [])
    assert {r["lane"] for r in await db.fetchall("SELECT lane FROM leases")} == {
        "server_lifecycle",
        "benchmark_lane",
        "profile_lane",
        "gpu_research_lane",
    }
    assert (await rounds.get("round-spec-1")).holder_task_id == "spec-1"
    assert (await rounds.get("round-spec-1")).state == OPEN


@pytest.mark.asyncio
async def test_a_round_wedged_by_lane_rows_stays_wedged_and_is_reported(db, monkeypatch):
    """The cost of refusing to guess, pinned one level up.

    A holder that ended with its cleanup unconfirmed keeps its lane, so a round
    whose only remaining obstacle is that lane no longer resolves itself. Seven
    rounds of review showed every cheap proof of "the lane is free" to be a
    proxy a served process slips out of, and releasing a lane wrongly puts two
    rounds on the same cards. So the round waits, and the operator is told which
    lane to look at.
    """
    monkeypatch.setattr(reconcile, "TERMINAL_HOLDER_CAP_SEC", 0.0)
    rec, rounds, tasks, _ = _build(db)
    await _wedge_round_with_lane_rows(db, rounds, tasks, cleanup_confirmed=True)

    report = await rec.run(_NOW + 10.0)

    assert report.leases_reaped == 0
    assert (report.handed_off, report.settled) == ([], [])
    assert (await rounds.get("round-spec-1")).state == OPEN
    # The lane rows are still there, and counted for the operator.
    assert report.leases_unverifiable >= 1


@pytest.mark.asyncio
async def test_a_revalidation_window_whose_result_is_still_to_be_booked_is_left_alone(db):
    """A terminal row whose result the dispatcher has not booked yet still reports."""
    state = _State()
    state.enablement.validation_pending = True
    state.enablement.revalidation_task_id = "reval-1"
    owed = {"reval-1"}
    rec, _, tasks, _ = _build(db, state=state, result_pending=lambda task_id: task_id in owed)
    await tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="k:reval-1", task_id="reval-1")
    )
    await tasks.transition("reval-1", "running")
    await tasks.transition("reval-1", "succeeded")

    report = await rec.run(_NOW + 1.0)

    assert report.closed_windows == []
    assert state.enablement.validation_pending is True
    assert state.enablement.revalidation_task_id == "reval-1"

    owed.clear()
    report = await rec.run(_NOW + 2.0)

    assert report.closed_windows == ["reval-1"]
    assert state.enablement.validation_pending is False
