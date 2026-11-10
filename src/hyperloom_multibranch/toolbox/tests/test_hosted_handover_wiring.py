# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted toolbox gives its sandbox pool what make-before-break needs: the grant to count the room in, what the
branch leases hold of it, and, after every change of the queue, which tests wait to start or run."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import threading

import pytest

from hyperloom.measure.sandboxes import AttemptTag, Handover, PoolLimits, SandboxPool
from hyperloom.sandbox.socket import CreateOptions, SandboxInfo, SandboxQuotaExceeded
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import LeaseCharge
from hyperloom_multibranch.toolbox.hosted_leases import DevSession, GrantLimits, HostedBranchLeases, LeaseTuning
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions, handover_for, parallel_tests_for
from hyperloom_multibranch.toolbox.queue import QueuedTest, State
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import _GRANT, _toolbox


_THREE = {**_GRANT, "max_sandboxes": 3, "max_gpus": 3}  # three tests of TP=1 fit
_SUBMISSION = {**_GRANT, "max_sandboxes": 3, "max_gpus": 2}  # the grant of a hosted submission: two fit


def test_the_backend_counts_the_grant_and_what_the_leases_hold(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits=_THREE)
    handover = toolbox.backend._config.handover
    assert handover == Handover(3, 3)
    assert toolbox.leases is not None and handover.reserved == toolbox.leases.reserved
    assert toolbox.backend.pool._handover is handover
    assert toolbox.backend._config.max_parallel_tests == 2, "two tests, and the room of the kept sandbox"
    assert toolbox.leases.tests_at_once() == 2
    assert toolbox.leases._make_room == toolbox.backend.pool.make_room
    toolbox.leases._promised["b1"] = 1  # a branch's dev sandbox holds a card of the grant
    assert handover.reserved() == (1, 1)
    assert toolbox.leases.tests_at_once() == 2, "two fit beside the lease, and both run: no room is left for a kept one"


@pytest.mark.parametrize(
    ("grant", "keep", "handover", "parallel"),
    [
        (_THREE, False, None, 3),  # off: every test the grant fits runs, and each sandbox is closed when it ends
        (_THREE, True, Handover(3, 3), 2),  # on where three fit: one test's room is held for the kept sandbox
        (_SUBMISSION, True, None, 2),  # on where two fit: no sandbox is kept, which would halve the tests at once
        (_SUBMISSION, False, None, 2),
    ],
)
def test_the_make_before_break_switch(
    tmp_path: Path, grant: dict[str, float], keep: bool, handover: Handover | None, parallel: int
) -> None:
    options = HostedOptions(framework_configs="", make_before_break=keep)
    toolbox = _toolbox(None, tmp_path, options=options, grant_limits=grant)
    assert toolbox.backend._config.handover == handover and toolbox.backend.pool._handover == handover
    assert toolbox.backend._config.max_parallel_tests == parallel
    assert toolbox.leases is not None and toolbox.leases.tests_at_once() == parallel
    assert (toolbox.leases._make_room is not None) == (handover is not None)


def test_a_grant_that_fits_one_test_keeps_no_sandbox(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3, "max_gpus": 1})
    assert toolbox.backend._config.handover is None
    assert toolbox.backend._config.max_parallel_tests == 1


def test_without_grant_limits_no_sandbox_is_kept(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits=None)
    assert toolbox.backend._config.handover is None
    grant = GrantLimits(max_sandboxes=3, max_gpus=3, gpu_hours=1.0, max_runtime_sec=7200)
    assert handover_for(grant, 1) == Handover(3, 3) and handover_for(grant, 1).reserved() == (0, 0)
    assert handover_for(None, 1) is None


def test_a_kept_sandbox_outlasts_several_gateway_cuts_of_the_next_create() -> None:
    # Hosted smoke #12: the service's gateway cut the blocking create of a sandbox that queued for a node with a 502
    # about every 600 s, and the create was sent again with the same key. One cycle is no wait to keep a node for.
    gateway_cut_sec = 600
    grant = GrantLimits(max_sandboxes=3, max_gpus=3, gpu_hours=1.0, max_runtime_sec=7200)
    assert handover_for(grant, 1).timeout_sec >= 3 * gateway_cut_sec
    assert handover_for(grant, 1, None, 2400).timeout_sec == 2400, "the argument is the timeout"
    short = dataclasses.replace(grant, max_runtime_sec=900)
    assert handover_for(short, 1).timeout_sec == 900, "no sandbox of the grant lives past max_runtime_sec"


async def test_the_pool_hears_which_tests_wait_or_run_after_every_change_of_the_queue(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3, "max_gpus": 1})  # one at a time
    heard: list[frozenset[str]] = []
    toolbox.backend.pool.note_upcoming = heard.append  # type: ignore[method-assign]
    blocked = asyncio.Event()

    async def run(item: QueuedTest) -> None:
        await blocked.wait()

    toolbox._run = run  # type: ignore[method-assign]
    for seq, request_id in enumerate(("r1", "r2", "r3")):
        state = State.PENDING_BUILD if request_id == "r3" else State.QUEUED
        toolbox._queue.push(QueuedTest(dataclasses.replace(req(), request_id=request_id), 2, seq, state=state))
    toolbox._pump()
    assert heard[-1] == {"r1", "r2"}, "r1 runs and r2 waits; r3 waits for its builds, not for a sandbox"
    toolbox._queue.remove("r1")
    toolbox._pump()
    assert heard[-1] == {"r2"}
    toolbox._queue.remove("r2")
    toolbox._pump()
    assert heard[-1] == frozenset()
    blocked.set()
    await asyncio.gather(*toolbox._tasks.values())


class _GrantService:
    """A sandbox service that enforces a grant as the hosted one does (``sandbox/pkg/grant/quota.go``): every live
    sandbox counts against ``max_sandboxes`` and its cards against ``max_gpus``, from its create until it ends; one
    more is refused with ``grant_quota_exceeded``. Records ``("running", id)`` and ``("close", id)`` in order, and the
    most sandboxes and GPUs that were ever live at once. With ``hold`` set, the next create is charged and listed at once
    and returns when ``hold`` is, as a create that queues for a node; ``pending`` is set once it waits."""

    name = "grant"

    def __init__(self, grant: GrantLimits, cards: int) -> None:
        self._grant, self._cards = grant, cards
        self.live: dict[str, _GrantSandbox] = {}
        self.events: list[tuple[str, str]] = []
        self.refusals = 0
        self.peak = (0, 0)
        self.hold: threading.Event | None = None
        self.pending = threading.Event()
        self._lock = threading.Lock()
        self._count = 0

    def create(self, template, *, role, timeout_sec, metadata, options=CreateOptions()):
        with self._lock:
            if (
                len(self.live) + 1 > self._grant.max_sandboxes
                or (len(self.live) + 1) * self._cards > self._grant.max_gpus
            ):
                self.refusals += 1
                raise SandboxQuotaExceeded(f"grant_quota_exceeded: {len(self.live)} running")
            self._count += 1
            sandbox = _GrantSandbox(self, f"sbx-{self._count}", dict(metadata), role)
            self.live[sandbox.sandbox_id] = sandbox
            self.peak = max(self.peak, (len(self.live), len(self.live) * self._cards))
            hold, self.hold = self.hold, None  # the next create is held, later ones are not
        if hold is not None:
            self.pending.set()
            assert hold.wait(10), "the test never let the create through"
        with self._lock:
            self.events.append(("running", sandbox.sandbox_id))
        return sandbox

    def ended(self, sandbox_id: str) -> None:
        with self._lock:
            if self.live.pop(sandbox_id, None) is not None:
                self.events.append(("close", sandbox_id))

    def connect(self, sandbox_id, *, role):
        return self.live[sandbox_id]

    def find(self, metadata, *, role):
        with self._lock:
            infos = [sandbox.info() for sandbox in self.live.values() if sandbox.role == role]
        return [info for info in infos if all(info.metadata.get(k) == v for k, v in metadata.items())]


class _GrantSandbox:
    def __init__(self, service: _GrantService, sandbox_id: str, metadata: dict[str, str], role: str) -> None:
        self.sandbox_id, self._service, self._metadata, self.role = sandbox_id, service, metadata, role

    def info(self) -> SandboxInfo:
        return SandboxInfo(self.sandbox_id, "t", dict(self._metadata))

    def kill(self) -> None:
        self._service.ended(self.sandbox_id)


#: The grants the hosted toolbox runs under, by (max_sandboxes, max_gpus), with the TP of their sessions.
_PRODUCTION = [((3, 3), 1), ((8, 8), 1), ((8, 8), 2), ((4, 16), 4)]


def _production_pool(sandboxes: int, gpus: int, tp: int) -> tuple[_GrantService, SandboxPool, int]:
    """The pool as ``build_hosted_toolbox`` sizes it, over a service that enforces the grant."""
    grant = GrantLimits(max_sandboxes=sandboxes, max_gpus=gpus, gpu_hours=100.0, max_runtime_sec=7200)
    handover = handover_for(grant, tp)
    assert handover is not None, "the grant fits three tests"
    parallel = parallel_tests_for(grant, tp, handover)
    service = _GrantService(grant, tp)
    pool = SandboxPool(
        service, {tp: "runner"}, run_id="run-1", limits=PoolLimits(max_parallel_tests=parallel, handover=handover)
    )
    return service, pool, parallel


@pytest.mark.parametrize(("grant", "tp"), _PRODUCTION)
def test_under_production_sizing_a_full_pool_hands_its_sandbox_to_the_waiting_test(
    grant: tuple[int, int], tp: int
) -> None:
    """Every slot is busy and a test waits, which is when the toolbox queues one: the sandbox of the test that finishes
    is kept, the waiting test's sandbox runs beside it within the grant, and only then is the old one closed."""
    service, pool, parallel = _production_pool(*grant, tp)
    running = [pool.acquire(tp, AttemptTag(f"req-{n}", 1), timeout_sec=60) for n in range(parallel)]
    waiting = f"req-{parallel}"
    pool.note_upcoming({*(f"req-{n}" for n in range(parallel)), waiting})
    pool.hand_over(running[0])
    assert pool.parked == running[0].sandbox_id, "the finished test's sandbox is kept for the waiting one"
    nxt = pool.acquire(tp, AttemptTag(waiting, 1), timeout_sec=5)
    assert service.events[-2:] == [("running", nxt.sandbox_id), ("close", running[0].sandbox_id)]
    assert service.refusals == 0, "the kept sandbox and the next one fit the grant together"
    assert service.peak == (parallel + 1, (parallel + 1) * tp)
    assert service.peak[0] <= grant[0] and service.peak[1] <= grant[1]
    for runner in (*running[1:], nxt):
        pool.release(runner)
    assert service.live == {} and pool.active == 0


def test_the_hosted_submission_grant_keeps_no_sandbox_and_runs_every_test_it_fits() -> None:
    """(3, 2) fits two tests at TP=1: holding one's room for a kept sandbox would halve the tests at once. At TP=2 it
    fits one, and there is no room for a kept sandbox at all."""
    grant = GrantLimits(max_sandboxes=3, max_gpus=2, gpu_hours=100.0, max_runtime_sec=7200)
    assert handover_for(grant, 1) is None and parallel_tests_for(grant, 1, None) == 2
    assert parallel_tests_for(grant, 1, Handover(3, 2)) == 2, "even a handover given here does not halve the tests"
    assert handover_for(grant, 2) is None and parallel_tests_for(grant, 2, None) == 1


async def test_a_branch_lease_takes_the_room_of_the_kept_sandbox_under_production_sizing() -> None:
    """Grant (3, 3), TP=1, two tests at once: req-c runs, and req-a's sandbox is kept while req-b's create queues for a
    node, so all three GPUs are charged. A branch's dev lease that does not wait is promised a card: the lease closes
    the kept sandbox before its create and gets its card, where it would have been refused for the quota."""
    grant = GrantLimits(max_sandboxes=3, max_gpus=3, gpu_hours=100.0, max_runtime_sec=7200)
    service = _GrantService(grant, 1)
    tuning = LeaseTuning(wait_sec=1.0, first_backoff_sec=0.02, max_backoff_sec=0.05)
    leases = HostedBranchLeases(service, grant, DevSession("run-1", 1, {1: "dev-1"}), tuning)
    handover = handover_for(grant, 1, leases)
    assert handover is not None
    pool = SandboxPool(
        service,
        {1: "runner"},
        run_id="run-1",
        limits=PoolLimits(max_parallel_tests=parallel_tests_for(grant, 1, handover), handover=handover),
    )
    leases.keep_room_for_handover(pool.make_room)
    assert pool._max == 2, "two tests at once, and the room of the kept sandbox"
    other = pool.acquire(1, AttemptTag("req-c", 1), timeout_sec=60)
    first = pool.acquire(1, AttemptTag("req-a", 1), timeout_sec=60)
    pool.note_upcoming({"req-a", "req-b", "req-c"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id
    service.hold = held = threading.Event()
    second = asyncio.create_task(asyncio.to_thread(pool.acquire, 1, AttemptTag("req-b", 1), timeout_sec=60))
    await asyncio.to_thread(service.pending.wait, 10)
    assert len(service.live) == 3, "req-c's, the kept one and req-b's queued one hold the grant's three GPUs"
    lease = await leases.lease("b1", 1, charge=LeaseCharge(LeaseKind.DEV, "b1"), wait_sec=0)
    assert first.sandbox_id not in service.live and service.refusals == 0, "the lease took the kept sandbox's room"
    held.set()
    runner = await second
    await leases.release(lease.lease_id)
    pool.release(runner)
    pool.release(other)
    assert service.live == {} and pool.active == 0


async def test_a_start_ends_what_an_earlier_process_left_before_any_test(tmp_path: Path) -> None:
    """A process that died while it kept a sandbox for its next test left that sandbox to nobody: the restarted
    toolbox ends it (``HostedMeasurementBackend.end_leftovers``) along with the leases' leftovers, before any test."""
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3, "max_gpus": 2})
    assert toolbox.leases is not None
    ended: list[str] = []
    toolbox.leases.end_leftovers = lambda: ended.append("leases")  # type: ignore[method-assign]
    toolbox.backend.end_leftovers = lambda: ended.append("measurement") or []  # type: ignore[method-assign]
    toolbox.backend.close = lambda: None  # type: ignore[method-assign]  # no service to ask
    await toolbox.start()
    try:
        assert ended == ["leases", "measurement"]
    finally:
        await toolbox.close()
