# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make-before-break between two tests: the sandbox a finished test handed over is kept until the next test's sandbox
runs, so that the cluster's queue does not give its node to another job in between, and it is never leaked.

The pool here runs over a fake service whose creates can be held open, refused for the quota or failed, and which
records every create and close in order.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import pytest

from hyperloom.measure.sandboxes import (
    HANDOVER_TIMER_NAME,
    AttemptTag,
    Handover,
    PoolClosed,
    PoolLimits,
    RunnerSandbox,
    SandboxPool,
)
from hyperloom.sandbox.socket import CreateOptions, SandboxInfo, SandboxQuotaExceeded, SandboxUnavailable


class _Sandbox:
    def __init__(self, service: _Service, sandbox_id: str, metadata: dict[str, str]) -> None:
        self.sandbox_id = sandbox_id
        self._service = service
        self._metadata = metadata

    def info(self) -> SandboxInfo:
        return SandboxInfo(self.sandbox_id, "t", dict(self._metadata))

    def kill(self) -> None:
        if self._service.kill_error is not None:
            raise self._service.kill_error
        self._service.closed(self.sandbox_id)


class _Service:
    """A sandbox service that records ``("create", id)``, ``("running", id)`` and ``("close", id)`` in order.

    ``hold`` makes the next create wait until it is set, ``quota`` refuses a create while that many run, ``fail``
    makes a create raise, ``kill_error`` makes a close raise, and ``on_find`` runs at the start of every ``find``.
    """

    name = "fake"

    def __init__(self) -> None:
        self.live: dict[str, _Sandbox] = {}
        self.events: list[tuple[str, str]] = []
        self.hold: threading.Event | None = None
        self.creating = threading.Event()
        self.quota: int | None = None
        self.refusals = 0
        self.fail: Exception | None = None
        self.kill_error: Exception | None = None
        self.on_find: Callable[[], None] = lambda: None
        self._lock = threading.Lock()
        self._count = 0

    def create(self, template, *, role, timeout_sec, metadata, options=CreateOptions()):
        self.creating.set()
        if self.hold is not None:
            assert self.hold.wait(10), "the test never let the create through"
        if self.fail is not None:
            raise self.fail
        with self._lock:
            if self.quota is not None and len(self.live) >= self.quota:
                self.refusals += 1
                raise SandboxQuotaExceeded("grant_quota_exceeded")
            self._count += 1
            sandbox = _Sandbox(self, f"sbx-{self._count}", dict(metadata))
            self.live[sandbox.sandbox_id] = sandbox
            self.events.append(("running", sandbox.sandbox_id))
        return sandbox

    def closed(self, sandbox_id: str) -> None:
        with self._lock:
            if self.live.pop(sandbox_id, None) is not None:
                self.events.append(("close", sandbox_id))

    def connect(self, sandbox_id, *, role):
        return self.live[sandbox_id]

    def find(self, metadata, *, role):
        self.on_find()
        with self._lock:
            infos = [sandbox.info() for sandbox in self.live.values()]
        return [info for info in infos if all(info.metadata.get(k) == v for k, v in metadata.items())]


def _pool(service: _Service, *, grant: tuple[int, int] | None = (3, 2), timeout: float = 30.0, **limits) -> SandboxPool:
    handover = None if grant is None else Handover(grant[0], grant[1], timeout_sec=timeout)
    limits.setdefault("max_parallel_tests", 1)
    return SandboxPool(service, {1: "runner-1"}, run_id="run-1", limits=PoolLimits(handover=handover, **limits))


def _acquire(pool: SandboxPool, request_id: str, **kwargs) -> RunnerSandbox:
    kwargs.setdefault("timeout_sec", 60)
    return pool.acquire(1, AttemptTag(request_id, 1), **kwargs)


def _until(condition: Callable[[], bool], what: str, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def _in_thread(target: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
    """Run ``target`` as a test's worker thread; what it returns or raises lands in the list."""
    landed: list[object] = []

    def run() -> None:
        try:
            landed.append(target())
        except Exception as exc:  # noqa: BLE001 - the test reads what the worker raised
            landed.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, landed


def _joined(worker: threading.Thread) -> None:
    worker.join(10)
    assert not worker.is_alive(), "the worker did not end"


def test_the_next_sandbox_runs_before_the_handed_over_one_is_closed() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-1", "req-2"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id and pool.active == 0, "kept, and its slot is free for the next test"
    second = _acquire(pool, "req-2")  # max_parallel_tests is 1: the kept sandbox is no second test
    assert pool.active == 1, "closing the kept sandbox gave back no slot of the test that runs"
    assert service.events == [
        ("running", first.sandbox_id),
        ("running", second.sandbox_id),
        ("close", first.sandbox_id),
    ]
    assert pool.parked is None and set(service.live) == {second.sandbox_id}
    pool.release(second)
    assert service.live == {} and pool.active == 0


def test_a_grant_without_room_for_two_sandboxes_releases_then_creates(caplog) -> None:
    for grant in ((1, 8), (3, 1)):
        service = _Service()
        pool = _pool(service, grant=grant)
        first = _acquire(pool, "req-1")
        pool.note_upcoming({"req-1", "req-2"})
        with caplog.at_level(logging.INFO, logger="hyperloom.measure.sandboxes"):
            pool.hand_over(first)
        assert pool.parked is None
        assert "has no room for 2 sandboxes" in caplog.text
        second = _acquire(pool, "req-2")
        assert service.events == [
            ("running", first.sandbox_id),
            ("close", first.sandbox_id),
            ("running", second.sandbox_id),
        ], grant
        caplog.clear()


def test_the_grant_counts_what_the_branch_leases_hold(caplog) -> None:
    service = _Service()
    handover = Handover(3, 2, reserved=lambda: (1, 1))  # a dev sandbox holds one GPU: no room for a second test's
    pool = SandboxPool(
        service, {1: "runner-1"}, run_id="run-1", limits=PoolLimits(max_parallel_tests=1, handover=handover)
    )
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    with caplog.at_level(logging.INFO, logger="hyperloom.measure.sandboxes"):
        pool.hand_over(first)
    assert pool.parked is None and service.live == {}
    assert "1 sandboxes, 1 GPUs reserved" in caplog.text


def test_without_a_next_test_the_sandbox_is_released(caplog) -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-1"})  # only the test that just finished
    with caplog.at_level(logging.INFO, logger="hyperloom.measure.sandboxes"):
        pool.hand_over(first)
    assert pool.parked is None and service.live == {}
    assert "no test waits for a sandbox" in caplog.text


def test_without_a_grant_nothing_is_kept() -> None:
    service = _Service()
    pool = _pool(service, grant=None)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    assert pool.parked is None and service.live == {}


def test_a_next_test_that_goes_away_closes_the_kept_sandbox() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id
    pool.note_upcoming(set())  # req-2 was withdrawn before it asked for a sandbox
    _until(lambda: service.live == {}, "the kept sandbox to close")
    assert pool.parked is None


def test_one_sandbox_is_kept_at_a_time() -> None:
    service = _Service()
    pool = _pool(service, grant=(8, 8), max_parallel_tests=2)
    first, second = _acquire(pool, "req-1"), _acquire(pool, "req-2")
    pool.note_upcoming({"req-3", "req-4"})
    pool.hand_over(first)
    pool.hand_over(second)
    assert pool.parked == first.sandbox_id and set(service.live) == {first.sandbox_id}


def test_a_next_sandbox_that_never_runs_closes_the_old_one_after_the_timeout(caplog) -> None:
    service = _Service()
    pool = _pool(service, timeout=0.3)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    service.hold = threading.Event()
    with caplog.at_level(logging.INFO, logger="hyperloom.measure.sandboxes"):
        worker, landed = _in_thread(lambda: _acquire(pool, "req-2"))
        _until(lambda: first.sandbox_id not in service.live, "the timeout to close the kept sandbox")
        assert worker.is_alive() and not landed, "the next test keeps waiting for its own sandbox"
        assert "keeps waiting for its own" in caplog.text
        service.hold.set()
        _joined(worker)
    [second] = landed
    assert isinstance(second, RunnerSandbox)
    assert service.events == [
        ("running", first.sandbox_id),
        ("close", first.sandbox_id),
        ("running", second.sandbox_id),
    ]
    pool.release(second)
    assert service.live == {} and pool.active == 0


def test_a_next_sandbox_that_fails_to_start_closes_the_old_one() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    service.fail = SandboxUnavailable("pending_timeout")
    with pytest.raises(SandboxUnavailable):
        _acquire(pool, "req-2")
    assert service.live == {} and pool.parked is None and pool.active == 0


def _timers() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name == HANDOVER_TIMER_NAME}


def test_a_session_cancelled_during_the_overlap_closes_both() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    before = _timers()
    pool.hand_over(first)
    [timer] = _timers() - before
    service.hold = threading.Event()
    worker, landed = _in_thread(lambda: _acquire(pool, "req-2"))
    _until(service.creating.is_set, "the next create to begin")
    pool.close()
    assert first.sandbox_id not in service.live
    service.hold.set()  # the create the service was making when the session stopped comes back
    _joined(worker)
    assert isinstance(landed[0], PoolClosed)
    assert service.live == {}, "the sandbox made while the session stopped is closed as well"
    timer.join(5)
    assert not timer.is_alive(), "the close stopped the timer of the kept sandbox"


def test_a_test_stopped_while_its_create_waits_for_quota_closes_the_old_one() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    service.quota = 1  # the kept sandbox fills it
    stop = threading.Event()
    worker, landed = _in_thread(lambda: _acquire(pool, "req-2", stop=stop))
    _until(lambda: service.refusals >= 1, "a refused create")
    stop.set()
    _joined(worker)
    assert isinstance(landed[0], PoolClosed)
    assert service.live == {} and pool.parked is None


def test_a_create_refused_for_quota_keeps_the_old_one_until_the_timeout() -> None:
    service = _Service()
    pool = _pool(service, timeout=0.5)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    service.quota = 1  # the kept sandbox fills the grant's quota
    began = time.monotonic()
    second = _acquire(pool, "req-2")
    assert service.refusals >= 1, "the create was refused while the old sandbox was kept"
    assert time.monotonic() - began >= 0.45, "the old sandbox was kept until the timeout"
    assert service.events == [
        ("running", first.sandbox_id),
        ("close", first.sandbox_id),
        ("running", second.sandbox_id),
    ]
    assert time.monotonic() - began < 5, "the closed sandbox woke the create at once, not after the quota retry"
    pool.release(second)
    assert service.live == {}


def test_a_create_in_flight_leaves_no_room_to_keep_a_sandbox(caplog) -> None:
    """Two tests at once on a grant of two GPUs: while the second test's sandbox is still being created, the first
    test's sandbox is not kept for a third, whose sandbox would make three of two GPUs and wait for the quota."""
    service = _Service()
    pool = _pool(service, grant=(3, 2), max_parallel_tests=2, timeout=30)
    first = _acquire(pool, "req-1")
    service.hold = threading.Event()
    worker, landed = _in_thread(lambda: _acquire(pool, "req-2"))
    _until(service.creating.is_set, "req-2's create to begin")
    pool.note_upcoming({"req-1", "req-2", "req-3"})
    with caplog.at_level(logging.INFO, logger="hyperloom.measure.sandboxes"):
        pool.hand_over(first)
    assert pool.parked is None and first.sandbox_id not in service.live
    assert "has no room for 3 sandboxes with 3 GPUs" in caplog.text
    service.hold.set()
    _joined(worker)
    [second] = landed
    assert isinstance(second, RunnerSandbox)
    service.hold, service.quota = None, 2  # the grant's two GPUs
    began = time.monotonic()
    third = _acquire(pool, "req-3")
    assert service.refusals == 0 and time.monotonic() - began < 5, "req-3's create ran at once"
    pool.release(second)
    pool.release(third)
    assert service.live == {} and pool.active == 0


def test_a_kept_sandbox_without_room_beside_two_creates_is_closed_before_the_second() -> None:
    """The kept sandbox waits for req-2's sandbox; req-3 starts before that one runs and the grant has no room for
    all three, so the kept one is closed before req-3's create instead of the service refusing it."""
    service = _Service()
    pool = _pool(service, grant=(3, 2), max_parallel_tests=2, timeout=30)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-1", "req-2", "req-3"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id
    service.hold, service.quota = threading.Event(), 2
    second_worker, second = _in_thread(lambda: _acquire(pool, "req-2"))
    _until(service.creating.is_set, "req-2's create to begin")
    third_worker, third = _in_thread(lambda: _acquire(pool, "req-3"))
    _until(lambda: first.sandbox_id not in service.live, "the kept sandbox to close before req-3's create")
    assert pool.parked is None
    service.hold.set()
    _joined(second_worker)
    _joined(third_worker)
    assert service.refusals == 0, "no create was refused for the grant's quota"
    assert service.events[:2] == [("running", first.sandbox_id), ("close", first.sandbox_id)]
    for runner in (*second, *third):
        assert isinstance(runner, RunnerSandbox)
        pool.release(runner)
    assert service.live == {} and pool.active == 0


def test_a_grant_that_shrank_after_the_hand_over_closes_the_kept_sandbox_before_the_create() -> None:
    service = _Service()
    reserved = [0, 0]
    handover = Handover(3, 2, reserved=lambda: (reserved[0], reserved[1]))
    pool = SandboxPool(
        service, {1: "runner-1"}, run_id="run-1", limits=PoolLimits(max_parallel_tests=1, handover=handover)
    )
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id
    reserved[:] = [1, 1]  # a branch's dev sandbox took a card of the grant meanwhile
    second = _acquire(pool, "req-2")
    assert service.events == [
        ("running", first.sandbox_id),
        ("close", first.sandbox_id),
        ("running", second.sandbox_id),
    ]
    assert pool.parked is None and pool.active == 1


@pytest.mark.parametrize("failing", ["close", "reserved"])
def test_a_claim_that_raises_gives_the_slot_back(failing: str) -> None:
    """Closing the kept sandbox before a create, or counting the grant, raises something unexpected: the test fails
    and its slot is free for the next one."""
    service = _Service()
    reserved: list[Callable[[], tuple[int, int]]] = [lambda: (0, 0)]
    handover = Handover(3, 2, reserved=lambda: reserved[0]())
    pool = SandboxPool(
        service, {1: "runner-1"}, run_id="run-1", limits=PoolLimits(max_parallel_tests=1, handover=handover)
    )
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2", "req-3"})
    pool.hand_over(first)
    assert pool.parked == first.sandbox_id

    def broken() -> tuple[int, int]:
        raise RuntimeError("the leases could not be counted")

    if failing == "close":
        reserved[0] = lambda: (1, 1)  # no room: the kept sandbox is closed before the create
        service.kill_error = RuntimeError("the service could not be reached")
    else:
        reserved[0] = broken
    with pytest.raises(RuntimeError):
        _acquire(pool, "req-2")
    assert pool.active == 0, "the failed test gave its slot back"
    service.kill_error = None
    service.closed(first.sandbox_id)  # a sandbox the pool could not close ends on its own timeout
    reserved[0] = lambda: (0, 0)
    third = _acquire(pool, "req-3", timeout_sec=2)
    pool.release(third)


def test_a_hand_over_while_the_pool_closes_keeps_nothing() -> None:
    service = _Service()
    pool = _pool(service)
    first = _acquire(pool, "req-1")
    pool.note_upcoming({"req-2"})
    handed: list[None] = []

    def hand_over_once() -> None:  # the test ends while the pool closes, before it forgets what it holds
        if not handed:
            handed.append(pool.hand_over(first))

    service.on_find = hand_over_once
    pool.close()
    assert handed and pool.parked is None and service.live == {}
