# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool of authoritative sandboxes (design 12.1 test_sandboxes)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.measure import sandboxes
from hyperloom.measure.sandboxes import (
    RUN_ID_KEY,
    AcquireTimedOut,
    AttemptTag,
    NoRunnerTemplate,
    PoolClosed,
    PoolLimits,
    SandboxPool,
    sandbox_priority,
)
from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.tests.test_sandbox_e2b import (
    CREATE_FAILED_BODY,
    CREATE_FAILED_MESSAGE,
    FAST,
    GATEWAY_CUTS,
    KEY_SPENT_BODY,
    KEY_MEASUREMENT,
    SANDBOX,
    _listed,
    _Service,
    service,  # noqa: F401 (the canned service the end-to-end case runs against)
)
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    CreateOptions,
    SandboxCreateFailed,
    SandboxGone,
    SandboxInfo,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)


class _Sandbox:
    def __init__(self, provider: _Provider, sandbox_id: str, metadata: dict[str, str]) -> None:
        self.sandbox_id = sandbox_id
        self._provider = provider
        self._metadata = metadata
        self.killed = 0

    def info(self) -> SandboxInfo:
        return SandboxInfo(self.sandbox_id, "t", dict(self._metadata))

    def kill(self) -> None:
        self.killed += 1
        self._provider.live.pop(self.sandbox_id, None)


class _Provider:
    """Counts how many sandboxes exist at once; creation can be made to fail.

    ``quota`` caps the live sandboxes as the session grant does, and ``spent`` holds the idempotency keys whose sandbox
    has ended.
    """

    name = "fake"

    def __init__(self) -> None:
        self.live: dict[str, _Sandbox] = {}
        self.calls: list[dict] = []
        self.peak = 0
        self.failures: list[Exception] = []
        self.quota: int | None = None
        self.on_refuse: Callable[[], None] | None = None  # runs inside a create that is refused for the quota
        self.spent: set[str] = set()
        self._lock = threading.Lock()
        self._count = 0

    def create(self, template, *, role, timeout_sec, metadata, options=CreateOptions()):
        self.calls.append(
            {
                "template": template,
                "role": role,
                "timeout_sec": timeout_sec,
                "metadata": dict(metadata),
                "envs": options.envs,
                "wait_sec": options.wait_sec,
            }
        )
        if self.failures:
            raise self.failures[0]
        if metadata.get(IDEMPOTENCY_KEY) in self.spent:
            raise SandboxGone("idempotency_key_spent")
        if self.quota is not None and len(self.live) >= self.quota:
            if self.on_refuse is not None:
                self.on_refuse()
            raise SandboxQuotaExceeded("grant_quota_exceeded")
        with self._lock:
            self._count += 1
            sandbox = _Sandbox(self, f"sbx-{self._count}", dict(metadata))
            self.live[sandbox.sandbox_id] = sandbox
            self.peak = max(self.peak, len(self.live))
        return sandbox

    def connect(self, sandbox_id, *, role):
        try:
            return self.live[sandbox_id]
        except KeyError:
            raise SandboxGone(sandbox_id) from None

    def find(self, metadata, *, role):
        infos = [s.info() for s in list(self.live.values())]
        return [info for info in infos if all(info.metadata.get(k) == v for k, v in metadata.items())]


_THIS_RUN = {RUN_ID_KEY: "run-1"}


def _pool(provider, create_envs=None, **kwargs) -> SandboxPool:
    run_id = kwargs.pop("run_id", "run-1")
    templates = {1: "runner-1", 8: "runner-8"}
    return SandboxPool(provider, templates, run_id=run_id, limits=PoolLimits(**kwargs), create_envs=create_envs)


def _acquire(pool: SandboxPool, request_id: str = "req-1", attempt: int = 1, **kwargs):
    kwargs.setdefault("timeout_sec", 100)
    tag = AttemptTag(request_id, attempt, kwargs.pop("avoid_nodes", ()), kwargs.pop("meta", None))
    return pool.acquire(1, tag, **kwargs)


def test_no_more_tests_run_at_once_than_the_cap_allows() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=2)
    done: list[str] = []

    def one(index: int) -> None:
        runner = _acquire(pool, f"req-{index}")
        time.sleep(0.15)
        pool.release(runner)
        done.append(runner.sandbox_id)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert len(done) == 6, "every test got its turn"
    assert provider.peak == 2, "two ran together, never three"
    assert pool.active == 0


def test_a_failed_create_gives_the_slot_back() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=1)
    provider.failures = [SandboxUnavailable("no capacity")]
    with pytest.raises(SandboxUnavailable):
        _acquire(pool)
    assert pool.active == 0
    provider.failures = []
    assert _acquire(pool).sandbox_id == "sbx-1"


def test_no_capacity_is_not_quota_back_pressure_and_is_raised_at_once() -> None:
    provider = _Provider()
    pool = _pool(provider)
    provider.failures = [SandboxUnavailable("503 no_capacity")]
    with pytest.raises(SandboxUnavailable) as refused:
        _acquire(pool)
    assert type(refused.value) is SandboxUnavailable
    assert len(provider.calls) == 1 and pool.active == 0


def test_a_create_over_the_quota_waits_for_a_sandbox_to_end_and_asks_again() -> None:
    provider = _Provider()
    provider.quota = 1
    pool = _pool(provider, max_parallel_tests=2)
    first = _acquire(pool, "first")
    made: list = []
    waiter = threading.Thread(target=lambda: made.append(_acquire(pool, "second")))
    waiter.start()
    deadline = time.monotonic() + 5
    while len(provider.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert waiter.is_alive() and made == [], "refused for the quota, the second test waits and is not failed"
    pool.release(first)
    waiter.join(5)
    assert [r.sandbox_id for r in made] == ["sbx-2"]
    assert len(provider.calls) == 3, "one refusal, then one create when the first sandbox ended; no polling"
    assert pool.active == 1


def test_a_full_quota_is_asked_again_after_the_retry_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandboxes, "_QUOTA_RETRY_SEC", 0.05)
    provider = _Provider()
    provider.quota = 0  # filled by sandboxes the pool does not hold: nothing of its own ends to wake it
    pool = _pool(provider)
    threading.Timer(0.3, lambda: setattr(provider, "quota", None)).start()
    assert _acquire(pool).sandbox_id == "sbx-1"
    assert len(provider.calls) > 2 and pool.active == 1


def test_waiting_for_the_quota_ends_at_the_tests_timeout_with_what_the_service_said(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sandboxes, "_QUOTA_RETRY_SEC", 0.05)
    monkeypatch.setattr(sandboxes, "WAIT_LOG_EVERY_SEC", 0.2)
    caplog.set_level(logging.INFO, logger=sandboxes.__name__)
    provider = _Provider()
    provider.quota = 0  # filled by sandboxes of another role or process: nothing this pool holds ends to wake it
    pool = _pool(provider)
    started = time.monotonic()
    with pytest.raises(AcquireTimedOut) as gave_up:
        _acquire(pool, "req-q", attempt=2, timeout_sec=1)
    assert time.monotonic() - started < 3, "the wait ended at the test's timeout"
    assert "room in the grant's quota" in str(gave_up.value) and "grant_quota_exceeded" in str(gave_up.value)
    assert pool.active == 0, "the slot is given back"
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("test req-q attempt 2 waits for room in the grant's quota") for line in lines), lines
    assert any("still waiting for room in the grant's quota" in line for line in lines), lines


def test_waiting_for_a_slot_ends_at_the_tests_timeout_naming_what_holds_the_slots(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=sandboxes.__name__)
    provider = _Provider()
    provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)  # one nobody released
    pool = _pool(provider, max_parallel_tests=1)
    with pytest.raises(
        AcquireTimedOut, match="a sandbox slot.*0 held by this pool and 1 running that it does not hold"
    ):
        _acquire(pool, "req-s", timeout_sec=0.5)
    assert pool.active == 0 and len(provider.calls) == 1, "no create was sent for the attempt"
    assert any("req-s attempt 1 waits for a sandbox slot" in r.getMessage() for r in caplog.records)


def test_a_reattach_waits_for_its_slot_no_longer_than_it_is_told() -> None:
    provider = _Provider()
    survivor = provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)
    provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)  # holds the only slot
    pool = _pool(provider, max_parallel_tests=1)
    with pytest.raises(AcquireTimedOut, match="a sandbox slot"):
        pool.reattach(survivor.sandbox_id, timeout_sec=0.3)
    assert pool.active == 0


def test_a_create_that_waits_for_its_sandbox_says_so_while_it_waits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sandboxes, "WAIT_LOG_EVERY_SEC", 0.1)
    caplog.set_level(logging.INFO, logger=sandboxes.__name__)
    provider = _Provider()
    slow = provider.create
    provider.create = lambda *a, **kw: (time.sleep(0.5), slow(*a, **kw))[1]  # type: ignore[method-assign]
    runner = _acquire(_pool(provider), "req-c", timeout_sec=30)
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("test req-c attempt 1: creating its sandbox") for line in lines), lines
    assert any("req-c attempt 1 still waiting for its sandbox to run" in line for line in lines), lines
    assert any(f"sandbox {runner.sandbox_id} is running" in line for line in lines), lines


def test_waiting_for_the_quota_ends_when_the_run_stops() -> None:
    provider = _Provider()
    provider.quota = 0
    pool = _pool(provider)
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(PoolClosed):
        _acquire(pool, stop=stop)
    assert pool.active == 0


def test_a_sandbox_that_ends_while_the_create_is_refused_is_not_missed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandboxes, "_QUOTA_RETRY_SEC", 60.0)
    provider = _Provider()
    provider.quota = 1
    pool = _pool(provider, max_parallel_tests=2)
    first = _acquire(pool, "first")
    provider.on_refuse = lambda: pool.release(first)  # it ends after the service looked and before the pool waits
    made: list = []
    waiter = threading.Thread(target=lambda: made.append(_acquire(pool, "second")), daemon=True)
    waiter.start()
    waiter.join(5)
    assert not waiter.is_alive(), "the pool asked again at once instead of sleeping the whole retry interval"
    assert [r.sandbox_id for r in made] == ["sbx-2"]


def test_a_stop_wins_over_a_sandbox_that_ended_while_the_quota_was_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandboxes, "_QUOTA_RETRY_SEC", 60.0)
    provider = _Provider()
    provider.quota = 1
    pool = _pool(provider, max_parallel_tests=2)
    first = _acquire(pool, "first")
    stop = threading.Event()

    def end_and_stop() -> None:
        pool.release(first)
        stop.set()

    provider.on_refuse = end_and_stop
    with pytest.raises(PoolClosed):
        _acquire(pool, "second", stop=stop)
    assert len(provider.calls) == 2 and pool.active == 0


def test_a_spent_idempotency_key_is_replaced_by_the_next_key_of_the_attempt() -> None:
    provider = _Provider()
    provider.spent = {"run-1:req-1:1", "run-1:req-1:1:r1"}
    pool = _pool(provider)
    runner = _acquire(pool)
    keys = [call["metadata"][IDEMPOTENCY_KEY] for call in provider.calls]
    assert keys == ["run-1:req-1:1", "run-1:req-1:1:r1", "run-1:req-1:1:r2"]
    assert runner.metadata[IDEMPOTENCY_KEY] == "run-1:req-1:1:r2"
    assert provider.calls[-1]["metadata"]["hyperloom.request-id"] == "req-1", "only the key changes"


def test_a_create_that_finds_every_key_spent_is_gone_and_gives_the_slot_back() -> None:
    provider = _Provider()
    provider.failures = [SandboxGone("idempotency_key_spent")]
    pool = _pool(provider, max_parallel_tests=1)
    with pytest.raises(SandboxGone):
        _acquire(pool)
    assert len(provider.calls) == sandboxes._MAX_SPENT_KEYS + 1 and pool.active == 0


def test_a_sandbox_that_failed_before_it_ran_ends_the_attempt_without_another_key() -> None:
    provider = _Provider()
    provider.failures = [SandboxCreateFailed(f"500: {CREATE_FAILED_MESSAGE}")]
    pool = _pool(provider, max_parallel_tests=1)
    with pytest.raises(SandboxCreateFailed, match="phase Failed"):
        _acquire(pool)
    assert [call["metadata"][IDEMPOTENCY_KEY] for call in provider.calls] == ["run-1:req-1:1"]
    assert pool.active == 0, "the slot is given back"
    provider.failures = []
    runner = _acquire(pool, attempt=2)
    assert runner.metadata[IDEMPOTENCY_KEY] == "run-1:req-1:2", "the next attempt is a new key"


def test_a_failed_sandbox_is_created_once_against_the_platform_of_joint_run_4(service: _Service) -> None:
    """Agent Sandbox answers a failed sandbox with a 500, then reports its key spent (410), and a new key makes another
    sandbox that fails the same way: one acquire made three sandboxes in 27 s until the session was killed."""
    _platform_answers(service, *[(500, CREATE_FAILED_BODY), (410, KEY_SPENT_BODY)] * 20)
    with pytest.raises(SandboxCreateFailed, match="sandbox_create_failed"):
        _acquire(_e2b_pool(service))
    assert _keys_created(service) == ["run-1:req-1:1"]


def test_a_sandbox_that_failed_while_its_create_was_cut_is_not_created_again_under_another_key(
    service: _Service,
) -> None:
    """The gateway cuts the create while the sandbox is pending; it then fails, and the replay is told the key is spent.
    That ends the attempt: replacing the key would make another sandbox that fails the same way."""
    _platform_answers(service, GATEWAY_CUTS[0], *[(410, KEY_SPENT_BODY), (500, CREATE_FAILED_BODY)] * 20)
    pool = _e2b_pool(service)
    with pytest.raises(SandboxCreateFailed, match="idempotency_key_spent"):
        _acquire(pool)
    assert _keys_created(service) == ["run-1:req-1:1", "run-1:req-1:1"]
    assert pool.active == 0


def test_a_create_cut_by_the_gateway_finds_its_pending_sandbox_through_the_pool(service: _Service) -> None:
    _platform_answers(service, GATEWAY_CUTS[0], (201, SANDBOX))
    service.reply("GET", "/sandboxes/sbx-1", (200, _listed("sbx-1", dict(_THIS_RUN))))
    runner = _acquire(_e2b_pool(service))
    assert runner.sandbox_id == "sbx-1"
    assert _keys_created(service) == ["run-1:req-1:1", "run-1:req-1:1"]


def _platform_answers(service: _Service, *answers) -> None:
    service.reply("POST", "/v2/sandboxes", *answers, (201, SANDBOX))
    service.reply("GET", "/v2/sandboxes", (200, []))  # the pool counts the run's live sandboxes before it creates


def _e2b_pool(service: _Service) -> SandboxPool:
    provider = E2BSandboxProvider(
        {"measurement": KEY_MEASUREMENT}, api_url=service.url, sandbox_url=service.url, retry=FAST
    )
    return SandboxPool(provider, {1: "runner-1"}, run_id="run-1", limits=PoolLimits(max_parallel_tests=1))


def _keys_created(service: _Service) -> list[str]:
    return [call.body["metadata"][IDEMPOTENCY_KEY] for call in service.calls("POST", "/v2/sandboxes")]


def test_a_slot_freed_twice_is_freed_once() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=2)
    first, second = _acquire(pool, "a"), _acquire(pool, "b")
    pool.release(first)
    pool.release(first)
    assert pool.active == 1, "the second release of the same sandbox did not free the other test's slot"
    assert first.handle.killed == 1
    pool.release(second)
    assert pool.active == 0


def test_waiting_for_a_slot_ends_when_the_run_stops_or_the_pool_closes() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=1)
    _acquire(pool)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    with pytest.raises(PoolClosed):
        _acquire(pool, "other", stop=stop)
    pool.close()
    with pytest.raises(PoolClosed):
        _acquire(pool, "late")


def test_a_sandbox_a_previous_process_left_running_keeps_its_slot_until_it_is_reattached_or_ends() -> None:
    provider = _Provider()
    survivor = provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)
    pool = _pool(provider, max_parallel_tests=1)
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(PoolClosed):
        _acquire(pool, "new", stop=stop)
    assert len(provider.live) == 1, "no second sandbox was created next to the one still measuring"
    again = pool.reattach(survivor.sandbox_id)
    assert again.sandbox_id == survivor.sandbox_id and pool.active == 1
    pool.release(again)
    assert _acquire(pool, "new").sandbox_id != survivor.sandbox_id


def test_a_slot_frees_when_the_sandbox_that_held_it_ends() -> None:
    provider = _Provider()
    survivor = provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)
    pool = _pool(provider, max_parallel_tests=1)
    threading.Timer(0.3, survivor.kill).start()
    runner = _acquire(pool, "new")
    assert runner.sandbox_id != survivor.sandbox_id and len(provider.live) == 1


def test_close_closes_every_measurement_sandbox_of_the_run_including_one_another_process_made() -> None:
    provider = _Provider()
    pool = _pool(provider)
    mine = _acquire(pool, "mine")
    stray = provider.create("runner-1", role="measurement", timeout_sec=10, metadata=_THIS_RUN)
    pool.close()
    assert provider.live == {}
    assert mine.handle.killed == 1 and stray.killed == 1
    pool.close()


def test_a_pool_neither_counts_nor_closes_the_sandboxes_of_another_run_that_its_key_sees() -> None:
    provider = _Provider()
    other = provider.create("runner-1", role="measurement", timeout_sec=10, metadata={RUN_ID_KEY: "run-2"})
    unmarked = provider.create("runner-1", role="measurement", timeout_sec=10, metadata={})
    pool = _pool(provider, max_parallel_tests=1)
    giveup = threading.Event()
    timer = threading.Timer(3, giveup.set)  # a pool that counted the others would wait for a slot: this ends the wait
    timer.start()
    try:
        mine = _acquire(pool, "mine", stop=giveup)
    finally:
        timer.cancel()
    assert pool.active == 1 and mine.metadata[RUN_ID_KEY] == "run-1", "the slot was free: the others are not this run's"
    pool.close()
    assert other.killed == unmarked.killed == 0 and mine.handle.killed == 1
    assert sorted(provider.live) == sorted([other.sandbox_id, unmarked.sandbox_id])


def test_the_create_call_carries_the_template_lifetime_and_metadata_of_the_design() -> None:
    provider = _Provider()
    pool = _pool(provider, timeout_slack_sec=900, request_timeout_sec=3000)
    _acquire(
        pool,
        "req-7",
        attempt=3,
        avoid_nodes=("n2", "n1", "n2"),
        meta={"priority": 3, "purpose": "confirm", "branch_id": "b9"},
        timeout_sec=600,
    )
    call = provider.calls[0]
    assert call["template"] == "runner-1" and call["role"] == "measurement"
    assert call["timeout_sec"] == 1500, "the test's timeout plus the slack"
    assert 590 < call["wait_sec"] <= 600, "the wait for the sandbox to run is bounded by the test's own timeout"
    _acquire(pool, "req-8", timeout_sec=5000)
    assert provider.calls[1]["wait_sec"] == 3000, "and by the pool's own when that is shorter"
    assert call["envs"] is None, "a pool given no create environment passes none"
    assert call["metadata"] == {
        "agent-sandbox.io/idempotency-key": "run-1:req-7:3",
        "hyperloom.run-id": "run-1",
        "agent-sandbox.io/avoid-nodes": "n1,n2",
        "agent-sandbox.io/priority": "1",
        "hyperloom.request-id": "req-7",
        "hyperloom.purpose": "confirm",
        "hyperloom.branch-id": "b9",
    }


def test_the_create_environment_reaches_every_create_of_the_pool() -> None:
    provider = _Provider()
    provider.spent = {"run-1:req-1:1"}  # the first key is spent: the same attempt creates again with its next key
    given = {"SOURCE_REPO": "https://source.invalid/repo.git", "SOURCE_SHA": "ab" * 20}
    pool = _pool(provider, create_envs=given)
    given["SOURCE_SHA"] = "changed after the pool was built"
    _acquire(pool)
    _acquire(pool, "req-2")
    assert len(provider.calls) == 3
    assert [call["envs"] for call in provider.calls] == [
        {"SOURCE_REPO": given["SOURCE_REPO"], "SOURCE_SHA": "ab" * 20}
    ] * 3


def test_the_sandbox_never_outlives_the_run() -> None:
    provider = _Provider()
    _acquire(_pool(provider), timeout_sec=600, remaining_sec=120.7)
    assert provider.calls[0]["timeout_sec"] == 120


def test_the_template_follows_the_card_count_and_an_unserved_count_takes_no_slot() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=1)
    held = pool.acquire(8, AttemptTag("r", 1), timeout_sec=10)
    assert provider.calls[0]["template"] == "runner-8"
    assert pool.active == 1, "only the served count holds a slot"
    with pytest.raises(NoRunnerTemplate, match="3 GPU"):
        pool.acquire(3, AttemptTag("r", 1), timeout_sec=10)
    assert pool.active == 1
    pool.release(held)


@pytest.mark.parametrize(("priority", "tier"), [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (None, 1)])
def test_test_priorities_map_to_the_three_claw_tiers(priority, tier) -> None:
    assert sandbox_priority(priority) == tier


def test_a_priority_outside_the_range_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="0 to 4"):
        sandbox_priority(5)


def test_reattach_takes_a_slot_and_a_gone_sandbox_gives_it_back() -> None:
    provider = _Provider()
    pool = _pool(provider, max_parallel_tests=1)
    first = _acquire(pool)
    pool.release(first)
    provider.live[first.sandbox_id] = first.handle
    again = pool.reattach(first.sandbox_id)
    assert again.sandbox_id == first.sandbox_id and pool.active == 1
    pool.release(again)
    with pytest.raises(SandboxGone):
        pool.reattach("sbx-unknown")
    assert pool.active == 0


# -- over the local sandbox provider --------------------------------------------------------------------------------


@pytest.fixture
def local(tmp_path: Path):
    template = LocalTemplate("runner-1", "measurement", gpus=1, sealed=True, signing=True)
    provider = LocalSandboxProvider([template], gpus=range(4), base_dir=tmp_path / "sb", hostname="node-a")
    yield provider
    provider.close()


def test_the_platforms_facts_reach_the_caller(local) -> None:
    pool = _pool(local)
    runner = _acquire(pool)
    assert runner.node_name == "node-a" and runner.same_node is False
    assert runner.public_key_pem.startswith("-----BEGIN PUBLIC KEY-----")
    pool.release(runner)


def test_a_node_that_could_not_be_avoided_is_marked_same_node(local) -> None:
    pool = _pool(local)
    first = _acquire(pool, "first")
    retest = _acquire(pool, "retest", avoid_nodes=(first.node_name,))
    assert retest.same_node is True and retest.node_name == "node-a"
    elsewhere = _acquire(pool, "other", avoid_nodes=("node-z",))
    assert elsewhere.same_node is False


def test_the_idempotency_key_returns_the_same_sandbox_until_the_attempt_number_changes(local) -> None:
    pool = _pool(local, max_parallel_tests=4)
    first = _acquire(pool, "req", attempt=1)
    repeat = _acquire(pool, "req", attempt=1)
    assert repeat.sandbox_id == first.sandbox_id
    retry = _acquire(pool, "req", attempt=2)
    assert retry.sandbox_id != first.sandbox_id
