# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool of authoritative sandboxes (design 12.1 test_sandboxes)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hyperloom.measure.sandboxes import (
    AttemptTag,
    NoRunnerTemplate,
    PoolClosed,
    PoolLimits,
    SandboxPool,
    sandbox_priority,
)
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.socket import CreateOptions, SandboxGone, SandboxInfo, SandboxUnavailable


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
    """Counts how many sandboxes exist at once; creation can be made to fail."""

    name = "fake"

    def __init__(self) -> None:
        self.live: dict[str, _Sandbox] = {}
        self.calls: list[dict] = []
        self.peak = 0
        self.failures: list[Exception] = []
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
        return [s.info() for s in list(self.live.values())]


def _pool(provider, **kwargs) -> SandboxPool:
    run_id = kwargs.pop("run_id", "run-1")
    return SandboxPool(provider, {1: "runner-1", 8: "runner-8"}, run_id=run_id, limits=PoolLimits(**kwargs))


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
    survivor = provider.create("runner-1", role="measurement", timeout_sec=10, metadata={})
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
    survivor = provider.create("runner-1", role="measurement", timeout_sec=10, metadata={})
    pool = _pool(provider, max_parallel_tests=1)
    threading.Timer(0.3, survivor.kill).start()
    runner = _acquire(pool, "new")
    assert runner.sandbox_id != survivor.sandbox_id and len(provider.live) == 1


def test_close_closes_every_measurement_sandbox_including_one_another_process_made() -> None:
    provider = _Provider()
    pool = _pool(provider)
    mine = _acquire(pool, "mine")
    stray = provider.create("runner-1", role="measurement", timeout_sec=10, metadata={})
    pool.close()
    assert provider.live == {}
    assert mine.handle.killed == 1 and stray.killed == 1
    pool.close()


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
    assert call["wait_sec"] == 3000
    assert call["envs"] is None, "a sealed template takes no caller variables"
    assert call["metadata"] == {
        "claw.io/idempotency-key": "run-1:req-7:3",
        "claw.io/avoid-nodes": "n1,n2",
        "claw.io/priority": "1",
        "hyperloom.request-id": "req-7",
        "hyperloom.purpose": "confirm",
        "hyperloom.branch-id": "b9",
    }


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
