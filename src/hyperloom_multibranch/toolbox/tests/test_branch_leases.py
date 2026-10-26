# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the local backend: cards out of the measurement pool, back on release or close.

The toolbox is the real local one (``build_local_toolbox``: the real queue, hosted backend, sandbox provider and card
pool) with the fake runner as the sandbox's command, on a two-card pool unless a test says otherwise.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hyperloom.measure.tests.hosted_fixtures import point_yaml, runner_template, spec_context
from hyperloom.sandbox.card_occupancy import CardGuard, Holder, read_holders
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, MeasurementCapacity, RequestRejected, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox import branch_leases, local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox, LocalOptions, LocalSession
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, build, req
from hyperloom_multibranch.toolbox.tests.card_list import INTRUDER, CardList
from hyperloom_multibranch.toolbox.tests.test_card_placement import (
    FAST,
    HANG,
    SLOW,
    _behaviour,
    _PointPerBranch,
    _until,
)
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

Make = Callable[..., Any]


@pytest_asyncio.fixture
async def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``make(cards=2, tp=1, runner_envs=None, **branches)``: a started local toolbox whose test of branch ``name``
    behaves as ``branches[name]`` (``SLOW``, ``HANG`` or ``FAST``)."""

    def fake_runner(name: str, *, gpus: int, python: str, envs: Any) -> Any:
        from dataclasses import replace

        return replace(runner_template(name, gpus=gpus), envs=dict(envs or {}))

    monkeypatch.setattr(local_hosted, "local_runner_template", fake_runner)
    made: list[LocalHostedToolbox] = []

    async def make(
        cards: int = 2,
        tp: int = 1,
        runner_envs: dict[str, str] | None = None,
        card_holders: Callable[..., Any] = read_holders,
        card_guard: CardGuard | None = None,
        **branches: str,
    ) -> Any:
        session = tmp_path / f"session-{len(made)}"
        session.mkdir()
        points = {
            branch: point_yaml(session, f"{branch}.yaml", tp=tp, **_behaviour(behaviour))
            for branch, behaviour in branches.items()
        }
        toolbox = local_hosted.build_local_toolbox(
            LocalSession(session, f"run-{len(made)}", range(cards), tp=tp),
            LocalNotebook(session / "notebook", Writer(Role.TOOLBOX, "toolbox")),
            AcceptanceSetup(FakeRepo(), _PointPerBranch(points), IMAGE, WorkloadKind.AGENTX),
            spec_context(session, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")),
            options=LocalOptions(
                poll_interval_sec=0.05, runner_envs=runner_envs, card_holders=card_holders, card_guard=card_guard
            ),
        )
        made.append(toolbox)
        await toolbox.start()
        return toolbox

    yield make
    for toolbox in made:
        if not toolbox._closed:
            await toolbox.withdraw_all()
            await toolbox.close()


def _running(box: LocalHostedToolbox) -> int:
    return len(box.provider.find({}, role="measurement"))


async def _cards_of(box: LocalHostedToolbox, request_id: str) -> tuple[int, ...]:
    result = await asyncio.wait_for(box.outcome(request_id), 60)
    assert result.sandbox is not None, result
    return result.sandbox.gpu_ids


async def test_a_lease_holds_cards_out_of_measurement_and_no_test_ever_runs_on_them(local: Make) -> None:
    box = await local(slow=SLOW, fast=FAST)
    lease = await box.lease_branch_sandbox("agent", 1)
    (card,) = lease.gpu_ids
    assert await box.measurement_capacity() == MeasurementCapacity(2, 2, leased_cards=1)
    assert box.provider.pool.free_cards() == (1 - card,)
    first, second = await box.submit_test(req("slow")), await box.submit_test(req("fast"))
    await _until(lambda: _running(box) == 1, "the first test on the one card left")
    state = await box.queue_state()
    assert (state.running, state.queued) == (1, 1), "the second test waits for the card instead of starting on none"
    assert [await _cards_of(box, first), await _cards_of(box, second)] == [(1 - card,), (1 - card,)]
    assert await box.measurement_capacity() == MeasurementCapacity(2, 2, leased_cards=1)


async def test_the_lease_names_its_cards_its_directory_and_what_is_left(local: Make, tmp_path: Path) -> None:
    box = await local(cards=4)
    lease = await box.lease_branch_sandbox("b02", 2)
    assert (lease.branch_id, len(lease.gpu_ids), lease.remaining_cards) == ("b02", 2, 2)
    assert lease.env["HIP_VISIBLE_DEVICES"] == ",".join(map(str, lease.gpu_ids))
    assert lease.root.is_dir() and lease.root.parent == tmp_path / "session-0" / "branch-sandboxes"
    other = await box.lease_branch_sandbox("b03", 1)
    assert other.remaining_cards == 1 and other.root != lease.root and not set(other.gpu_ids) & set(lease.gpu_ids)
    assert await box.measurement_capacity() == MeasurementCapacity(4, 4, leased_cards=3)


async def test_release_returns_the_cards_to_measurement_and_is_idempotent(local: Make) -> None:
    box = await local(a=HANG, b=HANG)
    lease = await box.lease_branch_sandbox("agent", 1)
    await box.release_branch_sandbox(lease.lease_id)
    await box.release_branch_sandbox(lease.lease_id)
    await box.release_branch_sandbox("a lease that never was")
    assert await box.measurement_capacity() == MeasurementCapacity(2, 2)
    assert box.provider.pool.free_cards() == (0, 1) and lease.root.is_dir()
    await box.submit_test(req("a"))
    await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2, "both cards measuring again")


async def test_two_releases_at_once_return_the_cards_once(local: Make) -> None:
    box = await local(cards=4)
    lease = await box.lease_branch_sandbox("agent", 2)
    await asyncio.gather(box.release_branch_sandbox(lease.lease_id), box.release_branch_sandbox(lease.lease_id))
    assert box.provider.pool.free_cards() == (0, 1, 2, 3)
    assert await box.measurement_capacity() == MeasurementCapacity(4, 4)


async def test_two_threads_releasing_one_lease_return_its_cards_exactly_once(local: Make) -> None:
    """The scheduler's stop path and the session-close backstop may release the same lease at the same moment."""
    box = await local(cards=4)
    lease = await box.lease_branch_sandbox("agent", 2)
    pool, returned = box.provider.pool, []
    give_back = pool.release

    def slow_release(cards: Any) -> None:
        returned.append(tuple(cards))
        time.sleep(0.2)
        give_back(cards)

    pool.release = slow_release  # type: ignore[method-assign]
    barrier = threading.Barrier(2)

    def release() -> None:
        barrier.wait()
        box.leases.release(lease.lease_id)

    await asyncio.gather(asyncio.to_thread(release), asyncio.to_thread(release))
    assert returned == [lease.gpu_ids], "the cards went back to the pool more than once"
    assert pool.free_cards() == (0, 1, 2, 3) and await box.measurement_capacity() == MeasurementCapacity(4, 4)


async def test_close_releases_every_lease(local: Make) -> None:
    box = await local(cards=4)
    await box.lease_branch_sandbox("b02", 1)
    await box.lease_branch_sandbox("b03", 2)
    assert (await box.measurement_capacity()).leased_cards == 3
    await box.close()
    assert box.provider.pool.free_cards() == (0, 1, 2, 3)
    assert await box.measurement_capacity() == MeasurementCapacity(4, 4)


async def test_a_branch_that_holds_a_lease_cannot_take_a_second(local: Make) -> None:
    box = await local(cards=4)
    await box.lease_branch_sandbox("b02", 1)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("b02", 1)
    assert refused.value.code == "lease_exists" and "lease_exists" in REJECTION_CODES
    assert await box.measurement_capacity() == MeasurementCapacity(4, 4, leased_cards=1)


@pytest.mark.parametrize(("cards", "tp", "asked", "left"), [(2, 1, 2, 2), (4, 2, 3, 4), (4, 1, 4, 4)])
async def test_a_lease_that_would_leave_less_than_one_test_needs_is_refused(
    local: Make, cards: int, tp: int, asked: int, left: int
) -> None:
    box = await local(cards=cards, tp=tp)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("b02", asked)
    assert refused.value.code == "lease_leaves_no_measurement" and "lease_leaves_no_measurement" in REJECTION_CODES
    assert (
        box.provider.pool.free_cards() == tuple(range(cards)) and (await box.measurement_capacity()).leased_cards == 0
    )


async def test_the_leases_together_may_not_take_the_last_test_either(local: Make) -> None:
    box = await local(cards=4, tp=2)
    await box.lease_branch_sandbox("b02", 2)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("b03", 1)
    assert refused.value.code == "lease_leaves_no_measurement" and "lease_leaves_no_measurement" in REJECTION_CODES
    assert (await box.measurement_capacity()).leased_cards == 2


async def test_cards_lost_for_good_count_against_what_a_lease_may_take(local: Make) -> None:
    box = await local(cards=2)
    box.provider.quarantined.add(0)
    capacity = await box.measurement_capacity()
    assert (capacity.usable_cards, capacity.leased_cards) == (1, 0)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("b02", 1)
    assert refused.value.code == "lease_leaves_no_measurement" and "lease_leaves_no_measurement" in REJECTION_CODES


async def test_a_lease_asks_for_at_least_one_card(local: Make) -> None:
    box = await local(cards=4)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("b02", 0)
    assert refused.value.code == "invalid_request"


async def test_usable_cards_keep_their_meaning_while_cards_are_leased(local: Make) -> None:
    box = await local(cards=4)
    await box.lease_branch_sandbox("b02", 3)
    capacity = await box.measurement_capacity()
    assert (capacity.total_cards, capacity.usable_cards, capacity.leased_cards) == (4, 4, 3)
    assert capacity.measurement_cards == 1


async def test_a_lease_waits_for_a_running_test_and_the_queue_keeps_the_card_for_it(local: Make) -> None:
    box = await local(a=HANG, b=HANG, c=FAST)
    first, _ = await box.submit_test(req("a")), await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2 and box.provider.pool.free_cards() == (), "both cards measuring")
    queued = await box.submit_test(req("c"))
    waiting = asyncio.ensure_future(box.lease_branch_sandbox("agent", 1))
    await _until(lambda: box.leases.capacity().leased_cards == 1, "the lease promised its card")
    assert not waiting.done()
    await box.cancel(first)
    lease = await asyncio.wait_for(waiting, 30)
    assert lease.remaining_cards == 1
    assert (await box.queue_state()).queued_by_branch == {"c": 1}, "the freed card went to the lease, not to a test"
    assert box.provider.pool.free_cards() == ()
    await box.release_branch_sandbox(lease.lease_id)
    assert await _cards_of(box, queued) == lease.gpu_ids


async def test_a_lease_that_stops_waiting_gives_its_promise_back(local: Make) -> None:
    box = await local(a=HANG, b=HANG)
    await box.submit_test(req("a"))
    await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2 and box.provider.pool.free_cards() == (), "both cards measuring")
    waiting = asyncio.ensure_future(box.lease_branch_sandbox("agent", 1))
    await _until(lambda: box.leases.capacity().leased_cards == 1, "the lease promised its card")
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert (await box.measurement_capacity()).leased_cards == 0
    await box.withdraw_all()
    await _until(lambda: box.provider.pool.free_cards() == (0, 1), "the withdrawn tests' cards back")


async def test_a_lease_still_waiting_when_the_toolbox_closes_is_refused_its_cards(local: Make) -> None:
    box = await local(a=HANG, b=HANG)
    await box.submit_test(req("a"))
    await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2 and box.provider.pool.free_cards() == (), "both cards measuring")
    waiting = asyncio.ensure_future(box.lease_branch_sandbox("agent", 1))
    await _until(lambda: box.leases.capacity().leased_cards == 1, "the lease promised its card")
    await box.withdraw_all()
    box.leases.close()
    with pytest.raises(SandboxUnavailable, match="closing"):
        await asyncio.wait_for(waiting, 30)


async def test_the_lease_env_is_the_image_env_without_secrets_or_a_mask_of_other_cards(local: Make) -> None:
    image = {
        "SGLANG_USE_AITER": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "HIP_VISIBLE_DEVICES": "0,1",
        "HSA_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "1",
        "ANTHROPIC_API_KEY": "sk-secret",
        "HF_TOKEN": "hf_secret",
        "SOME_PASSWORD": "hunter2",
        "LD_PRELOAD": "/tmp/hook.so",
    }
    box = await local(cards=4, runner_envs=image)
    lease = await box.lease_branch_sandbox("b02", 2)
    assert dict(lease.env) == {
        "SGLANG_USE_AITER": "1",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "HIP_VISIBLE_DEVICES": ",".join(map(str, lease.gpu_ids)),
    }


async def test_a_toolbox_without_cards_of_its_own_refuses_a_lease(tmp_path: Path) -> None:
    toolbox = build(tmp_path).toolbox
    with pytest.raises(RequestRejected) as refused:
        await toolbox.lease_branch_sandbox("b02", 1)
    assert refused.value.code == "unsupported_kind" and "unsupported_kind" in REJECTION_CODES
    await toolbox.release_branch_sandbox("a lease that never was")


# -- how long a lease waits ------------------------------------------------------------------------------------------


async def _both_cards_measuring(local: Make) -> LocalHostedToolbox:
    box = await local(a=HANG, b=HANG)
    await box.submit_test(req("a"))
    await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2 and box.provider.pool.free_cards() == (), "both cards measuring")
    return box


async def test_a_lease_that_waits_for_nothing_is_refused_at_once_when_no_card_is_free(
    local: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = await _both_cards_measuring(local)
    monkeypatch.setattr(branch_leases, "_POLL_SECONDS", 30.0)  # a lease that sleeps at all takes half a minute
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="not freed within 0 s"):
        await asyncio.wait_for(box.lease_branch_sandbox("agent", 1, wait_sec=0), 10)
    assert time.monotonic() - started < 5, "a lease that waits for nothing waited"
    assert (await box.measurement_capacity()).leased_cards == 0, "the refused lease gave its promise back"
    assert box.provider.pool.free_cards() == ()


async def test_a_lease_that_waits_for_nothing_takes_a_card_that_is_free(local: Make) -> None:
    box = await local(a=HANG, b=HANG)
    lease = await asyncio.wait_for(box.lease_branch_sandbox("agent", 1, wait_sec=0), 10)
    assert len(lease.gpu_ids) == 1 and (await box.measurement_capacity()).leased_cards == 1


async def test_a_positive_wait_bounds_how_long_a_lease_waits(local: Make) -> None:
    box = await _both_cards_measuring(local)
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="not freed within 0.5 s"):
        await asyncio.wait_for(box.lease_branch_sandbox("agent", 1, wait_sec=0.5), 20)
    waited = time.monotonic() - started
    assert 0.4 <= waited < 10, f"waited {waited:.2f} s for a wait of 0.5 s"
    assert (await box.measurement_capacity()).leased_cards == 0


async def test_a_positive_wait_takes_the_cards_a_running_test_gives_back_within_it(local: Make) -> None:
    box = await local(a=HANG, b=HANG)
    first, _ = await box.submit_test(req("a")), await box.submit_test(req("b"))
    await _until(lambda: _running(box) == 2 and box.provider.pool.free_cards() == (), "both cards measuring")
    waiting = asyncio.ensure_future(box.lease_branch_sandbox("agent", 1, wait_sec=30))
    await _until(lambda: box.leases.capacity().leased_cards == 1, "the lease promised its card")
    await box.cancel(first)
    assert len((await asyncio.wait_for(waiting, 30)).gpu_ids) == 1


async def test_no_wait_given_waits_as_long_as_a_measurement_sandbox_does(
    local: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = await _both_cards_measuring(local)
    monkeypatch.setattr(branch_leases, "LEASE_WAIT_SECONDS", 0.6)
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="not freed within 0.6 s"):
        await asyncio.wait_for(box.lease_branch_sandbox("agent", 1), 20)
    assert time.monotonic() - started >= 0.5, "a lease with no wait given did not wait"


@pytest.mark.parametrize("wait", [-1.0, -0.001, math.nan, math.inf])
async def test_a_wait_that_is_not_a_number_of_seconds_is_refused_before_anything_is_promised(
    local: Make, wait: float
) -> None:
    box = await local(cards=2)
    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("agent", 1, wait_sec=wait)
    assert refused.value.code == "invalid_request" and "wait_sec" in refused.value.reason
    assert (await box.measurement_capacity()).leased_cards == 0 and box.provider.pool.free_cards() == (0, 1)
    assert len((await box.lease_branch_sandbox("agent", 1)).gpu_ids) == 1, "the refusal left no lease behind"


@pytest.mark.parametrize("wait", [None, 0, 0.0, 12.5])
def test_the_wait_a_lease_takes_is_the_default_for_none_and_the_given_number_otherwise(wait: float | None) -> None:
    assert branch_leases.lease_wait(wait, 99.0) == (99.0 if wait is None else wait)


# -- a card some process holds is not leased ---------------------------------------------------------------------------


async def _watching_the_cards(
    local: Make, cards: int = 3, guard: CardGuard = CardGuard.ENFORCE
) -> tuple[LocalHostedToolbox, CardList]:
    holders = CardList()
    return await local(cards=cards, card_holders=holders, card_guard=guard), holders


async def test_a_lease_skips_a_card_a_process_holds(local: Make) -> None:
    box, holders = await _watching_the_cards(local)
    holders.holders = (INTRUDER,)

    lease = await box.lease_branch_sandbox("agent", 1, wait_sec=0)

    assert lease.gpu_ids == (1,)
    assert box.provider.pool.free_cards() == (0, 2), "the held card stays in the pool, un-leased"


async def test_a_lease_is_refused_at_once_when_every_free_card_is_held_and_names_who_holds_it(local: Make) -> None:
    box, holders = await _watching_the_cards(local)
    holders.holders = tuple(Holder(card=card, pid=4242 + card, command="python3 -c import torch") for card in range(3))
    started = time.monotonic()

    with pytest.raises(
        SandboxUnavailable, match=r"not freed within 0 s: .*card 0 by pid 4242 \(python3 -c import torch\)"
    ):
        await asyncio.wait_for(box.lease_branch_sandbox("agent", 1, wait_sec=0), 10)

    assert time.monotonic() - started < 5, "a lease that waits for nothing waited"
    assert (await box.measurement_capacity()).leased_cards == 0, "the refused lease gave its promise back"
    assert box.provider.pool.free_cards() == (0, 1, 2)


async def test_a_lease_that_waits_takes_the_card_when_the_process_lets_go(local: Make) -> None:
    box, holders = await _watching_the_cards(local)
    held = tuple(Holder(card=card, pid=4242 + card, command="python3 x.py") for card in range(3))
    holders.holders = held
    waiting = asyncio.ensure_future(box.lease_branch_sandbox("agent", 1, wait_sec=30))
    await asyncio.sleep(0.3)
    assert not waiting.done(), "the lease waits while every card is held"

    holders.holders = held[:2]

    assert (await asyncio.wait_for(waiting, 30)).gpu_ids == (2,)


async def test_a_lease_that_waits_gives_up_at_its_wait_and_says_who_held_the_cards(local: Make) -> None:
    box, holders = await _watching_the_cards(local)
    holders.holders = tuple(Holder(card=card, pid=4242 + card, command="") for card in range(3))

    with pytest.raises(SandboxUnavailable, match=r"not freed within 0.4 s: .*card 0 by pid 4242; card 1 by pid 4243"):
        await asyncio.wait_for(box.lease_branch_sandbox("agent", 1, wait_sec=0.4), 20)


async def test_a_lease_under_a_guard_that_only_observes_takes_a_card_a_process_holds_and_logs_it(
    local: Make, caplog: pytest.LogCaptureFixture
) -> None:
    box, holders = await _watching_the_cards(local, guard=CardGuard.OBSERVE)
    holders.holders = (INTRUDER,)

    with caplog.at_level("WARNING", logger="hyperloom.sandbox.gated_pool"):
        lease = await box.lease_branch_sandbox("agent", 1, wait_sec=0)

    assert lease.gpu_ids == (0,), "the held card was handed out as a plain pool would"
    assert [r.getMessage() for r in caplog.records] == [
        "cards a process holds were handed out (the card guard observes only): "
        "held card 0 by pid 4242 (python3 -c import torch)"
    ]


async def test_a_lease_with_no_guard_named_enforces(local: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_CARD_GUARD", raising=False)
    holders = CardList()
    holders.holders = tuple(Holder(card=card, pid=4242 + card, command="x") for card in range(3))
    box = await local(cards=3, card_holders=holders)

    with pytest.raises(SandboxUnavailable, match=r"card 0 by pid 4242 \(x\); card 1 by pid 4243"):
        await box.lease_branch_sandbox("agent", 1, wait_sec=0)
