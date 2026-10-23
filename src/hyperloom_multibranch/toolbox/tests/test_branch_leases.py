# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the local backend: cards out of the measurement pool, back on release or close.

The toolbox is the real local one (``build_local_toolbox``: the real queue, hosted backend, sandbox provider and card
pool) with the fake runner as the sandbox's command, on a two-card pool unless a test says otherwise.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hyperloom.measure.tests.hosted_fixtures import point_yaml, runner_template, spec_context
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, MeasurementCapacity, RequestRejected, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox, LocalOptions, LocalSession
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, build, req
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

    async def make(cards: int = 2, tp: int = 1, runner_envs: dict[str, str] | None = None, **branches: str) -> Any:
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
            options=LocalOptions(poll_interval_sec=0.05, runner_envs=runner_envs),
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
