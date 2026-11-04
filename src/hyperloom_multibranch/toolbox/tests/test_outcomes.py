# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Waiting for a test's outcome and withdrawing tests with their ids (the scheduler's ``result`` and ``withdraw``)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Validity
from hyperloom_multibranch.contracts.toolbox import TestWithdrawn, UnknownTestRequest
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, failed, req, settle

K1 = QueuePolicy(per_branch_k=1, global_concurrency=9)


def _gated() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    return executor


async def test_outcome_waits_for_the_result_and_returns_it(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    waiter = asyncio.ensure_future(rig.toolbox.outcome(request_id))
    await settle()
    assert not waiter.done()
    executor.gate.set()  # type: ignore[union-attr]
    result = await asyncio.wait_for(waiter, 5)
    assert result.request_id == request_id and result.validity is Validity.SUCCEEDED


async def test_outcome_of_a_finished_test_returns_at_once(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    assert (await asyncio.wait_for(rig.toolbox.outcome(request_id), 1)).request_id == request_id


async def test_outcome_returns_a_measurement_failed_result_instead_of_a_withdrawal(tmp_path: Path) -> None:
    executor = ScriptedExecutor(script=[lambda job: failed(job, FailureClass.INFRA)])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    result = await asyncio.wait_for(rig.toolbox.outcome(request_id), 5)
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert len(executor.jobs) == DEFAULT_MAX_INFRA_RETRIES + 1


async def test_outcome_of_a_withdrawn_test_raises_withdrawn_with_the_reason(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=_gated())
    request_id = await rig.toolbox.submit_test(req())
    waiter = asyncio.ensure_future(rig.toolbox.outcome(request_id))
    await settle()
    await rig.toolbox.withdraw_branch("b1", "stop for the test")
    with pytest.raises(TestWithdrawn) as raised:
        await asyncio.wait_for(waiter, 5)
    assert raised.value.reason == "stop for the test" and raised.value.request_id == request_id


async def test_outcome_of_an_unknown_id_is_an_unknown_request(tmp_path: Path) -> None:
    rig = build(tmp_path)
    with pytest.raises(UnknownTestRequest):
        await rig.toolbox.outcome("never-accepted")


async def test_withdraw_branch_returns_the_ids_of_the_queued_and_running_tests_of_that_branch_only(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path, executor=_gated(), policy=K1)
    stopped = [await rig.toolbox.submit_test(req("b1", server_args=f"--{n}")) for n in "ab"]
    other = await rig.toolbox.submit_test(req("b2"))
    await settle()
    assert set(await rig.toolbox.withdraw_branch("b1")) == set(stopped)
    assert await rig.toolbox.withdraw_branch("b1") == ()  # idempotent: nothing left to withdraw
    assert not rig.toolbox._log.get(other).outcome  # type: ignore[union-attr]


async def test_withdraw_all_returns_every_waiting_test_and_leaves_finished_ones(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=K1)
    first = await rig.toolbox.submit_test(req("b1"))
    second = await rig.toolbox.submit_test(req("b2"))
    await settle()
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()
    executor.gate.clear()  # type: ignore[union-attr]
    queued = [await rig.toolbox.submit_test(req(branch, server_args="--x")) for branch in ("b1", "b2")]
    await settle()
    assert set(await rig.toolbox.withdraw_all()) == set(queued)
    assert first not in queued and second not in queued
    assert (await rig.toolbox.outcome(first)).validity is Validity.SUCCEEDED
