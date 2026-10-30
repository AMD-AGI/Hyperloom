# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that wait for native builds: the ``pending_build`` state, its notices, its charges and its refusal."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import (
    BuildCharge,
    BuildPayment,
    FailureClass,
    Purpose,
    TestKind,
    TestRequest,
    Validity,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.blackboard.records import BuildNotice
from hyperloom_multibranch.contracts.blackboard.vocab import BuildOutcome
from hyperloom_multibranch.contracts.toolbox import (
    Backpressure,
    BuildState,
    BuildStatus,
    CancelOutcome,
    RequestRejected,
    TestProgress,
)
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring, MemoryBlackboard
from hyperloom_multibranch.toolbox.fake_builds import FakeBuild, FakeBuildDriver, FakeBuildEnd
from hyperloom_multibranch.toolbox.fake_model import FakeToolboxConfig
from hyperloom_multibranch.toolbox.pending_builds import BUILD_AFTER_DEADLINE
from hyperloom_multibranch.toolbox.queue import QueuePolicy

KERNEL = "sgl_kernel"
ETA = 34.0
CPU = 120.0
START = FakeToolboxConfig().start_time


def _request(branch: str = "b1", server_args: str = "--max-num-seqs 256", **overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.EXPLORE,
        "branch_id": branch,
        "parent_commit": "c0",
        "delta": RecipeDelta(config=ConfigDelta(server_args=server_args)),
    }
    return TestRequest(**{**fields, **overrides})


def _native(request: TestRequest) -> bool:
    return request.delta.config.server_args.startswith("--native")


def _rig(*, autorun: bool = True, **config: object) -> tuple[FakeToolbox, FakeBuildDriver, MemoryBlackboard]:
    driver, blackboard = FakeBuildDriver(eta=ETA), MemoryBlackboard()
    driver.need(_native, KERNEL, source="stack-a")
    box = FakeToolbox(
        FakeToolboxConfig(**config), blackboard=blackboard, autorun=autorun, wiring=FakeWiring(builds=driver)
    )
    return box, driver, blackboard


def _notices(blackboard: MemoryBlackboard) -> list[BuildNotice]:
    return [item for kind, item in blackboard.log if kind == "build"]  # type: ignore[misc]


def _kinds(blackboard: MemoryBlackboard) -> list[str]:
    return [kind for kind, _ in blackboard.log]


async def _let_builds_start() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _submit_held(box: FakeToolbox, driver: FakeBuildDriver, *requests: TestRequest) -> list[str]:
    driver.gate.clear()
    ids = [await box.submit_test(request) for request in requests]
    await _let_builds_start()
    return ids


async def _release(box: FakeToolbox, driver: FakeBuildDriver) -> None:
    driver.gate.set()
    await box.drain()


async def test_a_test_whose_stack_needs_a_build_waits_as_pending_build_holding_no_card() -> None:
    box, driver, _ = _rig()
    (request_id,) = await _submit_held(box, driver, _request(server_args="--native a"))
    progress = await box.progress(request_id)
    key = driver.needs(_request(server_args="--native a"), None)[0].key
    assert progress == TestProgress("pending_build", builds=(BuildStatus(key, KERNEL, BuildState.BUILDING, ETA),))
    state = await box.queue_state()
    assert (state.pending_build_by_branch, state.queued, state.running) == ({"b1": 1}, 0, 0)
    assert state.builds == (BuildStatus(key, KERNEL, BuildState.BUILDING, ETA),)
    await _release(box, driver)
    assert (await box.queue_state()).builds == (BuildStatus(key, KERNEL, BuildState.READY, 0.0, CPU),)


async def test_a_stack_that_needs_no_build_never_waits_and_no_notice_is_sent() -> None:
    box, driver, blackboard = _rig(autorun=False)
    request_id = await box.submit_test(_request())
    assert (await box.progress(request_id)).state == "running"  # type: ignore[union-attr]
    await box.drain()
    assert driver.ensured == [] and _notices(blackboard) == []
    assert box.results[request_id].builds == ()


async def test_ready_builds_queue_the_test_and_card_time_starts_then_charged_to_the_starter() -> None:
    box, driver, blackboard = _rig()
    (request_id,) = await _submit_held(box, driver, _request(server_args="--native a"))
    box.executor.now += 500.0  # the build took 500 s of the toolbox's clock
    await _release(box, driver)
    result = box.results[request_id]
    key = driver.ensured[0]
    assert result.validity is Validity.SUCCEEDED
    assert result.queued_at == START + 500.0
    assert result.builds == (BuildCharge(key, KERNEL, CPU, BuildPayment.PAID),)
    assert _notices(blackboard) == [BuildNotice(request_id, key, KERNEL, BuildOutcome.READY)]


async def test_equal_keys_of_two_branches_build_once_paid_by_the_first_and_each_is_told() -> None:
    box, driver, blackboard = _rig()
    first, second = await _submit_held(
        box, driver, _request("b1", server_args="--native a"), _request("b2", server_args="--native a")
    )
    await _release(box, driver)
    key = driver.ensured[0]
    assert driver.ensured == [key]
    assert box.results[first].builds == (BuildCharge(key, KERNEL, CPU, BuildPayment.PAID),)
    assert box.results[second].builds == (BuildCharge(key, KERNEL, 0.0, BuildPayment.REUSED),)
    assert {notice.request_id for notice in _notices(blackboard)} == {first, second}
    assert len(_notices(blackboard)) == 2


async def test_a_key_already_in_the_store_never_waits_and_is_charged_as_reused() -> None:
    box, driver, blackboard = _rig()
    await box.submit_test(_request("b1", server_args="--native a"))
    await box.drain()
    notices = len(_notices(blackboard))
    later = await box.submit_test(_request("b2", server_args="--native a"))
    assert (await box.progress(later)).state == "running"  # type: ignore[union-attr]
    await box.drain()
    assert len(_notices(blackboard)) == notices
    assert box.results[later].builds == (BuildCharge(driver.ensured[0], KERNEL, 0.0, BuildPayment.REUSED),)


async def test_a_candidate_compile_error_ends_the_test_with_a_result_and_no_notice() -> None:
    box, driver, blackboard = _rig()
    driver.script(KERNEL, FakeBuild(FakeBuildEnd.COMPILE_ERROR, "kernel.cu:3: error: no member named 'foo'"))
    request_id = await box.submit_test(_request(server_args="--native a"))
    await box.drain()
    result = box.results[request_id]
    key = driver.ensured[0]
    assert (result.validity, result.failure_class) == (Validity.CANDIDATE_FAILED, FailureClass.CANDIDATE)
    assert result.failure_reason == f"compile_error: {KERNEL} {key[:12]}: kernel.cu:3: error: no member named 'foo'"
    assert result.sandbox is None and result.gpu_seconds == 0.0 and result.queued_at is None
    assert result.builds == (BuildCharge(key, KERNEL, CPU, BuildPayment.PAID),)
    assert _notices(blackboard) == []
    assert _kinds(blackboard) == ["request", "result"]


async def test_a_machine_failure_is_told_to_the_waiting_test_and_the_build_is_retried() -> None:
    box, driver, blackboard = _rig()
    driver.script(KERNEL, FakeBuild(FakeBuildEnd.INFRA, "the build runner was evicted"))
    request_id = await box.submit_test(_request(server_args="--native a"))
    await box.drain()
    key = driver.ensured[0]
    assert driver.ensured == [key, key]
    assert _notices(blackboard) == [
        BuildNotice(request_id, key, KERNEL, BuildOutcome.FAILED, "BuildInfraError: the build runner was evicted"),
        BuildNotice(request_id, key, KERNEL, BuildOutcome.READY),
    ]
    assert box.results[request_id].validity is Validity.SUCCEEDED


async def test_machine_failures_past_the_retries_end_the_test_as_a_measurement_failure() -> None:
    box, driver, _ = _rig(max_infra_retries=1)
    driver.script(KERNEL, FakeBuild(FakeBuildEnd.INFRA), FakeBuild(FakeBuildEnd.INFRA))
    request_id = await box.submit_test(_request(server_args="--native a"))
    await box.drain()
    result = box.results[request_id]
    assert (result.validity, result.failure_class) == (Validity.MEASUREMENT_FAILED, FailureClass.INFRA)
    assert result.failure_reason is not None and "failed 2 times for the machine" in result.failure_reason
    assert len(driver.ensured) == 2


async def test_a_withdrawn_pending_test_gets_neither_notice_nor_result_when_its_build_ends() -> None:
    box, driver, blackboard = _rig()
    (request_id,) = await _submit_held(box, driver, _request(server_args="--native a"))
    assert await box.cancel(request_id) is CancelOutcome.WITHDRAWN
    assert (await box.queue_state()).pending_build_by_branch == {}
    await _release(box, driver)
    assert _notices(blackboard) == [] and request_id not in box.results
    assert _kinds(blackboard) == ["request", "withdrawal"]


async def test_a_pending_build_counts_against_the_branch_backlog() -> None:
    box, driver, blackboard = _rig(branch_queue_limit=1)
    await _submit_held(box, driver, _request(server_args="--native a"))
    with pytest.raises(Backpressure) as refused:
        await box.submit_test(_request(server_args="--native b"))
    assert (refused.value.queued, refused.value.limit) == (1, 1)
    await _release(box, driver)


def _deadline(box: FakeToolbox, seconds_from_now: float) -> QueuePolicy:
    return QueuePolicy(
        per_branch_k=2,
        global_concurrency=box.config.max_parallel_tests,
        max_queued_per_branch=box.config.branch_queue_limit,
        branch_deadline={"b1": box.now + seconds_from_now},
    )


async def test_a_build_that_would_end_past_the_deadline_is_refused_and_nothing_is_written() -> None:
    box, driver, blackboard = _rig()
    await box.set_policy(_deadline(box, ETA - 1))
    with pytest.raises(RequestRejected) as refused:
        await box.submit_test(_request(server_args="--native a"))
    assert refused.value.code == BUILD_AFTER_DEADLINE
    assert KERNEL in refused.value.reason and "eta 34 s" in refused.value.reason
    assert blackboard.log == [] and box.requests == {} and driver.ensured == []


async def test_the_deadline_counts_the_median_test_of_the_kind_after_the_build() -> None:
    box, driver, _ = _rig(attempt_seconds=600.0)
    await box.submit_test(_request(branch="b2"))
    await box.drain()
    await box.set_policy(_deadline(box, ETA + 300))
    with pytest.raises(RequestRejected) as refused:
        await box.submit_test(_request(server_args="--native a"))
    assert refused.value.code == BUILD_AFTER_DEADLINE and "one test (600 s)" in refused.value.reason
    await box.set_policy(_deadline(box, ETA + 700))
    await box.submit_test(_request(server_args="--native a"))
    await box.drain()


async def test_a_deadline_does_not_refuse_a_test_whose_builds_are_ready() -> None:
    box, driver, _ = _rig()
    await box.submit_test(_request("b2", server_args="--native a"))
    await box.drain()
    await box.set_policy(_deadline(box, 1.0))
    request_id = await box.submit_test(_request(server_args="--native a"))
    await box.drain()
    assert box.results[request_id].builds[0].payment is BuildPayment.REUSED


# --- a restart while a test waits for its build ------------------------------------------------------------------------


def _restartable(log: Path, driver: FakeBuildDriver, blackboard: MemoryBlackboard) -> FakeToolbox:
    """A toolbox over ``log``; a second one over the same log, driver (the shared store) and blackboard is a restart."""
    return FakeToolbox(FakeToolboxConfig(), blackboard=blackboard, wiring=FakeWiring(builds=driver, log_path=log))


async def _died_waiting(tmp_path: Path) -> tuple[FakeBuildDriver, MemoryBlackboard, str, str]:
    """A test accepted and waiting for its build when the toolbox dies; the build never got past the gate."""
    driver, blackboard = FakeBuildDriver(eta=ETA), MemoryBlackboard()
    driver.need(_native, KERNEL, source="stack-a")
    box = _restartable(tmp_path / "submissions.jsonl", driver, blackboard)
    (request_id,) = await _submit_held(box, driver, _request(server_args="--native a"))
    await box.close()
    assert driver.ensured == [] and box.results == {}
    return driver, blackboard, request_id, driver.needs(_request(server_args="--native a"), None)[0].key


async def _restart(tmp_path: Path, driver: FakeBuildDriver, blackboard: MemoryBlackboard) -> FakeToolbox:
    again = _restartable(tmp_path / "submissions.jsonl", driver, blackboard)
    await again.check_test(_request())  # the fake toolbox starts, and recovers its log, on first use
    await _let_builds_start()
    return again


async def test_a_test_waiting_for_a_build_nobody_ran_builds_it_after_a_restart(tmp_path: Path) -> None:
    driver, blackboard, request_id, key = await _died_waiting(tmp_path)
    driver.gate.set()  # nothing builds the key while the toolbox is down

    again = await _restart(tmp_path, driver, blackboard)
    await again.drain()

    assert driver.ensured == [key], "planned again from the logged request and built once"
    result = again.results[request_id]
    assert result.validity is Validity.SUCCEEDED
    assert result.builds == (BuildCharge(key, KERNEL, CPU, BuildPayment.PAID),)


async def test_a_test_whose_build_is_still_running_waits_for_it_after_a_restart(tmp_path: Path) -> None:
    driver, blackboard, request_id, key = await _died_waiting(tmp_path)

    again = await _restart(tmp_path, driver, blackboard)  # the gate is still closed: the key is being built
    progress = await again.progress(request_id)
    assert progress == TestProgress("pending_build", builds=(BuildStatus(key, KERNEL, BuildState.BUILDING, ETA),))
    assert again.results == {}

    await _release(again, driver)
    assert again.results[request_id].validity is Validity.SUCCEEDED
    assert driver.ensured == [key]


async def test_a_test_whose_build_landed_while_the_toolbox_was_down_is_queued_at_once(tmp_path: Path) -> None:
    driver, blackboard, request_id, key = await _died_waiting(tmp_path)
    driver.gate.set()
    (inputs,) = driver.needs(_request(server_args="--native a"), None)
    await driver.ensure(inputs)  # another toolbox on the store finished the key meanwhile
    ensured = list(driver.ensured)

    again = await _restart(tmp_path, driver, blackboard)
    await again.drain()

    assert driver.ensured == ensured, "a ready key is read from the store, not built again"
    result = again.results[request_id]
    assert result.validity is Validity.SUCCEEDED
    assert result.builds == (BuildCharge(key, KERNEL, 0.0, BuildPayment.REUSED),)
