# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, Validity
from hyperloom_multibranch.contracts.gate import ResultUnverified, ResultVerifier
from hyperloom_multibranch.contracts.notebook import Role
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, DedupStatus
from hyperloom_multibranch.contracts.toolbox import Backpressure, CancelOutcome, RequestRejected, UnknownTestRequest
from hyperloom_multibranch.notebook.tests.factories import TTL, intent
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.toolbox import RealToolbox
from hyperloom_multibranch.toolbox.tests.helpers import (
    PARENT,
    ScriptedExecutor,
    build,
    delta,
    failed,
    ok,
    req,
    settle,
)

K1 = QueuePolicy(per_branch_k=1, global_concurrency=9)


def _gated() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    return executor


async def _record(rig, request_id: str):
    return await rig.notebook.record(request_id)


# --- results and the notebook -----------------------------------------------------------------------------------


async def test_a_measured_test_is_written_to_the_notebook_as_a_result(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert record.result is not None and record.result.validity.value == "succeeded"
    assert record.result.sandbox.node == "n1" and record.result.queued_at is not None
    assert {(r.phase, metric) for r in record.result.rounds for metric in r.metrics} == {("measure", "output_tps")}
    assert [(a.task, a.metric, a.score) for a in record.result.accuracy] == [("gsm8k", "exact_match", 0.9)]
    assert rig.toolbox._log.get(request_id).settled  # type: ignore[union-attr]


async def test_the_claim_is_carried_by_the_request_renewed_and_completed_by_the_result(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=_gated())
    specialist = rig.open_notebook(Role.SPECIALIST_RUNTIME, "b1")
    target, action = intent("b1", PARENT), delta()
    claim = await specialist.acquire(target, action.fingerprint(), ttl_seconds=TTL)
    request_id = await rig.toolbox.submit_test(req(intent=target.fingerprint, change=action))
    await settle()
    held = (await specialist.lookup(claim.key, action.fingerprint())).claim
    assert held is not None and held.request_id == request_id
    rig.clock.advance(100)  # type: ignore[attr-defined]
    await rig.toolbox.renew_claims()
    renewed = (await specialist.lookup(claim.key, action.fingerprint())).claim
    assert renewed is not None and renewed.expires_at > held.expires_at
    rig.executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()
    done = (await specialist.lookup(claim.key, action.fingerprint())).claim
    assert done is None or done.status is ClaimStatus.COMPLETED


async def test_a_request_whose_claim_is_not_held_is_refused_and_withdrawn_and_its_number_is_not_reused(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path)
    with pytest.raises(RequestRejected, match="claim"):
        await rig.toolbox.submit_test(req(intent="nobody-claimed-this"))
    state = rig.notebook._state()
    (lost,) = state.withdrawals
    assert lost == rig.toolbox._log.request_id("b1", 1)
    assert await rig.toolbox.submit_test(req()) == rig.toolbox._log.request_id("b1", 2)


# --- priority, K, backpressure ------------------------------------------------------------------------------------


async def test_queued_tests_start_in_priority_order(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=QueuePolicy(per_branch_k=9, global_concurrency=1))
    first = await rig.toolbox.submit_test(req("b1", server_args="--a"))
    explore = await rig.toolbox.submit_test(req("b2", server_args="--b"))
    profile = await rig.toolbox.submit_test(req("b3", purpose=Purpose.PROFILE, server_args="--c"))
    integ = await rig.toolbox.submit_test(req("b4", purpose=Purpose.INTEGRATION, server_args="--d"))
    confirm = await rig.toolbox.submit_test(req("b5", purpose=Purpose.CONFIRM, retest_of=first, server_args="--e"))
    for _ in range(5):
        executor.gate.set()  # type: ignore[union-attr]
        await settle()
        executor.gate = asyncio.Event()
    executor.gate.set()
    await rig.toolbox.drain()
    assert executor.started == [first, confirm, integ, profile, explore]


async def test_a_branch_runs_at_most_k_and_a_confirm_retest_is_not_blocked_by_it(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=K1)
    first = await rig.toolbox.submit_test(req(server_args="--a"))
    second = await rig.toolbox.submit_test(req(server_args="--b"))
    await settle()
    assert executor.started == [first]
    state = await rig.toolbox.queue_state()
    assert (dict(state.running_by_branch), dict(state.queued_by_branch)) == ({"b1": 1}, {"b1": 1})
    confirm = await rig.toolbox.submit_test(req(purpose=Purpose.CONFIRM, retest_of=first, server_args="--c"))
    await settle()
    assert executor.started == [first, confirm], "the confirm-retest must run although the branch is at K"
    assert second not in executor.started
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()
    assert executor.started[-1] == second


async def test_backpressure_refuses_the_submit_and_writes_nothing(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(
        tmp_path, executor=executor, policy=QueuePolicy(per_branch_k=1, global_concurrency=9, max_queued_per_branch=2)
    )
    await rig.toolbox.submit_test(req(server_args="--a"))
    await settle()
    await rig.toolbox.submit_test(req(server_args="--b"))
    await rig.toolbox.submit_test(req(server_args="--c"))
    written = len(rig.notebook._state().requests)
    with pytest.raises(Backpressure) as refused:
        await rig.toolbox.submit_test(req(server_args="--d"))
    assert (refused.value.queued, refused.value.limit) == (2, 2)
    assert len(rig.notebook._state().requests) == written
    await rig.toolbox.submit_test(req("b2", server_args="--e"))  # another branch is unaffected
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()


async def test_a_new_policy_from_the_scheduler_takes_effect_on_the_queue(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=K1)
    for name in ("--a", "--b", "--c"):
        await rig.toolbox.submit_test(req(server_args=name))
    await settle()
    assert len(executor.started) == 1
    await rig.toolbox.set_policy(QueuePolicy(per_branch_k=3, global_concurrency=9))
    await settle()
    assert len(executor.started) == 3
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()


# --- attempts and reruns --------------------------------------------------------------------------------------------


async def test_an_unclear_failure_is_rerun_on_another_node_and_the_first_node_is_quarantined(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.UNCLEAR, "n1"), lambda j: ok(j, "n2")])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert record.result is not None and record.result.validity.value == "succeeded"
    assert [(a.node, a.failure and a.failure.value) for a in record.result.attempts] == [("n1", "infra"), ("n2", None)]
    assert executor.jobs[1].avoid_nodes == ("n1",) and executor.jobs[1].attempt == 2
    assert [event.node for event in await rig.notebook.quarantined()] == ["n1"]


async def test_an_unclear_failure_that_repeats_is_the_candidates_failure(tmp_path: Path) -> None:
    executor = ScriptedExecutor(
        [lambda j: failed(j, FailureClass.UNCLEAR, "n1"), lambda j: failed(j, FailureClass.UNCLEAR, "n2")]
    )
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert len(executor.jobs) == 2, "an unclear failure is rerun once, not more"
    assert record.result is not None
    assert (record.result.validity.value, record.result.failure.value) == ("candidate_failed", "unclear")  # type: ignore[union-attr]
    assert await rig.notebook.quarantined() == ()


async def test_a_candidate_failure_is_final_without_a_rerun(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.CANDIDATE, "n1", "import_error")])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert len(executor.jobs) == 1
    assert record.result is not None and record.result.failure_reason == "import_error"


async def test_infra_failures_retry_up_to_the_limit_then_measurement_failed_is_stored_as_the_result(
    tmp_path: Path,
) -> None:
    nodes = iter(["n1", "n2", "n3", "n4", "n5"])
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.INFRA, next(nodes), "pull failed")])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    assert len(executor.jobs) == 4, "one try and three retries"
    record = await _record(rig, request_id)
    assert record.result is not None and record.result.validity is Validity.MEASUREMENT_FAILED
    assert record.withdrawn_at is None and request_id not in rig.notebook._state().withdrawals
    assert rig.toolbox._log.get(request_id).settled  # type: ignore[union-attr]
    assert {event.node for event in await rig.notebook.quarantined()} == {"n1", "n2", "n3", "n4"}


async def test_a_measurement_failed_result_releases_its_claim_and_dedup_does_not_call_it_tested(
    tmp_path: Path,
) -> None:
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.INFRA, "n1", "pull failed")])
    rig = build(tmp_path, executor=executor)
    specialist = rig.open_notebook(Role.SPECIALIST_RUNTIME, "b1")
    target, action = intent("b1", PARENT), delta()
    claim = await specialist.acquire(target, action.fingerprint(), ttl_seconds=TTL)
    request_id = await rig.toolbox.submit_test(req(intent=target.fingerprint, change=action))
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert record.result is not None and record.result.validity is Validity.MEASUREMENT_FAILED
    answer = await specialist.lookup(claim.key, action.fingerprint())
    assert answer.claim is None, "the claim was released, not held or completed"
    assert answer.status is not DedupStatus.TESTED and answer.status is not DedupStatus.IN_PROGRESS
    await rig.toolbox.tick()
    assert len(rig.notebook._state().results) == 1, "written once, never again"


async def test_an_infra_failure_then_success_is_the_result(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.INFRA, "n1"), lambda j: ok(j, "n2")])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert record.result is not None and len(record.result.attempts) == 2


async def test_a_refused_measurement_is_not_retried_and_is_stored_as_measurement_failed(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    executor.refuse = "spec_rejected: model hash differs"
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert len(executor.jobs) == 1
    assert record.withdrawn_at is None and record.result is not None
    assert record.result.validity is Validity.MEASUREMENT_FAILED and "spec_rejected" in record.result.failure_reason


async def test_a_crashing_executor_is_a_measurement_failed_result_not_a_lost_test(tmp_path: Path) -> None:
    class Crashing(ScriptedExecutor):
        async def run(self, job):
            raise RuntimeError("executor bug")

    rig = build(tmp_path, executor=Crashing())
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    record = await _record(rig, request_id)
    assert record.withdrawn_at is None and record.result is not None
    assert record.result.validity is Validity.MEASUREMENT_FAILED and "executor bug" in record.result.failure_reason


async def test_on_a_single_node_avoiding_it_still_runs_there_and_marks_same_node(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: ok(j, "only-node")])
    rig = build(tmp_path, executor=executor)
    first = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    retest = await rig.toolbox.submit_test(req(purpose=Purpose.CONFIRM, retest_of=first, avoid_nodes=("only-node",)))
    await rig.toolbox.drain()
    assert executor.jobs[-1].avoid_nodes == ("only-node",)
    outcome = rig.toolbox._log.get(retest).outcome  # type: ignore[union-attr]
    assert outcome["result"]["sandbox"]["same_node"] is True  # type: ignore[index]
    first_outcome = rig.toolbox._log.get(first).outcome  # type: ignore[union-attr]
    assert first_outcome["result"]["sandbox"]["same_node"] is False  # type: ignore[index]


async def test_a_quarantined_node_is_passed_to_every_later_test_as_avoided(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.INFRA, "bad"), lambda j: ok(j, "good")])
    rig = build(tmp_path, executor=executor)
    await rig.toolbox.submit_test(req(server_args="--a"))
    await rig.toolbox.drain()
    executor.script = [lambda j: ok(j, "good")]
    await rig.toolbox.submit_test(req(server_args="--b"))
    await rig.toolbox.drain()
    assert executor.jobs[-1].avoid_nodes == ("bad",)
    assert rig.toolbox.quarantined == {"bad"}


async def test_the_sandbox_priority_tier_is_handed_to_the_executor(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await rig.toolbox.submit_test(req(priority=3))
    await rig.toolbox.drain()
    assert (rig.executor.jobs[0].priority, rig.executor.jobs[0].sandbox_priority) == (3, 1)


# --- cancel ----------------------------------------------------------------------------------------------------------


async def test_cancelling_a_queued_test_withdraws_it_and_is_idempotent(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=K1)
    await rig.toolbox.submit_test(req(server_args="--a"))
    queued = await rig.toolbox.submit_test(req(server_args="--b"))
    await settle()
    assert await rig.toolbox.cancel(queued) is CancelOutcome.WITHDRAWN
    assert await rig.toolbox.cancel(queued) is CancelOutcome.WITHDRAWN
    record = await _record(rig, queued)
    assert record.withdrawn_at is not None and record.result is None
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()
    assert queued not in executor.started


async def test_cancelling_a_running_test_stops_it_and_writes_no_result(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor)
    running = await rig.toolbox.submit_test(req())
    await settle()
    assert executor.started == [running]
    assert await rig.toolbox.cancel(running) is CancelOutcome.WITHDRAWN
    executor.gate.set()  # type: ignore[union-attr]
    await settle()
    record = await _record(rig, running)
    assert record.result is None and record.withdrawn_at is not None
    assert (await rig.toolbox.queue_state()).running == 0


async def test_cancelling_a_finished_test_changes_nothing_and_an_unknown_id_is_an_error(tmp_path: Path) -> None:
    rig = build(tmp_path)
    done = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    assert await rig.toolbox.cancel(done) is CancelOutcome.FINISHED
    assert (await _record(rig, done)).withdrawn_at is None
    with pytest.raises(UnknownTestRequest):
        await rig.toolbox.cancel("never-accepted")


async def test_a_stopped_branch_has_every_queued_and_running_test_withdrawn(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor, policy=K1)
    stopped = [await rig.toolbox.submit_test(req("b1", server_args=f"--{n}")) for n in "ab"]
    other = await rig.toolbox.submit_test(req("b2"))
    await settle()
    assert await rig.toolbox.cancel_branch("b1") == 2
    executor.gate.set()  # type: ignore[union-attr]
    await rig.toolbox.drain()
    for request_id in stopped:
        assert (await _record(rig, request_id)).withdrawn_at is not None
    assert (await _record(rig, other)).result is not None


# --- re-verifying a recorded result (L26) -----------------------------------------------------------------------

BUNDLE = "a" * 64


def _unverified(raised: pytest.ExceptionInfo[ResultUnverified]) -> str:
    return raised.value.reason


def test_the_toolbox_is_a_result_verifier(tmp_path: Path) -> None:
    verifier: ResultVerifier = build(tmp_path).toolbox  # the annotation is the static check
    assert isinstance(verifier, RealToolbox)


async def test_a_recorded_result_with_a_bundle_hash_goes_to_the_executor(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: ok(j, manifest_sha256=BUNDLE)])
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    await rig.toolbox.reverify(request_id)
    assert [r.request_id for r in executor.verified] == [request_id]
    assert executor.verified[0].manifest_sha256 == BUNDLE


async def test_what_the_executor_names_comes_back_as_the_reason(tmp_path: Path) -> None:
    executor = ScriptedExecutor([lambda j: ok(j, manifest_sha256=BUNDLE)])
    executor.unverified = "rounds[0].metrics.output_tps: recorded 1, the bundle has 2"
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    with pytest.raises(ResultUnverified) as raised:
        await rig.toolbox.reverify(request_id)
    assert raised.value.request_id == request_id
    assert _unverified(raised) == "rounds[0].metrics.output_tps: recorded 1, the bundle has 2"


async def test_a_request_the_notebook_never_heard_of_is_unverified(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    rig = build(tmp_path, executor=executor)
    with pytest.raises(ResultUnverified) as raised:
        await rig.toolbox.reverify("nope")
    assert _unverified(raised) == "the notebook holds no record of this request"
    assert executor.verified == []


async def test_a_record_without_a_result_is_unverified(tmp_path: Path) -> None:
    executor = _gated()
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    with pytest.raises(ResultUnverified) as raised:
        await rig.toolbox.reverify(request_id)
    assert _unverified(raised) == "the record holds no result"
    assert executor.verified == []
    executor.gate.set()
    await rig.toolbox.drain()


async def test_a_result_without_a_manifest_hash_is_unverified(tmp_path: Path) -> None:
    executor = ScriptedExecutor()  # ok() carries no hash
    rig = build(tmp_path, executor=executor)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    with pytest.raises(ResultUnverified) as raised:
        await rig.toolbox.reverify(request_id)
    assert _unverified(raised) == "the result names no manifest hash"
    assert executor.verified == []


async def test_a_toolbox_without_a_notebook_to_read_is_unverified(tmp_path: Path) -> None:
    rig = build(tmp_path)
    rig.toolbox._records = None
    with pytest.raises(ResultUnverified) as raised:
        await rig.toolbox.reverify("any")
    assert _unverified(raised) == "the toolbox reads no notebook records"
