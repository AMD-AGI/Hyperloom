# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import (
    FailureClass,
    Purpose,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook.records import QuarantineEvent, Withdrawal
from hyperloom_multibranch.contracts.notebook.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import (
    Backpressure,
    CancelOutcome,
    RequestRejected,
    Toolbox,
    UnknownTestRequest,
)
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate, FailureRatios, FakeToolboxConfig, NoiseModel

DELTA = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
QUIET = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)


class Notebook:
    """Records what the toolbox writes, in order."""

    def __init__(self) -> None:
        self.log: list[tuple[str, object]] = []

    async def write_request(self, request: TestRequest) -> None:
        self.log.append(("request", request))

    async def write_result(self, result: TestResult) -> None:
        self.log.append(("result", result))

    async def write_withdrawal(self, withdrawal: Withdrawal) -> None:
        self.log.append(("withdrawal", withdrawal))

    async def record_quarantine(self, event: QuarantineEvent) -> None:
        self.log.append(("quarantine", event))

    def of(self, kind: str) -> list:
        return [item for name, item in self.log if name == kind]


def _request(branch: str = "b1", **overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.EXPLORE,
        "branch_id": branch,
        "parent_commit": "c0",
        "delta": DELTA,
    }
    return TestRequest(**{**fields, **overrides})


def _toolbox(*, notebook: Notebook | None = None, autorun: bool = False, **config: object) -> FakeToolbox:
    return FakeToolbox(FakeToolboxConfig(**config), notebook=notebook, autorun=autorun)


async def _run_one(box: FakeToolbox, request: TestRequest | None = None) -> TestResult:
    request_id = await box.submit_test(request or _request())
    await box.drain()
    return box.results[request_id]


def test_the_fake_is_a_toolbox() -> None:
    box: Toolbox = FakeToolbox()
    assert box is not None


# --- acceptance --------------------------------------------------------------------------------------------------


async def test_submit_returns_a_stable_uuid_and_writes_the_request_before_returning() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    first = await box.submit_test(_request())
    assert book.of("request")[0].request_id == first
    assert box.requests[first].request_id == first
    twin = _toolbox()
    assert await twin.submit_test(_request()) == first
    assert await box.submit_test(_request()) != first


async def test_a_rejection_raises_with_its_reason_and_writes_nothing() -> None:
    book = Notebook()
    box = FakeToolbox(notebook=book, autorun=False, parent_exists=lambda commit: commit == "c0")
    with pytest.raises(RequestRejected) as raised:
        await box.submit_test(_request(parent_commit="c9"))
    assert raised.value.code == "unknown_parent"
    with pytest.raises(RequestRejected) as raised:
        await box.submit_test(_request(purpose=Purpose.CONFIRM, retest_of="nope"))
    assert raised.value.code == "unknown_retest"
    box.reject_if(
        lambda r: RequestRejected("patch does not apply", "delta_not_applicable") if r.delta.components == () else None
    )
    with pytest.raises(RequestRejected, match="patch does not apply"):
        await box.submit_test(_request())
    assert book.log == [] and box.requests == {}


async def test_confirm_retest_of_a_known_request_is_accepted() -> None:
    box = _toolbox()
    first = await box.submit_test(_request())
    again = await box.submit_test(_request(purpose=Purpose.CONFIRM, retest_of=first, avoid_nodes=("node-0",)))
    assert box.requests[again].retest_of == first


async def test_backpressure_is_distinct_and_per_branch() -> None:
    box = _toolbox(branch_queue_limit=2)
    await box.submit_test(_request("b1"))
    await box.submit_test(_request("b1"))
    with pytest.raises(Backpressure) as raised:
        await box.submit_test(_request("b1"))
    assert (raised.value.queued, raised.value.limit) == (2, 2)
    assert not isinstance(raised.value, RequestRejected)
    await box.submit_test(_request("b2"))
    assert len(box.requests) == 3
    await box.drain()
    await box.submit_test(_request("b1"))


# --- queue -------------------------------------------------------------------------------------------------------


async def test_default_priority_follows_the_purpose_and_queue_state_shows_it() -> None:
    box = _toolbox()
    await box.submit_test(_request(purpose=Purpose.EXPLORE))
    await box.submit_test(_request(purpose=Purpose.PROFILE, delta=DELTA))
    await box.submit_test(_request(purpose=Purpose.REPRO, kind=TestKind.REPRO, attachments=_attach()))
    await box.submit_test(_request(purpose=Purpose.EXPLORE, priority=1))
    state = await box.queue_state()
    assert dict(state.queued_by_priority) == {0: 0, 1: 2, 2: 0, 3: 1, 4: 1}
    assert dict(state.queued_by_branch) == {"b1": 4}
    assert state.running == 0


def _attach():
    from hyperloom.contracts.measurement import AttachmentRef

    return (AttachmentRef("case", "file", "a" * 64),)


async def test_more_urgent_tests_run_first() -> None:
    order: list[str] = []
    book = Notebook()
    box = FakeToolbox(FakeToolboxConfig(default_k=9, max_parallel_tests=1), notebook=book, autorun=False)
    slow = await box.submit_test(_request(priority=4))
    urgent = await box.submit_test(_request(priority=0))
    middle = await box.submit_test(_request(priority=2))
    await box.drain()
    order = [r.request_id for r in book.of("result")]
    assert order == [urgent, middle, slow]


async def test_a_branch_runs_at_most_k_tests_but_confirmations_do_not_count() -> None:
    box = _toolbox(default_k=1, run_delay_s=0.01, max_parallel_tests=9)
    box_auto = FakeToolbox(box.config, autorun=True)
    await box_auto.submit_test(_request())
    await box_auto.submit_test(_request())
    first = next(iter(box_auto.requests))
    await box_auto.submit_test(_request(purpose=Purpose.CONFIRM, retest_of=first))
    state = await box_auto.queue_state()
    assert state.running_by_branch == {"b1": 2}  # one in K, and the confirmation outside it
    assert state.queued == 1
    await box_auto.drain()
    assert len(box_auto.results) == 3


async def test_set_k_widens_a_branch() -> None:
    box = FakeToolbox(FakeToolboxConfig(default_k=1, run_delay_s=0.01), autorun=True)
    box.set_k("b1", 3)
    for _ in range(3):
        await box.submit_test(_request())
    assert (await box.queue_state()).running_by_branch == {"b1": 3}
    await box.drain()
    with pytest.raises(ValueError, match="at least one"):
        box.set_k("b1", 0)


async def test_the_global_limit_holds_across_branches() -> None:
    box = FakeToolbox(FakeToolboxConfig(max_parallel_tests=2, run_delay_s=0.01), autorun=True)
    for branch in ("b1", "b2", "b3"):
        await box.submit_test(_request(branch))
    state = await box.queue_state()
    assert (state.running, state.queued) == (2, 1)
    await box.drain()


# --- results and attempts ----------------------------------------------------------------------------------------


async def test_a_quiet_success_is_complete_and_written_to_the_notebook() -> None:
    book = Notebook()
    box = _toolbox(notebook=book, noise=QUIET, default_effect=0.02)
    result = await _run_one(box)
    assert result.validity is Validity.SUCCEEDED and result.failure_class is None
    assert book.of("result") == [result]
    assert result.sandbox is not None and result.sandbox.node_name in box.config.nodes
    assert len(result.attempts) == 1 and result.attempts[0].sandbox == result.sandbox
    assert [r.phase for r in result.rounds] == ["warmup", "measure", "measure", "measure"]
    assert result.rounds[1].metrics["output_tps"] == pytest.approx(1020.0)
    assert result.manifest_sha256 and result.output_dir == f"tests/{result.request_id}"
    assert result.gpu_seconds == box.config.attempt_seconds * box.config.gpu_count
    assert result.finished_at - result.started_at == box.config.attempt_seconds
    assert TestResult.from_document(result.to_document()) == result


async def test_a_profile_test_reports_where_the_profile_is() -> None:
    result = await _run_one(_toolbox(), _request(kind=TestKind.PROFILE, purpose=Purpose.PROFILE))
    assert result.profile_location == f"tests/{result.request_id}/profile"


async def test_infra_failure_retries_on_another_node_and_the_submitter_gets_one_result() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    box.script(lambda r: True, [Fate.INFRA, Fate.INFRA, Fate.OK])
    result = await _run_one(box)
    assert result.validity is Validity.SUCCEEDED
    assert [a.failure_class for a in result.attempts] == [FailureClass.INFRA, FailureClass.INFRA, None]
    nodes = [a.sandbox.node_name for a in result.attempts]
    assert len(set(nodes)) == 3
    assert len({a.sandbox.sandbox_id for a in result.attempts}) == 3
    assert len(book.of("result")) == 1
    assert [e.node for e in book.of("quarantine")] == nodes[:2]
    assert box.quarantined == set(nodes[:2])
    assert all(e.action is QuarantineAction.QUARANTINE for e in book.of("quarantine"))


async def test_infra_retries_run_out_as_measurement_failed_never_a_candidate_failure() -> None:
    box = _toolbox(max_infra_retries=3)
    box.script(lambda r: True, [Fate.INFRA] * 10)
    result = await _run_one(box)
    assert len(result.attempts) == 4
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert result.failure_class is FailureClass.INFRA
    assert result.manifest_sha256 is None and result.rounds == ()


async def test_unclear_failure_reruns_once_and_the_rerun_wins() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    box.script(lambda r: True, [Fate.UNCLEAR, Fate.OK])
    result = await _run_one(box)
    assert result.validity is Validity.SUCCEEDED and result.failure_class is None
    first, second = result.attempts
    assert first.failure_class is FailureClass.UNCLEAR and second.failure_class is None
    assert second.sandbox.node_name != first.sandbox.node_name
    assert [e.node for e in book.of("quarantine")] == [first.sandbox.node_name]


async def test_unclear_failure_that_repeats_is_the_candidates() -> None:
    box = _toolbox()
    box.script(lambda r: True, [Fate.UNCLEAR, Fate.UNCLEAR, Fate.OK])
    result = await _run_one(box)
    assert len(result.attempts) == 2
    assert result.validity is Validity.CANDIDATE_FAILED
    assert result.failure_class is FailureClass.UNCLEAR
    assert box.quarantined == set()


async def test_candidate_failure_is_final_with_a_signature_reason() -> None:
    box = _toolbox()
    box.script(lambda r: True, [Fate.CANDIDATE, Fate.OK])
    result = await _run_one(box)
    assert len(result.attempts) == 1
    assert (result.validity, result.failure_class) == (Validity.CANDIDATE_FAILED, FailureClass.CANDIDATE)
    assert result.failure_reason == "serve_flag"


async def test_a_failed_measurement_is_never_retried_and_never_the_candidates() -> None:
    box = _toolbox()
    box.script(lambda r: True, [Fate.MEASUREMENT, Fate.OK])
    result = await _run_one(box)
    assert len(result.attempts) == 1
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert result.failure_class is not FailureClass.CANDIDATE
    assert box.quarantined == set()


async def test_avoid_nodes_are_honoured_by_the_first_attempt() -> None:
    nodes = tuple(f"n{i}" for i in range(4))
    box = _toolbox(nodes=nodes)
    result = await _run_one(box, _request(avoid_nodes=("n0", "n1", "n2")))
    assert result.sandbox.node_name == "n3" and result.sandbox.same_node is False
    result = await _run_one(box, _request(avoid_nodes=nodes))
    assert result.sandbox.same_node is True


async def test_a_fleet_with_every_node_quarantined_ends_in_measurement_failed() -> None:
    box = _toolbox(nodes=("a", "b"))
    box.script(lambda r: True, [Fate.INFRA] * 10)
    result = await _run_one(box)
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert box.quarantined == {"a", "b"}
    assert result.attempts[-1].sandbox is None


async def test_scripts_only_touch_matching_requests() -> None:
    box = _toolbox()
    box.script(lambda r: r.branch_id == "bad", [Fate.CANDIDATE])
    good = await _run_one(box, _request("good"))
    bad = await _run_one(box, _request("bad"))
    assert (good.validity, bad.validity) == (Validity.SUCCEEDED, Validity.CANDIDATE_FAILED)


async def test_random_failure_ratios_reach_every_class_and_replay_with_the_seed() -> None:
    config = FakeToolboxConfig(
        seed=11,
        failures=FailureRatios(infra=0.15, unclear=0.15, candidate=0.15, measurement=0.1),
        branch_queue_limit=500,
        max_parallel_tests=500,
        default_k=500,
    )

    async def outcomes() -> list[tuple[str, str | None]]:
        box = FakeToolbox(config, autorun=False)
        ids = [await box.submit_test(_request()) for _ in range(300)]
        await box.drain()
        return [(box.results[i].validity.value, box.results[i].failure_reason) for i in ids] + [
            ("attempt", a.failure_reason) for i in ids for a in box.results[i].attempts
        ]

    first = await outcomes()
    assert first == await outcomes()
    assert {v for v, _ in first if v != "attempt"} == {"succeeded", "candidate_failed", "measurement_failed"}
    reasons = {reason for _, reason in first}
    assert {"serve_flag", "nonzero_exit_unmatched", "result_validation_failed"} <= reasons


async def test_results_do_not_depend_on_run_order() -> None:
    config = FakeToolboxConfig(seed=4, max_parallel_tests=1, default_k=9)

    async def values(priorities: list[int]) -> dict[str, float]:
        box = FakeToolbox(config, autorun=False)
        ids = [await box.submit_test(_request(priority=p)) for p in priorities]
        await box.drain()
        return {i: box.results[i].rounds[1].metrics["output_tps"] for i in ids}

    assert await values([4, 3, 2]) == await values([2, 3, 4])


# --- cancel ------------------------------------------------------------------------------------------------------


async def test_cancel_of_a_queued_test_withdraws_it_and_it_never_runs() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    keep = await box.submit_test(_request())
    drop = await box.submit_test(_request())
    assert await box.cancel(drop) is CancelOutcome.WITHDRAWN
    assert await box.cancel(drop) is CancelOutcome.WITHDRAWN
    await box.drain()
    assert set(box.results) == {keep}
    assert [w.request_id for w in book.of("withdrawal")] == [drop]
    assert (await box.queue_state()).queued == 0


async def test_cancel_of_a_running_test_withdraws_it_and_writes_no_result() -> None:
    book = Notebook()
    box = FakeToolbox(FakeToolboxConfig(run_delay_s=0.05), notebook=book, autorun=True)
    request_id = await box.submit_test(_request())
    assert (await box.queue_state()).running == 1
    assert await box.cancel(request_id) is CancelOutcome.WITHDRAWN
    await box.drain()
    assert box.results == {} and book.of("result") == []
    assert [w.request_id for w in book.of("withdrawal")] == [request_id]


async def test_cancel_of_a_finished_test_changes_nothing_and_unknown_ids_raise() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    result = await _run_one(box)
    assert await box.cancel(result.request_id) is CancelOutcome.FINISHED
    assert book.of("withdrawal") == []
    with pytest.raises(UnknownTestRequest):
        await box.cancel("nope")


# --- the whole thing ---------------------------------------------------------------------------------------------


async def test_autorun_finishes_in_the_background() -> None:
    box = FakeToolbox(FakeToolboxConfig(), autorun=True)
    request_id = await box.submit_test(_request())
    await box.drain()
    assert request_id in box.results and (await box.queue_state()).running == 0
