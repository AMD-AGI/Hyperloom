# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
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
    TestProgress,
    Toolbox,
    UnknownTestRequest,
)
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import TTL, FakeClock, intent
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
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

    async def quarantined(self) -> list[QuarantineEvent]:
        return []

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
    await box.drain()
    await twin.drain()


async def test_a_rejection_raises_with_its_reason_and_writes_nothing() -> None:
    book = Notebook()
    box = FakeToolbox(notebook=book, autorun=False, wiring=FakeWiring(parent_exists=lambda commit: commit == "c0"))
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
    await box.drain()


async def test_backpressure_is_distinct_and_per_branch() -> None:
    box = _toolbox(branch_queue_limit=2, max_parallel_tests=1)  # one runs (held), the rest queue
    await box.submit_test(_request("b1"))
    await box.submit_test(_request("b1"))
    await box.submit_test(_request("b1"))
    with pytest.raises(Backpressure) as raised:
        await box.submit_test(_request("b1"))
    assert (raised.value.queued, raised.value.limit) == (2, 2)
    assert not isinstance(raised.value, RequestRejected)
    await box.submit_test(_request("b2"))
    assert len(box.requests) == 4
    await box.drain()
    await box.drain()
    await box.submit_test(_request("b1"))


# --- queue -------------------------------------------------------------------------------------------------------


async def test_default_priority_follows_the_purpose_and_queue_state_shows_it() -> None:
    box = _toolbox(max_parallel_tests=1)  # the first test runs (held); the rest queue
    await box.submit_test(_request(purpose=Purpose.EXPLORE))
    await box.submit_test(_request(purpose=Purpose.PROFILE, delta=DELTA))
    await box.submit_test(_request(purpose=Purpose.REPRO, kind=TestKind.REPRO, attachments=_attach()))
    await box.submit_test(_request(purpose=Purpose.EXPLORE, priority=1))
    state = await box.queue_state()
    assert dict(state.queued_by_priority) == {0: 0, 1: 2, 2: 0, 3: 1, 4: 0}
    assert dict(state.queued_by_branch) == {"b1": 3}
    assert state.running == 1
    await box.drain()


async def test_the_cold_start_is_unknown_until_a_test_settled_and_then_the_attempt_time() -> None:
    box = _toolbox(attempt_seconds=420.0)
    assert (await box.queue_state()).cold_start_seconds is None
    await box.submit_test(_request())
    await box.drain()
    assert (await box.queue_state()).cold_start_seconds == 420.0


async def test_progress_counts_the_tests_queued_ahead_and_times_the_running_one() -> None:
    box = _toolbox(max_parallel_tests=1, default_k=9)
    running = await box.submit_test(_request(priority=2))  # starts at once, held by the gate
    later = await box.submit_test(_request(priority=4))
    sooner = await box.submit_test(_request(priority=1))
    assert await box.progress(running) == TestProgress("running", running_seconds=0.0)
    assert await box.progress(sooner) == TestProgress("queued", queued_ahead=0)
    assert await box.progress(later) == TestProgress("queued", queued_ahead=1)
    await box.drain()
    assert [await box.progress(request_id) for request_id in (running, later, sooner)] == [None, None, None]


def _attach():
    from hyperloom.contracts.measurement import AttachmentRef

    return (AttachmentRef("case", "file", "a" * 64),)


async def test_more_urgent_tests_run_first() -> None:
    order: list[str] = []
    book = Notebook()
    box = FakeToolbox(FakeToolboxConfig(default_k=9, max_parallel_tests=1), notebook=book, autorun=False)
    blocker = await box.submit_test(_request(priority=1))  # starts at once: a running test is not preempted
    slow = await box.submit_test(_request(priority=4))
    urgent = await box.submit_test(_request(priority=0))
    middle = await box.submit_test(_request(priority=2))
    await box.drain()
    order = [r.request_id for r in book.of("result")]
    assert order == [blocker, urgent, middle, slow]


async def test_a_branch_runs_at_most_k_tests_but_confirmations_do_not_count() -> None:
    box_auto = FakeToolbox(FakeToolboxConfig(default_k=1, run_delay_s=0.01, max_parallel_tests=9), autorun=True)
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
    await box.set_k(3)
    for _ in range(3):
        await box.submit_test(_request())
    assert (await box.queue_state()).running_by_branch == {"b1": 3}
    await box.drain()
    with pytest.raises(ValueError, match="at least one"):
        await box.set_k(0)


async def test_set_k_after_submit_starts_the_queued_tests() -> None:
    box = FakeToolbox(FakeToolboxConfig(default_k=1, run_delay_s=0.01), autorun=True)
    for _ in range(3):
        await box.submit_test(_request())
    assert (await box.queue_state()).running_by_branch == {"b1": 1}
    await box.set_k(3)
    assert (await box.queue_state()).running_by_branch == {"b1": 3}
    await box.drain()


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
    assert result.rounds[1].metrics[GRADED_OUTPUT] == pytest.approx(1020.0)
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
    # the real runner re-labels the first attempt once the rerun passed: an infrastructure failure of its node
    assert first.failure_class is FailureClass.INFRA and second.failure_class is None
    assert first.failure_reason.startswith("unclear failure did not reproduce on a rerun")
    assert second.sandbox.node_name != first.sandbox.node_name
    assert book.of("quarantine") == [] and box.quarantined == set()


async def _reran_clear(box: FakeToolbox, request: TestRequest | None = None) -> str:
    box.script(lambda r: True, [Fate.UNCLEAR, Fate.OK])
    result = await _run_one(box, request)
    assert result.validity is Validity.SUCCEEDED
    assert result.attempts[0].failure_reason.startswith("unclear failure did not reproduce on a rerun")
    return result.attempts[0].sandbox.node_name


async def test_a_second_unclear_failure_that_did_not_reproduce_on_one_node_quarantines_it() -> None:
    book = Notebook()
    box = _toolbox(notebook=book, nodes=("a", "b"))
    node = await _reran_clear(box)
    assert box.quarantined == set()
    other = ({"a", "b"} - {node}).pop()
    assert await _reran_clear(box, _request("b2", avoid_nodes=(other,))) == node
    assert box.quarantined == {node}
    assert [e.node for e in book.of("quarantine")] == [node]
    assert box.node_failures == {node: 2}


async def test_two_non_reproducing_failures_on_the_only_node_are_counted_and_do_not_quarantine_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    book = Notebook()
    box = _toolbox(notebook=book, nodes=("a",))
    assert box.single_node
    with caplog.at_level("INFO", logger="hyperloom_multibranch.toolbox.toolbox"):
        assert await _reran_clear(box) == "a"
        assert await _reran_clear(box, _request("b2")) == "a"
    assert box.quarantined == set() and book.of("quarantine") == []
    assert box.node_failures == {"a": 2}
    assert [r.getMessage().split(" (")[0] for r in caplog.records] == ["node a not quarantined: single node"] * 2
    result = await _run_one(box, _request("b3"))
    assert result.validity is Validity.SUCCEEDED and result.attempts[0].sandbox.same_node is False


async def test_an_infra_failure_on_the_only_node_is_counted_and_does_not_quarantine_it() -> None:
    book = Notebook()
    box = _toolbox(notebook=book, nodes=("a",))
    box.script(lambda r: True, [Fate.INFRA, Fate.OK])
    result = await _run_one(box)
    assert result.validity is Validity.SUCCEEDED and result.attempts[0].failure_class is FailureClass.INFRA
    assert box.quarantined == set() and book.of("quarantine") == []
    assert box.node_failures == {"a": 1}


def test_a_fleet_of_several_nodes_is_not_a_single_node() -> None:
    assert not _toolbox(nodes=("a", "b")).single_node and _toolbox(nodes=("a",)).single_node


async def test_unclear_failures_that_did_not_reproduce_on_different_nodes_quarantine_none() -> None:
    box = _toolbox(nodes=("a", "b"))
    first = await _reran_clear(box)
    other = ({"a", "b"} - {first}).pop()
    assert await _reran_clear(box, _request("b2", avoid_nodes=(first,))) == other
    assert box.quarantined == set()


async def test_an_infra_failure_quarantines_its_node_at_once() -> None:
    book = Notebook()
    box = _toolbox(notebook=book)
    box.script(lambda r: True, [Fate.INFRA, Fate.OK])
    result = await _run_one(box)
    assert box.quarantined == {result.attempts[0].sandbox.node_name}
    assert len(book.of("quarantine")) == 1


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
    assert result.attempts == ()  # the real runner records an attempt only when the executor returned a result
    assert result.failure_reason == "measurement refused: result_validation_failed"
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
    # quarantine only widens avoid_nodes (design 4.9.3): the last attempt runs on a node it was to avoid
    assert result.attempts[-1].sandbox.same_node is True


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

    async def run() -> tuple[list[tuple[str, str | None]], list[dict]]:
        box = FakeToolbox(config, autorun=False)
        ids = [await box.submit_test(_request()) for _ in range(120)]
        await box.drain()
        outcomes = [(box.results[i].validity.value, box.results[i].failure_reason) for i in ids] + [
            ("attempt", a.failure_reason) for i in ids for a in box.results[i].attempts
        ]
        return outcomes, [box.results[i].to_document() for i in ids]

    (first, documents), (again, documents_again) = await run(), await run()
    assert first == again
    assert documents == documents_again  # same seed: every value, node and timestamp is identical
    assert {v for v, _ in first if v != "attempt"} == {"succeeded", "candidate_failed", "measurement_failed"}
    reasons = {reason for _, reason in first}
    assert {"serve_flag", "nonzero_exit_unmatched", "measurement refused: result_validation_failed"} <= reasons


async def test_results_do_not_depend_on_run_order() -> None:
    config = FakeToolboxConfig(seed=4, max_parallel_tests=1, default_k=9)

    async def values(priorities: list[int]) -> dict[str, float]:
        box = FakeToolbox(config, autorun=False)
        ids = [await box.submit_test(_request(priority=p)) for p in priorities]
        await box.drain()
        return {i: box.results[i].rounds[1].metrics[GRADED_OUTPUT] for i in ids}

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
    assert box.withdrawn == {drop: "cancelled by the submitter"}
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


# --- the real notebook -------------------------------------------------------------------------------------------


async def test_the_fake_writes_requests_results_withdrawals_and_quarantine_into_a_local_notebook(
    tmp_path: Path,
) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    box = FakeToolbox(
        FakeToolboxConfig(noise=QUIET),
        notebook=notebook,
        wiring=FakeWiring(claims=notebook, records=notebook),
        autorun=False,
    )
    box.script(lambda r: r.branch_id == "flaky", [Fate.INFRA, Fate.OK])
    box.script(lambda r: r.branch_id == "broken", [Fate.CANDIDATE])
    target, action = intent("b1", "c0"), DELTA
    specialist = LocalNotebook(
        tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, "specialist", "b1"), clock=FakeClock()
    )
    claim = await specialist.acquire(target, action.fingerprint(), ttl_seconds=TTL)
    measured = await box.submit_test(_request(intent_fingerprint=target.fingerprint))
    flaky = await box.submit_test(_request("flaky"))
    broken = await box.submit_test(_request("broken"))
    dropped = await box.submit_test(_request("b3"))
    assert await box.cancel(dropped) is CancelOutcome.WITHDRAWN
    await box.drain()

    record = await notebook.record(measured)
    assert record.result is not None and record.result.validity.value == "succeeded"
    assert record.result.sandbox.node in box.config.nodes
    assert record.result.rounds == box.results[measured].rounds
    assert {(r.phase, metric) for r in record.result.rounds for metric in r.metrics} == {
        (phase, metric) for phase in ("warmup", "measure") for metric in box.config.baseline
    }
    held = (await specialist.lookup(claim.key, action.fingerprint())).claim
    assert held is None or held.status is ClaimStatus.COMPLETED or held.request_id == measured
    retried = (await notebook.record(flaky)).result
    assert retried is not None and [a.failure.value if a.failure else None for a in retried.attempts] == ["infra", None]
    failed = (await notebook.record(broken)).result
    assert failed is not None and failed.validity.value == "candidate_failed"
    assert (await notebook.record(dropped)).withdrawn_at is not None
    assert {event.node for event in await notebook.quarantined()} == box.quarantined != set()


# --- re-verifying a result the fake produced ---------------------------------------------------------------------


async def test_the_fake_reverifies_the_result_it_produced_and_names_a_mismatch_or_a_stranger() -> None:
    from dataclasses import replace

    from hyperloom_multibranch.contracts.gate import ResultUnverified

    box = _toolbox(autorun=True)
    result = await _run_one(box)
    assert await box.executor.reverify(result) is None
    with pytest.raises(ResultUnverified) as edited:
        await box.executor.reverify(replace(result, rounds=()))
    assert edited.value.reason == "rounds: differs from what the fake executor produced"
    with pytest.raises(ResultUnverified) as stranger:
        await box.executor.reverify(replace(result, request_id="other"))
    assert stranger.value.reason == "the fake executor produced no result for this request"


async def test_the_fake_keeps_an_attachment_and_returns_a_usable_ref() -> None:
    fake = FakeToolbox()
    ref = await fake.put_attachment("note-1", "file", b"hello")

    assert (ref.id, ref.kind) == ("note-1", "file") and fake.blobs.get(ref.sha256) == b"hello"
    assert await fake.put_attachment("note-1", "file", b"hello") == ref
    with pytest.raises(RequestRejected) as refused:
        await fake.put_attachment("case", "dir", b"not a tar")
    assert refused.value.code == "attachment_unreadable"

    escaping, buffer = b"x", io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = len(escaping)
        tar.addfile(info, io.BytesIO(escaping))
    with pytest.raises(RequestRejected) as unsafe:
        await fake.put_attachment("case", "dir", buffer.getvalue())
    assert unsafe.value.code == "attachment_unsafe"


async def test_the_fake_refuses_an_attachment_id_the_sheet_would_refuse_and_a_truncated_archive() -> None:
    fake = FakeToolbox()
    with pytest.raises(RequestRejected, match="must match"):
        await fake.put_attachment("../x", "file", b"hello")
    with pytest.raises(RequestRejected) as truncated:
        await fake.put_attachment("case", "dir", gzip.compress(b"x" * 2000)[:20])
    assert truncated.value.code == "attachment_unreadable"
