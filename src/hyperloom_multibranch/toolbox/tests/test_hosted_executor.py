# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted executor over W7's backend, the local sandbox provider and the fake runner, under the real toolbox."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, SandboxFacts, TestResult, Validity
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.socket import TestCall, TestOutcome
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import (
    DEFAULT_TEST_TIMEOUT_SEC,
    TEST_SILENCE_TIMEOUT_SEC,
    HostedExecutor,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import FakePlanner, Rig as ToolboxRig, build, req

POLICY = QueuePolicy(per_branch_k=1, global_concurrency=1)
OK, NODE_FAULT, UNKNOWN, CANDIDATE = "ok", "node_fault", "fail_unknown", "fail_test"


class PointPlanner(FakePlanner):
    """Plans every request to one point file."""

    def __init__(self, point: Path, environment: object = None) -> None:
        super().__init__()
        self.point, self.environment = point, environment

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        super().plan(request, parent)
        return AcceptedMeasurement(self.environment, (self.point,))  # type: ignore[arg-type]


@dataclass
class ScriptedBackend:
    """The real backend; before each attempt the point is rewritten to the next mode of the script."""

    inner: object
    point: Path
    modes: list[str]
    calls: int = 0
    seen: list[TestOutcome] = field(default_factory=list)

    name = "hosted"

    @property
    def journal(self):  # type: ignore[no-untyped-def]
        return self.inner.journal  # type: ignore[attr-defined]

    def run_test(self, call):  # type: ignore[no-untyped-def]
        mode = self.modes[min(self.calls, len(self.modes) - 1)]
        self.calls += 1
        point_yaml(self.point.parent, self.point.name, fake_mode=mode)
        outcome = self.inner.run_test(call)  # type: ignore[attr-defined]
        self.seen.append(outcome)
        return outcome


@pytest.fixture
def rig(tmp_path: Path):
    made: list[Rig] = []

    def make(**config: object) -> Rig:
        r = make_rig(tmp_path, **config)
        made.append(r)
        return r

    yield make
    for r in made:
        r.backend.close()
        r.provider.close()


def _toolbox(tmp_path: Path, r: Rig, *modes: str, policy: QueuePolicy = POLICY) -> tuple[ToolboxRig, ScriptedBackend]:
    point = point_yaml(tmp_path, "planned.yaml")
    scripted = ScriptedBackend(r.backend, point, list(modes))
    executor = HostedExecutor(scripted, tmp_path / "session")  # type: ignore[arg-type]
    return build(
        tmp_path, executor=executor, planner=PointPlanner(point), policy=policy, workload_kind=WorkloadKind.AGENTX
    ), scripted  # type: ignore[arg-type]


async def _run(tb: ToolboxRig) -> tuple[str, TestResult]:
    await tb.toolbox.start()
    request_id = await tb.toolbox.submit_test(req())
    await asyncio.wait_for(tb.toolbox.drain(), 60)
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    outcome = entry.outcome
    return request_id, TestResult.from_document(outcome["result"])


def _journal_attempts(r: Rig, request_id: str) -> dict[int, str]:
    """attempt number -> sandbox id, from the journal."""
    return {rec.attempt: str(rec.sandbox_id) for rec in r.backend.journal.records() if rec.request_id == request_id}


async def test_a_verified_measurement_is_a_succeeded_result_with_its_audit_data(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, OK)
    request_id, result = await _run(tb)
    record = r.backend.journal.latest(request_id)
    assert result.validity is Validity.SUCCEEDED and result.failure_class is None
    assert record is not None and record.extra["verified"] is True
    assert result.manifest_sha256 == record.extra["manifest_sha256"]
    assert result.signing_key_fingerprint == record.extra["signing_key_fingerprint"]
    assert result.spec_fingerprint, "the fingerprint is read from the archived manifest"
    assert Path(str(record.extra["archive"])).is_dir()
    assert result.output_dir == f"tests/{request_id}/attempt-1" and (tmp_path / "session" / result.output_dir).is_dir()
    assert result.sandbox is not None and result.sandbox.node_name == "node-a" and result.sandbox.backend == "hosted"
    assert result.sandbox.sandbox_id == record.sandbox_id and result.sandbox.gpu_count == 1
    (measured,) = result.rounds
    assert measured.phase == "measure" and measured.metrics["output_throughput"] == 5000.0
    assert len(result.attempts) == 1 and scripted.calls == 1
    notebook_record = await tb.notebook.record(request_id)
    assert notebook_record.result is not None and notebook_record.result.manifest_sha256 == result.manifest_sha256


class _Overlay:
    """A compiled environment as the sheet reads it: one root with one upserted file."""

    attachments = ()

    def to_document(self) -> dict[str, object]:
        return {
            "roots": [
                {
                    "id": "yaml",
                    "kind": "package",
                    "import_name": "yaml",
                    "dist": {"name": "PyYAML", "version": "0"},
                    "entries": [],
                }
            ],
            "runtime": None,
            "blob_root": "/blobs",
        }


async def test_the_compiled_environment_reaches_the_instruction_sheet(tmp_path: Path, rig) -> None:
    r = rig()
    point = point_yaml(tmp_path, "planned.yaml")
    executor = HostedExecutor(r.backend, tmp_path / "session")
    tb = build(
        tmp_path,
        executor=executor,
        planner=PointPlanner(point, _Overlay()),
        policy=POLICY,
        workload_kind=WorkloadKind.AGENTX,
    )
    request_id, result = await _run(tb)
    assert result.validity is Validity.SUCCEEDED
    sheet = json.loads(r.backend.journal.load_spec(request_id, 1))
    assert [root["id"] for root in sheet["environment"]["roots"]] == ["yaml"]
    assert sheet["procedure"]["silence_timeout_sec"] == TEST_SILENCE_TIMEOUT_SEC


async def test_an_infrastructure_failure_retries_in_a_new_sandbox(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, NODE_FAULT, OK)
    request_id, result = await _run(tb)
    assert result.validity is Validity.SUCCEEDED and scripted.calls == 2
    first, second = result.attempts
    assert first.failure_class is FailureClass.INFRA and second.failure_class is None
    sandboxes = _journal_attempts(r, request_id)
    assert sorted(sandboxes) == [1, 2] and len(set(sandboxes.values())) == 2, "a retry never reuses a sandbox"
    assert first.sandbox is not None and second.sandbox is not None
    assert first.sandbox.sandbox_id != second.sandbox.sandbox_id
    assert second.sandbox.same_node, "one node: the node to avoid carried the retry and says so"
    assert tb.toolbox.quarantined == {"node-a"}
    assert result.output_dir == f"tests/{request_id}/attempt-2"


async def test_infrastructure_failures_end_as_measurement_failed_after_the_retries(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, NODE_FAULT)
    request_id, result = await _run(tb)
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert scripted.calls == 4 and len(result.attempts) == 4 and "node_fault" in (result.failure_reason or "")
    assert len(set(_journal_attempts(r, request_id).values())) == 4
    notebook_record = await tb.notebook.record(request_id)
    assert notebook_record.withdrawn_at is None and notebook_record.result == result


async def test_an_unclear_failure_is_rerun_once_and_the_rerun_decides(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, UNKNOWN, OK)
    request_id, result = await _run(tb)
    assert result.validity is Validity.SUCCEEDED and scripted.calls == 2
    assert result.attempts[0].failure_class is FailureClass.INFRA
    assert "did not reproduce" in (result.attempts[0].failure_reason or "")
    assert scripted.seen[0].failure_class is FailureClass.UNCLEAR, "the executor was given the unclear class"
    assert len(set(_journal_attempts(r, request_id).values())) == 2


async def test_an_unclear_failure_that_repeats_is_a_candidate_failure(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, UNKNOWN)
    _, result = await _run(tb)
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR
    assert scripted.calls == 2 and result.failure_reason == "no_known_signature"
    assert result.manifest_sha256, "the failed bundle was verified too"


async def test_a_candidate_failure_is_final_at_once_and_keeps_its_signature_name(tmp_path: Path, rig) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, CANDIDATE, OK)
    _, result = await _run(tb)
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
    assert scripted.calls == 1 and result.failure_reason == "import_error"
    assert result.rounds == ()


@pytest.mark.parametrize(
    ("mode", "reason"), [("spec_rejected", "spec_rejected"), ("tamper", "result_rejected")], ids=["spec", "result"]
)
async def test_a_rejected_sheet_or_result_is_a_measurement_failure_without_a_retry(
    tmp_path: Path, rig, mode: str, reason: str
) -> None:
    r = rig()
    tb, scripted = _toolbox(tmp_path, r, mode, OK)
    request_id, result = await _run(tb)
    assert result.validity is Validity.MEASUREMENT_FAILED and scripted.calls == 1
    assert f"measurement refused: {reason}" in (result.failure_reason or "")
    assert result.manifest_sha256 is None
    assert list(_journal_attempts(r, request_id)) == [1]


async def test_a_timeout_is_an_unclear_failure_of_the_attempt(tmp_path: Path, rig) -> None:
    class Timing(ScriptedBackend):
        def run_test(self, call):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                self.inner.run_test(call)  # type: ignore[attr-defined]
                raise subprocess.TimeoutExpired("test", 1)
            return super().run_test(call)

    r = rig()
    point = point_yaml(tmp_path, "planned.yaml")
    backend = Timing(r.backend, point, [OK])
    executor = HostedExecutor(backend, tmp_path / "session")  # type: ignore[arg-type]
    tb = build(
        tmp_path, executor=executor, planner=PointPlanner(point), policy=POLICY, workload_kind=WorkloadKind.AGENTX
    )
    _, result = await _run(tb)
    assert result.validity is Validity.SUCCEEDED, "the rerun succeeded"
    assert result.attempts[0].failure_class is FailureClass.INFRA, "an unclear failure that did not reproduce"
    assert result.attempts[0].failure_reason.endswith(": timeout")  # type: ignore[union-attr]
    assert len(result.attempts) == 2


class _Crash(BaseException):
    """What a killed application process looks like to the code under test: it unwinds without cleaning up."""


async def test_a_restarted_toolbox_re_attaches_to_the_sandbox_that_is_still_measuring(tmp_path: Path, rig) -> None:
    r = rig()
    point = point_yaml(tmp_path, "planned.yaml", fake_delay=0.5)
    dying = r.another_backend()
    dying._pool.release = lambda runner: None  # a dead process closes nothing

    def die(attempt):  # type: ignore[no-untyped-def]
        raise _Crash

    dying._wait = die  # type: ignore[method-assign]
    first = build(
        tmp_path,
        executor=HostedExecutor(dying, tmp_path / "session"),
        planner=PointPlanner(point),
        policy=POLICY,
        workload_kind=WorkloadKind.AGENTX,
    )
    await first.toolbox.start()
    request_id = await first.toolbox.submit_test(req())
    await asyncio.wait_for(first.toolbox.drain(), 30)
    submitted = r.backend.journal.latest(request_id)
    assert submitted is not None and submitted.state == STATE_SENT
    entry = first.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is None, "the crash left the test unfinished in the submit log"
    await first.toolbox.close()
    created: list[str] = []
    real_create = r.provider.create
    r.provider.create = lambda *a, **k: created.append("x") or real_create(*a, **k)  # type: ignore[method-assign]
    second_backend = r.another_backend()
    second = build(
        tmp_path,
        executor=HostedExecutor(second_backend, tmp_path / "session"),
        planner=PointPlanner(point),
        policy=POLICY,
        workload_kind=WorkloadKind.AGENTX,
    )
    await second.toolbox.start()
    await asyncio.wait_for(second.toolbox.drain(), 60)
    entry = second.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    result = TestResult.from_document(entry.outcome["result"])
    assert result.validity is Validity.SUCCEEDED and result.sandbox is not None
    assert result.sandbox.sandbox_id == submitted.sandbox_id, "the unfinished sandbox carried the test"
    assert created == [] and _journal_attempts(r, request_id) == {1: str(submitted.sandbox_id)}
    assert result.output_dir == f"tests/{request_id}/attempt-1"
    assert second_backend.journal.latest(request_id).state == STATE_FINISHED  # type: ignore[union-attr]


# -- what the executor refuses to call a success --------------------------------------------------------------------


class _Journal:
    def __init__(self, record: object = None) -> None:
        self.record = record

    def latest(self, request_id: str) -> object:
        return self.record

    def next_attempt(self, request_id: str) -> int:
        return 1


class _Stub:
    name = "hosted"

    def __init__(self, outcome: TestOutcome, record: object = None) -> None:
        self.outcome, self.journal = outcome, _Journal(record)
        self.calls: list[TestCall] = []

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        return self.outcome


def _job(point: Path) -> TestJob:
    return TestJob(
        req().accepted("b1-000001"),
        1,
        (),
        2,
        2,
        AcceptedMeasurement(None, (point,)),  # type: ignore[arg-type]
    )


def _outcome(rc: int = 0, failure: str | None = None, reason: str | None = None) -> TestOutcome:
    return TestOutcome(rc, "", "", (), SandboxFacts("sbx-1", node_name="n"), "b1-000001", failure, reason)  # type: ignore[arg-type]


async def test_a_success_the_journal_does_not_call_verified_is_refused(tmp_path: Path) -> None:
    executor = HostedExecutor(_Stub(_outcome()), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(MeasurementRefused, match="not recorded as a verified bundle"):
        await executor.run(_job(point_yaml(tmp_path)))


async def test_every_attempt_carries_the_ten_minute_silence_timeout_under_the_hard_one(tmp_path: Path) -> None:
    backend = _Stub(_outcome(1, FailureClass.UNCLEAR.value, "no_known_signature"))
    await HostedExecutor(backend, tmp_path).run(_job(point_yaml(tmp_path)))  # type: ignore[arg-type]
    (call,) = backend.calls
    assert (call.silence_timeout_sec, call.timeout_sec) == (600.0, DEFAULT_TEST_TIMEOUT_SEC)
    assert TEST_SILENCE_TIMEOUT_SEC == 600.0


@pytest.mark.parametrize("failure", list(FailureClass), ids=lambda f: f.value)
async def test_every_failure_class_reaches_the_result(tmp_path: Path, failure: FailureClass) -> None:
    executor = HostedExecutor(_Stub(_outcome(1, failure.value, "why")), tmp_path)  # type: ignore[arg-type]
    result = await executor.run(_job(point_yaml(tmp_path)))
    assert result.failure_class is failure and result.failure_reason == "why"
    expected = Validity.MEASUREMENT_FAILED if failure is FailureClass.INFRA else Validity.CANDIDATE_FAILED
    assert result.validity is expected


async def test_a_stop_is_a_cancellation_not_a_result(tmp_path: Path) -> None:
    executor = HostedExecutor(_Stub(_outcome(-917)), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await executor.run(_job(point_yaml(tmp_path)))


async def test_an_end_with_no_class_and_no_success_is_refused_with_its_reason(tmp_path: Path) -> None:
    executor = HostedExecutor(_Stub(_outcome(-923)), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(MeasurementRefused, match="result_rejected"):
        await executor.run(_job(point_yaml(tmp_path)))


async def test_a_job_without_the_accepted_measurement_is_refused(tmp_path: Path) -> None:
    executor = HostedExecutor(_Stub(_outcome()), tmp_path)  # type: ignore[arg-type]
    job = TestJob(req().accepted("b1-000001"), 1, (), 2, 2, {"env": "x"})
    with pytest.raises(MeasurementRefused, match="accepted measurement"):
        await executor.run(job)
