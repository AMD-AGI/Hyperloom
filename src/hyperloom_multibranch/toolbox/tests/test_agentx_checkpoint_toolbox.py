# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tests the toolbox holds to the AgentX checkpoint, and what a judged test's result carries."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.agentx_early_stop import EarlyStopMode
from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_INTVTY_P50_CHECKPOINT
from hyperloom.contracts.measurement import (
    AgentxCheckpointRuling,
    AttachmentRef,
    FailureClass,
    Purpose,
    SandboxFacts,
    TestKind,
    TestResult,
    Validity,
)
from hyperloom.measure.checkpoint import CheckpointPlan
from hyperloom.measure.failures import classify_test_failure
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.socket import TestCall, TestOutcome
from hyperloom.measure.tests.hosted_fixtures import point_yaml
from hyperloom_multibranch.contracts.blackboard.interfaces import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import RecordedValue
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID
from hyperloom_multibranch.rulegate.rules import RETRACTION_ID
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.agentx_checkpoint import CheckpointReferences
from hyperloom_multibranch.toolbox.attempts import AttemptRunner, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import FakePlanner, FakeRepo, req

P50, TOTAL = AgentXObjective.P50_INTERACTIVITY, AgentXObjective.TOTAL_TOKEN_THROUGHPUT
RULING = AgentxCheckpointRuling(1800.0, 50.0, 45.0, -10.0, 2.0, trails=True, stopped=True)
CASE = AttachmentRef(FIX_CASE_ID, "dir", "b" * 64)
RETRACTION = AttachmentRef(RETRACTION_ID, "file", "c" * 64)


class _Commits:
    """The recorded values of the commits, by id; any other commit is not recorded."""

    def __init__(self, **values: dict[str, float]) -> None:
        self.values = values

    async def recorded_value(self, commit_id: str) -> RecordedValue:
        if commit_id not in self.values:
            raise NotRecorded(f"no commit {commit_id}")
        return RecordedValue(commit_id, self.values[commit_id], ("r0",))


class _Points(FakePlanner):
    def plan(self, request, parent):  # type: ignore[no-untyped-def]
        super().plan(request, parent)
        return AcceptedMeasurement(None, (Path("point.yaml"),))  # type: ignore[arg-type]


async def _accepted(request, mode=EarlyStopMode.SHADOW, objective=P50, commits=None) -> CheckpointPlan | None:  # type: ignore[no-untyped-def]
    references = CheckpointReferences(mode, objective, commits or _Commits(c0={GRADED_INTVTY_P50_CHECKPOINT: 50.0}))
    setup = AcceptanceSetup(FakeRepo(), _Points(), None, WorkloadKind.AGENTX, checkpoint=references)
    prepared = await Acceptance(setup, is_accepted=lambda _: True).prepare(request)
    return prepared.environment.checkpoint  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", [EarlyStopMode.SHADOW, EarlyStopMode.ENFORCE])
async def test_an_explore_under_p50_is_held_to_its_parents_checkpoint_median_in_the_sessions_mode(mode) -> None:
    assert await _accepted(req(), mode) == CheckpointPlan(50.0, mode)


@pytest.mark.parametrize(
    "request_",
    [
        req(purpose=Purpose.CONFIRM, retest_of="b1-000001"),
        req(purpose=Purpose.FINAL),
        req(purpose=Purpose.INTEGRATION),
        req(purpose=Purpose.BASELINE),
        req(INTEGRATION_BRANCH),
        req(attachments=(CASE,)),
        req(attachments=(RETRACTION,)),
        req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE),
    ],
    ids=["confirm", "final", "integration", "baseline", "explore_on_integration", "fix", "retraction", "profile"],
)
async def test_a_test_that_measures_on_purpose_is_never_judged(request_) -> None:
    assert await _accepted(request_, EarlyStopMode.ENFORCE) is None


async def test_no_test_is_judged_under_total_token_throughput_or_with_the_checkpoint_off() -> None:
    assert await _accepted(req(), EarlyStopMode.ENFORCE, TOTAL) is None
    assert await _accepted(req(), EarlyStopMode.OFF) is None


@pytest.mark.parametrize("commits", [_Commits(c0={GRADED_INTVTY_P50: 60.0}), _Commits()], ids=["no_median", "no_value"])
async def test_an_explore_whose_parent_recorded_no_median_runs_unjudged_and_says_so(commits, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert await _accepted(req(), EarlyStopMode.ENFORCE, commits=commits) is None
    assert "an explore on c0 runs unjudged" in caplog.text


# -- the executor ---------------------------------------------------------------------------------------------------


class _Journal:
    def latest(self, request_id: str) -> object:
        return None

    def next_attempt(self, request_id: str) -> int:
        return 1


class _Stopping:
    """A backend whose sandbox stopped the test at the checkpoint and left its ruling in the measured round's slot."""

    name = "hosted"
    journal = _Journal()

    def __init__(self) -> None:
        self.calls: list[TestCall] = []

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        slot = Path(call.output_dir) / "variant_00_test"
        slot.mkdir(parents=True)
        (slot / "agentx_checkpoint.json").write_text(json.dumps(RULING.to_document()), encoding="utf-8")
        verdict = classify_test_failure(returncode=-919)
        failure = verdict.failure_class.value if verdict.failure_class else None
        return TestOutcome(
            -919, "", "", (), SandboxFacts("sbx-1", node_name="n"), "b1-000001", failure, verdict.failure_reason
        )  # type: ignore[arg-type]


def _job(point: Path, plan: CheckpointPlan | None) -> TestJob:
    accepted = AcceptedMeasurement(None, (point,), checkpoint=plan)  # type: ignore[arg-type]
    return TestJob(req().accepted("b1-000001"), 1, (), 2, 2, accepted)


async def test_a_test_the_checkpoint_stopped_is_the_candidates_failure_and_carries_the_ruling(tmp_path: Path) -> None:
    backend = _Stopping()
    plan = CheckpointPlan(50.0, EarlyStopMode.ENFORCE)

    result = await HostedExecutor(backend, tmp_path / "session").run(_job(point_yaml(tmp_path), plan))  # type: ignore[arg-type]

    (call,) = backend.calls
    assert call.checkpoint == plan
    assert (result.validity, result.failure_reason) == (Validity.CANDIDATE_FAILED, "agentx_early_stop")
    assert result.agentx_checkpoint == RULING
    assert TestResult.from_document(result.to_document()) == result


async def test_a_test_that_carried_no_checkpoint_claims_no_ruling(tmp_path: Path) -> None:
    result = await HostedExecutor(_Stopping(), tmp_path / "session").run(_job(point_yaml(tmp_path), None))  # type: ignore[arg-type]
    assert result.agentx_checkpoint is None


async def test_a_checkpoint_stop_is_never_retried() -> None:
    class _Executor:
        def __init__(self) -> None:
            self.jobs: list[TestJob] = []

        async def run(self, job: TestJob) -> TestResult:
            self.jobs.append(job)
            verdict = classify_test_failure(returncode=-919)
            return TestResult(
                request_id="r1",
                validity=Validity.CANDIDATE_FAILED,
                failure_class=verdict.failure_class,
                failure_reason=verdict.failure_reason,
                sandbox=SandboxFacts(sandbox_id="s1", node_name="n1"),
                agentx_checkpoint=RULING,
            )

        async def reverify(self, result: TestResult) -> None:
            return None

        def withdraw(self, request_id: str) -> None:
            return None

    async def on_node_failure(node: str, reason: str, reproduced: bool) -> None:
        raise AssertionError("a ruling on the candidate is no node's failure")

    executor = _Executor()
    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, max_infra_retries=3)
    job = replace(_job(Path("p.yaml"), None), request=req())

    result = await runner.run(job)

    assert len(executor.jobs) == 1 and result.failure_class is FailureClass.CANDIDATE
    assert result.agentx_checkpoint == RULING
