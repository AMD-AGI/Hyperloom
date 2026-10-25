# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which infrastructure failures :class:`AttemptRunner` retries."""

from __future__ import annotations

from hyperloom.contracts.measurement import (
    FailureClass,
    Purpose,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.runner.import_outcome import INERT_PATCH
from hyperloom_multibranch.toolbox.attempts import AttemptRunner, TestJob

INERT_REASON = f"{INERT_PATCH}: process 1 imported aiter from /sgl-workspace/aiter/aiter/__init__.py, not from the farm"
PORT_REASON = "port_in_use: [Errno 98] address already in use"


class _ScriptedExecutor:
    """Fails every attempt with ``reason`` as an infrastructure failure, each on its own node."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.jobs: list[TestJob] = []

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED,
            failure_class=FailureClass.INFRA,
            failure_reason=self.reason,
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
        )

    async def reverify(self, result: TestResult) -> None:
        return None

    def withdraw(self, request_id: str) -> None:
        return None


def _job() -> TestJob:
    request = TestRequest(
        kind=TestKind.TEST,
        purpose=Purpose.EXPLORE,
        branch_id="b1",
        parent_commit="c0",
        delta=RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256")),
    )
    return TestJob(request=request, attempt=1, avoid_nodes=(), priority=0, sandbox_priority=0)


async def _run(reason: str) -> tuple[TestResult, _ScriptedExecutor, list[str]]:
    executor = _ScriptedExecutor(reason)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, max_infra_retries=3)
    return await runner.run(_job()), executor, reported


async def test_an_inert_patch_is_not_retried_and_stays_an_infrastructure_failure() -> None:
    result, executor, reported = await _run(INERT_REASON)
    assert len(executor.jobs) == 1 and len(result.attempts) == 1
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == INERT_REASON
    # the patch, not the node, is at fault: the node stays usable
    assert reported == []


async def test_another_infrastructure_failure_is_still_retried_on_other_nodes() -> None:
    result, executor, reported = await _run(PORT_REASON)
    assert len(executor.jobs) == 4
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {PORT_REASON}"
    assert reported == ["n1", "n2", "n3", "n4"]
