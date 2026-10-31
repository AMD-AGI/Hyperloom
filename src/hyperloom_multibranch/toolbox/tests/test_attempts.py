# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which infrastructure failures :class:`AttemptRunner` retries."""

from __future__ import annotations

import pytest

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
from hyperloom.measure.boot_watch import HUNG_BEFORE_READY
from hyperloom.measure import hosted
from hyperloom.measure.failures import HUB_RATE_LIMITED, INFRA_ERROR_CLASSES, JIT_BUILD_TIMEOUT
from hyperloom.measure.runner.import_outcome import INERT_PATCH
from hyperloom_multibranch.toolbox.attempts import (
    NODE_BLAMELESS_INFRA_REASONS,
    RUNNER_LOST,
    RUNNER_START_FAILED,
    SANDBOX_CREATE_FAILED,
    AttemptRunner,
    TestJob,
)

INERT_REASON = f"{INERT_PATCH}: process 1 imported aiter from /sgl-workspace/aiter/aiter/__init__.py, not from the farm"
PORT_REASON = "port_in_use: [Errno 98] address already in use"
DISK_FULL_REASON = "disk_full: OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/k.json'"
HUNG_REASON = f"{HUNG_BEFORE_READY}: no output for 901s before the server was ready"


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


@pytest.mark.parametrize(
    "reason", [PORT_REASON, DISK_FULL_REASON, HUNG_REASON], ids=["port_in_use", "disk_full", "hung_before_ready"]
)
async def test_another_infrastructure_failure_is_still_retried_on_other_nodes(reason) -> None:
    result, executor, reported = await _run(reason)
    assert len(executor.jobs) == 4
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {reason}"
    assert reported == ["n1", "n2", "n3", "n4"]


JIT_REASON = f"{JIT_BUILD_TIMEOUT}: [TP0] Scheduler watchdog timeout (self.watchdog_timeout=300)"


async def test_a_jit_build_timeout_is_retried_once_and_not_held_against_the_node() -> None:
    result, executor, reported = await _run(JIT_REASON)
    assert len(executor.jobs) == 2 and len(result.attempts) == 2
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == JIT_REASON
    assert reported == [], "the tables, not the node, made the server build past its watchdog"


#: What W227's hosted runs reported, word for word but for the cache path.
KEY_READ_ONLY_REASON = f"{RUNNER_START_FAILED}: signing key file cannot be removed: OSError"
CACHE_REASON = (
    f"{RUNNER_START_FAILED}: runner bootstrap, step cache, exit 1: the source in /cache/2ebadc612/source is not what "
    "the manifest recorded: 3980 files and 54915363 bytes, the manifest has 3457 and 46986171"
)
LOST_REASON = f"{RUNNER_LOST}: the sandbox could not be read"
#: What the hosted backend reports for the Agent Sandbox answer of joint hosted run 4, word for word.
#: What a runner reports when the Hub refused the replay client's download for its rate limit (board #161).
HUB_REFUSED_REASON = f"{HUB_RATE_LIMITED}: 429 Too Many Requests: you have reached your 'api' rate limit."
CREATE_FAILED_REASON = (
    f"{SANDBOX_CREATE_FAILED}: 500: sandbox_create_failed: workload sbx-f7f56db77ec7244ae34d9886d2ed9dde phase Failed"
)


@pytest.mark.parametrize(
    "reason",
    [KEY_READ_ONLY_REASON, CACHE_REASON, LOST_REASON, HUB_REFUSED_REASON],
    ids=["signing_key", "runner_cache", "runner_lost", "hub_rate_limited"],
)
async def test_a_runner_that_could_not_start_or_was_lost_is_retried_and_not_held_against_the_node(reason) -> None:
    result, executor, reported = await _run(reason)
    assert len(executor.jobs) == 4, "retried in fresh sandboxes like any infrastructure failure"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {reason}"
    assert reported == [], "the sandbox's delivery or the platform's reclaim, not the node, ended the attempt"


async def test_a_sandbox_that_failed_before_it_ran_is_retried_once_with_a_new_key_and_not_held_against_the_node() -> (
    None
):
    result, executor, reported = await _run(CREATE_FAILED_REASON)
    assert [job.attempt for job in executor.jobs] == [1, 2], "one more sandbox, under the next attempt's key, then stop"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == CREATE_FAILED_REASON
    assert reported == [], "the template or the platform, not a node, failed the sandbox"


def test_the_blameless_names_are_the_ones_the_runner_and_the_backend_report() -> None:
    assert NODE_BLAMELESS_INFRA_REASONS == {
        "runner_start_failed",
        "runner_lost",
        "sandbox_create_failed",
        "hub_rate_limited",
        "sandbox_acquire_timeout",
    }
    assert NODE_BLAMELESS_INFRA_REASONS - {HUB_RATE_LIMITED} <= INFRA_ERROR_CLASSES, (
        "the one a log, not a status, names"
    )
    assert RUNNER_LOST == hosted.RUNNER_LOST
    assert SANDBOX_CREATE_FAILED == hosted.SANDBOX_CREATE_FAILED
