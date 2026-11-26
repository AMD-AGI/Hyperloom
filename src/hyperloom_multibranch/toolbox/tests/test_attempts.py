# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which infrastructure failures :class:`AttemptRunner` retries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hyperloom.contracts.measurement import (
    FailureClass,
    PhaseTiming,
    Purpose,
    RoundValue,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.boot_watch import HUNG_BEFORE_READY
from hyperloom.measure import hosted
from hyperloom.measure.failures import (
    FOREIGN_PROCESS_ON_CARD,
    FRAMEWORK_VERSION_MISMATCH,
    HF_CACHE_PERMISSION,
    HUB_RATE_LIMITED,
    INFRA_ERROR_CLASSES,
    JIT_BUILD_TIMEOUT,
    PROFILER_DIED,
    PROFILER_HUNG,
    PROFILER_SNAPSHOT_FAILED,
    RESULT_DOWNLOAD_INCOMPLETE,
    TEST_TIMEOUT_REASON,
    FailureEvidence,
    StepFailure,
    classify_test_failure,
)
from hyperloom.measure.phases import MEASURED, SERVER_BOOT
from hyperloom.measure.runner.import_outcome import INERT_PATCH
from inference_suite.contracts import FailedStep
from hyperloom_multibranch.toolbox.attempts import (
    CAPACITY_UNAVAILABLE,
    NODE_BLAMELESS_INFRA_REASONS,
    RUNNER_START_TIMEOUT,
    WAITED_INFRA_REASONS,
    UNAVOIDED_INFRA_REASONS,
    RUNNER_LOST,
    RUNNER_START_FAILED,
    SANDBOX_CREATE_FAILED,
    INFRA_BACKOFF_SECONDS,
    AttemptRunner,
    InfraBackoff,
    TestJob,
    node_unquarantined,
)

INERT_REASON = f"{INERT_PATCH}: process 1 imported aiter from /sgl-workspace/aiter/aiter/__init__.py, not from the farm"
PORT_REASON = "port_in_use: [Errno 98] address already in use"
DISK_FULL_REASON = "disk_full: OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/k.json'"
HUNG_REASON = f"{HUNG_BEFORE_READY}: no output for 600s before the server was ready"
#: Failures that are not the node's: cards a process outside the test held, no sandbox from the platform in time.
FOREIGN_REASON = f"{FOREIGN_PROCESS_ON_CARD}: card 3 held by pid 4242 (python)"
START_TIMEOUT_REASON = RUNNER_START_TIMEOUT
CAPACITY_REASON = f"{CAPACITY_UNAVAILABLE}: no node with 8 free GPUs"
#: The phases of an attempt that failed while its server booted, and of one that failed after its server was ready.
BOOTED = (PhaseTiming(SERVER_BOOT, 1000.0, 60.0),)
SERVED = (*BOOTED, PhaseTiming(MEASURED, 1060.0, 300.0))


class _FakeTime:
    """A clock that only a wait moves, and the waits it was asked for."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class _ScriptedExecutor:
    """Fails every attempt with ``reason`` as an infrastructure failure, each on its own node; ``started`` is the clock
    at each attempt. A sequence of reasons gives one per attempt, the last for every later one. Each attempt stamps
    ``phases``."""

    def __init__(
        self, reason: str | tuple[str, ...], clock: _FakeTime | None = None, phases: tuple[PhaseTiming, ...] = ()
    ) -> None:
        self.reasons = (reason,) if isinstance(reason, str) else reason
        self.reason = self.reasons[0]
        self.jobs: list[TestJob] = []
        self.clock = clock or _FakeTime()
        self.started: list[float] = []
        self.phases = phases

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        self.started.append(self.clock())
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED,
            failure_class=FailureClass.INFRA,
            failure_reason=self.reasons[min(job.attempt, len(self.reasons)) - 1],
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
            phases=self.phases,
            unaccounted_seconds=0.0 if self.phases else None,
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
    return TestJob(request=request, attempt=1, avoid_nodes=(), priority=0)


def _backoff(time: _FakeTime, latest: float | None = None) -> InfraBackoff:
    return InfraBackoff(INFRA_BACKOFF_SECONDS, lambda _request: latest, clock=time, sleep=time.sleep)


async def _run(
    reason: str | tuple[str, ...],
    latest: float | None = None,
    kind: TestKind = TestKind.TEST,
    phases: tuple[PhaseTiming, ...] = (),
) -> tuple[TestResult, _ScriptedExecutor, list[str]]:
    executor = _ScriptedExecutor(reason, phases=phases)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    backoff = _backoff(executor.clock, latest)
    runner = AttemptRunner(
        executor, avoid=frozenset, on_node_failure=on_node_failure, max_infra_retries=3, backoff=backoff
    )
    job = _job()
    return await runner.run(replace(job, request=replace(job.request, kind=kind))), executor, reported


@pytest.mark.parametrize(
    "reason",
    [FOREIGN_REASON, START_TIMEOUT_REASON, CAPACITY_REASON, f"{RUNNER_LOST}: the sandbox is gone"],
    ids=["busy_cards", "no_sandbox_in_time", "no_capacity", "node_blameless"],
)
async def test_retries_after_a_failure_that_is_not_the_node_s_wait_longer_each_time_before_they_start(reason) -> None:
    result, executor, _ = await _run(reason)
    assert len(executor.jobs) == 4 and result.validity is Validity.MEASUREMENT_FAILED
    assert executor.clock.waits == [60.0, 120.0, 240.0]
    gaps = [later - earlier for earlier, later in zip(executor.started, executor.started[1:])]
    assert gaps == [60.0, 120.0, 240.0], "each retry starts only after its wait"


@pytest.mark.parametrize(
    "reason", [PORT_REASON, DISK_FULL_REASON, HUNG_REASON, "pull failed"], ids=["port", "disk", "hung", "unnamed"]
)
async def test_a_retry_after_a_failure_held_against_its_node_goes_at_once_elsewhere(reason) -> None:
    result, executor, reported = await _run(reason)
    assert len(executor.jobs) == 4 and result.validity is Validity.MEASUREMENT_FAILED
    assert executor.clock.waits == [], "the avoid set already moves the retry; a wait would only idle its slot"
    assert executor.started == [1000.0] * 4
    assert reported == ["n1", "n2", "n3", "n4"], "each node was held to its failure"
    assert [job.avoid_nodes for job in executor.jobs][1:] == [("n1",), ("n1", "n2"), ("n1", "n2", "n3")]


async def test_in_a_mixed_run_only_the_retries_after_a_failure_that_is_not_the_node_s_wait() -> None:
    result, executor, reported = await _run((FOREIGN_REASON, PORT_REASON, CAPACITY_REASON, PORT_REASON))
    assert len(executor.jobs) == 4 and result.validity is Validity.MEASUREMENT_FAILED
    assert executor.clock.waits == [60.0, 120.0], "the waits count the waiting retries only"
    assert executor.started == [1000.0, 1060.0, 1060.0, 1180.0]
    assert "n1" not in reported and "n2" in reported and "n4" in reported
    assert [job.avoid_nodes for job in executor.jobs] == [(), (), ("n2",), ("n2", "n3")], "busy cards do not avoid"


async def test_cards_a_process_outside_the_test_held_neither_avoid_nor_report_their_node() -> None:
    result, executor, reported = await _run(FOREIGN_REASON)
    assert len(executor.jobs) == 4 and result.validity is Validity.MEASUREMENT_FAILED
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {FOREIGN_REASON}"
    assert [job.avoid_nodes for job in executor.jobs] == [()] * 4, "the retry waits and may land on the same node"
    assert reported == [], "the card guard refuses the foreign holder; its node is not quarantined"
    assert executor.clock.waits == [60.0, 120.0, 240.0]


class _SucceedsAfter(_ScriptedExecutor):
    """Fails the first attempts with the scripted reasons, each on its own node, then succeeds on the next node."""

    async def run(self, job: TestJob) -> TestResult:
        if job.attempt <= len(self.reasons):
            return await super().run(job)
        self.jobs.append(job)
        return TestResult(
            request_id="r1",
            validity=Validity.SUCCEEDED,
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
        )


async def test_a_retry_after_cards_a_process_outside_the_test_held_waits_and_carries_no_avoid_for_that_node() -> None:
    executor = _SucceedsAfter((FOREIGN_REASON,))
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, backoff=_backoff(executor.clock))
    result = await runner.run(_job())
    assert result.validity is Validity.SUCCEEDED
    assert [job.avoid_nodes for job in executor.jobs] == [(), ()]
    assert reported == [] and executor.clock.waits == [60.0]


async def test_a_waited_failure_does_not_lift_the_avoid_a_node_held_failure_put_on_its_node() -> None:
    result, executor, reported = await _run((PORT_REASON, FOREIGN_REASON, PORT_REASON, CAPACITY_REASON))
    assert [job.avoid_nodes for job in executor.jobs] == [(), ("n1",), ("n1",), ("n1", "n3")]
    assert reported == ["n1", "n3", "n4"], "the busy cards of n2 do not report their node"


@pytest.mark.parametrize(
    "reason",
    [FOREIGN_REASON, f"{HUB_RATE_LIMITED}: 429 Too Many Requests"],
    ids=["busy_cards_on_other_nodes", "hub_rate_limited"],
)
async def test_a_failure_that_says_nothing_about_the_node_puts_no_node_in_the_avoid_set(reason) -> None:
    _, executor, _ = await _run(reason)
    assert [job.avoid_nodes for job in executor.jobs] == [()] * 4, "the retry may land on the same node"


@pytest.mark.parametrize(
    "reason",
    [CAPACITY_REASON, f"{RUNNER_LOST}: the sandbox is gone", f"{RUNNER_START_FAILED}: exec format error"],
    ids=["no_capacity", "runner_lost", "runner_start_failed"],
)
async def test_a_lost_or_unstarted_runner_and_no_capacity_still_avoid_their_node(reason) -> None:
    _, executor, _ = await _run(reason)
    assert [job.avoid_nodes for job in executor.jobs][1:] == [("n1",), ("n1", "n2"), ("n1", "n2", "n3")]


class _PlacingExecutor(_ScriptedExecutor):
    """Places each attempt on the first of ``nodes`` its avoid set leaves and fails it with the next reason scripted for
    that node; a node with no reason left succeeds."""

    def __init__(self, nodes: tuple[str, ...], failures: dict[str, list[str]]) -> None:
        super().__init__(PORT_REASON)
        self.nodes = nodes
        self.failures = failures

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        self.started.append(self.clock())
        node = next(node for node in self.nodes if node not in job.avoid_nodes)
        sandbox = SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=node)
        left = self.failures.get(node, [])
        if not left:
            return TestResult(request_id="r1", validity=Validity.SUCCEEDED, sandbox=sandbox)
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED,
            failure_class=FailureClass.INFRA,
            failure_reason=left.pop(0),
            sandbox=sandbox,
        )


async def _place(failures: dict[str, list[str]]) -> tuple[TestResult, _PlacingExecutor, list[str]]:
    executor = _PlacingExecutor(("n1", "n2"), failures)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    runner = AttemptRunner(
        executor,
        avoid=frozenset,
        on_node_failure=on_node_failure,
        max_infra_retries=3,
        backoff=_backoff(executor.clock),
    )
    return await runner.run(_job()), executor, reported


@pytest.mark.parametrize(
    "reason",
    [f"{RUNNER_LOST}: the sandbox is gone", f"{RUNNER_START_FAILED}: exec format error"],
    ids=["runner_lost", "runner_start_failed"],
)
async def test_a_node_that_loses_every_runner_sends_the_retry_to_the_free_node(reason) -> None:
    result, executor, reported = await _place({"n1": [reason] * 10})
    assert result.validity is Validity.SUCCEEDED
    assert [job.avoid_nodes for job in executor.jobs] == [(), ("n1",)]
    assert reported == [], "a lost runner is still not held against its node"


async def test_a_process_that_stays_on_the_cards_of_a_node_is_avoided_after_two_refusals_there() -> None:
    result, executor, reported = await _place({"n1": [FOREIGN_REASON] * 10})
    assert result.validity is Validity.SUCCEEDED and result.sandbox.node_name == "n2"
    assert [job.avoid_nodes for job in executor.jobs] == [(), (), ("n1",)]
    assert reported == [] and executor.clock.waits == [60.0, 120.0], "still no report and no quarantine"


async def test_a_single_refusal_for_busy_cards_puts_no_node_in_the_avoid_set() -> None:
    result, executor, reported = await _place({"n1": [FOREIGN_REASON]})
    assert result.validity is Validity.SUCCEEDED and result.sandbox.node_name == "n1"
    assert [job.avoid_nodes for job in executor.jobs] == [(), ()]
    assert reported == []


async def test_the_avoid_after_two_refusals_for_busy_cards_lasts_one_attempt() -> None:
    result, executor, reported = await _place({"n1": [FOREIGN_REASON] * 10, "n2": [FOREIGN_REASON] * 10})
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert [attempt.node for attempt in result.attempts] == ["n1", "n1", "n2", "n1"]
    assert [job.avoid_nodes for job in executor.jobs] == [(), (), ("n1",), ()]
    assert reported == []


async def test_a_runner_start_timeout_waits_and_still_puts_its_node_in_the_next_attempt_s_avoid_set() -> None:
    _, executor, _ = await _run(START_TIMEOUT_REASON)
    assert executor.clock.waits == [60.0, 120.0, 240.0]
    assert [job.avoid_nodes for job in executor.jobs][1:] == [("n1",), ("n1", "n2"), ("n1", "n2", "n3")], (
        "a slow or faulty node (an image pull, a GPU that does not come up) is one way a runner does not start in time"
    )


def test_the_unavoided_names_are_the_busy_cards_and_the_blameless_ones_less_a_lost_or_unstarted_runner() -> None:
    assert UNAVOIDED_INFRA_REASONS == (NODE_BLAMELESS_INFRA_REASONS - {"runner_lost", "runner_start_failed"}) | {
        "foreign_process_on_card"
    }
    assert WAITED_INFRA_REASONS - UNAVOIDED_INFRA_REASONS == {
        "runner_start_timeout",
        "capacity_unavailable",
        "runner_lost",
        "runner_start_failed",
    }


async def test_an_infrastructure_retry_never_waits_past_the_latest_start_and_goes_at_once_after_it() -> None:
    result, executor, _ = await _run(FOREIGN_REASON, latest=1000.0 + 100.0)
    assert len(executor.jobs) == 4, "a retry that cannot wait still goes"
    assert executor.clock.waits == [60.0, 40.0], "the second wait ends at the latest start, the third is none"
    assert executor.started == [1000.0, 1060.0, 1100.0, 1100.0]


def test_the_default_backoff_is_minutes_and_bounded() -> None:
    backoff = InfraBackoff()
    request = _job().request
    assert [backoff.before(retry, request) for retry in (0, 1, 2, 3, 4)] == [0.0, 60.0, 120.0, 240.0, 240.0]
    assert InfraBackoff(()).before(1, request) == 0.0


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


async def test_a_dataset_cache_lock_the_sandbox_may_not_take_is_not_retried_and_not_held_against_the_node() -> None:
    lock = "/x/hf/datasets/_x_hf_datasets_openai___gsm8k_main_0.0.0_e53f048.lock"
    log = f"PermissionError: [Errno 13] Permission denied: '{lock}'"
    reason = classify_test_failure(returncode=1, evidence=FailureEvidence(logs=(log,))).failure_reason
    assert reason is not None and reason.startswith(f"{HF_CACHE_PERMISSION}: ")
    result, executor, reported = await _run(reason)
    assert len(executor.jobs) == 1 and len(result.attempts) == 1, "every sandbox shares the cache and is refused alike"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == reason
    assert reported == [], "the shared cache's permissions, not the node, refused the lock"


async def test_a_framework_of_another_version_than_the_patch_was_compiled_against_ends_the_test_at_once() -> None:
    verdict = classify_test_failure(returncode=1, error_class=FRAMEWORK_VERSION_MISMATCH)
    assert (verdict.failure_class, verdict.failure_reason) == (FailureClass.INFRA, FRAMEWORK_VERSION_MISMATCH)
    result, executor, reported = await _run(FRAMEWORK_VERSION_MISMATCH)
    assert len(executor.jobs) == 1 and len(result.attempts) == 1, "every sandbox of the image has the same version"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == FRAMEWORK_VERSION_MISMATCH
    assert reported == [], "the image, not the node, carries the other version"
    assert executor.clock.waits == []


JIT_REASON = f"{JIT_BUILD_TIMEOUT}: [TP0] Scheduler watchdog timeout (self.watchdog_timeout=300)"


async def test_a_jit_build_timeout_is_retried_once_and_not_held_against_the_node() -> None:
    result, executor, reported = await _run(JIT_REASON)
    assert len(executor.jobs) == 2 and len(result.attempts) == 2
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == JIT_REASON
    assert reported == [], "the tables, not the node, made the server build past its watchdog"


@pytest.mark.parametrize("reason", [INERT_REASON, JIT_REASON], ids=["unretried", "retried_once"])
async def test_an_unretried_failure_or_a_jit_build_timeout_does_not_wait(reason) -> None:
    _, executor, _ = await _run(reason)
    assert executor.clock.waits == []


def test_the_waited_names_are_the_blameless_ones_and_the_busy_cards_and_no_sandbox_ones_the_backend_reports() -> None:
    assert WAITED_INFRA_REASONS == NODE_BLAMELESS_INFRA_REASONS | {
        "foreign_process_on_card",
        "runner_start_timeout",
        "capacity_unavailable",
    }
    assert {FOREIGN_PROCESS_ON_CARD, RUNNER_START_TIMEOUT, CAPACITY_UNAVAILABLE} <= INFRA_ERROR_CLASSES


#: Reasons a hosted runner can report on a failed start.
KEY_READ_ONLY_REASON = f"{RUNNER_START_FAILED}: signing key file cannot be removed: OSError"
CACHE_REASON = (
    f"{RUNNER_START_FAILED}: runner bootstrap, step cache, exit 1: the source in /cache/abc123/source is not what "
    "the manifest recorded: 100 files and 1000000 bytes, the manifest has 90 and 900000"
)
LOST_REASON = f"{RUNNER_LOST}: the sandbox could not be read"
#: What a runner reports when the Hub refused the replay client's download for its rate limit.
HUB_REFUSED_REASON = f"{HUB_RATE_LIMITED}: 429 Too Many Requests: you have reached your 'api' rate limit."
#: What the hosted backend reports when the sandbox service answers a create with an error.
CREATE_FAILED_REASON = (
    f"{SANDBOX_CREATE_FAILED}: 500: sandbox_create_failed: workload sbx-0123456789abcdef0123456789abcdef phase Failed"
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


#: What the hosted backend reports when a result file's every copy out of the sandbox came back short.
DOWNLOAD_REASON = f"{RESULT_DOWNLOAD_INCOMPLETE}: report.json: 4194301 bytes received, the manifest lists 4194304"


async def test_a_result_download_that_came_back_short_is_retried_after_waits_and_not_held_against_the_node() -> None:
    result, executor, reported = await _run(DOWNLOAD_REASON)
    assert len(executor.jobs) == 4, "a new attempt downloads the result again, up to the infrastructure retries"
    assert executor.clock.waits == [60.0, 120.0, 240.0]
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {DOWNLOAD_REASON}"
    assert reported == [], "the platform's file API, not the node, cut the download short"
    assert node_unquarantined(DOWNLOAD_REASON), "and it never quarantines the node"
    assert (
        classify_test_failure(returncode=0, error_class=RESULT_DOWNLOAD_INCOMPLETE).failure_class is FailureClass.INFRA
    )


def test_the_blameless_names_are_the_ones_the_runner_and_the_backend_report() -> None:
    assert NODE_BLAMELESS_INFRA_REASONS == {
        "runner_start_failed",
        "runner_lost",
        "sandbox_create_failed",
        "hub_rate_limited",
        "sandbox_acquire_timeout",
        "hf_cache_permission",
        "framework_version_mismatch",
        "result_download_incomplete",
    }
    assert NODE_BLAMELESS_INFRA_REASONS - {HUB_RATE_LIMITED, HF_CACHE_PERMISSION} <= INFRA_ERROR_CLASSES, (
        "the ones a log, not a status, names"
    )
    assert RUNNER_LOST == hosted.RUNNER_LOST
    assert SANDBOX_CREATE_FAILED == hosted.SANDBOX_CREATE_FAILED


class _TimedOutExecutor(_ScriptedExecutor):
    """Runs every attempt past the test's timeout, each on its own node."""

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        return TestResult(
            request_id="r1",
            validity=Validity.CANDIDATE_FAILED,
            failure_class=FailureClass.UNCLEAR,
            failure_reason=TEST_TIMEOUT_REASON,
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
        )


@pytest.mark.parametrize(
    "kind, runs, validity",
    [(TestKind.SWEEP, 1, Validity.MEASUREMENT_FAILED), (TestKind.TEST, 2, Validity.CANDIDATE_FAILED)],
    ids=["sweep", "test"],
)
async def test_a_sweep_that_ran_out_of_time_is_not_rerun(kind, runs, validity) -> None:
    executor = _TimedOutExecutor(TEST_TIMEOUT_REASON)
    time = _FakeTime()
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    job = _job()
    concs = (4, 8, 16) if kind is TestKind.SWEEP else ()
    job = replace(job, request=replace(job.request, kind=kind, concurrency=concs))
    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, backoff=_backoff(time))
    result = await runner.run(job)
    assert len(executor.jobs) == runs and len(result.attempts) == runs
    assert time.waits == [], "an unclear rerun does not wait"
    assert result.validity is validity and result.failure_class is FailureClass.UNCLEAR
    assert result.failure_reason == TEST_TIMEOUT_REASON
    assert reported == [], "a timeout is held against neither node"


class _ClassifyingExecutor(_ScriptedExecutor):
    """Ends every attempt with what :func:`classify_test_failure` makes of ``log`` and the step ``step`` names, as the
    executors record it."""

    def __init__(self, log: str, step: StepFailure | None = None, phases: tuple[PhaseTiming, ...] = ()) -> None:
        super().__init__("unused", phases=phases)
        self.log = log
        self.step = step

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        verdict = classify_test_failure(
            returncode=1, evidence=FailureEvidence(logs=(self.log,), step_failure=self.step)
        )
        infra = verdict.failure_class is FailureClass.INFRA
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED if infra else Validity.CANDIDATE_FAILED,
            failure_class=verdict.failure_class,
            failure_reason=verdict.failure_reason,
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
            phases=self.phases,
            unaccounted_seconds=0.0 if self.phases else None,
        )


async def _profile(
    log: str, step: StepFailure | None = None, phases: tuple[PhaseTiming, ...] = ()
) -> tuple[TestResult, _ClassifyingExecutor, list[str]]:
    executor = _ClassifyingExecutor(log, step, phases)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    runner = AttemptRunner(
        executor,
        avoid=frozenset,
        on_node_failure=on_node_failure,
        max_infra_retries=3,
        backoff=_backoff(executor.clock),
    )
    job = _job()
    job = replace(job, request=replace(job.request, kind=TestKind.PROFILE, purpose=Purpose.PROFILE))
    return await runner.run(job), executor, reported


#: A profile's server that died in the profiler's memory-snapshot dump after the capture wrote its traces.
_SNAPSHOT_LOG = (
    '  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py", line 872, '
    "in _post_process_after_profile\n"
    '    torch.cuda.memory._dump_snapshot("cuda_graph_runner_memory_usage.pickle")\n'
    '  File "/opt/venv/lib/python3.10/site-packages/torch/cuda/memory.py", line 1059, in _dump_snapshot\n'
    "    s = _snapshot()\n"
    '  File "/opt/venv/lib/python3.10/site-packages/torch/cuda/memory.py", line 1043, in _snapshot\n'
    "    return _C._cuda_memorySnapshot(None)\n"
    "RuntimeError: stoi\n"
    "Exception: Capture cuda graph failed: stoi\n"
)


async def test_a_profile_that_died_in_the_profilers_snapshot_dump_ends_unmeasured_not_against_the_candidate() -> None:
    result, executor, reported = await _profile(_SNAPSHOT_LOG)

    assert len(executor.jobs) == 1, "every sandbox of the image dies in the same dump"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"{PROFILER_SNAPSHOT_FAILED}: RuntimeError: stoi"
    assert reported == [], "the image's profiler, not the node, failed"


_TRACER_HANG = "queue_interposition.cpp:412 Async signal handler still waiting"
_SILENT = "no server log or client output for 600.4s"


async def test_a_profile_whose_server_stalled_inside_its_window_is_not_retried_and_ends_unmeasured() -> None:
    """The tracer hangs the server at the same point in every sandbox of the image; a retry only holds the cards again."""
    stall = StepFailure(FailedStep.SERVER_STALLED, _SILENT, profile_stall=_TRACER_HANG)

    result, executor, reported = await _profile("", stall)

    assert len(executor.jobs) == 1, "a hang in the profile window is not retried"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"{PROFILER_HUNG}: {_TRACER_HANG}"
    assert reported == [], "the image's profiler, not the node, hung the server"


async def test_a_profile_whose_server_stalled_outside_its_window_is_not_rerun_and_ends_unmeasured() -> None:
    result, executor, reported = await _profile("", StepFailure(FailedStep.SERVER_STALLED, _SILENT), SERVED)

    assert len(executor.jobs) == 1, "a profile's rerun fails again for the same cause"
    assert result.validity is Validity.MEASUREMENT_FAILED, "an unclear profile is not held against its candidate"
    assert result.failure_class is FailureClass.UNCLEAR and result.failure_reason == "server_stalled"
    assert reported == []


_ENGINE_DIED = "RuntimeError: Engine core process EngineCore died unexpectedly.\n"
_FATAL = "EngineCore encountered a fatal error.\nRuntimeError: cancelled"


async def test_a_profile_whose_server_died_inside_its_window_is_retried_once_elsewhere_then_ends_unmeasured() -> None:
    """The tracer kills a worker when it starts on live serving steps, most often in every sandbox of the image; one
    retry on another node separates that from a fault of the first node."""
    death = StepFailure(FailedStep.SERVER_DIED, _FATAL, profile_death=True)

    result, executor, reported = await _profile(_ENGINE_DIED, death)

    assert len(executor.jobs) == 2, "a death in the profile window is retried once, and only once"
    assert "n1" in executor.jobs[1].avoid_nodes, "the retry goes to a node other than the first"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == f"{PROFILER_DIED}: EngineCore encountered a fatal error. RuntimeError: cancelled"
    assert reported == [], "the image's profiler, not the node, killed the server"


async def test_a_profile_whose_server_died_outside_its_window_is_not_rerun_and_ends_unmeasured() -> None:
    result, executor, reported = await _profile(_ENGINE_DIED, StepFailure(FailedStep.SERVER_DIED, _FATAL), SERVED)

    assert len(executor.jobs) == 1, "a profile's rerun fails again for the same cause"
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert result.failure_class is FailureClass.UNCLEAR and result.failure_reason.startswith("python_exception: ")
    assert reported == []


_NCCL = "RuntimeError: NCCL error: unhandled cuda error\n"


@pytest.mark.parametrize("phases, runs", [(BOOTED, 4), ((), 2)], ids=["booted", "no_phases"])
async def test_a_profile_whose_server_failed_before_it_was_ready_is_still_rerun(phases, runs) -> None:
    """A boot also fails for a passing fault of its node; the reruns keep the session's profile."""
    exited = StepFailure(FailedStep.SERVER_EXITED_BEFORE_READY, "the server exited with code -9 before it was ready")

    result, executor, _ = await _profile(_NCCL, exited, phases)

    assert len(executor.jobs) == runs and "n1" in executor.jobs[1].avoid_nodes
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR
    assert executor.clock.waits == ([60.0, 120.0, 240.0] if runs == 4 else [])


#: The unclear failure of a server whose collective failed to initialise while it booted.
_BOOT_REASON = "python_exception: RuntimeError: NCCL error: unhandled cuda error"


class _UnclearExecutor(_ScriptedExecutor):
    """Fails its first ``failures`` attempts as an unclear failure, each on its own node, then measures on the next
    node. ``phases`` gives the phases of each attempt, the last for every later one."""

    def __init__(self, failures: int, phases: tuple[tuple[PhaseTiming, ...], ...], reason: str = _BOOT_REASON) -> None:
        super().__init__(reason)
        self.failures = failures
        self.phases_by_attempt = phases

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        self.started.append(self.clock())
        sandbox = SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}")
        if job.attempt > self.failures:
            return TestResult(
                request_id="r1",
                validity=Validity.SUCCEEDED,
                rounds=(RoundValue("measure", 64, {"output_tps": 1000.0}),),
                sandbox=sandbox,
            )
        phases = self.phases_by_attempt[min(job.attempt, len(self.phases_by_attempt)) - 1]
        return TestResult(
            request_id="r1",
            validity=Validity.CANDIDATE_FAILED,
            failure_class=FailureClass.UNCLEAR,
            failure_reason=self.reason,
            sandbox=sandbox,
            phases=phases,
            unaccounted_seconds=0.0 if phases else None,
        )


async def _unclear(
    failures: int, phases: tuple[tuple[PhaseTiming, ...], ...], reason: str = _BOOT_REASON
) -> tuple[TestResult, _UnclearExecutor, list[tuple[str, bool]]]:
    executor = _UnclearExecutor(failures, phases, reason)
    reported: list[tuple[str, bool]] = []

    async def on_node_failure(node: str, _reason: str, reproduced: bool) -> None:
        reported.append((node, reproduced))

    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, backoff=_backoff(executor.clock))
    return await runner.run(_job()), executor, reported


async def test_a_server_that_failed_to_boot_three_times_is_rerun_after_waits_and_its_success_is_the_result() -> None:
    """Boots fail in bursts of minutes whatever the candidate; the reruns are spread over the infrastructure backoff."""
    result, executor, reported = await _unclear(3, (BOOTED,))

    assert result.validity is Validity.SUCCEEDED and result.failure_class is None
    assert executor.clock.waits == [60.0, 120.0, 240.0]
    gaps = [later - earlier for earlier, later in zip(executor.started, executor.started[1:])]
    assert gaps == [60.0, 120.0, 240.0], "each rerun starts only after its wait, no attempt is running meanwhile"
    assert [job.avoid_nodes for job in executor.jobs] == [(), ("n1",), ("n1", "n2"), ("n1", "n2", "n3")]
    assert [a.failure_class for a in result.attempts] == [FailureClass.INFRA] * 3 + [None]
    assert all(a.failure_reason.startswith("unclear failure did not reproduce on a rerun") for a in result.attempts[:3])
    assert reported == [("n1", False), ("n2", False), ("n3", False)], "each one a failure that did not reproduce"


async def test_a_server_that_fails_every_boot_is_still_a_candidate_failure_after_its_last_rerun() -> None:
    result, executor, reported = await _unclear(4, (BOOTED,))

    assert len(executor.jobs) == 4 and len(result.attempts) == 4
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR
    assert result.failure_reason == _BOOT_REASON
    assert executor.clock.waits == [60.0, 120.0, 240.0]
    assert reported == []


@pytest.mark.parametrize("phases", [SERVED, ()], ids=["served", "no_phases"])
async def test_an_unclear_failure_that_did_not_end_a_boot_is_rerun_once_at_once(phases) -> None:
    result, executor, _ = await _unclear(2, (phases,))

    assert len(executor.jobs) == 2 and executor.jobs[1].avoid_nodes == ("n1",)
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR
    assert executor.clock.waits == []


async def test_a_test_that_ran_out_of_its_time_while_it_booted_is_rerun_once_at_once() -> None:
    result, executor, _ = await _unclear(2, (BOOTED,), reason=TEST_TIMEOUT_REASON)

    assert len(executor.jobs) == 2 and executor.clock.waits == []
    assert result.validity is Validity.CANDIDATE_FAILED


async def test_a_failed_boot_after_a_failure_once_served_gets_the_boot_reruns_left() -> None:
    result, executor, _ = await _unclear(4, (SERVED, BOOTED))

    assert len(executor.jobs) == 4, "the served failure took one of the reruns"
    assert executor.clock.waits == [60.0, 120.0], "only the reruns after a failed boot wait"
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR


@pytest.mark.parametrize(
    "reason", [DISK_FULL_REASON, f"{RUNNER_LOST}: the sandbox is gone"], ids=["disk_full", "runner_lost"]
)
async def test_a_profile_ends_unmeasured_when_its_retry_failed_for_the_same_reason_after_its_server_was_ready(
    reason,
) -> None:
    result, executor, reported = await _run(reason, kind=TestKind.PROFILE, phases=SERVED)

    assert len(executor.jobs) == 2, "a third attempt would hold cards for the cause two nodes already met"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == reason
    assert "n2" not in reported, "a cause two nodes met is not the second node's"


@pytest.mark.parametrize("phases", [BOOTED, ()], ids=["booted", "no_phases"])
@pytest.mark.parametrize("reason", [PORT_REASON, HUNG_REASON], ids=["port", "hung_before_ready"])
async def test_a_profile_retries_a_failure_before_its_server_was_ready_like_any_test(reason, phases) -> None:
    result, executor, reported = await _run(reason, kind=TestKind.PROFILE, phases=phases)

    assert len(executor.jobs) == 4
    assert result.failure_reason == f"4 attempts failed for infrastructure reasons: {reason}"
    assert reported == ["n1", "n2", "n3", "n4"]


async def test_a_profile_retries_another_infrastructure_reason_after_a_first_one() -> None:
    reasons = (DISK_FULL_REASON, f"{RUNNER_LOST}: the sandbox is gone", DISK_FULL_REASON)

    result, executor, reported = await _run(reasons, kind=TestKind.PROFILE, phases=SERVED)

    assert len(executor.jobs) == 3, "the third attempt meets the first one's cause again"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_reason == DISK_FULL_REASON
    assert reported == ["n1"], "a lost runner and a cause two nodes met report no node"
    assert executor.jobs[2].avoid_nodes == ("n1", "n2")


async def test_a_profile_whose_candidate_crashed_is_still_a_candidate_failure() -> None:
    crash = (
        "torch.AcceleratorError: HIP error: an illegal memory access was encountered\n"
        "Search for `hipErrorIllegalAddress' for more information.\n"
    )

    result, executor, _ = await _profile(_SNAPSHOT_LOG + crash)

    assert len(executor.jobs) == 1
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
