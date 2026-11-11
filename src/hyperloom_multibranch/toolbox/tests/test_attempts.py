# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which infrastructure failures :class:`AttemptRunner` retries."""

from __future__ import annotations

from dataclasses import replace

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
from hyperloom.measure.failures import (
    FOREIGN_PROCESS_ON_CARD,
    HF_CACHE_PERMISSION,
    HUB_RATE_LIMITED,
    INFRA_ERROR_CLASSES,
    JIT_BUILD_TIMEOUT,
    PROFILER_SNAPSHOT_FAILED,
    TEST_TIMEOUT_REASON,
    classify_test_failure,
)
from hyperloom.measure.runner.import_outcome import INERT_PATCH
from hyperloom_multibranch.toolbox.attempts import (
    CAPACITY_UNAVAILABLE,
    NODE_BLAMELESS_INFRA_REASONS,
    RUNNER_START_TIMEOUT,
    WAITED_INFRA_REASONS,
    RUNNER_LOST,
    RUNNER_START_FAILED,
    SANDBOX_CREATE_FAILED,
    INFRA_BACKOFF_SECONDS,
    AttemptRunner,
    InfraBackoff,
    TestJob,
)

INERT_REASON = f"{INERT_PATCH}: process 1 imported aiter from /sgl-workspace/aiter/aiter/__init__.py, not from the farm"
PORT_REASON = "port_in_use: [Errno 98] address already in use"
DISK_FULL_REASON = "disk_full: OSError: [Errno 122] Disk quota exceeded: '/work/.cache/triton/k.json'"
HUNG_REASON = f"{HUNG_BEFORE_READY}: no output for 600s before the server was ready"
#: Failures that are not the node's: cards a process outside the test held, no sandbox from the platform in time.
FOREIGN_REASON = f"{FOREIGN_PROCESS_ON_CARD}: card 3 held by pid 4242 (python)"
START_TIMEOUT_REASON = RUNNER_START_TIMEOUT
CAPACITY_REASON = f"{CAPACITY_UNAVAILABLE}: no node with 8 free GPUs"


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
    at each attempt. A sequence of reasons gives one per attempt, the last for every later one."""

    def __init__(self, reason: str | tuple[str, ...], clock: _FakeTime | None = None) -> None:
        self.reasons = (reason,) if isinstance(reason, str) else reason
        self.reason = self.reasons[0]
        self.jobs: list[TestJob] = []
        self.clock = clock or _FakeTime()
        self.started: list[float] = []

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        self.started.append(self.clock())
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED,
            failure_class=FailureClass.INFRA,
            failure_reason=self.reasons[min(job.attempt, len(self.reasons)) - 1],
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


def _backoff(time: _FakeTime, latest: float | None = None) -> InfraBackoff:
    return InfraBackoff(INFRA_BACKOFF_SECONDS, lambda _request: latest, clock=time, sleep=time.sleep)


async def _run(
    reason: str | tuple[str, ...], latest: float | None = None
) -> tuple[TestResult, _ScriptedExecutor, list[str]]:
    executor = _ScriptedExecutor(reason)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    backoff = _backoff(executor.clock, latest)
    runner = AttemptRunner(
        executor, avoid=frozenset, on_node_failure=on_node_failure, max_infra_retries=3, backoff=backoff
    )
    return await runner.run(_job()), executor, reported


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
    assert "n2" in reported and "n4" in reported


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
    reason = classify_test_failure(returncode=1, logs=[log]).failure_reason
    assert reason is not None and reason.startswith(f"{HF_CACHE_PERMISSION}: ")
    result, executor, reported = await _run(reason)
    assert len(executor.jobs) == 1 and len(result.attempts) == 1, "every sandbox shares the cache and is refused alike"
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.failure_reason == reason
    assert reported == [], "the shared cache's permissions, not the node, refused the lock"


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


def test_the_blameless_names_are_the_ones_the_runner_and_the_backend_report() -> None:
    assert NODE_BLAMELESS_INFRA_REASONS == {
        "runner_start_failed",
        "runner_lost",
        "sandbox_create_failed",
        "hub_rate_limited",
        "sandbox_acquire_timeout",
        "hf_cache_permission",
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
    """Ends every attempt with what :func:`classify_test_failure` makes of ``log``, as the executors record it."""

    def __init__(self, log: str) -> None:
        super().__init__("unused")
        self.log = log

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        verdict = classify_test_failure(returncode=1, logs=(self.log,))
        infra = verdict.failure_class is FailureClass.INFRA
        return TestResult(
            request_id="r1",
            validity=Validity.MEASUREMENT_FAILED if infra else Validity.CANDIDATE_FAILED,
            failure_class=verdict.failure_class,
            failure_reason=verdict.failure_reason,
            sandbox=SandboxFacts(sandbox_id=f"s{job.attempt}", node_name=f"n{job.attempt}"),
        )


async def _profile(log: str) -> tuple[TestResult, _ClassifyingExecutor, list[str]]:
    executor = _ClassifyingExecutor(log)
    reported: list[str] = []

    async def on_node_failure(node: str, _reason: str, _reproduced: bool) -> None:
        reported.append(node)

    runner = AttemptRunner(executor, avoid=frozenset, on_node_failure=on_node_failure, max_infra_retries=3)
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


async def test_a_profile_whose_candidate_crashed_is_still_a_candidate_failure() -> None:
    crash = (
        "torch.AcceleratorError: HIP error: an illegal memory access was encountered\n"
        "Search for `hipErrorIllegalAddress' for more information.\n"
    )

    result, executor, _ = await _profile(_SNAPSHOT_LOG + crash)

    assert len(executor.jobs) == 1
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
