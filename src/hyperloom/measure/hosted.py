# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted measurement backend: every test runs in a fresh authoritative sandbox (design 4.4.4).

``HostedMeasurementBackend.run_test(call)`` is the blocking entry point the tool layer calls once per attempt (from
``asyncio.to_thread``). It builds the instruction sheet, gets a sandbox from the :class:`~.sandboxes.SandboxPool`,
writes the sheet and the ``READY`` marker, waits for the runner's ``status.json``, and accepts a result only after
:func:`~hyperloom.measure.bundle.verify_bundle` proved that the runner of *that* sandbox signed *this* measurement. The
verified bundle goes to the result archive first and to the output directory second, so what a decision rests on can
always be audited from the archive.

A journal (:mod:`.hosted_log`) records every sandbox before the sheet is written, so a restarted application re-attaches
to the sandbox that is still measuring instead of paying for the measurement twice, and a test is never lost.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hyperloom.common.env import env_bool
from hyperloom.contracts.measurement import AccuracyValue, FailureClass, SandboxFacts
from hyperloom.measure import bundle as bundle_mod
from hyperloom.measure.archive import ArchiveError, ResultArchive
from hyperloom.measure.bundle import (
    BundleExpectation,
    MaterializeError,
    RecordedBundle,
    RejectReason,
    ResultRejected,
    VerifiedBundle,
)
from hyperloom.measure.failures import (
    RUNNER_EXITED,
    SANDBOX_ACQUIRE_TIMEOUT,
    Disposition,
    FailureClassification,
    classify_test_failure,
)
from hyperloom.measure.hosted_live import LiveOutputs, finite_number
from hyperloom.measure.hosted_log import (
    STATE_FINISHED,
    STATE_SENT,
    STATE_SUBMITTED,
    MeasurementJournal,
    Record,
    bounded,
    runner_status,
    status_evidence,
    where,
)
from hyperloom.measure.hosted_spec import SpecContext, SpecRefused, build_spec, split_result_dir
from hyperloom.measure.jit.publish import publish_contributions
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.phases import ACQUIRED, LANDED, RUNNER_DONE, HostTimeline
from hyperloom.measure.boot_watch import HUNG_BEFORE_READY, PRE_READY_SILENCE_SEC, PreReadySilence
from hyperloom.measure.runner.rounds import ROUNDS_FILE, WARMUP_DIR
from hyperloom.measure.returncodes import (
    HOSTED_INFRA_FAILURE_RC,
    HOSTED_RESULT_REJECTED_RC,
    HOSTED_RUNNER_UNAVAILABLE_RC,
    HOSTED_SPEC_REJECTED_RC,
)
from hyperloom.measure.sandboxes import (
    DEFAULT_MAX_PARALLEL_TESTS,
    DEFAULT_RUNNER_START_TIMEOUT_SEC,
    DEFAULT_TIMEOUT_SLACK_SEC,
    AcquireTimedOut,
    AttemptTag,
    NoRunnerTemplate,
    PoolClosed,
    PoolLimits,
    RunnerSandbox,
    SandboxPool,
)
from hyperloom.measure.signing import SigningError, verify
from hyperloom.measure.socket import MeasurementTimeout, RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.spec import MeasurementSpec, blob_input_path, fingerprint, spec_sha256
from hyperloom.sandbox.socket import (
    Sandbox,
    SandboxCreateFailed,
    SandboxDenied,
    SandboxError,
    SandboxGone,
    SandboxProvider,
    SandboxUnavailable,
)

log = logging.getLogger(__name__)

__all__ = [
    "HostedConfig",
    "HOSTED_DIR_NAME",
    "HostedMeasurementBackend",
    "HostedStartError",
    "Reverified",
    "STOPPED_RETURNCODE",
]

#: ``ORCHESTRATOR_CANCELLED_RETURNCODE`` of the local flow; a stopped hosted test returns the same code.
STOPPED_RETURNCODE = -917
STOPPED_REASON = "orchestrator_cancelled"
RUNNER_LOST = "runner_lost"
#: The platform made a test's sandbox and it failed before it ran (``SandboxCreateFailed``); the reason carries the
#: platform's message.
SANDBOX_CREATE_FAILED = "sandbox_create_failed"
HOSTED_DIR_NAME = "hosted"
BACKEND_NAME = "hosted"
READY_PATH, SPEC_PATH = "/in/READY", "/in/spec.json"
STATUS_PATH, PROGRESS_PATH = "/out/status.json", "/out/progress.json"
MANIFEST_PATH, SIGNATURE_PATH, FILES_ROOT = "/out/manifest.json", "/out/manifest.sig", "/out/files"
_LOSS_POLLS = 3
_LOG_SUFFIX = ".log"

EventEmitter = Callable[[str, Mapping[str, Any]], None]


class HostedStartError(RuntimeError):
    """The hosted backend cannot start in this environment (4.4.4 start checks)."""


@dataclass(frozen=True)
class Reverified:
    """What an archived bundle says once it verified again: its signed manifest, its rounds and the sheet's points.

    The rounds' directories lie below the directory the bundle was landed in; ``point_yaml`` is the YAML of every point
    of the sheet that was sent, in order, so a round's ``point`` indexes it.
    """

    manifest: ResultManifest
    rounds: tuple[RoundOutcome, ...]
    point_yaml: tuple[str, ...]


class _Events(Protocol):
    def emit(self, type: str, data: Mapping[str, Any]) -> None: ...  # noqa: A002 - the events socket's name


@dataclass(frozen=True)
class HostedConfig:
    """Configuration of the hosted backend; defaults are those of design 4.4.4.

    ``allow_unsigned`` exists for the table of 4.4.4 but is refused when true: no result is ever accepted without a
    signature that verified against the key the platform reported for its sandbox. ``rule_pack_digest`` is the digest
    of the rule pack the session reads: a result whose runner did not read a pack with that digest is rejected
    (``rule_pack_mismatch``); ``None`` accepts any pack. ``create_envs`` is the environment every authoritative sandbox is
    created with; the service's sealed runner template must accept those names (``accepted_envs``). ``jit_store`` is the
    shared compiled-code cache (design 4.21): the compiled files a signed result says its test added are published to it
    after the result is accepted; ``None`` publishes nothing.
    """

    session_dir: Path
    run_id: str
    spec: SpecContext
    runner_templates: Mapping[int, str]
    archive: ResultArchive
    runner_start_timeout_sec: float = DEFAULT_RUNNER_START_TIMEOUT_SEC
    timeout_slack_sec: int = DEFAULT_TIMEOUT_SLACK_SEC
    poll_interval_sec: float = 5.0
    pre_ready_silence_sec: float = PRE_READY_SILENCE_SEC
    max_parallel_tests: int = DEFAULT_MAX_PARALLEL_TESTS
    allow_unsigned: bool = False
    nodes: int = 1
    runner_version: str | None = None
    rule_pack_digest: str | None = None
    create_envs: Mapping[str, str] = field(default_factory=dict)
    staging_root: Path | None = None
    jit_store: DirJitStore | None = None
    should_stop: Callable[[], bool] = field(default=lambda: False, compare=False)

    def __post_init__(self) -> None:
        if self.allow_unsigned:
            raise ValueError("allow_unsigned is not supported: a result without a verified signature is never accepted")
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")

    @property
    def hosted_dir(self) -> Path:
        return Path(self.session_dir) / HOSTED_DIR_NAME


class _Stops:
    """The stop switch of every test that is running: one per request, and closing the backend sets them all.

    A test is told to stop by :meth:`withdraw`. The switch is read where the test waits (for a slot, for its sandbox's
    result) so that a withdrawn test gives its sandbox, and with it its cards, back at once instead of measuring on
    for minutes after nobody wants the result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, threading.Event] = {}
        self._withdrawn: set[str] = set()
        self._closed = False

    def begin(self, request_id: str) -> threading.Event:
        """The switch of ``request_id``'s run; already set when the backend closed or the test was withdrawn first."""
        with self._lock:
            switch = self._running[request_id] = threading.Event()
            if self._closed or request_id in self._withdrawn:
                switch.set()
            return switch

    def end(self, request_id: str) -> None:
        """The run is over; its switch and a withdrawal that reached it are dropped."""
        with self._lock:
            self._running.pop(request_id, None)
            self._withdrawn.discard(request_id)

    def withdraw(self, request_id: str) -> None:
        """Set the switch of ``request_id``'s run, or, when it has not begun yet (its worker thread may still be
        queued), make it begin with the switch set. A withdrawal that arrives after the run ended is kept: one short
        string, as a request id never runs again after it was withdrawn."""
        with self._lock:
            self._withdrawn.add(request_id)
            switch = self._running.get(request_id)
        if switch is not None:
            switch.set()

    def withdrawn(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._withdrawn

    def close(self) -> None:
        with self._lock:
            self._closed = True
            switches = list(self._running.values())
        for switch in switches:
            switch.set()


@dataclass
class _Attempt:
    """Everything one attempt needs while it runs."""

    call: TestCall
    request_id: str
    number: int
    spec: MeasurementSpec
    spec_raw: bytes
    runner: RunnerSandbox
    started_unix: float
    stop: threading.Event
    timeline: HostTimeline


def _outcome(
    call: TestCall,
    rc: int,
    *,
    facts: SandboxFacts | None = None,
    stderr: str = "",
    failure: FailureClassification | None = None,
) -> TestOutcome:
    failure_class = failure.failure_class if failure and failure.disposition is Disposition.FAILED else None
    reason = failure.failure_reason if failure else None
    return TestOutcome(
        returncode=rc,
        stdout="",
        stderr=stderr,
        rounds=(),
        sandbox=facts or SandboxFacts(sandbox_id=f"hosted-{call.task_id}", backend=BACKEND_NAME),
        request_id=call.task_id,
        failure_class=failure_class,
        failure_reason=reason,
    )


def _refusal(call: TestCall, rc: int, reason: str, detail: str = "") -> TestOutcome:
    """An end that is not a measurement: the failure class is empty and the reason names what was wrong."""
    outcome = _outcome(call, rc, stderr=f"{reason}: {detail}" if detail else reason)
    return replace(outcome, failure_reason=reason)


def _infra(call: TestCall, rc: int, reason: str, detail: str = "", facts: SandboxFacts | None = None) -> TestOutcome:
    verdict = classify_test_failure(returncode=0, error_class=reason)
    return _outcome(call, rc, facts=facts, stderr=f"{reason}: {detail}" if detail else reason, failure=verdict)


class HostedMeasurementBackend:
    """Runs each test in a fresh authoritative sandbox and accepts only a verified, signed result."""

    name = BACKEND_NAME

    def __init__(
        self,
        provider: SandboxProvider,
        config: HostedConfig,
        *,
        events: _Events | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._events = events
        self._pool = SandboxPool(
            provider,
            config.runner_templates,
            run_id=config.run_id,
            limits=PoolLimits(
                max_parallel_tests=config.max_parallel_tests,
                timeout_slack_sec=config.timeout_slack_sec,
                request_timeout_sec=config.runner_start_timeout_sec,
            ),
            create_envs=config.create_envs,
        )
        self._journal: MeasurementJournal | None = None
        self._stops = _Stops()
        self._live = LiveOutputs(clock)

    # -- lifecycle ----------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Check the three conditions of 4.4.4 that are known before any test; refuse to start when one fails."""
        if self._config.nodes != 1:
            raise HostedStartError(f"hosted measurement runs single-node activities, this one has {self._config.nodes}")
        if env_bool("INFERENCE_OPTIMIZER_RAY_EXEC", default=False):
            raise HostedStartError("INFERENCE_OPTIMIZER_RAY_EXEC is on; hosted measurement does not use Ray")
        if not self._config.runner_templates:
            raise HostedStartError("no runner template list: set the host information or runner_templates")
        if self._config.archive is None:
            raise HostedStartError("no result archive: set the host information or archive_dir")
        self._journal = MeasurementJournal(self._config.hosted_dir)

    def close(self) -> None:
        """Stop accepting tests and close every authoritative sandbox still running."""
        self._stops.close()
        self._pool.close()

    @property
    def pool(self) -> SandboxPool:
        return self._pool

    @property
    def journal(self) -> MeasurementJournal:
        if self._journal is None:
            raise HostedStartError("the backend was not started")
        return self._journal

    def live_output(self, request_id: str) -> tuple[float, str] | None:
        """``(seconds since the newest live log was written, its last line)`` of a test that is running, as its runner
        reported at the latest poll; ``None`` when the test is not running or its runner has reported no log yet."""
        return self._live.get(request_id)

    # -- the entry point ----------------------------------------------------------------------------------------

    def run_test(self, call: TestCall) -> TestOutcome:
        """Run one attempt of a test to the end.

        Returns the outcome; a failed measurement, a refused sheet and a rejected result are all outcomes with their
        own return code (``returncodes``). A timeout raises :class:`subprocess.TimeoutExpired`, as the local flow does.
        A withdrawn test (:meth:`withdraw`) returns the stop code once its sandbox is closed.
        """
        stop = self._stops.begin(call.task_id)
        timeline = HostTimeline(self._clock)
        try:
            outcome = self._attempt(call, stop, timeline)
        except subprocess.TimeoutExpired as exc:
            runner = exc.phases if isinstance(exc, MeasurementTimeout) else ()
            raise MeasurementTimeout(str(exc.cmd), exc.timeout, phases=timeline.compose(runner)) from exc
        finally:
            self._stops.end(call.task_id)
            self._live.forget(call.task_id)
        return replace(outcome, phases=timeline.compose(outcome.phases))

    def _attempt(self, call: TestCall, stop: threading.Event, timeline: HostTimeline) -> TestOutcome:
        journal = self.journal
        previous = journal.latest(call.task_id)
        if previous is not None and previous.state in (STATE_SUBMITTED, STATE_SENT):
            resumed = self._resume(call, previous, stop, timeline)
            if resumed is not None:
                return resumed
        return self._fresh(call, journal.next_attempt(call.task_id), stop, timeline)

    def withdraw(self, request_id: str) -> None:
        """Tell the test of ``request_id`` that nobody wants its result any more; it returns at once.

        The test ends at its next poll (or leaves the wait for a slot): its sandbox is closed and its cards go back to
        the pool, so the next test can have them. Without this a withdrawn test measures on to the end, holding its
        card the whole time. Calling it for a test that is not running, or twice, changes nothing.
        """
        self._stops.withdraw(request_id)

    # -- a new attempt ------------------------------------------------------------------------------------------

    def _fresh(self, call: TestCall, number: int, stop: threading.Event, timeline: HostTimeline) -> TestOutcome:
        try:
            spec = self._build(call)
            gpus = spec.hardware.gpu_count
            self._pool.template_for(gpus)
        except SpecRefused as exc:
            return _refusal(call, HOSTED_SPEC_REJECTED_RC, exc.reason, exc.detail)
        except NoRunnerTemplate as exc:
            return _refusal(call, HOSTED_SPEC_REJECTED_RC, "no_runner_template", str(exc))
        acquired = self._acquire(call, gpus, number, stop)
        if isinstance(acquired, TestOutcome):
            return acquired
        runner = acquired
        timeline.mark(ACQUIRED)
        try:
            if stop.is_set():  # told to stop while it waited for the sandbox: nothing is sent to it
                return _refusal(call, STOPPED_RETURNCODE, STOPPED_REASON, "the test was stopped before it started")
            if not runner.public_key_pem:
                return _refusal(
                    call, HOSTED_RESULT_REJECTED_RC, "no_signing_key", "the platform reported no signing key"
                )
            attempt = _Attempt(call, call.task_id, number, spec, spec.to_bytes(), runner, self._clock(), stop, timeline)
            self._submit(attempt)
            return self._finish(attempt)
        except SandboxError as exc:
            return self._lost(call, runner, f"{type(exc).__name__}: {exc}")
        finally:
            self._pool.release(runner)

    def _build(self, call: TestCall) -> MeasurementSpec:
        remaining = None if call.session_deadline_sec is None else call.session_deadline_sec - time.monotonic()
        window = call.timeout_sec + self._config.timeout_slack_sec if remaining is None else remaining
        return build_spec(call, request_id=call.task_id, ctx=self._config.spec, deadline_unix=self._clock() + window)

    def _acquire(self, call: TestCall, gpus: int, number: int, stop: threading.Event) -> RunnerSandbox | TestOutcome:
        remaining = None if call.session_deadline_sec is None else call.session_deadline_sec - time.monotonic()
        try:
            return self._pool.acquire(
                gpus,
                AttemptTag(call.task_id, number, tuple(call.avoid_nodes), _meta_of(call)),
                timeout_sec=call.timeout_sec,
                remaining_sec=remaining,
                stop=stop,
            )
        except AcquireTimedOut as exc:  # before SandboxUnavailable, its base: no sandbox was made, so the line is safe
            return self._create_failed(call, number, SANDBOX_ACQUIRE_TIMEOUT, str(exc))
        except SandboxUnavailable as exc:
            return _infra(call, HOSTED_RUNNER_UNAVAILABLE_RC, "runner_start_timeout", str(exc))
        except SandboxGone as exc:
            return self._create_failed(call, number, RUNNER_LOST, f"the sandbox ended during creation: {exc}")
        except SandboxCreateFailed as exc:
            return self._create_failed(call, number, SANDBOX_CREATE_FAILED, str(exc))
        except SandboxDenied as exc:
            return _refusal(call, HOSTED_SPEC_REJECTED_RC, "sandbox_denied", str(exc))
        except PoolClosed:
            return _refusal(call, STOPPED_RETURNCODE, STOPPED_REASON, "the pool is closed")

    def _create_failed(self, call: TestCall, number: int, reason: str, message: str) -> TestOutcome:
        """A create that made no usable sandbox: the platform's message goes to the outcome and to the journal, the only
        place it outlives this process."""
        outcome = _infra(call, HOSTED_INFRA_FAILURE_RC, reason, bounded(message))
        fields = {"verdict": reason, "detail": bounded(outcome.stderr)}
        self._journal_end(call.task_id, number, None, call, _end_fields(outcome, self._clock(), fields))
        return outcome

    def _submit(self, attempt: _Attempt) -> None:
        """Journal the sandbox, then write the sheet, then the READY marker (so the runner never reads half a sheet)."""
        call = attempt.call
        self.journal.save_spec(attempt.request_id, attempt.number, attempt.spec_raw)
        self._append(
            attempt, STATE_SUBMITTED, {"submitted_unix": attempt.started_unix, "timeout_sec": call.timeout_sec}
        )
        self._send(attempt)
        self._emit("hyperloom.measurement_started", attempt, {})

    def _send(self, attempt: _Attempt) -> None:
        """Blobs, then the sheet, then the marker; every write replaces what is there, so sending again is safe."""
        handle = _handle(attempt.runner)
        self._upload_blobs(handle, attempt.spec)
        handle.write_file(SPEC_PATH, attempt.spec_raw)
        handle.write_file(READY_PATH, b"")
        self._append(attempt, STATE_SENT, {"submitted_unix": attempt.started_unix})

    def _upload_blobs(self, handle: Sandbox, spec: MeasurementSpec) -> None:
        """Every attachment and overlay file goes to ``/in/blobs/<sha256>`` before the sheet that refers to it."""
        for blob in sorted(_blob_refs(spec)):
            handle.write_file(blob_input_path(blob), self._config.spec.blobs.get(blob.removeprefix("sha256:")))

    # -- a re-attached attempt ----------------------------------------------------------------------------------

    def _resume(
        self, call: TestCall, previous: Record, stop: threading.Event, timeline: HostTimeline
    ) -> TestOutcome | None:
        """Re-attach to the sandbox of an unfinished attempt; ``None`` when it is gone and a new attempt must start."""
        try:
            runner = self._pool.reattach(str(previous.sandbox_id), stop=stop, timeout_sec=call.timeout_sec)
        except AcquireTimedOut as exc:  # its sandbox measures on with no slot to hold it: end it, and the attempt
            self._pool.discard(str(previous.sandbox_id))
            outcome = _infra(call, HOSTED_INFRA_FAILURE_RC, SANDBOX_ACQUIRE_TIMEOUT, bounded(str(exc)))
            self._mark_finished(previous, _end_fields(outcome, self._clock(), {"verdict": SANDBOX_ACQUIRE_TIMEOUT}))
            return outcome
        except SandboxGone:
            self._mark_finished(previous, {"verdict": RUNNER_LOST, "reason": "sandbox gone on resume"})
            return None
        except PoolClosed:
            if self._stops.withdrawn(call.task_id):  # nobody wants it: it is closed, where a closing run leaves it
                self._pool.discard(str(previous.sandbox_id))
                self._mark_finished(previous, {"verdict": "withdrawn"})
            return _refusal(call, STOPPED_RETURNCODE, STOPPED_REASON, "the pool is closed")
        timeline.mark(ACQUIRED)
        try:
            try:
                raw = self.journal.load_spec(previous.request_id, previous.attempt)
                spec = MeasurementSpec.from_bytes(raw)
            except (OSError, ValueError) as exc:
                self._mark_finished(previous, {"verdict": "resume_failed", "reason": type(exc).__name__})
                return _refusal(
                    call, HOSTED_SPEC_REJECTED_RC, "resume_failed", f"the journalled sheet is unusable: {exc}"
                )
            started = float(previous.extra.get("submitted_unix", self._clock()))
            attempt = _Attempt(call, previous.request_id, previous.attempt, spec, raw, runner, started, stop, timeline)
            if not runner.public_key_pem:
                return _refusal(
                    call, HOSTED_RESULT_REJECTED_RC, "no_signing_key", "the platform reported no signing key"
                )
            if previous.state == STATE_SUBMITTED:
                self._send(attempt)
            return self._finish(attempt)
        except SandboxError as exc:
            return self._lost(call, runner, f"{type(exc).__name__}: {exc}")
        finally:
            self._pool.release(runner)

    # -- waiting and accepting ----------------------------------------------------------------------------------

    def _finish(self, attempt: _Attempt) -> TestOutcome:
        status = self._wait(attempt)
        attempt.timeline.mark(RUNNER_DONE)
        try:
            if isinstance(status, TestOutcome):
                return self._ended_without_result(attempt, status)
            if status.get("state") == "succeeded":
                return self._accept(attempt)
            return self._runner_failed(attempt, status)
        finally:
            attempt.timeline.mark(LANDED)

    def _ended_without_result(self, attempt: _Attempt, outcome: TestOutcome) -> TestOutcome:
        """A failed test with no status read: what ``status.json`` holds now, or that it is missing or unreadable, goes
        with the outcome and its journal line, since ``/out`` is gone once the sandbox is."""
        fields: dict[str, Any] = {"verdict": "ended_without_result"}
        if outcome.failure_class is not None:  # a stopped test is no failure to explain
            evidence = status_evidence(attempt.runner.handle, STATUS_PATH)
            fields["runner_status"] = evidence
            told = "; ".join(f"{key} {value}" for key, value in evidence.items() if value)
            outcome = replace(outcome, stderr=f"{outcome.stderr} ({where(*_place(attempt.runner))}; {told})")
        self._finish_record(attempt, outcome, fields)
        return outcome

    def _wait(self, attempt: _Attempt) -> dict[str, Any] | TestOutcome:
        """Poll until the runner writes ``status.json``; a stop, a timeout, a lost sandbox or a test silent before its server
        is ready (``boot_watch``) ends the wait early."""
        call, config = attempt.call, self._config
        handle = _handle(attempt.runner)
        last_output = 0.0
        unreachable_since: float | None = None
        silence = PreReadySilence(self._clock(), config.pre_ready_silence_sec)
        while True:
            exited = None
            try:
                # Asked before the read: a runner writes status.json before it exits, so one that has exited and left
                # none never will.
                exited = _startup_exit(handle)
                status = _read_status(handle)
                last_output = self._report_progress(handle, call, last_output, silence)
                unreachable_since = None
            except (SandboxGone, SandboxUnavailable):
                unreachable_since = unreachable_since or self._clock()
                if self._clock() - unreachable_since >= _LOSS_POLLS * config.poll_interval_sec:
                    return self._lost(call, attempt.runner, "the sandbox could not be read")
                status = None
            if status is not None:
                return status
            if exited is not None:
                self._save_logs(handle, Path(call.output_dir))
                detail = f"the runner exited with code {exited} without writing status.json"
                return _infra(
                    call, HOSTED_INFRA_FAILURE_RC, RUNNER_EXITED, detail, facts=_facts_of(attempt.runner, None)
                )
            if config.should_stop() or attempt.stop.is_set():
                self._save_logs(handle, Path(call.output_dir))
                return _refusal(call, STOPPED_RETURNCODE, STOPPED_REASON, "the run is stopping")
            silent = silence.silent_for(self._clock())
            if silent is not None:
                self._save_logs(handle, Path(call.output_dir))
                detail = f"no output for {silent:.0f}s before the server was ready"
                return _infra(
                    call, HOSTED_INFRA_FAILURE_RC, HUNG_BEFORE_READY, detail, facts=_facts_of(attempt.runner, None)
                )
            if self._timed_out(attempt):
                attempt.timeline.mark(RUNNER_DONE)
                self._save_logs(handle, Path(call.output_dir))
                evidence = status_evidence(handle, STATUS_PATH)
                ending = {"verdict": "timeout", "runner_status": evidence}
                self._record_end(attempt, HOSTED_INFRA_FAILURE_RC, None, "timeout", ending)
                attempt.timeline.mark(LANDED)
                self._pool.release(attempt.runner)
                raise subprocess.TimeoutExpired(cmd=f"hosted test {call.task_id}", timeout=call.timeout_sec)
            self._sleep(config.poll_interval_sec)

    def _timed_out(self, attempt: _Attempt) -> bool:
        call = attempt.call
        if self._clock() - attempt.started_unix >= call.timeout_sec:
            return True
        deadline = call.session_deadline_sec
        return deadline is not None and time.monotonic() >= deadline

    def _report_progress(self, handle: Sandbox, call: TestCall, last_output: float, silence: PreReadySilence) -> float:
        try:
            progress = _read_json(handle, PROGRESS_PATH)
        except (ValueError, KeyError):
            return last_output
        silence.observe(progress, self._clock())
        self._live.note(call.task_id, progress)
        latest = finite_number((progress or {}).get("last_output_unix")) or 0.0
        if latest > last_output and call.on_output is not None:
            call.on_output()
        return max(latest, last_output)

    def _lost(self, call: TestCall, runner: RunnerSandbox, detail: str) -> TestOutcome:
        return _infra(call, HOSTED_INFRA_FAILURE_RC, RUNNER_LOST, detail, facts=_facts_of(runner, None))

    def _runner_failed(self, attempt: _Attempt, status: Mapping[str, Any]) -> TestOutcome:
        call = attempt.call
        error_class = str(status.get("error_class") or "runner_failed")
        verdict = classify_test_failure(returncode=0, error_class=error_class)
        self._save_logs(_handle(attempt.runner), Path(call.output_dir))
        rc = HOSTED_INFRA_FAILURE_RC if verdict.disposition is Disposition.FAILED else HOSTED_SPEC_REJECTED_RC
        if verdict.disposition is Disposition.PROGRAM_ERROR and error_class != "spec_rejected":
            rc = HOSTED_RESULT_REJECTED_RC
        reported = runner_status(status)
        text = f"{error_class}: {reported['message']}"
        if rc == HOSTED_INFRA_FAILURE_RC:  # where it failed matters for the node; a refused sheet is the sheet's
            text = f"{text} ({where(*_place(attempt.runner))})"
        outcome = _outcome(call, rc, facts=_facts_of(attempt.runner, None), stderr=text, failure=verdict)
        self._finish_record(attempt, outcome, {"verdict": error_class, "runner_status": reported})
        return outcome

    # -- the result -----------------------------------------------------------------------------------------------

    def _accept(self, attempt: _Attempt) -> TestOutcome:
        """Read, verify, archive and land the bundle; any failure on the way is an outcome, never a half-accepted result."""
        call = attempt.call
        staging = self._staging_dir(attempt)
        try:
            verified = self._read_and_verify(attempt, staging)
            location = self._archive(attempt, verified, staging)
            self._land(attempt, verified)
            self._publish_jit(attempt, verified, staging)
            rounds = _rounds_of(verified, Path(call.output_dir))
        except ResultRejected as exc:
            outcome = _refusal(call, HOSTED_RESULT_REJECTED_RC, "result_rejected", str(exc))
            self._finish_record(attempt, outcome, {"verdict": "result_rejected", "reject": exc.reason.value})
            self._emit("hyperloom.measurement_finished", attempt, {"accepted": False})
            return outcome
        except (ArchiveError, MaterializeError, SandboxError, OSError, ValueError) as exc:
            error_class = "archive_failed" if isinstance(exc, ArchiveError) else "result_read_failed"
            outcome = _infra(call, HOSTED_INFRA_FAILURE_RC, error_class, f"{type(exc).__name__}: {exc}")
            self._finish_record(attempt, outcome, {"verdict": error_class})
            return outcome
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self._decided(attempt, verified, location, rounds)

    def _publish_jit(self, attempt: _Attempt, verified: VerifiedBundle, staging: Path) -> None:
        """Add what the accepted result's test compiled to the shared cache; a result never fails on it."""
        store, facts = self._config.jit_store, verified.result.jit_cache
        if store is None or facts is None:
            return
        try:
            scratch = staging / "jit"
            scratch.mkdir()
            publish_contributions(store, _handle(attempt.runner), facts, scratch)
        except OSError as exc:
            log.warning("jit cache: nothing was published (%s)", type(exc).__name__)

    def _staging_dir(self, attempt: _Attempt) -> Path:
        root = self._config.staging_root or Path(tempfile.gettempdir()) / "hyperloom-staging"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{attempt.request_id}-", dir=root))

    def _read_and_verify(self, attempt: _Attempt, staging: Path) -> VerifiedBundle:
        handle, runner = _handle(attempt.runner), attempt.runner
        public_key = str(runner.public_key_pem)
        manifest_raw = handle.read_file(MANIFEST_PATH)
        signature = handle.read_file(SIGNATURE_PATH)
        (staging / "manifest.json").write_bytes(manifest_raw)
        (staging / "manifest.sig").write_bytes(signature)
        listed = self._signed_file_list(manifest_raw, signature, public_key, attempt.spec)
        for entry in listed.files:
            target = staging / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            handle.copy_out(f"{FILES_ROOT}/{entry.path}", target)
        return bundle_mod.verify_bundle(
            staging,
            bundle_mod.BundleExpectation(
                public_key_pem=public_key,
                request_id=attempt.request_id,
                spec_fingerprint=self._expected_fingerprint(attempt.spec, listed.runner.hyperloom_version),
                spec_sha256=spec_sha256(attempt.spec_raw),
                sandbox_id=runner.sandbox_id,
                rule_pack_digest=self._config.rule_pack_digest,
            ),
            max_total_bytes=attempt.spec.outputs.max_total_bytes,
        )

    @staticmethod
    def _signed_file_list(
        manifest_raw: bytes, signature: bytes, public_key: str, spec: MeasurementSpec
    ) -> ResultManifest:
        """The file list of a manifest whose signature already verified; nothing is downloaded on an unsigned one."""
        try:
            valid = verify(manifest_raw, signature.decode("ascii"), public_key)
        except (SigningError, UnicodeDecodeError):
            raise ResultRejected(RejectReason.PUBLIC_KEY_UNUSABLE, "the signature or the key cannot be read") from None
        if not valid:
            raise ResultRejected(RejectReason.BAD_SIGNATURE, "the signature does not match the manifest bytes")
        try:
            manifest = ResultManifest.from_bytes(manifest_raw)
        except ValueError as exc:
            raise ResultRejected(RejectReason.MANIFEST_INVALID, str(exc)) from None
        if manifest.total_size() > spec.outputs.max_total_bytes:
            raise ResultRejected(RejectReason.TOO_LARGE, f"{manifest.total_size()} > {spec.outputs.max_total_bytes}")
        return manifest

    def _expected_fingerprint(self, spec: MeasurementSpec, claimed_version: str) -> str:
        """The fingerprint the sheet must have in the manifest.

        The runner's Hyperloom version only salts the fingerprint. When the run pins it (``runner_version``), the
        pinned value is used and a runner of another version fails the comparison; when it does not, the version the
        signed manifest names is used, and the content is still bound by the sheet's own bytes (``spec_sha256``).
        """
        return fingerprint(spec, hyperloom_version=self._config.runner_version or claimed_version)

    def _archive(self, attempt: _Attempt, verified: VerifiedBundle, staging: Path) -> str:
        return self._config.archive.put_bundle(attempt.request_id, verified.manifest_sha256, staging)

    def _land(self, attempt: _Attempt, verified: VerifiedBundle) -> None:
        """Move the verified files into the output directory; files already there, byte for byte, are a resumed landing."""
        output = Path(attempt.call.output_dir)
        try:
            bundle_mod.materialize(verified, output)
        except MaterializeError:
            if not _already_landed(verified, output):
                raise
        relative, external = split_result_dir(attempt.call)
        if external is not None and (output / relative).is_dir():
            shutil.copytree(output / relative, external, dirs_exist_ok=True)

    def _decided(
        self, attempt: _Attempt, verified: VerifiedBundle, location: str, rounds: tuple[RoundOutcome, ...]
    ) -> TestOutcome:
        call, result = attempt.call, verified.result
        out = result.outcome
        facts = _facts_of(attempt.runner, result)
        failure = _failure_of(out)
        outcome = TestOutcome(
            returncode=out.returncode,
            stdout=out.stdout_tail,
            stderr=out.stderr_tail,
            rounds=rounds,
            sandbox=facts,
            request_id=call.task_id,
            failure_class=failure.failure_class if failure.disposition is Disposition.FAILED else None,
            failure_reason=failure.failure_reason if failure.disposition is Disposition.FAILED else None,
            phases=result.timing.phases,
            notes=result.notes,
        )
        self._finish_record(
            attempt,
            outcome,
            {
                "verdict": "accepted",
                "verified": True,
                "manifest_sha256": verified.manifest_sha256,
                "signing_key_fingerprint": verified.signing_key_fingerprint,
                # What reverify needs later: the key the platform reported for this sandbox, as the verification
                # above used it, and the fingerprint the sheet had to carry. Neither is read from a live runner again.
                "public_key_pem": str(attempt.runner.public_key_pem),
                "spec_fingerprint": verified.result.spec_fingerprint,
                "archive": location,
            },
        )
        self._emit("hyperloom.measurement_finished", attempt, {"accepted": True, "returncode": out.returncode})
        if out.timed_out:
            raise MeasurementTimeout(f"hosted test {call.task_id}", call.timeout_sec, phases=outcome.phases)
        return outcome

    # -- verifying an accepted result again -----------------------------------------------------------------------

    def reverify(self, request_id: str, recorded: RecordedBundle, output_dir: Path) -> Reverified:
        """Verify the archived bundle of an accepted result again, from what was journalled at acceptance (L26).

        The key, the sandbox and the sheet come from the journal and the saved sheet, never from a live runner, so
        this works after the sandbox is gone. The bundle must verify for the same request, sheet and sandbox and be
        the one ``recorded`` names; its files are then landed in ``output_dir`` (a directory the caller owns and
        removes) so the rounds can be read the way acceptance read them.

        Raises :class:`ResultRejected` for any bundle or journal that does not support the result,
        :class:`~hyperloom.measure.archive.ArchiveError` or ``OSError`` when the archive cannot be read.
        """
        record = self._accepted_record(request_id, recorded.manifest_sha256)
        expected = self._expectation(record)
        bundle_dir = self._config.archive.bundle_path(request_id, recorded.manifest_sha256)
        verified = bundle_mod.reverify_record(bundle_dir, expected, recorded)
        try:
            bundle_mod.materialize(verified, Path(output_dir))
            spec = MeasurementSpec.from_bytes(self.journal.load_spec(request_id, record.attempt))
            rounds = _rounds_of(verified, Path(output_dir))
        except (MaterializeError, ValueError, KeyError) as exc:
            raise ResultRejected(RejectReason.NOT_VERIFIED, f"the archived bundle cannot be read: {exc}") from None
        return Reverified(verified.result, rounds, tuple(point.yaml for point in spec.config))

    def _accepted_record(self, request_id: str, manifest_sha256: str) -> Record:
        """The journal line that accepted this bundle; nothing else supports a result."""
        found = [
            record
            for record in self.journal.records()
            if record.request_id == request_id
            and record.state == STATE_FINISHED
            and record.extra.get("verified") is True
            and record.extra.get("manifest_sha256") == manifest_sha256
        ]
        if not found:
            raise ResultRejected(RejectReason.RECORD_MISMATCH, "the journal holds no acceptance of this bundle")
        return found[-1]

    def _expectation(self, record: Record) -> BundleExpectation:
        key, fingerprint = record.extra.get("public_key_pem"), record.extra.get("spec_fingerprint")
        if not isinstance(key, str) or not key:
            raise ResultRejected(RejectReason.PUBLIC_KEY_UNUSABLE, "no signing key was kept at acceptance")
        if not isinstance(fingerprint, str) or not fingerprint or not record.sandbox_id:
            raise ResultRejected(RejectReason.RECORD_MISMATCH, "the acceptance kept no sheet fingerprint or sandbox")
        sheet = self.journal.load_spec(record.request_id, record.attempt)
        return BundleExpectation(key, record.request_id, fingerprint, spec_sha256(sheet), record.sandbox_id)

    # -- journal and events -------------------------------------------------------------------------------------

    def _append(self, attempt: _Attempt, state: str, extra: Mapping[str, Any]) -> None:
        self.journal.append(
            Record(
                attempt.request_id,
                attempt.number,
                state,
                attempt.runner.sandbox_id,
                str(attempt.call.output_dir),
                attempt.call.task_id,
                extra,
            )
        )

    def _finish_record(self, attempt: _Attempt, outcome: TestOutcome, extra: Mapping[str, Any]) -> None:
        """The journal line of an attempt's end; a failure also keeps what was said about it (``detail``)."""
        failure_class = getattr(outcome.failure_class, "value", outcome.failure_class)
        said = {"detail": bounded(outcome.stderr)} if failure_class is not None and outcome.stderr else {}
        self._record_end(attempt, outcome.returncode, failure_class, outcome.failure_reason, {**said, **extra})

    def _record_end(
        self,
        attempt: _Attempt,
        returncode: int,
        failure_class: str | None,
        reason: str | None,
        extra: Mapping[str, Any],
    ) -> None:
        fields = {
            "returncode": returncode,
            "failure_class": failure_class,
            "failure_reason": reason,
            "finished_unix": self._clock(),
            "node": attempt.runner.node_name,
            **extra,
        }
        self._journal_end(attempt.request_id, attempt.number, attempt.runner.sandbox_id, attempt.call, fields)

    def _journal_end(
        self, request_id: str, number: int, sandbox_id: str | None, call: TestCall, fields: Mapping[str, Any]
    ) -> None:
        """One ``finished`` line for attempt ``number``; ``sandbox_id`` is ``None`` when no sandbox ran."""
        record = Record(request_id, number, STATE_FINISHED, sandbox_id, str(call.output_dir), call.task_id, fields)
        self.journal.append(record)

    def _mark_finished(self, previous: Record, extra: Mapping[str, Any]) -> None:
        self.journal.append(
            Record(
                previous.request_id,
                previous.attempt,
                STATE_FINISHED,
                previous.sandbox_id,
                previous.output_dir,
                previous.task_id,
                {"finished_unix": self._clock(), **extra},
            )
        )

    def _emit(self, event: str, attempt: _Attempt, data: Mapping[str, Any]) -> None:
        if self._events is None:
            return
        payload = {
            "request_id": attempt.request_id,
            "attempt": attempt.number,
            "sandbox_id": attempt.runner.sandbox_id,
            "node": attempt.runner.node_name,
            **data,
        }
        try:
            self._events.emit(event, payload)
        except Exception:
            log.exception("the events socket refused %s", event)

    @staticmethod
    def _save_logs(handle: Sandbox, output: Path) -> None:
        """Best effort: keep the logs the runner wrote, so a failed or timed-out test can be diagnosed."""
        try:
            names = [n for n in handle.list_dir(FILES_ROOT) if n.endswith(_LOG_SUFFIX)]
            output.mkdir(parents=True, exist_ok=True)
            for name in names:
                handle.copy_out(f"{FILES_ROOT}/{name}", output / f"hosted-{name}")
        except FileNotFoundError:
            log.debug("the runner wrote no files directory, so there are no logs to save")  # it failed before measuring
        except (SandboxError, OSError):
            log.warning("the runner's logs could not be saved", exc_info=True)


# -- module helpers ------------------------------------------------------------------------------------------------


def _blob_refs(spec: MeasurementSpec) -> set[str]:
    """Every ``sha256:<hex>`` blob the sheet refers to: attachments and the entries of every environment root."""
    refs = [attachment.blob for attachment in spec.attachments]
    refs.extend(entry.blob for root in spec.environment.roots for entry in root.entries if entry.blob)
    return set(refs)


def _handle(runner: RunnerSandbox) -> Sandbox:
    if runner.handle is None:
        raise SandboxGone(f"sandbox {runner.sandbox_id} has no handle")
    return runner.handle


@runtime_checkable
class _WatchesStartup(Protocol):
    """A sandbox that can tell when its template's startup command, the runner, has exited (the local one can)."""

    def startup_exit_code(self) -> int | None: ...


def _startup_exit(handle: Sandbox) -> int | None:
    """The runner's exit code once it has exited, where the sandbox can tell; else ``None``."""
    return handle.startup_exit_code() if isinstance(handle, _WatchesStartup) else None


def _read_status(handle: Sandbox) -> dict[str, Any] | None:
    """``status.json`` once the runner finished; a half-written or unreadable one counts as not there yet."""
    try:
        return _read_json(handle, STATUS_PATH)
    except ValueError:
        return None


def _read_json(handle: Sandbox, path: str) -> dict[str, Any] | None:
    try:
        raw = handle.read_file(path)
    except FileNotFoundError:
        return None
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON object")
    return document


def _place(runner: RunnerSandbox) -> tuple[str, str | None]:
    return runner.sandbox_id, runner.node_name


def _end_fields(outcome: TestOutcome, finished_unix: float, extra: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a ``finished`` line for ``outcome``."""
    return {
        "returncode": outcome.returncode,
        "failure_class": getattr(outcome.failure_class, "value", outcome.failure_class),
        "failure_reason": outcome.failure_reason,
        "finished_unix": finished_unix,
        **extra,
    }


def _meta_of(call: TestCall) -> dict[str, object]:
    meta = call.meta
    return {"priority": meta.priority, "purpose": meta.purpose, "branch_id": meta.branch_id}


def _facts_of(runner: RunnerSandbox, result: ResultManifest | None) -> SandboxFacts:
    facts = result.runner if result is not None else None
    occupancy = facts.card_occupancy if facts else None
    return SandboxFacts(
        sandbox_id=runner.sandbox_id,
        node_name=(facts.node_name if facts else None) or runner.node_name,
        gpu_model=facts.gpu_model if facts else None,
        gpu_count=len(facts.gpu_ids) if facts else 0,
        power_cap_w=facts.power_cap_w if facts else None,
        partition_mode=facts.partition_mode if facts else None,
        image_digest=facts.image_digest if facts else None,
        same_node=runner.same_node,
        gpu_ids=tuple(facts.gpu_ids) if facts else (),
        backend=BACKEND_NAME,
        card_occupancy=occupancy.state if occupancy else "",
        card_occupancy_reason=(occupancy.reason or "") if occupancy else "",
        card_occupancy_method=(occupancy.method or "") if occupancy else "",
    )


def _failure_of(out: Any) -> FailureClassification:
    """The class of a verified outcome: the runner's own verdict when it gave one, else the classification rules."""
    if out.error_class is not None:
        return classify_test_failure(returncode=out.returncode, error_class=out.error_class)
    if out.failure_class is not None:
        return FailureClassification(Disposition.FAILED, FailureClass(out.failure_class), out.failure_reason)
    if out.timed_out:
        return classify_test_failure(returncode=out.returncode, timed_out=True)
    return classify_test_failure(returncode=out.returncode, logs=[out.stderr_tail, out.stdout_tail])


def _rounds_of(verified: VerifiedBundle, output: Path) -> tuple[RoundOutcome, ...]:
    """The rounds of a landed result: from ``rounds.json`` when the runner listed it, else from the report layout.

    The layout alone only says which reports there are; a report in the warm-up directory is not a measuring round.
    """
    listed = {entry.path for entry in verified.result.files}
    returncode = verified.result.outcome.returncode
    if ROUNDS_FILE in listed:
        document = json.loads(verified.read_file(ROUNDS_FILE).decode("utf-8"))
        return tuple(
            RoundOutcome(
                int(r["point"]),
                r["phase"],
                output / r["dir"],
                int(r["returncode"]),
                tuple(AccuracyValue.from_document(v) for v in r.get("accuracy", ())),
                tuple(str(w) for w in r.get("nonfatal_warnings", ())),
            )
            for r in document
        )
    reports = sorted(
        path for path in listed if path.endswith("benchmark_report.json") and WARMUP_DIR not in Path(path).parts
    )
    return tuple(RoundOutcome(0, "measure", output / Path(path).parent, returncode) for path in reports)


def _already_landed(verified: VerifiedBundle, output: Path) -> bool:
    import hashlib

    for entry in verified.result.files:
        target = output / entry.path
        try:
            data = target.read_bytes()
        except OSError:
            return False
        if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.sha256:
            return False
    return True
