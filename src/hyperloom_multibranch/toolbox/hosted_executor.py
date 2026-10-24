# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The executor that runs one attempt of a test through the hosted measurement implementation (design 3.7, 4.4, 4.5).

``HostedExecutor.run`` turns the accepted request and what acceptance compiled into W6's ``TestCall``, runs it with
``HostedMeasurementBackend.run_test`` (in a worker thread) and turns the ``TestOutcome`` back into the attempt's
``TestResult``. It never retries: the same ``task_id`` again is the next attempt, with a fresh sandbox, and
:class:`~.attempts.AttemptRunner` decides when to ask for it.

A result is reported as succeeded only when the backend's journal says its bundle was verified, archived and was made by
the sandbox the outcome names. An end that is not a measurement (a rejected sheet or result) is a
:class:`~.attempts.MeasurementRefused`; a stop is a cancellation, so the test stays in the submit log and runs on the
next start.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common.outside_facts import ENV_CONC
from hyperloom.contracts.measurement import (
    CARD_FOREIGN,
    CARD_UNKNOWN,
    AccuracyValue,
    FailureClass,
    PhaseTiming,
    RoundValue,
    SandboxFacts,
    Purpose,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.measure.agentx_requests import read_round_classes
from hyperloom.measure.fake_gain import PREFIX_CACHE_OFF_ERROR_CLASS, baseline_prefix_cache_refusal
from hyperloom.measure.failures import Disposition, classify_test_failure
from hyperloom.measure.archive import ArchiveError
from hyperloom.measure.boot_watch import boot_seconds
from hyperloom.measure.bundle import RecordedBundle, ResultRejected
from hyperloom.measure.hosted import HostedMeasurementBackend, Reverified
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT, STATE_SUBMITTED, Record
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.phases import RESULT_READ, account
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.profile_traces import PROFILE_DIR, has_traces, is_trace_file
from hyperloom.measure.repro import REPRO_DIR, VERDICT_FILE, ReproError, read_verdict
from hyperloom.measure.socket import MeasurementTimeout, RoundOutcome, RoundPlan, TestCall, TestMeta, TestOutcome
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.acceptance import backend_kind
from hyperloom_multibranch.toolbox.attempts import (
    UNMEASURED_KINDS,
    MeasurementRefused,
    TestJob,
    carries_request_classes,
)
from hyperloom_multibranch.toolbox.reverify import (
    first_consistency_difference,
    first_hash_difference,
    first_request_classes_difference,
    first_value_difference,
    first_warning_difference,
)
from hyperloom_multibranch.toolbox.short_prompts import ReferenceStore, consistency_of, record_reference, rederive
from hyperloom_multibranch.toolbox.table_hits import read_table_hits, runtime_findings

__all__ = ["DEFAULT_TEST_TIMEOUT_SEC", "HostedExecutor", "OUTPUTS_DIR", "TEST_SILENCE_TIMEOUT_SEC"]

DEFAULT_TEST_TIMEOUT_SEC = 3600
#: How long a round may go without server-log growth or benchmark output once the server is ready before it is reaped
#: as hung. A serving benchmark writes ``server.log`` continuously, so ten quiet minutes (the user's choice) is a hang,
#: not a slow test.
TEST_SILENCE_TIMEOUT_SEC = 600.0
#: Where a test's landed files go, below the session directory: ``tests/<request_id>/attempt-<n>``.
OUTPUTS_DIR = "tests"
#: Where the outputs of finished short-prompt checks are kept, below the session directory.
REFERENCES_DIR = "short_prompt_refs"
_REPORT = "benchmark_report.json"
_MANIFEST = "manifest.json"
_UNFINISHED = (STATE_SUBMITTED, STATE_SENT)
_SWEEP = "conc_sweep"
_ACCURACY = "accuracy"
_MEASURE = "measure"
_UTF8 = "utf-8"
#: A profiled round, and a repro case's server, run once with no warmup: neither is a statistic.
_NO_WARMUP = RoundPlan(warmup_rounds=0)
_REASON_DETAIL_CHARS = 300
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")


class _Span(NamedTuple):
    """The instants an attempt was started and its backend returned, on the executor's clock."""

    started: float
    returned: float


class HostedExecutor:
    """Runs one attempt of a test through ``backend``.

    Args:
        backend: A started hosted backend; the executor never starts or closes it.
        session_dir: The session directory; test output lands below ``<session_dir>/tests`` and a result's
            ``output_dir`` is relative to it.
        timeout_sec: The timeout of one attempt.
        magpie_python: The interpreter the measurement runs with (``TestCall.magpie_python``).
        workload_kind: The campaign's workload; an AgentX baseline served with the prefix cache off does not end
            ``succeeded``.
    """

    def __init__(
        self,
        backend: HostedMeasurementBackend,
        session_dir: Path,
        *,
        timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC,
        magpie_python: str = sys.executable,
        workload_kind: WorkloadKind = WorkloadKind.INFERENCEX,
    ) -> None:
        self._backend = backend
        self._session = Path(session_dir)
        self._timeout = timeout_sec
        self._python = magpie_python
        self._clock: Callable[[], float] = time.time
        self._workload = workload_kind
        self._references = ReferenceStore(self._session / REFERENCES_DIR)

    async def run(self, job: TestJob) -> TestResult:
        request = job.request
        request_id = request.request_id or ""
        accepted = job.environment
        if not isinstance(accepted, AcceptedMeasurement):
            raise MeasurementRefused(
                f"the hosted executor needs the accepted measurement, got {type(accepted).__name__}"
            )
        call = self._call(job, accepted, self._output_dir(request_id))
        started = self._clock()
        try:
            outcome = await asyncio.to_thread(self._backend.run_test, call)
        except subprocess.TimeoutExpired as expired:
            return self._timed_out(request_id, started, expired)
        result = self._result(request, accepted, call, outcome, _Span(started, self._clock()))
        result = await self._with_table_hits(result, accepted, call)
        result = await self._with_short_prompts(result, accepted, call)
        return await self._with_request_classes(result, request, outcome, call)

    def withdraw(self, request_id: str) -> None:
        """End the running attempt of ``request_id``: its sandbox is closed and its cards go back to the pool."""
        self._backend.withdraw(request_id)

    # -- verifying a recorded result again -------------------------------------------------------------------------

    def difference(self, result: TestResult) -> str | None:
        """Why ``result`` does not follow from its archived bundle, or ``None`` when it does (blocking).

        The bundle is verified again from what acceptance journalled, its rounds and accuracy are derived again, and
        every recorded field is compared. Raises what the backend raises for a bundle that does not verify.
        """
        if not result.manifest_sha256 or not result.signing_key_fingerprint:
            return "manifest_sha256: the record names no bundle"
        recorded = RecordedBundle(result.manifest_sha256, result.signing_key_fingerprint)
        with tempfile.TemporaryDirectory(prefix="reverify-") as scratch:
            landed = Path(scratch) / "bundle"
            derived = self._backend.reverify(result.request_id, recorded, landed)
            return (
                _unlike_bundle(result, derived, landed)
                or _unlike_classes(result, derived, landed)
                or _unlike_consistency(result, landed)
            )

    async def reverify(self, result: TestResult) -> None:
        """Raise :class:`ResultUnverified` naming why ``result`` does not follow from its archived bundle, or why the
        bundle cannot be checked (a rejected bundle, an archive that cannot be read)."""
        request_id = result.request_id
        try:
            difference = await asyncio.to_thread(self.difference, result)
        except ResultRejected as exc:
            raise ResultUnverified(
                request_id, f"the bundle was rejected ({exc.reason.value}): {_one_line(exc.detail)}"
            ) from None
        except ArchiveError as exc:
            raise ResultUnverified(request_id, f"the archive cannot be read: {_one_line(str(exc))}") from None
        except OSError as exc:
            raise ResultUnverified(request_id, f"the archive cannot be read: {type(exc).__name__}") from None
        if difference:
            raise ResultUnverified(request_id, difference)

    # -- the call --------------------------------------------------------------------------------------------------

    def _output_dir(self, request_id: str) -> Path:
        """Where this attempt lands: the directory of an unfinished attempt (a resume), else a new one."""
        journal = self._backend.journal
        previous = journal.latest(request_id)
        if previous is not None and previous.state in _UNFINISHED and previous.output_dir:
            return Path(previous.output_dir)
        return self._session / OUTPUTS_DIR / request_id / f"attempt-{journal.next_attempt(request_id)}"

    def _call(self, job: TestJob, accepted: AcceptedMeasurement, output_dir: Path) -> TestCall:
        request = job.request
        return TestCall(
            points=accepted.points,
            environment=accepted.environment,  # type: ignore[arg-type]  # the sheet reads its overlay
            output_dir=output_dir,
            timeout_sec=self._timeout,
            silence_timeout_sec=TEST_SILENCE_TIMEOUT_SEC,
            cwd=str(self._session),
            magpie_python=self._python,
            task_id=request.request_id or "",
            kind=backend_kind(request.kind),  # type: ignore[arg-type]
            prep=_SWEEP if request.kind is TestKind.SWEEP or len(accepted.points) > 1 else "grid",
            accuracy=request.accuracy,
            rounds=_NO_WARMUP if request.kind in UNMEASURED_KINDS else RoundPlan.from_env(),
            meta=TestMeta(
                purpose=request.purpose.value,
                priority=job.priority,
                branch_id=request.branch_id,
                parent_commit=request.parent_commit,
                retest_of=request.retest_of,
            ),
            avoid_nodes=tuple(job.avoid_nodes),
            attachments=request.attachments if request.kind is TestKind.REPRO else (),
            short_prompts=accepted.short_prompts,
        )

    # -- the outcome -----------------------------------------------------------------------------------------------

    def _result(
        self, request: TestRequest, accepted: AcceptedMeasurement, call: TestCall, outcome: TestOutcome, span: _Span
    ) -> TestResult:
        request_id = request.request_id or ""
        failure = _failure_class_of(outcome)
        if failure is None:
            _refuse_unless_measured(outcome)
        facts = _facts_of(outcome)
        record = self._record_of(request_id, outcome)
        audit = self._audit(request_id, record, call) if record is not None else {}
        if failure is None and not audit:
            raise MeasurementRefused("the result is not recorded as a verified bundle of this sandbox")
        refusal = self._cacheless_baseline(request, accepted, outcome) if failure is None else ""
        if refusal:
            failure = FailureClass.CANDIDATE
        finished = self._clock()
        rounds, accuracy = _measured_values(request, accepted, outcome) if failure is None else ((), ())
        location = self._profile_location(call) if failure is None and request.kind is TestKind.PROFILE else ""
        verdict = _landed_verdict(call) if failure is None and request.kind is TestKind.REPRO else ""
        started = span.started
        gpu_seconds = max(0.0, finished - started) * facts.gpu_count
        return TestResult(
            request_id=request_id,
            validity=_validity_of(failure),
            failure_class=failure,
            failure_reason=refusal or _failure_reason(outcome, failure),
            rounds=rounds,
            accuracy=accuracy,
            sandbox=facts,
            output_dir=_relative(call.output_dir, self._session),
            started_at=started,
            finished_at=finished,
            gpu_seconds=gpu_seconds,
            profile_location=location,
            repro_verdict=verdict,
            nonfatal_warnings=(
                *_warnings_of(outcome.rounds),
                *_card_warnings(facts.card_occupancy, facts.card_occupancy_reason),
            )
            if failure is None
            else (),
            cold_start_seconds=boot_seconds(call.output_dir),
            **_phase_fields(outcome.phases, span, finished, facts.gpu_count),
            **audit,
        )

    async def _with_table_hits(self, result: TestResult, accepted: AcceptedMeasurement, call: TestCall) -> TestResult:
        """``result`` with how much of each kernel table its recipe loads the test's server hit; a failed test has none.

        The logs are read in a worker thread: a server log of a long test is large.
        """
        if result.validity is not Validity.SUCCEEDED:
            return result
        hits = await asyncio.to_thread(read_table_hits, accepted.environment, accepted.points, Path(call.output_dir))
        return replace(result, table_hits=hits, runtime_findings=runtime_findings(hits))

    async def _with_short_prompts(
        self, result: TestResult, accepted: AcceptedMeasurement, call: TestCall
    ) -> TestResult:
        """``result`` with the short-prompt consistency check the test ran, read from its landed outputs and compared with
        the parent's; a test that asked for none, or did not succeed, carries none.

        A check that completed is kept as the reference of the tests whose parent is this test's recipe.
        """
        plan = accepted.short_prompts
        if plan is None or result.validity is not Validity.SUCCEEDED:
            return result
        check = await asyncio.to_thread(consistency_of, plan, Path(call.output_dir), self._references)
        await asyncio.to_thread(
            record_reference, plan, check, Path(call.output_dir), self._references, result.request_id
        )
        return replace(result, short_prompt_consistency=check)

    async def _with_request_classes(
        self, result: TestResult, request: TestRequest, outcome: TestOutcome, call: TestCall
    ) -> TestResult:
        """``result`` with, for each measured round of an AgentX test, how many of its requests the client marked failed
        by class and where the round's per-request export is; any other result has none.

        The exports are read in a worker thread: one holds a line for every request of a round.
        """
        if not carries_request_classes(self._workload, request, result.validity):
            return result
        concurrencies = _concurrencies_of(call.points)
        classes = await asyncio.to_thread(read_round_classes, outcome.rounds, Path(call.output_dir), concurrencies)
        return replace(result, request_classes=classes)

    def _profile_location(self, call: TestCall) -> str:
        """Where the landed traces of a profile test are, relative to the session; a profile with none is no result."""
        directory = Path(call.output_dir) / PROFILE_DIR
        location = _relative(directory, self._session)
        if not location or not has_traces(directory):
            raise MeasurementRefused("the verified profile result holds no trace file")
        return location

    def _cacheless_baseline(self, request: TestRequest, accepted: AcceptedMeasurement, outcome: TestOutcome) -> str:
        """Why an AgentX baseline measured with the prefix cache off is refused, or ``""`` (main's #1884).

        Every candidate is graded against the baseline, so a cold-cache one would credit the cache itself as a gain.
        """
        if self._workload is not WorkloadKind.AGENTX or request.purpose is not Purpose.BASELINE:
            return ""
        for item in outcome.rounds:
            if item.phase != _MEASURE or item.point >= len(accepted.points):
                continue
            reason = baseline_prefix_cache_refusal(accepted.points[item.point], Path(item.output_dir))
            if reason:
                return f"{PREFIX_CACHE_OFF_ERROR_CLASS}: {reason}"
        return ""

    def _timed_out(self, request_id: str, started: float, expired: subprocess.TimeoutExpired) -> TestResult:
        """A timeout is an unclear failure (4.5): the sandbox is read from the journal, the node is not known."""
        record = self._backend.journal.latest(request_id)
        verdict = classify_test_failure(returncode=0, timed_out=True)
        sandbox = record.sandbox_id if record is not None and record.sandbox_id else f"hosted-{request_id}"
        finished = self._clock()
        phases = expired.phases if isinstance(expired, MeasurementTimeout) else ()
        return TestResult(
            request_id=request_id,
            validity=Validity.CANDIDATE_FAILED,
            failure_class=verdict.failure_class,
            failure_reason=verdict.failure_reason,
            sandbox=SandboxFacts(sandbox_id=sandbox, backend=self._backend.name),
            started_at=started,
            finished_at=finished,
            **_phase_fields(phases, _Span(started, finished), finished, 0),
        )

    def _record_of(self, request_id: str, outcome: TestOutcome) -> Record | None:
        record = self._backend.journal.latest(request_id)
        if record is None or record.state != STATE_FINISHED:
            return None
        return record if record.sandbox_id == outcome.sandbox.sandbox_id else None

    def _audit(self, request_id: str, record: Record, call: TestCall) -> dict[str, Any]:
        """What the journal says about a verified, archived bundle; empty when it does not say that."""
        extra = record.extra
        manifest_sha = extra.get("manifest_sha256")
        if extra.get("verified") is not True or not manifest_sha or Path(record.output_dir) != Path(call.output_dir):
            return {}
        fields: dict[str, Any] = {
            "manifest_sha256": manifest_sha,
            "signing_key_fingerprint": extra.get("signing_key_fingerprint"),
        }
        manifest = _archived_manifest(str(extra.get("archive") or ""), manifest_sha)
        if manifest is not None:
            fields.update(spec_fingerprint=manifest.spec_fingerprint, surface_sha256=manifest.surface_sha256)
        return fields


def _phase_fields(phases: tuple[PhaseTiming, ...], span: _Span, finished: float, gpus: int) -> dict[str, Any]:
    """The ``TestResult`` fields that say where an attempt's wall time went: the backend's phases, the executor's own
    reading of the result, and the seconds left over; none for a backend that stamped no phases."""
    if not phases:
        return {}
    read = (PhaseTiming(RESULT_READ, span.returned, finished - span.returned),) if finished > span.returned else ()
    accounted = account((*phases, *read), finished - span.started, gpus)
    return {"phases": accounted.phases, "unaccounted_seconds": accounted.unaccounted_seconds}


def _one_line(text: str) -> str:
    return _CONTROL_CHARS.sub(" ", text.strip())[:_REASON_DETAIL_CHARS]


def _unlike_bundle(result: TestResult, derived: Reverified, landed: Path) -> str | None:
    """The first field of ``result`` that the verified bundle does not support.

    The warnings are checked on every path: a result that did not succeed records none, any other lists those of the
    bundle's rounds and the one its signed card occupancy asks for, whatever kind of test it was.
    """
    difference = _unlike_values(result, derived, landed)
    if difference:
        return difference
    if result.validity is not Validity.SUCCEEDED:
        return first_warning_difference(result, ())
    occupancy = derived.manifest.runner.card_occupancy
    card = _card_warnings(occupancy.state, occupancy.reason) if occupancy else ()
    return first_warning_difference(result, (*_warnings_of(derived.rounds), *card))


def _unlike_classes(result: TestResult, derived: Reverified, landed: Path) -> str | None:
    """The first field of the request classes ``result`` records that the verified bundle's exports do not support.

    A record that holds none, as every one written before the classes existed, is not checked: it makes no claim.
    """
    if not result.request_classes:
        return None
    concurrencies = tuple(_concurrency_in(text) for text in derived.point_yaml)
    return first_request_classes_difference(result, read_round_classes(derived.rounds, landed, concurrencies))


def _unlike_consistency(result: TestResult, landed: Path) -> str | None:
    """The first field of the short-prompt check ``result`` records that the verified bundle's outputs do not support.

    A record that holds none is not checked: it makes no claim.
    """
    recorded = result.short_prompt_consistency
    return None if recorded is None else first_consistency_difference(result, rederive(recorded, landed))


def _unlike_values(result: TestResult, derived: Reverified, landed: Path) -> str | None:
    difference = first_hash_difference(result, derived.manifest)
    if difference or result.validity is not Validity.SUCCEEDED:
        return difference or _unmeasured_difference(result)
    if derived.manifest.outcome.returncode != 0:
        return (
            f"validity: recorded succeeded, the bundle's outcome has return code {derived.manifest.outcome.returncode}"
        )
    if result.profile_location:
        return _profile_difference(result, derived.manifest)
    if result.repro_verdict or _lists_verdict(derived.manifest):
        return _repro_difference(result, derived.manifest, landed)
    values = _round_values_at(derived.rounds, tuple(_concurrency_in(text) for text in derived.point_yaml))
    return first_value_difference(result, values, _scores_of(derived.rounds))


def _profile_difference(result: TestResult, manifest: ResultManifest) -> str | None:
    """A profile result holds no measured value and names the traces the signed manifest lists."""
    if result.rounds or result.accuracy:
        return "rounds: a profile result records no measured value"
    expected = f"{result.output_dir}/{PROFILE_DIR}"
    if result.profile_location != expected:
        return f"profile_location: recorded {result.profile_location!r}, the bundle's traces are at {expected!r}"
    if not any(f.path.startswith(f"{PROFILE_DIR}/") and is_trace_file(f.path) for f in manifest.files):
        return "profile_location: the bundle lists no trace file"
    return None


def _lists_verdict(manifest: ResultManifest) -> bool:
    return any(f.path == f"{REPRO_DIR}/{VERDICT_FILE}" for f in manifest.files)


def _repro_difference(result: TestResult, manifest: ResultManifest, landed: Path) -> str | None:
    """A repro result holds no measured value and says what the signed verdict file says, no more and no less."""
    if result.rounds or result.accuracy:
        return "rounds: a repro result records no measured value"
    if not _lists_verdict(manifest):
        return "repro_verdict: the bundle lists no verdict file"
    try:
        signed = read_verdict(landed)
    except ReproError as exc:
        return f"repro_verdict: {exc}"
    if result.repro_verdict != signed:
        return f"repro_verdict: recorded {result.repro_verdict!r}, the bundle says {signed!r}"
    return None


def _unmeasured_difference(result: TestResult) -> str | None:
    """A result that is not a success holds no measured value; one that does was not made by the runner."""
    if result.rounds:
        return "rounds: a result that did not succeed records measured rounds"
    return "accuracy: a result that did not succeed records scores" if result.accuracy else None


# -- mapping helpers -----------------------------------------------------------------------------------------------


def _failure_class_of(outcome: TestOutcome) -> FailureClass | None:
    """The outcome's failure class, ``None`` for a success; an end that is neither raises."""
    if outcome.failure_class is not None:
        return FailureClass(getattr(outcome.failure_class, "value", outcome.failure_class))
    return None


def _measured_values(
    request: TestRequest, accepted: AcceptedMeasurement, outcome: TestOutcome
) -> tuple[tuple[RoundValue, ...], tuple[AccuracyValue, ...]]:
    """The rounds and accuracy a successful test reports.

    A profile test reports none: its one round ran with the profiler on, so its numbers are not a measurement and the
    rule gate, which reads measure-phase rounds, finds nothing to compare. A repro test measures nothing either: it
    reports a verdict. Every other kind must hold a readable measured round.
    """
    if request.kind in UNMEASURED_KINDS:
        return (), ()
    rounds = _round_values(outcome.rounds, accepted.points)
    if not any(r.phase == _MEASURE for r in rounds):
        raise MeasurementRefused("the verified result holds no readable measured round")
    return rounds, _accuracy_values(outcome.rounds, wanted=request.accuracy)


def _landed_verdict(call: TestCall) -> str:
    """The verdict of a repro test, read from the files its verified bundle landed; none is no result."""
    try:
        return read_verdict(Path(call.output_dir))
    except ReproError as exc:
        raise MeasurementRefused(f"the verified repro result holds no verdict: {exc}") from None


def _refuse_unless_measured(outcome: TestOutcome) -> None:
    """An outcome with no failure class is a success only at return code 0; any other end is not a measurement."""
    if outcome.returncode == 0:
        return
    verdict = classify_test_failure(returncode=outcome.returncode)
    if verdict.disposition is Disposition.STOPPED:
        raise asyncio.CancelledError(verdict.failure_reason)  # the run is stopping: the test is picked up at restart
    detail = (outcome.stderr or "").strip()[-300:]
    raise MeasurementRefused(f"{verdict.failure_reason or 'unclassified end'}: {detail}".strip(": "))


def _failure_reason(outcome: TestOutcome, failure: FailureClass | None) -> str | None:
    """The reason of a failed outcome; an infrastructure failure also says what the runner reported.

    Without the runner's own message a runner that could not start is only ``runner_start_failed``, and nobody can tell
    a missing tool checkout from a broken interpreter.
    """
    if failure is None:
        return None
    reason = outcome.failure_reason
    detail = _CONTROL_CHARS.sub(" ", (outcome.stderr or "").strip()[-_REASON_DETAIL_CHARS:]).strip()
    if failure is not FailureClass.INFRA or not detail or detail == reason:
        return reason
    return detail if reason and detail.startswith(reason) else f"{reason}: {detail}"


def _validity_of(failure: FailureClass | None) -> Validity:
    if failure is None:
        return Validity.SUCCEEDED
    return Validity.MEASUREMENT_FAILED if failure is FailureClass.INFRA else Validity.CANDIDATE_FAILED


def _facts_of(outcome: TestOutcome) -> SandboxFacts:
    return replace(outcome.sandbox, backend=outcome.sandbox.backend or "hosted")


def _relative(path: Path, session: Path) -> str:
    try:
        return Path(path).relative_to(session).as_posix()
    except ValueError:
        return ""


def _archived_manifest(location: str, manifest_sha256: str) -> ResultManifest | None:
    """The manifest in the archive, taken only when its bytes hash to the verified ``manifest_sha256``."""
    if not location:
        return None
    try:
        raw = (Path(location) / _MANIFEST).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            return None
        return ResultManifest.from_bytes(raw)
    except (OSError, ValueError):
        return None


def _concurrency_of(point: Path) -> int | None:
    try:
        return _concurrency_in(Path(point).read_text(encoding=_UTF8))
    except OSError:
        return None


def _concurrencies_of(points: tuple[Path, ...]) -> tuple[int | None, ...]:
    return tuple(_concurrency_of(point) for point in points)


def _concurrency_in(text: str) -> int | None:
    try:
        document = yaml.safe_load(text) or {}
        value = int(str(document["benchmark"]["envs"][ENV_CONC]))
    except (ValueError, KeyError, TypeError, yaml.YAMLError):
        return None
    return value if value >= 1 else None


def _report_of(directory: Path) -> dict[str, Any] | None:
    for path in sorted(Path(directory).rglob(_REPORT)) if Path(directory).is_dir() else []:
        try:
            document = json.loads(path.read_text(encoding=_UTF8))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict):
            return document
    return None


def _metrics_of(measurement: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in measurement.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }


def _warnings_of(rounds: tuple[RoundOutcome, ...]) -> tuple[str, ...]:
    """The warnings ``run_grid`` wrote for the rounds of this test, in round order, as written."""
    return tuple(warning for r in rounds for warning in r.nonfatal_warnings)


def _card_warnings(state: str, reason: str | None) -> tuple[str, ...]:
    """The warning of a result whose cards were not shown to be the test's alone: ``foreign`` is a result that was kept
    although a process outside the test held its cards, which only a runner whose card guard observes signs (one that
    enforces refuses it), and ``unknown`` is one nobody could look at the cards of, in either mode. The same text is derived
    again from the signed manifest when the result is verified again."""
    if state == CARD_FOREIGN:
        return (f"foreign_process_on_card, kept because the card guard only observes: {reason}",)
    if state == CARD_UNKNOWN:
        return (f"card_occupancy_unknown, the test's cards were not shown to be free of other processes: {reason}",)
    return ()


def _scores_of(rounds: tuple[RoundOutcome, ...]) -> tuple[AccuracyValue, ...]:
    return tuple(value for r in rounds if r.phase == _ACCURACY for value in r.accuracy)


def _accuracy_values(rounds: tuple[RoundOutcome, ...], *, wanted: bool) -> tuple[AccuracyValue, ...]:
    """The scores of the accuracy rounds, in round order.

    A request that asked for accuracy and got no round, or a round with no score, is not a result: a test that
    "succeeds" without the number it was run for would be read as a pass of the accuracy gate.
    """
    evaluated = [r for r in rounds if r.phase == _ACCURACY]
    if not wanted:
        return ()
    if not evaluated:
        raise MeasurementRefused("the request asked for accuracy but the verified result has no accuracy round")
    if any(not r.accuracy for r in evaluated):
        raise MeasurementRefused("the verified result's accuracy round holds no accuracy score")
    return tuple(value for r in evaluated for value in r.accuracy)


def _round_values(rounds: tuple[RoundOutcome, ...], points: tuple[Path, ...]) -> tuple[RoundValue, ...]:
    """One value per warmup or measured round whose report is readable and valid; the metric names are the report's."""
    return _round_values_at(rounds, _concurrencies_of(points))


def _round_values_at(rounds: tuple[RoundOutcome, ...], concurrencies: tuple[int | None, ...]) -> tuple[RoundValue, ...]:
    """:func:`_round_values` for a sheet whose points' concurrencies are already known, in order."""
    values = []
    for item in rounds:
        if item.phase not in ("warmup", _MEASURE):
            continue
        report = _report_of(item.output_dir)
        if report is None:
            continue
        measurement = extract_benchmark_measurement(report, workspace=Path(item.output_dir))
        if not measurement.get("valid_measurement"):
            continue
        completed = measurement.get("completed_requests")
        requested = measurement.get("requested_requests")
        failed = requested - completed if isinstance(requested, int) and isinstance(completed, int) else None
        values.append(
            RoundValue(
                item.phase,
                concurrencies[item.point] if item.point < len(concurrencies) else None,
                _metrics_of(measurement),
                completed_requests=completed if isinstance(completed, int) else None,
                failed_requests=max(0, failed) if failed is not None else None,
            )
        )
    return tuple(values)
