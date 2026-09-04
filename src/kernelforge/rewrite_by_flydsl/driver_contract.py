# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic verification of the dual-path measurement driver contract.

A rewrite driver must compare the source against the FlyDSL candidate on the
same cases, time the source alone under ``--ref-bench-mode``, and time the
candidate alone under ``--bench-mode``. An ordinary forge-loop driver has
neither bench mode, and since drivers conventionally ignore unknown arguments
it answers a bench request by silently running its correctness path — a
mismatch that, unchecked, only surfaces after PORT has spent its budget.

This module is the one place the driver is executed and the one place its
output is read, so every stage sees the same timing, case ids, and correctness
verdict, and every rejection names a failure class rather than an opaque error.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    calculate_mean_case_speedup,
    parse_case_timings,
)
from kernelforge.rewrite_by_flydsl import protocol
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

DRIVER_MISSING = "driver_missing"
DRIVER_NOT_INDEPENDENT = "driver_not_independent"
SOURCE_CANDIDATE_COLLISION = "source_candidate_collision"
REF_MODE_UNSUPPORTED = "ref_mode_unsupported"
REF_MODE_FAILED = "ref_mode_failed"
REF_MODE_TIMEOUT = "ref_mode_timeout"
REF_TIMING_UNPARSEABLE = "ref_timing_unparseable"
CANDIDATE_MODE_UNSUPPORTED = "candidate_mode_unsupported"
CANDIDATE_MODE_FAILED = "candidate_mode_failed"
CANDIDATE_MODE_TIMEOUT = "candidate_mode_timeout"
CANDIDATE_TIMING_UNPARSEABLE = "candidate_timing_unparseable"
CANDIDATE_NOT_ISOLATED = "candidate_not_isolated"
CANDIDATE_SHADOWED = "candidate_shadowed"
CASE_COVERAGE_MISMATCH = "case_coverage_mismatch"
MALFORMED_CASE_TIMINGS = "malformed_case_timings"

REF_BENCH_FLAG = "--ref-bench-mode"
BENCH_FLAG = "--bench-mode"

# The canonical aggregate timing key. ``mean_ms`` predates it and is still read,
# but a driver emitting it is reported so the spelling can be migrated.
CANONICAL_TIMING_METRIC = "median_ms"
DEPRECATED_TIMING_METRIC = "mean_ms"

_TIMING_RE = re.compile(r"\b(median_ms|mean_ms):\s*([-+\d.eE]+)")
_CASE_COMMENT_RE = re.compile(r"^[^\S\n]*#[^\S\n]*case[^\S\n]+([^\s:]+)[^\S\n]*:", re.M)
_SNR_RE = re.compile(r"SNR:\s*([-+\d.eE]+)\s*dB")
_ALLCLOSE_RE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
_REJECTED_ARGUMENT_RE = re.compile(
    r"unrecognized arguments|no such option|unknown option|invalid choice|"
    r"unexpected argument",
    re.IGNORECASE,
)

_OUTPUT_TAIL_CHARS = 1200


@dataclass
class DriverRun:
    """One driver invocation, reduced to what the contract checks look at."""

    returncode: int | None
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    @property
    def rejected_arguments(self) -> bool:
        """The driver's own parser refused a mode flag it does not define."""
        return bool(_REJECTED_ARGUMENT_RE.search(self.output))

    @property
    def tail(self) -> str:
        return self.output[-_OUTPUT_TAIL_CHARS:].strip()


@dataclass
class DriverReading:
    """Everything the contract reads out of one driver invocation's output.

    ``case_times`` keeps the value on each ``case_ms:`` line, not just the id.
    The aggregate is a sum over cases whose times can span orders of magnitude,
    so a ratio of two aggregates is dominated by the largest case; the equal
    weight per case that decides KEEP can only be computed from the per-case
    times, which is why they are retained rather than reduced to coverage.
    """

    timing_ms: float | None = None
    timing_metric: str = ""
    case_ids: tuple[str, ...] = ()
    # Every case the driver timed, including the ones it marked out of the
    # score. Deliberately unfiltered: the exclusion belongs to the comparison,
    # not to the reading. Dropping them here would make this side's case set
    # disagree with a side that kept them -- forge-loop's own best_case_times
    # keep them -- and an equal-set check would then refuse to score at all.
    case_times: dict[str, float] = field(default_factory=dict)
    # Which cases were marked out of the score, carried so the comparison can
    # exclude them the way forge-loop does: a case whose run-to-run spread
    # swamps a real change lets noise carry the verdict.
    unscored_cases: tuple[str, ...] = ()
    # Reported, never resolved. One case printed twice, or a case whose time is
    # not a number, means the suite that came back is not the one the task
    # declared, and keeping the first reading would score a subset.
    duplicate_case_ids: tuple[str, ...] = ()
    unparseable_case_ids: tuple[str, ...] = ()
    snr_db: float | None = None
    allclose: bool | None = None

    @property
    def has_timing(self) -> bool:
        return self.timing_ms is not None

    @property
    def has_correctness_verdict(self) -> bool:
        return self.snr_db is not None or self.allclose is not None


@dataclass
class PreflightReport:
    """Outcome of one contract stage, carrying an explicit failure class."""

    ok: bool
    failure_class: str = ""
    detail: str = ""
    timing_ms: float | None = None
    timing_metric: str = ""
    case_ids: tuple[str, ...] = ()
    case_times: dict[str, float] = field(default_factory=dict)
    unscored_cases: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)


def _failed(failure_class: str, detail: str) -> PreflightReport:
    return PreflightReport(ok=False, failure_class=failure_class, detail=detail)


def read_driver_output(text: str) -> DriverReading:
    """Parse the canonical timing, case ids, and correctness verdict."""
    reading = DriverReading()
    for metric, raw in _TIMING_RE.findall(text or ""):
        try:
            value = float(raw)
        except ValueError:
            continue
        # A canonical key always wins over the deprecated spelling.
        if reading.timing_ms is None or (
            metric == CANONICAL_TIMING_METRIC and reading.timing_metric != CANONICAL_TIMING_METRIC
        ):
            reading.timing_ms = value
            reading.timing_metric = metric

    timings = parse_case_timings(text)
    reading.case_times = dict(timings.case_times)
    reading.unscored_cases = tuple(timings.unscored)
    reading.duplicate_case_ids = tuple(timings.duplicates)
    reading.unparseable_case_ids = tuple(timings.unparseable)

    # A case whose time did not parse still declares coverage, so its id stays
    # here. Omitting it would let both paths report the same malformed case and
    # pass the coverage check on the subset that happened to parse.
    case_ids: list[str] = list(timings.case_times)
    for case_id in timings.unparseable:
        if case_id not in case_ids:
            case_ids.append(case_id)
    for case_id in _CASE_COMMENT_RE.findall(text or ""):
        if case_id not in case_ids:
            case_ids.append(case_id)
    reading.case_ids = tuple(case_ids)

    snr = _SNR_RE.search(text or "")
    if snr:
        try:
            reading.snr_db = float(snr.group(1))
        except ValueError:
            reading.snr_db = None
    allclose = _ALLCLOSE_RE.search(text or "")
    if allclose:
        reading.allclose = allclose.group(1).lower() == "true"
    return reading


# Why no per-case mean was produced. The two are not interchangeable: with no
# per-case timings at all the aggregate ratio is the only number available and
# using it is honest, while a coverage disagreement means the two paths measured
# different work and NO comparison should be published.
CASE_SCORE_UNAVAILABLE = "no_case_timings"
CASE_SCORE_INCOMPARABLE = "case_coverage_mismatch"


def cross_language_mean_case_speedup(
    source_case_times: dict[str, float],
    candidate_case_times: dict[str, float],
    unscored_cases: set[str] | tuple[str, ...] | list[str] | None = None,
) -> tuple[float | None, str]:
    """Score the FlyDSL candidate against the source with equal weight per case.

    Returns ``(speedup, reason)``; exactly one is set. This is the same
    statistic forge-loop decides KEEP and REVERT on, computed by the same
    function, so the number this pipeline reports for "did the rewrite help?"
    answers the question the loop was optimizing. Dividing the two aggregate
    timings instead answers a different one: the aggregate is a sum over cases
    whose times routinely span an order of magnitude or two, so it is dominated
    by the largest case, and a candidate that transforms the cheap cases while
    leaving the expensive one alone scores near 1.0 on it while scoring far
    above 1.0 on the mean.

    Both sides must pass their FULL case set, including cases marked out of the
    score, and name the exclusions in ``unscored_cases``. Pre-filtering one side
    makes the two sets unequal and the comparison is refused; filtering both
    without declaring it lets an excluded case's noise into the mean, which is
    the failure the marker exists to prevent.

    The two failure reasons are kept apart because they call for opposite
    handling. No per-case timings means the driver only reported an aggregate,
    and falling back to its ratio is the best available answer. A coverage
    mismatch means the source and the candidate were timed on different case
    sets, so their ratio -- aggregate or otherwise -- compares different work
    and must not be published at all.
    """
    if not source_case_times or not candidate_case_times:
        return None, CASE_SCORE_UNAVAILABLE
    try:
        speedup = calculate_mean_case_speedup(
            candidate_case_times,
            source_case_times,
            unscored_cases or (),
        )
    except CaseCoverageError:
        return None, CASE_SCORE_INCOMPARABLE
    if speedup is None:
        return None, CASE_SCORE_UNAVAILABLE
    return speedup, ""


SPEEDUP_BASIS_MEAN_CASE = "mean_case_speedup"
SPEEDUP_BASIS_AGGREGATE = "aggregate_ratio"


@dataclass(frozen=True)
class CrossLanguageScore:
    """How fast the rewrite is, and which statistic says so.

    ``basis`` is empty exactly when ``speedup`` is None, and ``reason`` then
    names why nothing could be published. Consumers gate on ``publishable``
    rather than on a number, so a score that could not be computed cannot be
    mistaken for one that could.
    """

    speedup: float | None = None
    basis: str = ""
    reason: str = ""

    @property
    def publishable(self) -> bool:
        return self.speedup is not None


def resolve_cross_language_score(
    *,
    source_case_times: dict[str, float] | None,
    candidate_case_times: dict[str, float] | None,
    unscored_cases: set[str] | tuple[str, ...] | list[str] | None = None,
    source_ms: float | None = None,
    candidate_ms: float | None = None,
) -> CrossLanguageScore:
    """The ONE place the aggregate ratio may stand in for the per-case mean.

    Every consumer that decides something on the rewrite's speed resolves it
    here: the result, the apply-back manifest, the submission gate and the KB
    record. Four sites each writing ``mean if mean is not None else aggregate``
    is how a path that must refuse ends up publishing instead -- the point of
    preferring the mean is lost the moment one of them quietly falls back.

    Three outcomes, and only the middle one substitutes a statistic:

      * per-case timings on both sides -> the equal-weight mean over cases
      * no per-case timings at all -> the aggregate ratio, labelled as such,
        because it is the only number the driver reported
      * the two sides timed different case sets -> NOTHING. Their aggregates
        compare the same mismatched work by another route, so substituting one
        publishes a comparison that does not exist.
    """
    mean, reason = cross_language_mean_case_speedup(
        dict(source_case_times or {}),
        dict(candidate_case_times or {}),
        unscored_cases,
    )
    if mean is not None:
        return CrossLanguageScore(speedup=mean, basis=SPEEDUP_BASIS_MEAN_CASE)
    if reason == CASE_SCORE_INCOMPARABLE:
        return CrossLanguageScore(reason=reason)
    if source_ms and candidate_ms and source_ms > 0 and candidate_ms > 0:
        return CrossLanguageScore(
            speedup=source_ms / candidate_ms,
            basis=SPEEDUP_BASIS_AGGREGATE,
        )
    return CrossLanguageScore(reason=reason or CASE_SCORE_UNAVAILABLE)


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the driver and anything it spawned."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (AttributeError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        proc.kill()


def export_driver_environment(spec: RewriteSpec) -> None:
    """Publish the producer-owned variables to every driver forge launches.

    The correctness suite and the nested loop's own bench and test tools spawn
    the driver with the ambient environment, so exporting once here is what
    makes the contract hold for those invocations too, not only the ones this
    module runs directly.
    """
    os.environ.update(
        protocol.driver_environment(
            source_kernel=spec.source_kernel,
            candidate_kernel=spec.flydsl_kernel,
            logical_op_name=spec.op_name,
        )
    )


def run_driver(
    spec: RewriteSpec,
    driver_path: str,
    mode_args: list[str],
    *,
    warmup: int | None = None,
    iters: int | None = None,
    timeout_sec: int,
) -> DriverRun:
    """Invoke the driver once with the producer-owned environment."""
    cmd = [sys.executable, str(driver_path), *mode_args]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if iters is not None:
        cmd += ["--iters", str(iters)]
    env = {
        **os.environ,
        **protocol.driver_environment(
            source_kernel=spec.source_kernel,
            candidate_kernel=spec.flydsl_kernel,
            logical_op_name=spec.op_name,
        ),
    }
    proc = subprocess.Popen(
        cmd,
        cwd=spec.workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=max(1, int(timeout_sec)))
    except subprocess.TimeoutExpired:
        _terminate(proc)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = ""
        return DriverRun(returncode=None, output=output or "", timed_out=True)
    return DriverRun(returncode=proc.returncode, output=output or "", timed_out=False)


def check_driver_independence(spec: RewriteSpec, driver_path: str) -> PreflightReport:
    """Reject a driver or candidate layout that cannot gate anything.

    The driver must be a file of its own: one that is the source kernel, the
    generated candidate, or a produced forge artifact would be judging itself. A
    candidate path equal to the source is the same defect one level down — the
    port would overwrite the kernel it is measured against.
    """
    driver = Path(driver_path)
    if not driver.is_file():
        return _failed(DRIVER_MISSING, f"measurement driver not found: {driver_path}")

    resolved_driver = driver.resolve()
    source = Path(spec.source_kernel).resolve()
    candidate = Path(spec.flydsl_kernel).resolve()
    if resolved_driver in (source, candidate):
        return _failed(
            DRIVER_NOT_INDEPENDENT,
            f"the measurement driver is the same file as the kernel it measures: {resolved_driver}",
        )
    if "forge_experiments" in resolved_driver.parts:
        return _failed(
            DRIVER_NOT_INDEPENDENT,
            "the measurement driver is a generated forge artifact and cannot own "
            f"the correctness gate: {resolved_driver}",
        )
    if source == candidate:
        return _failed(
            SOURCE_CANDIDATE_COLLISION,
            f"the FlyDSL candidate would overwrite the source kernel it is compared against: {candidate}",
        )

    # Python resolves the driver's own directory before anything the producer
    # exports, so a same-named module there would be imported instead of the
    # candidate — typically a kernel left behind by an earlier run.
    for directory in (resolved_driver.parent, Path(spec.workspace).resolve()):
        if directory == candidate.parent:
            continue
        shadow = directory / candidate.name
        if shadow.is_file():
            return _failed(
                CANDIDATE_SHADOWED,
                f"{shadow} would be imported instead of the FlyDSL candidate at "
                f"{candidate}; remove it so the driver measures this run's port",
            )
    return PreflightReport(ok=True)


def _timing_report(reading: DriverReading) -> PreflightReport:
    if reading.duplicate_case_ids:
        # bench.py refuses a measurement with duplicate case timings, so the
        # contract has to refuse it too; accepting here would let a suite the
        # loop rejects be scored through the rewrite's own path.
        return _failed(
            MALFORMED_CASE_TIMINGS,
            f"the driver reported more than one timing for {', '.join(reading.duplicate_case_ids)}",
        )
    if reading.unparseable_case_ids:
        return _failed(
            MALFORMED_CASE_TIMINGS,
            f"the driver reported an unparseable timing for {', '.join(reading.unparseable_case_ids)}",
        )
    report = PreflightReport(
        ok=True,
        timing_ms=reading.timing_ms,
        timing_metric=reading.timing_metric,
        case_ids=reading.case_ids,
        case_times=dict(reading.case_times),
        unscored_cases=reading.unscored_cases,
    )
    if reading.timing_metric == DEPRECATED_TIMING_METRIC:
        report.warnings.append(
            f"the driver reports {DEPRECATED_TIMING_METRIC}; the canonical "
            f"aggregate timing key is {CANONICAL_TIMING_METRIC}"
        )
    return report


def preflight_reference(
    spec: RewriteSpec,
    driver_path: str,
    *,
    warmup: int = 10,
    iters: int = 30,
    timeout_sec: int,
) -> PreflightReport:
    """Prove the source path is measurable before any PORT budget is spent.

    The returned timing is the speedup baseline, so the contract check and the
    baseline measurement are one driver invocation rather than two.
    """
    run = run_driver(
        spec,
        driver_path,
        [REF_BENCH_FLAG],
        warmup=warmup,
        iters=iters,
        timeout_sec=timeout_sec,
    )
    if run.timed_out:
        return _failed(
            REF_MODE_TIMEOUT,
            f"the driver did not finish {REF_BENCH_FLAG} within {timeout_sec}s",
        )
    if run.rejected_arguments:
        return _failed(
            REF_MODE_UNSUPPORTED,
            f"the driver does not accept {REF_BENCH_FLAG}: {run.tail}",
        )
    if not run.ok:
        return _failed(
            REF_MODE_FAILED,
            f"the driver failed in {REF_BENCH_FLAG} (exit {run.returncode}): {run.tail}",
        )

    reading = read_driver_output(run.output)
    if not reading.has_timing:
        # A driver that ignores the flag runs its correctness path instead, which
        # is a missing mode rather than a broken timing report.
        if reading.has_correctness_verdict:
            return _failed(
                REF_MODE_UNSUPPORTED,
                f"the driver ignored {REF_BENCH_FLAG} and ran its correctness path instead of timing the source",
            )
        return _failed(
            REF_TIMING_UNPARSEABLE,
            f"the driver reported no {CANONICAL_TIMING_METRIC} in {REF_BENCH_FLAG}: {run.tail}",
        )
    return _timing_report(reading)


def probe_candidate_arguments(
    spec: RewriteSpec,
    driver_path: str,
    *,
    timeout_sec: int,
) -> PreflightReport:
    """Check the candidate mode while the candidate is still an unbuilt stub.

    The driver must recognize ``--bench-mode`` here but must not produce a
    timing: the seeded skeleton cannot run, so a successful measurement proves
    the driver never reaches the candidate and is timing the source on both
    paths, which would make every later speedup meaningless.
    """
    run = run_driver(
        spec,
        driver_path,
        [BENCH_FLAG],
        warmup=1,
        iters=1,
        timeout_sec=timeout_sec,
    )
    if run.timed_out:
        return _failed(
            CANDIDATE_MODE_TIMEOUT,
            f"the driver did not finish {BENCH_FLAG} within {timeout_sec}s",
        )
    if run.rejected_arguments:
        return _failed(
            CANDIDATE_MODE_UNSUPPORTED,
            f"the driver does not accept {BENCH_FLAG}: {run.tail}",
        )

    reading = read_driver_output(run.output)
    if run.ok and reading.has_timing:
        return _failed(
            CANDIDATE_NOT_ISOLATED,
            f"the driver timed {BENCH_FLAG} at {reading.timing_ms} ms while the "
            "FlyDSL candidate is still an unimplemented skeleton, so it is not "
            "running the candidate",
        )
    # Any other outcome is the expected "candidate not ready".
    return PreflightReport(ok=True, case_ids=reading.case_ids)


def check_case_coverage(
    reference_case_ids: tuple[str, ...],
    candidate_case_ids: tuple[str, ...],
) -> PreflightReport:
    """Require both benchmark paths to report the same cases.

    The cases the driver reports while running are the authority on coverage;
    the task's shapes are agent context. Timing different case sets on the two
    paths turns the reported speedup into a comparison between different work.

    A reference reporting no cases carries no coverage claim, so it passes. A
    candidate reporting none is the mismatch this gate exists to catch: treating it
    as "nothing to compare" publishes a smaller workload's timing as a speedup.
    """
    if not reference_case_ids:
        return PreflightReport(ok=True, case_ids=candidate_case_ids)
    missing = sorted(set(reference_case_ids) - set(candidate_case_ids))
    unexpected = sorted(set(candidate_case_ids) - set(reference_case_ids))
    if missing or unexpected:
        return _failed(
            CASE_COVERAGE_MISMATCH,
            "the driver benchmarked different cases for the source and the "
            f"candidate (missing: {missing or 'none'}, "
            f"unexpected: {unexpected or 'none'})",
        )
    return PreflightReport(ok=True, case_ids=candidate_case_ids)


def preflight_candidate(
    spec: RewriteSpec,
    driver_path: str,
    *,
    reference_case_ids: tuple[str, ...] = (),
    warmup: int = 10,
    iters: int = 30,
    timeout_sec: int,
) -> PreflightReport:
    """Measure the ported candidate and prove it covered the reference cases."""
    run = run_driver(
        spec,
        driver_path,
        [BENCH_FLAG],
        warmup=warmup,
        iters=iters,
        timeout_sec=timeout_sec,
    )
    if run.timed_out:
        return _failed(
            CANDIDATE_MODE_TIMEOUT,
            f"the driver did not finish {BENCH_FLAG} within {timeout_sec}s",
        )
    if run.rejected_arguments:
        return _failed(
            CANDIDATE_MODE_UNSUPPORTED,
            f"the driver does not accept {BENCH_FLAG}: {run.tail}",
        )
    if not run.ok:
        return _failed(
            CANDIDATE_MODE_FAILED,
            f"the driver failed in {BENCH_FLAG} (exit {run.returncode}): {run.tail}",
        )

    reading = read_driver_output(run.output)
    if not reading.has_timing:
        if reading.has_correctness_verdict:
            return _failed(
                CANDIDATE_MODE_UNSUPPORTED,
                f"the driver ignored {BENCH_FLAG} and ran its correctness path instead of timing the candidate",
            )
        return _failed(
            CANDIDATE_TIMING_UNPARSEABLE,
            f"the driver reported no {CANONICAL_TIMING_METRIC} in {BENCH_FLAG}: {run.tail}",
        )

    coverage = check_case_coverage(reference_case_ids, reading.case_ids)
    if not coverage.ok:
        return coverage
    return _timing_report(reading)
