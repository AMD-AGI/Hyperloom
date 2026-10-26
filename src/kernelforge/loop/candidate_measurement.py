# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build, validate, benchmark and score one candidate in the workspace it sits in."""

from __future__ import annotations

import asyncio
import textwrap
import time
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.evaluation.benchmark.runner import (
    BenchRun,
    CaseCoverageError,
    calculate_measurement_case_speedups,
    measure_wallclock,
)
from kernelforge.evaluation.correctness.canonical import accept_candidate
from kernelforge.evaluation.correctness.pipeline import ValidationReport, run_validation_pipeline
from kernelforge.evaluation.profiling.registers import check_registers
from kernelforge.evaluation.scoring import (
    KEEP_MEASUREMENT_COUNT,
    SIGMA_REMEASURE_BATCH,
    SIGMA_REMEASURE_MAX_ROUNDS,
    SigmaAttribution,
    attribute_sigma,
    beats_current_best,
    keep_score,
    measurement_sigma,
    passes_keep_threshold,
    required_keep_speedup,
    rescaled_sigma,
    runs_task_suite_acceptance,
)
from kernelforge.loop.iteration_result import IterationResult
from kernelforge.loop.jit_rebuild import force_jit_rebuild, tracked_source_changes
from kernelforge.tools.process_group import communicate_process_group


@dataclass(frozen=True)
class MeasurementSite:
    """The workspace one candidate is built, validated and benchmarked in, with its paths inside it."""

    workspace_dir: str
    driver_script: str
    build_dir: str | None
    declared_sources: tuple[str, ...]

    def rebased(self, workspace_dir: str) -> "MeasurementSite":
        """The same task paths inside another checkout of the same repository."""
        return MeasurementSite(
            workspace_dir=str(workspace_dir),
            driver_script=self._rebase(self.driver_script, workspace_dir),
            build_dir=self._rebase(self.build_dir, workspace_dir) if self.build_dir else None,
            declared_sources=tuple(self._rebase(path, workspace_dir) for path in self.declared_sources),
        )

    def jit_sources(self) -> list[str]:
        """Declared hints plus actual tracked edits that may require recompilation."""
        return list(dict.fromkeys([*self.declared_sources, *tracked_source_changes(self.workspace_dir)]))

    def _rebase(self, path: str, workspace_dir: str) -> str:
        root = Path(self.workspace_dir).resolve()
        candidate = Path(path)
        absolute = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            # A path outside the workspace (a system driver, a shared build tree) is the same file from every checkout.
            return str(path)
        return str(Path(workspace_dir) / relative)


def measurement_case_times(
    bench_detail: dict | None,
) -> dict[str, tuple[float, ...]]:
    """Per-case times from each independent measurement of one bench."""
    measurements = (bench_detail or {}).get("measurements")
    if not isinstance(measurements, list):
        return {}
    per_case: dict[str, list[float]] = {}
    for measurement in measurements:
        if not isinstance(measurement, dict):
            continue
        for case_id, value in (measurement.get("case_times") or {}).items():
            try:
                time_ms = float(value)
            except (TypeError, ValueError):
                continue
            if time_ms > 0:
                per_case.setdefault(str(case_id), []).append(time_ms)
    return {case_id: tuple(times) for case_id, times in per_case.items()}


@dataclass(frozen=True)
class SigmaResolution:
    """The sigma the KEEP bar was charged to, and how it was arrived at."""

    sigma: float | None
    measured_sigma: float | None
    dominant_case: str | None
    variance_share: float | None
    wall_share: float | None
    rounds: int
    sample_size: int
    unstable: bool
    detail: str = ""


def _sigma_attribution_note(resolution: SigmaResolution) -> str:
    """The clause the bench line carries when one case set the bar."""
    if resolution.dominant_case is None:
        if not resolution.detail:
            return ""
        return f"sigma not attributed per case ({resolution.detail}); "
    parts = [
        f"sigma attributed to case {resolution.dominant_case!r} "
        f"({resolution.variance_share:.1%} of variance on "
        f"{resolution.wall_share:.1%} of wall time)"
    ]
    if resolution.rounds:
        parts.append(
            f"bought {resolution.rounds} extra bench(es), sigma over "
            f"{resolution.sample_size} samples per case: "
            f"{resolution.measured_sigma:.6f} -> {resolution.sigma:.6f}"
        )
        if resolution.detail:
            parts.append(f"stopped early: {resolution.detail}")
    else:
        parts.append(f"not re-measured ({resolution.detail})")
    if resolution.unstable:
        parts.append(
            "case still dominates after the bound and the larger sample did "
            "not lower its spread, so the bar below is inflated by one case "
            "rather than by this candidate -- read it as a hint, not a "
            "finding: at nine samples that comparison misfires on about half "
            "of the cases that are merely noisy"
        )
    return "; ".join(parts) + "; "


def bench_failure_detail(bench_result: dict) -> str:
    """The driver's own evidence for why a bench run produced nothing usable."""
    message = str(bench_result.get("message") or "no failure message reported")
    output = str(bench_result.get("output") or "").strip()
    if not output:
        return message
    return f"{message}\n{textwrap.indent(output[-2000:], '    ')}"


def build_failure_tail(stdout: bytes, stderr: bytes, limit: int) -> str:
    """The tail of a failed build, taken from whichever stream carried it.

    Both streams are read. ninja prints the compiler's own output on stdout, so
    reading stderr alone reports a ninja failure to the agent as
    ``BUILD FAILED:`` and nothing else.
    """
    combined = b"\n".join(part.strip() for part in (stdout or b"", stderr or b"") if part.strip())
    text = combined.decode("utf-8", errors="replace").strip()
    return text[-limit:] if text else "no build output"


@dataclass(frozen=True)
class _ScoredBench:
    """One bench after scoring: what the KEEP bar and the result are built from."""

    bench_result: dict
    measurement_scores: list[float]
    sigma_resolution: SigmaResolution
    error_output: str


class CandidateMeasurementMixin:
    """Measure a candidate at a site and rule whether it clears the incumbent's KEEP bar.

    The host supplies ``ic``, ``best_mean_case_speedup``, ``_baseline_case_times``, ``_target_source_files``,
    ``_observe_measurement``, ``_working_tree_diff`` and ``_can_reuse_insession_benchmark``.
    """

    def _canonical_site(self) -> MeasurementSite:
        """The campaign workspace, where the linear search measures every candidate."""
        return MeasurementSite(
            workspace_dir=self.ic.workspace_dir,
            driver_script=self.ic.driver_script,
            build_dir=self.ic.build_dir,
            declared_sources=tuple(self._target_source_files()),
        )

    def _jit_source_files(self) -> list[str]:
        """Declared hints plus actual tracked edits in the campaign workspace."""
        return self._canonical_site().jit_sources()

    async def run_one_iteration(
        self,
        iteration: int,
        plan: str = "",
        *,
        benchmark_measurement: dict | None = None,
        site: MeasurementSite | None = None,
    ) -> IterationResult:
        """Execute a single build→validate→bench→canonical→decide iteration at ``site`` (the workspace by default)."""
        site = site or self._canonical_site()
        iter_start = time.time()
        force_jit_rebuild(site.jit_sources())
        build_failure = await self._build_candidate(iteration, iter_start, site)
        if build_failure is not None:
            return build_failure
        measurement_started = time.time()
        report = await self._validate_candidate(site)
        if not report.all_passed:
            self._observe_measurement(measurement_started)
            return IterationResult(
                iteration=iteration,
                duration_sec=time.time() - iter_start,
                validation_passed=False,
                validation_summary=report.summary(),
                validation_outcome=report.failed_outcome,
                error_output=report.failed_output,
                kept=False,
            )
        bench_result = await self._bench_candidate(site, benchmark_measurement)
        self._observe_measurement(measurement_started)
        self.last_case_bandwidth = dict(bench_result.get("case_bandwidth") or {})
        return await self._decide_iteration_keep(iteration, iter_start, report, bench_result, site)

    async def _build_candidate(
        self,
        iteration: int,
        iter_start: float,
        site: MeasurementSite,
    ) -> IterationResult | None:
        """Run the task's build at ``site``; the failed result, or ``None`` when it built."""
        if not self.ic.build_command:
            return None
        proc = await asyncio.create_subprocess_exec(
            *self.ic.build_command,
            cwd=site.build_dir or site.workspace_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await communicate_process_group(
            proc,
            timeout=self.ic.build_timeout_sec,
        )
        if proc.returncode == 0:
            return None
        return IterationResult(
            iteration=iteration,
            duration_sec=time.time() - iter_start,
            validation_passed=False,
            validation_summary=f"BUILD FAILED: {build_failure_tail(stdout, stderr, 500)}",
            kept=False,
        )

    async def _validate_candidate(self, site: MeasurementSite) -> ValidationReport:
        print("  [validate] Running full correctness suite...")
        report = await run_validation_pipeline(
            driver_script=site.driver_script,
            snr_threshold=self.ic.snr_threshold,
            timeout_per_stage=self.ic.validate_stage_timeout_sec,
        )
        for result in report.results:
            snr_str = f" SNR={result.snr_db:.1f}dB" if result.snr_db is not None else ""
            print(f"  [validate] Stage {result.stage} {result.stage_name}: {result.status_label}{snr_str}")
        if not report.all_passed:
            print(f"  [validate] FAILED at stage {report.failed_stage} — skipping bench")
        return report

    async def _bench_candidate(self, site: MeasurementSite, benchmark_measurement: dict | None) -> dict:
        if benchmark_measurement is not None and self._can_reuse_insession_benchmark(
            benchmark_measurement,
            attempt_diff=self._working_tree_diff(),
        ):
            print("  [bench] Reusing in-session three-measurement result...")
            bench_result = dict(benchmark_measurement)
            bench_result["reused_from_insession"] = True
            return bench_result
        print("  [bench] Running three independent benchmark suites...")
        return await measure_wallclock(
            BenchRun(site.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
            measurements=KEEP_MEASUREMENT_COUNT,
            timeout_sec=self.ic.bench_timeout_sec,
        )

    def _apply_mean_case_speedup_metric(self, bench_result: dict | None) -> None:
        """Attach three pristine-relative scores and their mean."""
        if not isinstance(bench_result, dict):
            return
        try:
            measurement_scores = calculate_measurement_case_speedups(
                bench_result,
                self._baseline_case_times,
                expected_measurements=KEEP_MEASUREMENT_COUNT,
            )
        except CaseCoverageError as error:
            bench_result["success"] = False
            bench_result["mean_case_speedup"] = None
            bench_result["measurement_mean_case_speedups"] = []
            bench_result["message"] = f"CASE COVERAGE FAILED: {error}"
            bench_result["case_coverage_complete"] = False
            return
        mean_case_speedup = keep_score(measurement_scores)
        if mean_case_speedup is None:
            bench_result["success"] = False
            bench_result["mean_case_speedup"] = None
            bench_result["message"] = "MEAN CASE SCORING FAILED: pristine per-case timings are unavailable"
            bench_result["case_coverage_complete"] = False
            return
        bench_result["mean_case_speedup"] = mean_case_speedup
        bench_result["measurement_mean_case_speedups"] = measurement_scores
        bench_result["case_coverage_complete"] = True

    def _scored_baseline_case_times(self, bench_result: dict) -> dict[str, float]:
        """The pristine per-case times the objective's mean is actually taken over."""
        excluded: set[str] = set()
        for measurement in bench_result.get("measurements") or ():
            if isinstance(measurement, dict):
                excluded.update(str(case_id) for case_id in (measurement.get("unscored_cases") or ()))
        return {
            case_id: float(baseline_ms)
            for case_id, baseline_ms in self._baseline_case_times.items()
            if case_id not in excluded and isinstance(baseline_ms, (int, float)) and float(baseline_ms) > 0.0
        }

    async def _remeasure_keep_sigma(
        self,
        found: SigmaResolution,
        base: SigmaAttribution,
        series: dict[str, list[float]],
        baseline: dict[str, float],
        site: MeasurementSite,
    ) -> SigmaResolution:
        """Buy extra benches until the dominant case's sigma settles or the budget ends."""
        current = base
        rounds = 0
        stopped = ""
        while rounds < SIGMA_REMEASURE_MAX_ROUNDS and current.dominant_case is not None:
            remeasure_started = time.time()
            extra = await measure_wallclock(
                BenchRun(site.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
                measurements=SIGMA_REMEASURE_BATCH,
                timeout_sec=self.ic.bench_timeout_sec,
            )
            self._observe_measurement(remeasure_started)
            rounds += 1
            extra_series = measurement_case_times(extra)
            stopped = _remeasure_stop_reason(extra, series, extra_series)
            if stopped:
                break
            for case_id in series:
                series[case_id].extend(extra_series[case_id])
            refreshed = attribute_sigma(series, baseline)
            if refreshed is None:
                stopped = "re-measure produced no usable spread"
                break
            current = refreshed
        if current is base:
            return replace(found, rounds=rounds, detail=stopped)
        return _settled_resolution(found, base, current, rounds=rounds, detail=stopped)

    async def _resolve_keep_sigma(
        self,
        bench_result: dict,
        measurement_scores: list[float],
        site: MeasurementSite,
    ) -> SigmaResolution:
        """Estimate the objective's sigma from the case that supplies it."""
        measured = measurement_sigma(measurement_scores)
        idle = SigmaResolution(
            sigma=measured,
            measured_sigma=measured,
            dominant_case=None,
            variance_share=None,
            wall_share=None,
            rounds=0,
            sample_size=len(measurement_scores),
            unstable=False,
        )
        if measured is None or not bench_result.get("success"):
            return idle
        baseline = self._scored_baseline_case_times(bench_result)
        series = {
            case_id: list(times)
            for case_id, times in measurement_case_times(bench_result).items()
            if case_id in baseline
        }
        base = attribute_sigma(series, baseline)
        if base is None:
            return replace(idle, detail="per-case times resolve no spread to attribute")
        if base.dominant_case is None:
            return idle
        dominant = base.dominant_case
        found = replace(
            idle,
            dominant_case=dominant,
            variance_share=base.variance_shares[dominant],
            wall_share=base.wall_shares[dominant],
            sample_size=base.sample_size,
        )
        incumbent = self.best_mean_case_speedup or 1.0
        if not beats_current_best(keep_score(measurement_scores), best_mean_case_speedup=incumbent):
            return replace(found, detail="reverted at every sigma")
        if passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=incumbent,
            sigma=measured,
            sigma_sample_size=base.sample_size,
        ):
            return replace(found, detail="kept at the measured sigma")
        return await self._remeasure_keep_sigma(found, base, series, baseline, site)

    def _emit_bench_scores(self, scored: _ScoredBench) -> None:
        """Print the KEEP-bar line and any pre-scoring bench-failure tail."""
        bench_result = scored.bench_result
        selected_raw_mean_ms = bench_result.get("median_ms")
        mean_case_speedup = bench_result.get("mean_case_speedup")
        score_text = f"{mean_case_speedup:.6f}x" if mean_case_speedup is not None else "n/a"
        sigma = scored.sigma_resolution.sigma
        required_score = required_keep_speedup(
            self.best_mean_case_speedup or 1.0,
            scored.measurement_scores,
            sigma=sigma,
            sigma_sample_size=scored.sigma_resolution.sample_size,
        )
        # The bar is a t multiple of the standard error of the mean, so a REVERT is only readable next to the spread
        # that set it and, when one case supplied that spread, next to the case: a weak candidate and a noisy 10 us
        # dispatch print the same mean score.
        sigma_text = f"{sigma:.6f}" if sigma is not None else "n/a"
        print(
            f"  [bench] pristine-relative scores="
            f"{[round(score, 6) for score in scored.measurement_scores]}; "
            f"sigma={sigma_text}; "
            f"{_sigma_attribution_note(scored.sigma_resolution)}"
            f"mean score={score_text}; required={required_score:.6f}x; "
            f"raw mean={selected_raw_mean_ms} ms  "
            f"({bench_result.get('message', '')})"
        )
        if scored.error_output:
            # The scoring verdict has already overwritten ``message`` with "candidate emitted no per-case timings",
            # which describes the symptom of a crash as if it were a formatting choice.
            print(f"  [bench] {scored.error_output}")

    def _clears_keep_bar(self, scored: _ScoredBench) -> bool:
        """Whether a measured candidate is a KEEP.

        The mean of the independent pristine-relative scores must clear the current best by the candidate's own
        measurement noise. A second-stage ASM result must also beat the original caller's aggregate time.
        """
        bench_result = scored.bench_result
        improved = bool(bench_result.get("success")) and passes_keep_threshold(
            scored.measurement_scores,
            best_mean_case_speedup=(self.best_mean_case_speedup or 1.0),
            sigma=scored.sigma_resolution.sigma,
            sigma_sample_size=scored.sigma_resolution.sample_size,
        )
        if improved and self.ic.kernel_backend == "assembly":
            source_ms = self.ic.pristine_baseline_wall_ms
            raw_mean_ms = bench_result.get("median_ms")
            improved = source_ms is not None and raw_mean_ms is not None and raw_mean_ms < source_ms
        return improved

    async def _score_bench(self, bench_result: dict, site: MeasurementSite) -> _ScoredBench:
        """Score the bench against the pristine baseline and resolve the sigma its KEEP bar is charged to."""
        # A candidate whose bench crashed is reverted for "no speedup" unless the crash itself reaches the agent; the
        # tool's output tail is the only place the traceback exists.
        error_output = "" if bench_result.get("success") else "BENCH FAILED: " + bench_failure_detail(bench_result)
        # Collapse per-case times into an equal-weight mean of per-case speedups, rather than allowing expensive cases
        # to dominate an aggregate ratio.
        self._apply_mean_case_speedup_metric(bench_result)
        measurement_scores = list(bench_result.get("measurement_mean_case_speedups") or [])
        sigma_resolution = await self._resolve_keep_sigma(bench_result, measurement_scores, site)
        scored = _ScoredBench(bench_result, measurement_scores, sigma_resolution, error_output)
        self._emit_bench_scores(scored)
        return scored

    async def _decide_iteration_keep(
        self,
        iteration: int,
        iter_start: float,
        report: ValidationReport,
        bench_result: dict,
        site: MeasurementSite,
    ) -> IterationResult:
        """Score the bench, check registers, apply the KEEP bar, and build the IterationResult."""
        scored = await self._score_bench(bench_result, site)
        snr_db = report.results[-1].snr_db if report.results else None
        reg_result = await check_registers(build_dir=site.build_dir)
        vgpr = reg_result.get("vgpr") if reg_result.get("success") else None
        if vgpr:
            print(f"  [registers] VGPR={vgpr}")
        result = IterationResult(
            iteration=iteration,
            duration_sec=time.time() - iter_start,
            validation_passed=True,
            validation_summary=report.summary(),
            wall_ms=bench_result.get("median_ms"),
            mean_case_speedup=bench_result.get("mean_case_speedup"),
            snr_db=snr_db,
            vgpr=vgpr,
            kept=self._clears_keep_bar(scored),
            bench_detail=bench_result,
            error_output=scored.error_output,
        )
        # The numerical contract, which only assembly declares and only it needs. Every other backend was judged by
        # the driver and measured through it since; re-running that verdict here would answer the same question with
        # the same command, while reading a task configuration whose shape the engine has no business knowing.
        if result.kept and runs_task_suite_acceptance(self.ic.kernel_backend):
            return await self._accept_on_task_suite(result, site)
        return result

    async def _accept_on_task_suite(self, result: IterationResult, site: MeasurementSite) -> IterationResult:
        """Run the task's canonical correctness suite on a candidate that cleared the bar."""
        canonical_started = time.time()
        canonical = await accept_candidate(
            site.workspace_dir,
            timeout_cap_sec=self.ic.validate_stage_timeout_sec,
            candidate_label=f"iteration {result.iteration}",
            kernel_backend=self.ic.kernel_backend,
        )
        # The suite only runs for a candidate the round produced, so it is part of that round's measurement and
        # has to be priced into the next round's admission alongside the validate-and-bench cycle.
        self._observe_measurement(canonical_started)
        if canonical.numerical_evidence is not None:
            result.bench_detail["numerical_validation"] = canonical.numerical_evidence
        duration_sec = result.duration_sec + (time.time() - canonical_started)
        if canonical.passed:
            return replace(
                result,
                duration_sec=duration_sec,
                validation_summary=f"{result.validation_summary}\n  Canonical correctness suite: {canonical.detail}",
            )
        return replace(
            result,
            duration_sec=duration_sec,
            validation_passed=False,
            validation_summary=(
                f"{result.validation_summary}\n  Canonical correctness suite: FAILED — {canonical.detail}"
            ),
            validation_outcome=(canonical.outcome or "canonical_correctness_failure"),
            error_output=canonical.output,
            kept=False,
        )


def _remeasure_stop_reason(
    extra: dict,
    series: dict[str, list[float]],
    extra_series: dict[str, tuple[float, ...]],
) -> str:
    """Why one extra bench cannot sharpen the sigma, or "" when it can."""
    if not extra.get("success"):
        return "re-measure bench failed"
    if set(series) - set(extra_series):
        return "re-measure lost a scored case"
    return ""


def _settled_resolution(
    found: SigmaResolution,
    base: SigmaAttribution,
    current: SigmaAttribution,
    *,
    rounds: int,
    detail: str,
) -> SigmaResolution:
    """The resolution a completed re-measure reports."""
    dominant = found.dominant_case
    share = current.variance_shares.get(dominant)
    # A case's *share* of the variance is structural -- q61's speedup term is 8x the others', so it holds most of
    # the variance at any sample size -- and re-measuring cannot be expected to move it.
    settled = current.case_sigmas[dominant] < base.case_sigmas[dominant]
    return replace(
        found,
        sigma=rescaled_sigma(found.measured_sigma, base, current),
        variance_share=share if share is not None else found.variance_share,
        wall_share=current.wall_shares.get(dominant, found.wall_share),
        rounds=rounds,
        sample_size=current.sample_size,
        unstable=current.dominant_case is not None and not settled,
        detail=detail,
    )


__all__ = [
    "CandidateMeasurementMixin",
    "MeasurementSite",
    "SigmaResolution",
    "bench_failure_detail",
    "build_failure_tail",
    "measurement_case_times",
]
