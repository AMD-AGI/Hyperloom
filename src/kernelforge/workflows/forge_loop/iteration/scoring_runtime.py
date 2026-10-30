# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoring a candidate against the pristine anchor: the baseline it divides by, the sigma a KEEP clears, and the cases it was configured for."""

from __future__ import annotations

import asyncio
import hashlib
import textwrap
import time
from dataclasses import dataclass, replace

from kernelforge.evaluation.benchmark.baseline import (
    BASELINE_DRIFT_TOLERANCE,
    BASELINE_DRIFT_TOLERANCE_ENV,
    check_baseline_against_reference,
)
from kernelforge.evaluation.benchmark.runner import (
    BenchRun,
    CaseCoverageError,
    calculate_mean_case_speedup,
    calculate_measurement_case_speedups,
    measure_wallclock,
)
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
)
from kernelforge.workflows.forge_loop.iteration.case_coverage import _measurement_case_times
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.campaign.reporting import KeepScore
from kernelforge.utils.process_group import communicate_process_group


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


def _bench_failure_detail(bench_result: dict) -> str:
    """The driver's own evidence for why a bench run produced nothing usable."""
    message = str(bench_result.get("message") or "no failure message reported")
    output = str(bench_result.get("output") or "").strip()
    if not output:
        return message
    return f"{message}\n{textwrap.indent(output[-2000:], '    ')}"


def _build_failure_tail(stdout: bytes, stderr: bytes, limit: int) -> str:
    """The tail of a failed build, taken from whichever stream carried it.

    Both streams are read. ninja prints the compiler's own output on stdout, so
    reading stderr alone reports a ninja failure to the agent as
    ``BUILD FAILED:`` and nothing else.
    """
    combined = b"\n".join(part.strip() for part in (stdout or b"", stderr or b"") if part.strip())
    text = combined.decode("utf-8", errors="replace").strip()
    return text[-limit:] if text else "no build output"


class ScoringMixin:
    """Own the keep/revert arithmetic: the measured baseline, the incumbent's per-case times, the objective's sigma and the coverage a KEEP bought."""

    def _can_reuse_insession_benchmark(
        self,
        measurement: dict | None,
        *,
        attempt_diff: str,
    ) -> bool:
        """Return whether a gate measurement belongs to this exact candidate."""
        if not isinstance(measurement, dict) or not measurement.get("success"):
            return False
        if not attempt_diff.strip():
            return False
        # The gate's tracked-diff fingerprint does not bind untracked source bytes.
        if self.ic.commit_new_paths and self._new_paths()[0]:
            return False
        if self.ic.build_command:
            return False
        if measurement.get("measurement_count") != KEEP_MEASUREMENT_COUNT:
            return False
        if len(measurement.get("measurements") or []) != KEEP_MEASUREMENT_COUNT:
            return False
        if measurement.get("bench_repeat") != self.ic.bench_repeat:
            return False
        expected_fingerprint = hashlib.sha256(attempt_diff.encode()).hexdigest()
        if measurement.get("candidate_diff_sha256") != expected_fingerprint:
            return False
        if measurement.get("driver_sha256") != self._driver_sha256():
            return False
        return (
            measurement.get("baseline_case_times") == self._baseline_case_times
            and measurement.get("best_mean_case_speedup") == self.best_mean_case_speedup
        )

    async def _measure_baseline(self) -> float | None:
        """Bench the pristine kernel before any agent edit — the speedup anchor."""
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *self.ic.build_command,
                cwd=self.ic.build_dir or self.ic.workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await communicate_process_group(
                proc,
                timeout=self.ic.build_timeout_sec,
            )
            if proc.returncode != 0:
                print(f"  Baseline build FAILED: {_build_failure_tail(stdout, stderr, 300)}")
                return None
        bench_result = await measure_wallclock(
            BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
            measurements=KEEP_MEASUREMENT_COUNT,
            timeout_sec=self.ic.bench_timeout_sec,
        )
        if not bench_result.get("success"):
            print("  Baseline bench FAILED: " + _bench_failure_detail(bench_result))
            return None
        baseline_case_times = dict(bench_result.get("case_times") or {})
        if not baseline_case_times:
            print(
                "  Baseline bench FAILED: the driver ran but printed no "
                "'case_ms: <case_id> <ms>' line: " + _bench_failure_detail(bench_result)
            )
            return None
        if bench_result.get("median_ms") is None:
            print(
                "  Baseline bench FAILED: the driver printed per-case timings "
                "but no aggregate 'median_ms:'/'mean_ms:' line: " + _bench_failure_detail(bench_result)
            )
            return None
        self.last_case_bandwidth = dict(bench_result.get("case_bandwidth") or {})
        unscored_cases = set(bench_result.get("unscored_cases") or [])
        try:
            baseline_score = calculate_mean_case_speedup(
                baseline_case_times,
                self._baseline_case_times or baseline_case_times,
                unscored_cases,
            )
        except CaseCoverageError as error:
            print(f"  Baseline bench FAILED: {error}")
            return None
        if baseline_score is None:
            print("  Baseline bench FAILED: mean case speedup is unavailable")
            return None

        if not self._baseline_case_times:
            self._baseline_case_times = dict(baseline_case_times)
            self.ic.baseline_case_times = dict(baseline_case_times)
        self.search_start_mean_case_speedup = baseline_score
        self._best_case_times = dict(baseline_case_times)
        self._unscored_cases = set(unscored_cases)
        self._persist_scoring_state()
        return bench_result.get("median_ms")

    def _verify_baseline_anchor(self) -> None:
        """Check the pristine anchor against the task reference, and report what it covered.

        Every speedup this run reports is a ratio against those timings, so an
        anchor that drifted from the task's own reference poisons the whole
        campaign. Drift is reported rather than fatal; a missing anchor is fatal.
        """
        # After the resume restore and the warm-start adoption above, never before them.
        if not self._baseline_case_times:
            raise RuntimeError(
                "mean case scoring requires pristine per-case timings before starting an optimization iteration"
            )

        # Every speedup this run reports is a ratio against those timings, so a baseline that drifted from the task's
        # own reference poisons the whole campaign.
        baseline_check = check_baseline_against_reference(
            self.ic.workspace_dir,
            self._baseline_case_times,
        )
        if baseline_check.unverified_reason:
            print(
                "  [baseline] the pristine anchor every speedup divides by is "
                f"unverified: {baseline_check.unverified_reason}"
            )
        else:
            print(
                "  [baseline] pristine anchor agrees with the task reference on "
                f"{baseline_check.compared_case_count} of "
                f"{baseline_check.measured_case_count} measured case(s); the "
                f"reference declares {baseline_check.reference_case_count}"
            )
        if baseline_check.unusable_entries:
            unusable = baseline_check.unusable_entries
            declared = baseline_check.reference_case_count + len(unusable)
            print(
                f"  [baseline] could not read {len(unusable)} of the "
                f"{declared} entries the task reference declares, so this "
                "check covers less of the anchor than the file does: " + "; ".join(unusable)
            )
        if baseline_check.tolerance_overridden and not baseline_check.unverified_reason:
            print(
                "  [baseline] drift tolerance widened to "
                f"{baseline_check.drift_tolerance * 100:.0f}% by "
                f"{BASELINE_DRIFT_TOLERANCE_ENV}, from the "
                f"{BASELINE_DRIFT_TOLERANCE * 100:.0f}% default; the "
                "anchor was accepted under the widened bound"
            )

    async def _anchor_speedup_reporting(self) -> None:
        """Settle the pristine anchor and the bar a candidate has to clear.

        The bar is read off the incumbent's own per-case times rather than
        assumed to be 1.0: assuming it would KEEP a candidate that loses to the
        kernel the campaign began with, on any run whose anchor came from the
        caller and on any resume that has not recorded a KEEP yet.
        """
        # The CLI constructs IterationLoop before applying a KB warm-start, then records the freshly measured pristine
        # case timings on IterationConfig.
        self._set_baseline_case_times(self.ic.baseline_case_times)
        if not self.resume and not self.ic.warm_start_commit and self._baseline_case_times:
            self._best_case_times = dict(self._baseline_case_times)
            self._unscored_cases = {str(case_id) for case_id in self.ic.preloop_baseline_unscored_cases}
        if self._best_case_times:
            self._persist_scoring_state()

        # Anchor speedup reporting.
        if not self.resume and self.ic.baseline_wall_ms is None:
            print("Measuring baseline on unmodified kernel...")
            baseline_ms = await self._measure_baseline()
            if baseline_ms is not None:
                self.ic.baseline_wall_ms = baseline_ms
                self.experiment.baseline_wall_ms = baseline_ms
                self.tracker.set_baseline(self.experiment.experiment_id, baseline_ms)
                print(f"  Baseline: {baseline_ms:.3f} ms\n")
            else:
                print(
                    "  Baseline measurement unavailable — see the "
                    "'Baseline build FAILED'/'Baseline bench FAILED' line above "
                    "for what the driver actually did\n"
                )

        if self.ic.pristine_baseline_wall_ms is None:
            self.ic.pristine_baseline_wall_ms = self.ic.baseline_wall_ms

        # The bar a candidate has to clear is whatever the incumbent scores against the anchor, so it is read off the
        # incumbent's own per-case times rather than assumed. It comes out at exactly 1.0 when the incumbent IS the
        # anchor, which is every run that did not supply one. Assuming 1.0 instead would KEEP a candidate that loses
        # to the kernel the campaign began with, on a fresh run whose anchor came from the caller and on any resume
        # that has not recorded a KEEP yet.
        if self.ic.baseline_wall_ms is not None:
            self.best_wall_ms = self.ic.baseline_wall_ms
        if self._baseline_case_times:
            self.best_mean_case_speedup = self._incumbent_mean_case_speedup()

    def _promote_best(self, result: IterationResult) -> None:
        """Make a kept candidate's aggregate case medians the new incumbent."""
        self.best_wall_ms = result.wall_ms
        detail = result.bench_detail or {}
        cases = detail.get("case_times") or {}
        if cases:
            self._best_case_times = dict(cases)
        self._persist_scoring_state()

    def _persist_scoring_state(self) -> None:
        """Checkpoint the state that decides keep/revert."""
        self.run_state.baseline_case_times = dict(self._baseline_case_times)
        self.run_state.best_case_times = dict(self._best_case_times)
        self.run_state.unscored_cases = sorted(self._unscored_cases)
        if self.search_start_mean_case_speedup is not None:
            self.run_state.search_start_mean_case_speedup = self.search_start_mean_case_speedup
        self.state_store.save(self.run_state)

    def _incumbent_mean_case_speedup(self) -> float:
        """Score the kernel currently in hand against the anchor every ratio divides by.

        A coverage mismatch is left to propagate: the incumbent's timings and the anchor are both written by this
        loop over the same case set, so they can only disagree on a checkpoint that no longer describes this
        campaign, and a KEEP bar guessed from that would admit a regression.
        """
        if not self._best_case_times:
            return 1.0
        return (
            calculate_mean_case_speedup(
                self._best_case_times,
                self._baseline_case_times,
                self._unscored_cases,
            )
            or 1.0
        )

    def _restore_scoring_state(self) -> None:
        """Rehydrate the keep/revert state recorded by a previous session."""
        state = self.run_state
        if state.best_case_times:
            self._best_case_times = dict(state.best_case_times)
        if state.unscored_cases:
            self._unscored_cases = {str(case_id) for case_id in state.unscored_cases}
        # A resume cannot re-measure the kernel the campaign started from -- the workspace holds the incumbent now --
        # so the score that anchors every ratio this session publishes has to come back from the checkpoint.
        if state.search_start_mean_case_speedup is not None:
            self.search_start_mean_case_speedup = state.search_start_mean_case_speedup
        self._scoring_state_restored = True
        if state.best_case_times:
            print(f"  [run-state] restored scoring state: {len(self._best_case_times)} case(s)")

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

    async def _resolve_keep_sigma(
        self,
        bench_result: dict,
        measurement_scores: list[float],
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
            for case_id, times in _measurement_case_times(bench_result).items()
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
        return await self._remeasure_keep_sigma(found, measured, base, series, baseline)

    def _keep_score(self, result: IterationResult, pending: dict, best_before: float | None) -> KeepScore | None:
        """Score a KEEP against its baseline and the search start; None when either side is unmeasured."""
        baseline = (
            pending.get("pristine_baseline_wall_ms")
            or pending.get("baseline_wall_ms")
            or self.ic.pristine_baseline_wall_ms
            or self.ic.publication_baseline_wall_ms
            or self.ic.baseline_wall_ms
            or self.run_state.baseline_wall_ms
            or best_before
        )
        if baseline is None or result.wall_ms is None or result.mean_case_speedup is None:
            return None
        return KeepScore(
            baseline_wall_ms=baseline,
            best_wall_ms=result.wall_ms,
            mean_case_speedup=result.mean_case_speedup,
            search_start_mean_case_speedup=(
                self.ic.warm_start_mean_case_speedup or self.search_start_mean_case_speedup or 1.0
            ),
            snr_db=result.snr_db,
            search_start_ms=(self.ic.warm_start_wall_ms or self.ic.baseline_wall_ms),
        )

    async def _remeasure_keep_sigma(
        self,
        found: SigmaResolution,
        measured: float,
        base: SigmaAttribution,
        series: dict[str, list[float]],
        baseline: dict[str, float],
    ) -> SigmaResolution:
        """Buy extra benches until the dominant case's sigma settles or the budget ends."""
        dominant = found.dominant_case
        current = base
        rounds = 0
        stopped = ""
        while rounds < SIGMA_REMEASURE_MAX_ROUNDS and current.dominant_case is not None:
            remeasure_started = time.time()
            extra = await measure_wallclock(
                BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
                measurements=SIGMA_REMEASURE_BATCH,
                timeout_sec=self.ic.bench_timeout_sec,
            )
            self._observe_measurement(remeasure_started)
            rounds += 1
            extra_series = _measurement_case_times(extra)
            if not extra.get("success"):
                stopped = "re-measure bench failed"
                break
            if set(series) - set(extra_series):
                stopped = "re-measure lost a scored case"
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
        share = current.variance_shares.get(dominant)
        # A case's *share* of the variance is structural -- q61's speedup term is 8x the others', so it holds most of
        # the variance at any sample size -- and re-measuring cannot be expected to move it.
        settled = current.case_sigmas[dominant] < base.case_sigmas[dominant]
        return replace(
            found,
            sigma=rescaled_sigma(measured, base, current),
            variance_share=share if share is not None else found.variance_share,
            wall_share=current.wall_shares.get(dominant, found.wall_share),
            rounds=rounds,
            sample_size=current.sample_size,
            unstable=current.dominant_case is not None and not settled,
            detail=stopped,
        )

    def _emit_bench_scores(
        self,
        bench_result: dict,
        sigma_resolution: SigmaResolution,
        measurement_scores: list[float],
        bench_error_output: str,
    ) -> None:
        """Print the KEEP-bar line and any pre-scoring bench-failure tail."""
        selected_raw_mean_ms = bench_result.get("median_ms")
        mean_case_speedup = bench_result.get("mean_case_speedup")
        score_text = f"{mean_case_speedup:.6f}x" if mean_case_speedup is not None else "n/a"
        sigma = sigma_resolution.sigma
        required_score = required_keep_speedup(
            self.best_mean_case_speedup or 1.0,
            measurement_scores,
            sigma=sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )
        # The bar is a t multiple of the standard error of the mean, so a REVERT is only readable next to the spread
        # that set it and, when one case supplied that spread, next to the case: a weak candidate and a noisy 10 us
        # dispatch print the same mean score.
        sigma_text = f"{sigma:.6f}" if sigma is not None else "n/a"
        print(
            f"  [bench] pristine-relative scores="
            f"{[round(score, 6) for score in measurement_scores]}; "
            f"sigma={sigma_text}; "
            f"{_sigma_attribution_note(sigma_resolution)}"
            f"mean score={score_text}; required={required_score:.6f}x; "
            f"raw mean={selected_raw_mean_ms} ms  "
            f"({bench_result.get('message', '')})"
        )
        if bench_error_output:
            # The scoring verdict has already overwritten ``message`` with "candidate emitted no per-case timings",
            # which describes the symptom of a crash as if it were a formatting choice.
            print(f"  [bench] {bench_error_output}")

    def _clears_keep_bar(
        self,
        bench_result: dict,
        measurement_scores: list[float],
        sigma_resolution: SigmaResolution,
    ) -> bool:
        """Whether a measured candidate is a KEEP.

        The mean of the independent pristine-relative scores must clear the current best by the candidate's own
        measurement noise. A second-stage ASM result must also beat the original caller's aggregate time.
        """
        improved = bool(bench_result.get("success")) and passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=(self.best_mean_case_speedup or 1.0),
            sigma=sigma_resolution.sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )
        if improved and self.ic.kernel_backend == "assembly":
            source_ms = self.ic.pristine_baseline_wall_ms
            raw_mean_ms = bench_result.get("median_ms")
            improved = source_ms is not None and raw_mean_ms is not None and raw_mean_ms < source_ms
        return improved
