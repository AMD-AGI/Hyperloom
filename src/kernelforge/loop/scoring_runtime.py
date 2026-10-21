# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoring a candidate against the pristine anchor: the baseline, the sigma a KEEP clears, and its case coverage."""

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
    attribute_sigma,
    beats_current_best,
    keep_score,
    measurement_sigma,
    passes_keep_threshold,
    rescaled_sigma,
)
from kernelforge.loop.iteration_types import (
    CONFIG_COVERAGE_DISPERSION_MULTIPLE,
    CaseConfigCoverage,
    IterationResult,
)
from kernelforge.tools.process_group import communicate_process_group


def _measurement_case_times(
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

    def _set_baseline_case_times(self, case_times: dict | None) -> None:
        """Record (once) the pristine per-case wall times and persist them."""
        if not case_times or self._baseline_case_times:
            return
        self._baseline_case_times = dict(case_times)
        self.run_state.baseline_case_times = dict(case_times)
        self.state_store.save(self.run_state)

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

    def _scored_case_ids(self) -> list[str]:
        """The cases the suite actually scores, sorted."""
        return sorted(case_id for case_id in self._baseline_case_times if case_id not in self._unscored_cases)

    def _scored_incumbent_case_times(self) -> dict[str, float]:
        """The incumbent's per-case times, restricted to the scored cases."""
        scored = set(self._scored_case_ids())
        return {
            case_id: float(time_ms)
            for case_id, time_ms in (self._best_case_times or {}).items()
            if case_id in scored and isinstance(time_ms, (int, float)) and float(time_ms) > 0.0
        }

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
        if not beats_current_best(
            keep_score(measurement_scores),
            best_mean_case_speedup=incumbent,
        ):
            return replace(found, detail="reverted at every sigma")
        if passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=incumbent,
            sigma=measured,
            sigma_sample_size=base.sample_size,
        ):
            return replace(found, detail="kept at the measured sigma")

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
            extra_series = _measurement_case_times(extra if isinstance(extra, dict) else {})
            if not (isinstance(extra, dict) and extra.get("success")):
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

    def _case_move_rule(
        self,
        before: float,
        measured: float,
        per_run: tuple[float, ...] | list[float],
    ) -> str | None:
        """Which rule, if any, admits one KEEP's move on one case as real."""
        if before <= 0 or measured <= 0:
            return None
        move = before - measured
        if move / before < self.ic.config_coverage_min_move_ratio:
            return None
        times = [float(value) for value in per_run if float(value) > 0]
        if len(times) < 2:
            return "floor"
        if max(times) >= before:
            return None
        spread = max(times) - min(times)
        if move >= CONFIG_COVERAGE_DISPERSION_MULTIPLE * spread:
            return "dispersion"
        return None

    def _case_config_coverage(self) -> CaseConfigCoverage:
        """Read per-case configuration coverage off this session's KEEPs."""
        scored = self._scored_case_ids()
        previous = dict(self._baseline_case_times)
        moved_by: dict[str, list[int]] = {case_id: [] for case_id in scored}
        # Covered cases whose every admitting KEEP was admitted by the floor alone.
        dispersion_tested: set[str] = set()
        keeps: list[int] = []
        unreadable: list[int] = []
        unmeasured = set(scored)
        for result in self.results:
            if not result.kept:
                continue
            detail = result.bench_detail or {}
            case_times = dict(detail.get("case_times") or {})
            per_run = _measurement_case_times(detail)
            if not case_times:
                # A KEEP replayed from a pending record can arrive without its per-case timings.
                unreadable.append(result.iteration)
                continue
            keeps.append(result.iteration)
            for case_id in scored:
                measured = case_times.get(case_id)
                before = previous[case_id]
                if not measured:
                    continue
                unmeasured.discard(case_id)
                rule = self._case_move_rule(
                    before,
                    float(measured),
                    per_run.get(case_id, ()),
                )
                if rule is not None:
                    moved_by[case_id].append(result.iteration)
                    if rule == "dispersion":
                        dispersion_tested.add(case_id)
            previous.update(case_times)

        if not keeps:
            # Before the first readable KEEP there is nothing to read coverage off.
            return CaseConfigCoverage(
                covered={},
                fallback=(),
                undifferentiated=(),
                keeps=(),
                unmeasured=(),
                unreadable=tuple(unreadable),
                floor_only=(),
            )

        groups: dict[tuple[int, ...], list[str]] = {}
        for case_id, iterations in moved_by.items():
            if iterations:
                groups.setdefault(tuple(iterations), []).append(case_id)
        return CaseConfigCoverage(
            covered={case_id: iterations[-1] for case_id, iterations in moved_by.items() if iterations},
            fallback=tuple(case_id for case_id in scored if not moved_by[case_id] and case_id not in unmeasured),
            undifferentiated=tuple(
                tuple(sorted(members)) for _signature, members in sorted(groups.items()) if len(members) > 1
            ),
            keeps=tuple(keeps),
            unmeasured=tuple(sorted(unmeasured)),
            unreadable=tuple(unreadable),
            floor_only=tuple(
                sorted(
                    case_id
                    for case_id, iterations in moved_by.items()
                    if iterations and case_id not in dispersion_tested
                )
            ),
        )

    def _case_config_coverage_flags(self) -> dict[str, tuple[str, ...]]:
        """Per-case coverage flags for the planning context's case evidence."""
        coverage = self._case_config_coverage()
        flags: dict[str, list[str]] = {}
        for case_id, iteration in coverage.covered.items():
            flags.setdefault(case_id, []).append(f"config_coverage_keep_{iteration}")
        for case_id in coverage.floor_only:
            # Covered, but on the floor ratio alone.
            flags.setdefault(case_id, []).append("config_coverage_floor_only")
        for case_id in coverage.fallback:
            flags.setdefault(case_id, []).append("config_coverage_fallback")
        for case_id in coverage.unmeasured:
            flags.setdefault(case_id, []).append("config_coverage_unmeasured")
        for group in coverage.undifferentiated:
            for case_id in group:
                flags.setdefault(case_id, []).append("config_coverage_undifferentiated")
        if coverage.unreadable:
            # Every other flag on this ledger was read off a partial record, so the planner is told which cases were
            # classified without it rather than being handed the classification alone.
            for case_id in self._scored_case_ids():
                flags.setdefault(case_id, []).append("config_coverage_partial_record")
        return {case_id: tuple(dict.fromkeys(values)) for case_id, values in flags.items()}

    def _with_case_config_coverage(self, context):
        """Attach measured configuration coverage to a planning context."""
        flags = self._case_config_coverage_flags()
        if not flags:
            return context
        return replace(
            context,
            cases=tuple(
                replace(
                    case,
                    flags=tuple(dict.fromkeys([*case.flags, *flags.get(case.case_id, ())])),
                )
                for case in context.cases
            ),
        )

    def _render_case_config_coverage(self) -> str:
        """Render the configuration-coverage ledger for the Implementer."""
        coverage = self._case_config_coverage()
        scored = self._scored_case_ids()
        if not scored:
            return ""
        lines = [
            "## Per-case configuration coverage (measured, this session)",
            (
                "A scored case counts as covered once some KEEP improved its "
                "measured time by at least "
                f"{self.ic.config_coverage_min_move_ratio:.1%}. "
                "What was established beyond that depends on the record, and "
                "each covered case below says which. A case no KEEP has "
                "improved has never had a configuration chosen for it: it is "
                "running on whatever generic path the canonical falls "
                "through to, and the suite mean averages it in at 1.00x "
                "without saying so."
            ),
        ]
        if coverage.unreadable:
            lines.append(
                "INCOMPLETE RECORD: KEEP iteration(s) "
                + ", ".join(str(iteration) for iteration in coverage.unreadable)
                + " carried no per-case timings, so nothing below accounts "
                "for what they changed. Treat every case listed as uncovered "
                "as unconfirmed until a KEEP with per-case timings lands."
            )
        if not coverage.keeps:
            lines.append(
                "No KEEP with per-case timings is on this session's record, "
                "so no scored case has been shown to own a configuration "
                "yet: " + ", ".join(scored)
            )
            return "\n".join(lines)
        lines.append(
            "Read off KEEP iteration(s) "
            + ", ".join(str(iteration) for iteration in coverage.keeps)
            + ". A resumed campaign restarts this record, so earlier "
            "sessions are absent from it rather than counted as uncovered."
        )
        lines.append(
            f"Covered {len(coverage.covered)}/{len(scored)}: "
            + (
                ", ".join(f"{case_id} (iter {iteration})" for case_id, iteration in sorted(coverage.covered.items()))
                or "(none)"
            )
        )
        strongly_covered = [case_id for case_id in sorted(coverage.covered) if case_id not in coverage.floor_only]
        if strongly_covered:
            lines.append(
                "Faster in every independent measurement of the KEEP that "
                "covered them, by more than those measurements disagree "
                "among themselves: " + ", ".join(strongly_covered)
            )
        if coverage.floor_only:
            lines.append(
                "Admitted by the "
                f"{self.ic.config_coverage_min_move_ratio:.1%} floor alone -- "
                "the KEEP that moved them carried no per-measurement detail, "
                "so their run-to-run spread was never tested and only the "
                "size of the move is established: " + ", ".join(coverage.floor_only)
            )
        if coverage.fallback:
            lines.append("No configuration of its own: " + ", ".join(coverage.fallback))
        if coverage.unmeasured:
            lines.append("Coverage unknown -- no KEEP on record timed them: " + ", ".join(coverage.unmeasured))
        for group in coverage.undifferentiated:
            lines.append(
                "Never distinguished by any KEEP, so one configuration currently serves them all: " + ", ".join(group)
            )
        return "\n".join(lines)
