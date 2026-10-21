# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline and incumbent case times, mean case speedup and case-configuration coverage for the forge-loop."""

from __future__ import annotations

import logging
from dataclasses import replace


from kernelforge.evaluation.benchmark.runner import (
    CaseCoverageError,
    calculate_mean_case_speedup,
    calculate_measurement_case_speedups,
)
from kernelforge.evaluation.scoring import (
    KEEP_MEASUREMENT_COUNT,
    keep_score,
)
from kernelforge.loop.iteration_types import (
    CaseConfigCoverage,
    IterationResult,
)

log = logging.getLogger(__name__)


CONFIG_COVERAGE_DISPERSION_MULTIPLE = 1.0


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


class ScoringRuntimeMixin:
    """Own the scoring state: baseline and best case times, mean case speedup, and case coverage."""

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

    def _scored_case_ids(self) -> list[str]:
        """The cases the suite actually scores, sorted."""
        return sorted(case_id for case_id in self._baseline_case_times if case_id not in self._unscored_cases)
