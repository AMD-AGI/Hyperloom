# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run-state hydration, validated warm starts and the speedup anchor for the forge-loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path


from kernelforge.loop.run_state import (
    BestRecord,
    make_event,
    should_resume,
)
from kernelforge.evaluation.benchmark.baseline import (
    BASELINE_DRIFT_TOLERANCE,
    BASELINE_DRIFT_TOLERANCE_ENV,
    check_baseline_against_reference,
)

log = logging.getLogger(__name__)


class WarmStartRuntimeMixin:
    """Own how a campaign starts from prior work: run-state hydration, warm starts, the anchor."""

    def _seed_and_hydrate_run_state(self) -> None:
        """Seed a fresh session or hydrate an explicitly validated resume."""
        head_out = self._git("rev-parse", "HEAD").strip()
        head = head_out.splitlines()[0] if head_out else ""
        if self.resume and self.run_state.baseline_case_times:
            self._baseline_case_times = dict(self.run_state.baseline_case_times)
        if self.resume and should_resume(self.run_state, head):
            self.best_wall_ms = self.run_state.best.wall_ms
            self.best_mean_case_speedup = self.run_state.best.mean_case_speedup
            print(
                f"  [run-state] resumed best from {self.run_state.best.commit_hash[:8]}: "
                f"mean case speedup={self.run_state.best.mean_case_speedup:.6f}x, "
                f"raw mean={self.run_state.best.wall_ms} ms "
                f"(iter {self.run_state.best.iteration})"
            )
        if self.monitor is not None:
            self.monitor.no_improve_streak = self.run_state.stall.no_improvement_iters
            self.monitor.last_attempt_iter = self.run_state.stall.last_supervisor_attempt_iter or -10_000
            self.monitor.intervention_count = self.run_state.intervention_count
        if self.resume:
            # Without this the resumed session re-derives its own noise floor, incumbents and SNR reference, so it
            # judges candidates by different rules than the session it continues.
            if not self._scoring_state_restored:
                self._restore_scoring_state()
        if self.ic.baseline_wall_ms is not None:
            self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
        if self._baseline_case_times:
            self.run_state.baseline_case_times = dict(self._baseline_case_times)
        if self.ic.pristine_baseline_wall_ms is not None:
            self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
        if not self.resume:
            self.state_store.append_event(
                make_event(
                    "baseline_measured",
                    0,
                    baseline_wall_ms=self.ic.baseline_wall_ms,
                    mean_case_speedup=1.0,
                )
            )
        self.state_store.save(self.run_state)

    def _validate_pre_published_warm_start(
        self,
        *,
        commit_hash: str,
        baseline_ms: float,
        best_ms: float,
        mean_case_speedup: float,
    ) -> bool:
        """Validate the CLI's kill-recoverable warm-start publication."""
        publication = self.ic.warm_start_publication
        if not publication:
            return False

        try:
            manifest_path = Path(str(publication["best_manifest"]))
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("best manifest is not an object")
            checks = (
                int(publication.get("best_iteration", -1)) == 0,
                str(publication.get("best_commit") or "") == commit_hash,
                float(publication.get("baseline_ms")) == float(baseline_ms),
                float(publication.get("best_ms")) == float(best_ms),
                float(publication.get("mean_case_speedup")) == float(mean_case_speedup),
                int(manifest.get("iteration", -1)) == 0,
                str(manifest.get("commit_hash") or "") == commit_hash,
                float(manifest.get("baseline_wall_ms")) == float(baseline_ms),
                float(manifest.get("best_wall_ms")) == float(best_ms),
                float(manifest.get("mean_case_speedup")) == float(mean_case_speedup),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError("pre-published warm-start best artifact is unreadable") from error
        if not all(checks):
            raise RuntimeError("pre-published warm-start best artifact does not match the validated workspace state")
        return True

    def _stage_validated_warm_start_state(self) -> None:
        """Make iteration-zero warm-start state durable on the first save."""
        if self.resume or not self.ic.warm_start_commit:
            return
        head = self._git("rev-parse", "HEAD").strip()
        if head != self.ic.warm_start_commit:
            raise RuntimeError("validated warm-start commit is not the current workspace HEAD")
        if (
            self.ic.baseline_wall_ms is None
            or self.ic.pristine_baseline_wall_ms is None
            or self.ic.warm_start_wall_ms is None
            or self.ic.warm_start_mean_case_speedup is None
        ):
            raise RuntimeError("validated warm-start is missing performance baselines")
        warm_start_bench = dict(self.ic.warm_start_bench or {})
        self.run_state.best = BestRecord(
            iteration=0,
            wall_ms=self.ic.warm_start_wall_ms,
            mean_case_speedup=self.ic.warm_start_mean_case_speedup,
            commit_hash=head,
            plan=f"Validated start: {self.ic.warm_start_solution_slug}".strip(),
            source="warm_start",
        )
        self.run_state.head_commit = head
        self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
        self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
        self.run_state.baseline_case_times = dict(self.ic.baseline_case_times)
        self.run_state.best_case_times = dict(warm_start_bench.get("case_times") or {})
        self.run_state.unscored_cases = [str(case_id) for case_id in (warm_start_bench.get("unscored_cases") or [])]

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
