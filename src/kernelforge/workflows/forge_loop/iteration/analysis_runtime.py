# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Policy-controlled Analysis refresh coordination for the iteration loop."""

from __future__ import annotations

import logging

from kernelforge.workflows.forge_loop.roles.analysis_evidence import AnalysisEvidenceMixin
from kernelforge.workflows.forge_loop.policy.analysis_refresh_policy import (
    ANALYSIS_REFRESH_THRESHOLD,
    AnalysisRefreshDecision,
    decide_analysis_refresh,
)
from kernelforge.workflows.forge_loop.campaign.run_state import make_event
from kernelforge.workflows.forge_loop.roles.analysis import AnalysisConfigurationError, AnalysisSources
from kernelforge.workflows.forge_loop.roles.analysis_session import AnalysisAttemptLimitError


log = logging.getLogger(__name__)


class AnalysisRuntimeMixin(AnalysisEvidenceMixin):
    """Own Analysis refresh admission, execution, retries, and checkpoints."""

    def _analysis_refresh_decision(
        self,
        context,
        *,
        supervisor_due: bool = False,
        iteration: int | None = None,
    ) -> AnalysisRefreshDecision:
        planning_iteration = self.run_state.iteration if iteration is None else int(iteration)
        return decide_analysis_refresh(
            self.run_state.analysis,
            canonical_commit=context.analysis_commit,
            current_mean_case_speedup=self.best_mean_case_speedup,
            supervisor_due=supervisor_due,
            current_iteration=planning_iteration,
        )

    def _record_analysis_refresh_decision(
        self,
        context,
        decision: AnalysisRefreshDecision,
        *,
        iteration: int | None = None,
    ) -> None:
        if getattr(self, "state_store", None) is None:
            return
        planning_iteration = self.run_state.iteration if iteration is None else int(iteration)
        self.state_store.append_event(
            make_event(
                "analysis_refresh_decision",
                planning_iteration,
                action="refresh" if decision.refresh else "reuse",
                reasons=list(decision.reasons),
                canonical_commit=context.analysis_commit,
                evidence_commit=(self.run_state.analysis.evidence_commit),
                evidence_stale=decision.evidence_stale,
                gain_since_evidence=decision.gain_since_evidence,
                cumulative_diff_path=context.cumulative_diff_path,
                cumulative_diff_error=context.cumulative_diff_error,
                refresh_threshold=ANALYSIS_REFRESH_THRESHOLD,
                current_mean_case_speedup=(self.best_mean_case_speedup),
                evidence_mean_case_speedup=(self.run_state.analysis.evidence_mean_case_speedup),
            )
        )

    def _record_analysis_result(self, iteration: int, **fields) -> None:
        """Append one analysis_result event when a state store is attached."""
        if getattr(self, "state_store", None) is None:
            return
        self.state_store.append_event(make_event("analysis_result", iteration, **fields))

    def _finish_successful_refresh(self, context, decision, planning_iteration, analysis_service) -> None:
        """Record the just-built bundle on run_state and emit its analysis_result event."""
        outcome = self._analysis_bundle.outcome
        published = (
            outcome is not None and outcome.checkpoint_level == "published"
        ) or self._published_analysis_bundle_root(context.analysis_commit) is not None
        manifest_status = str(self._analysis_bundle.manifest.get("status") or "READY").upper()
        if outcome is None:
            available_tier = ""
            upgrade_exhausted = False
        else:
            available_tier = str(outcome.available_tier or "")
            upgrade_exhausted = bool(outcome.upgrade_exhausted)
        if not analysis_service.profiling_enabled:
            evidence_status, attempt_status = available_tier or "static", "success"
        elif manifest_status != "PARTIAL":
            evidence_status, attempt_status = available_tier or "ready", "success"
        else:
            evidence_status, attempt_status = (
                ("partial_exhausted", "success") if upgrade_exhausted else ("partial", "partial")
            )
        analysis_state = self.run_state.analysis
        if published:
            self._last_published_analysis_commit = context.analysis_commit
            analysis_state.evidence_commit = context.analysis_commit
            analysis_state.evidence_mean_case_speedup = self.best_mean_case_speedup or 1.0
            analysis_state.evidence_status = evidence_status
        analysis_state.last_attempt_status = attempt_status
        event_payload = {
            "status": "published" if published else "partial",
            "analysis_commit": context.analysis_commit,
            "artifact_path": str(self._analysis_bundle.root),
            "refresh_reasons": list(decision.reasons),
            "mean_case_speedup_at_collection": (self.best_mean_case_speedup or 1.0),
        }
        event_payload.update(outcome.to_dict() if outcome is not None else {})
        self._record_analysis_result(planning_iteration, **event_payload)
        print(f"  [analysis] ready: {self._analysis_bundle.root}")

    async def _run_analysis_refresh(self, analysis_service, context, decision, planning_iteration, incremental):
        """ensure_bundle plus outcome recording; return the pre-refresh evidence context for failure fallback."""
        analysis_state = self.run_state.analysis
        analysis_state.last_attempt_commit = context.analysis_commit
        analysis_state.last_attempt_status = "running"
        analysis_state.last_attempt_iteration = planning_iteration
        stale_context = self._apply_last_analysis_evidence(context)
        self._analysis_bundle = None
        requested_tier = "profiled" if analysis_service.profiling_enabled else "static"
        try:
            self._analysis_bundle = await analysis_service.ensure_bundle(
                context,
                sources=AnalysisSources(self.ic.kernel_file, self.ic.driver_script, self._target_source_files()),
                usage=self._usage,
                deadline_unix=self._analysis_deadline_unix(),
                incremental=incremental,
            )
            self._finish_successful_refresh(context, decision, planning_iteration, analysis_service)
        except AnalysisAttemptLimitError as error:
            analysis_state.last_attempt_status = "exhausted"
            print(f"  [analysis] attempt budget exhausted ({error})")
            self._record_analysis_result(
                planning_iteration,
                status="attempts_exhausted",
                analysis_commit=context.analysis_commit,
                requested_tier=requested_tier,
                available_tier="none",
                checkpoint_level="work",
                failure_type=type(error).__name__,
                error=str(error),
                refresh_reasons=list(decision.reasons),
            )
        except AnalysisConfigurationError:
            analysis_state.last_attempt_status = "fatal"
            raise
        except Exception as error:  # isolation boundary: one iteration's analysis refresh
            log.warning("analysis refresh failed: %s", error, exc_info=True)
            analysis_state.last_attempt_status = "failed"
            print(
                f"  [analysis] unavailable ({error}); "
                "using the last published bundle plus completed checkpoint artifacts"
            )
            self._record_analysis_result(
                planning_iteration,
                status="failed",
                analysis_commit=context.analysis_commit,
                requested_tier=requested_tier,
                available_tier="none",
                attempt=0,
                checkpoint_level="none",
                failure_type=f"{type(error).__name__}",
                error=f"{type(error).__name__}: {error}",
                refresh_reasons=list(decision.reasons),
            )
        finally:
            self._checkpoint_llm_usage()
            if getattr(self, "state_store", None) is not None:
                self.state_store.save(self.run_state)
        return stale_context

    async def _resolve_analysis_context(
        self,
        analysis_service,
        *,
        supervisor_due: bool = False,
        iteration: int | None = None,
    ):
        """Refresh Analysis when policy requires it, otherwise reuse evidence."""
        planning_iteration = self.run_state.iteration if iteration is None else int(iteration)
        context = self._build_orchestration_context()
        restore_published = getattr(
            analysis_service,
            "apply_published_evidence",
            None,
        )
        if (
            self._active_analysis_context is None
            and self.run_state.analysis.evidence_commit
            and callable(restore_published)
        ):
            restored = restore_published(
                context,
                evidence_commit=(self.run_state.analysis.evidence_commit),
            )
            if restored is not context:
                self._active_analysis_context = restored

        decision = self._analysis_refresh_decision(
            context,
            supervisor_due=supervisor_due,
            iteration=planning_iteration,
        )
        self._record_analysis_refresh_decision(
            context,
            decision,
            iteration=planning_iteration,
        )
        if not decision.refresh:
            context = self._apply_last_analysis_evidence(context)
            context = analysis_service.apply_checkpoint(context)
            self._active_analysis_context = context
            return context

        previous_commit = self.run_state.analysis.evidence_commit or self._last_published_analysis_commit
        incremental = None
        if not context.cumulative_diff_error and previous_commit != context.analysis_commit:
            incremental = self._incremental_analysis_input(
                current_commit=context.analysis_commit,
                previous_commit=previous_commit,
            )
        mode = "cumulative post-KEEP incremental" if incremental is not None else "commit-bound"
        print(f"  [analysis] building {mode} analysis bundle ({', '.join(decision.reasons)})...")
        stale_context = await self._run_analysis_refresh(
            analysis_service, context, decision, planning_iteration, incremental
        )
        if self._analysis_bundle is not None and self._analysis_bundle.analysis_commit == context.analysis_commit:
            # A published bundle, including PARTIAL, is the evidence view for its own commit.
            context = self._analysis_bundle.apply(context)
        else:
            context = stale_context
            context = analysis_service.apply_checkpoint(context)
        self._active_analysis_context = self._with_evidence_standing(context)
        return self._active_analysis_context
