# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Policy-controlled Analysis refresh coordination for the iteration loop."""

from __future__ import annotations

import logging
from dataclasses import replace

from kernelforge.loop.analysis_evidence import AnalysisEvidenceMixin
from kernelforge.loop.analysis_refresh_policy import (
    ANALYSIS_REFRESH_THRESHOLD,
    AnalysisRefreshDecision,
    decide_analysis_refresh,
)
from kernelforge.loop.run_state import make_event
from kernelforge.orchestrator.analysis import AnalysisConfigurationError
from kernelforge.orchestrator.analysis_session import AnalysisAttemptLimitError


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
        state = self.run_state.analysis
        planning_iteration = self.run_state.iteration if iteration is None else int(iteration)
        decision = decide_analysis_refresh(
            canonical_commit=context.analysis_commit,
            evidence_commit=state.evidence_commit,
            evidence_mean_case_speedup=(state.evidence_mean_case_speedup),
            evidence_status=state.evidence_status,
            current_mean_case_speedup=self.best_mean_case_speedup,
            supervisor_due=supervisor_due,
            last_attempt_commit=state.last_attempt_commit,
            last_attempt_status=state.last_attempt_status,
            last_attempt_iteration=state.last_attempt_iteration,
            current_iteration=planning_iteration,
        )
        return decision

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
        try:
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
        except Exception as error:
            message = f"persist analysis refresh decision for iteration {planning_iteration}: {error}"
            self.persistence_degraded = True
            self.persistence_errors.append(message)
            self.persistence_errors = self.persistence_errors[-10:]
            log.warning(message, exc_info=True)

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
        if analysis_service is None:
            self._active_analysis_context = self._apply_last_analysis_evidence(context)
            return self._active_analysis_context

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
            try:
                context = analysis_service.apply_checkpoint(context)
            except Exception as error:  # noqa: BLE001 - best-effort evidence
                log.debug("invalid Analysis checkpoint ignored: %s", error)
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

        analysis_state = self.run_state.analysis
        analysis_state.last_attempt_commit = context.analysis_commit
        analysis_state.last_attempt_status = "running"
        analysis_state.last_attempt_iteration = planning_iteration
        stale_context = self._apply_last_analysis_evidence(context)
        self._analysis_bundle = None
        try:
            self._analysis_bundle = await analysis_service.ensure_bundle(
                context,
                kernel_file=self.ic.kernel_file,
                driver_script=self.ic.driver_script,
                source_files=self._target_source_files(),
                usage=self._usage,
                deadline_unix=self._analysis_deadline_unix(),
                incremental=incremental,
            )
            outcome = getattr(self._analysis_bundle, "outcome", None)
            published = (
                outcome is not None and outcome.checkpoint_level == "published"
            ) or self._published_analysis_bundle_root(context.analysis_commit) is not None
            manifest = getattr(self._analysis_bundle, "manifest", {}) or {}
            manifest_status = str(manifest.get("status") or "READY").upper()
            available_tier = str(getattr(outcome, "available_tier", "") or "")
            upgrade_exhausted = bool(getattr(outcome, "upgrade_exhausted", False))
            profiling_enabled = bool(getattr(analysis_service, "profiling_enabled", True))
            if not profiling_enabled:
                evidence_status = available_tier or "static"
                attempt_status = "success"
            elif manifest_status == "PARTIAL" and not upgrade_exhausted:
                evidence_status = "partial"
                attempt_status = "partial"
            elif manifest_status == "PARTIAL":
                evidence_status = "partial_exhausted"
                attempt_status = "success"
            else:
                evidence_status = available_tier or "ready"
                attempt_status = "success"

            if published:
                self._last_published_analysis_commit = context.analysis_commit
                analysis_state.evidence_commit = context.analysis_commit
                analysis_state.evidence_mean_case_speedup = self.best_mean_case_speedup or 1.0
                analysis_state.evidence_status = evidence_status
            analysis_state.last_attempt_status = attempt_status
            if getattr(self, "state_store", None) is not None:
                event_payload = {
                    "status": "published" if published else "partial",
                    "analysis_commit": context.analysis_commit,
                    "artifact_path": str(self._analysis_bundle.root),
                    "refresh_reasons": list(decision.reasons),
                    "mean_case_speedup_at_collection": (self.best_mean_case_speedup or 1.0),
                }
                if outcome is not None:
                    event_payload.update(outcome.to_dict())
                self.state_store.append_event(
                    make_event(
                        "analysis_result",
                        planning_iteration,
                        **event_payload,
                    )
                )
            print(f"  [analysis] ready: {self._analysis_bundle.root}")
        except AnalysisAttemptLimitError as error:
            analysis_state.last_attempt_status = "exhausted"
            print(f"  [analysis] attempt budget exhausted ({error})")
            if getattr(self, "state_store", None) is not None:
                self.state_store.append_event(
                    make_event(
                        "analysis_result",
                        planning_iteration,
                        status="attempts_exhausted",
                        analysis_commit=context.analysis_commit,
                        requested_tier=(
                            "profiled"
                            if getattr(
                                analysis_service,
                                "profiling_enabled",
                                True,
                            )
                            else "static"
                        ),
                        available_tier="none",
                        checkpoint_level="work",
                        failure_type=type(error).__name__,
                        error=str(error),
                        refresh_reasons=list(decision.reasons),
                    )
                )
        except AnalysisConfigurationError:
            analysis_state.last_attempt_status = "fatal"
            raise
        except Exception as error:  # noqa: BLE001 - partial checkpoint may remain
            analysis_state.last_attempt_status = "failed"
            print(
                f"  [analysis] unavailable ({error}); "
                "using the last published bundle plus completed checkpoint artifacts"
            )
            if getattr(self, "state_store", None) is not None:
                self.state_store.append_event(
                    make_event(
                        "analysis_result",
                        planning_iteration,
                        status="failed",
                        analysis_commit=context.analysis_commit,
                        requested_tier=(
                            "profiled"
                            if getattr(
                                analysis_service,
                                "profiling_enabled",
                                True,
                            )
                            else "static"
                        ),
                        available_tier="none",
                        attempt=0,
                        checkpoint_level="none",
                        failure_type=f"{type(error).__name__}",
                        error=f"{type(error).__name__}: {error}",
                        refresh_reasons=list(decision.reasons),
                    )
                )
        finally:
            self._checkpoint_llm_usage()
            if getattr(self, "state_store", None) is not None:
                try:
                    self.state_store.save(self.run_state)
                except Exception as error:
                    message = f"persist Analysis refresh state for iteration {planning_iteration}: {error}"
                    self.persistence_degraded = True
                    self.persistence_errors.append(message)
                    self.persistence_errors = self.persistence_errors[-10:]
                    log.warning(message, exc_info=True)

        if self._analysis_bundle is not None and self._analysis_bundle.analysis_commit == context.analysis_commit:
            # A published bundle, including PARTIAL, is the evidence view for
            # its own commit. Do not seed it with refs from the prior evidence
            # commit: that would report a current/non-stale commit while quietly
            # retaining older paths. Failed unpublished attempts still merge
            # their checkpoint with ``stale_context`` in the branch below.
            context = self._analysis_bundle.apply(context)
        else:
            context = stale_context
            try:
                context = analysis_service.apply_checkpoint(context)
            except Exception as error:  # noqa: BLE001 - Analysis is best-effort
                log.debug("invalid Analysis checkpoint ignored: %s", error)
        cumulative_diff = self._analysis_cumulative_diff(
            evidence_commit=analysis_state.evidence_commit,
            canonical_commit=context.analysis_commit,
        )
        context = replace(
            context,
            canonical_commit=context.analysis_commit,
            evidence_commit=analysis_state.evidence_commit,
            evidence_stale=bool(
                analysis_state.evidence_commit and analysis_state.evidence_commit != context.analysis_commit
            ),
            evidence_status=analysis_state.evidence_status,
            evidence_mean_case_speedup=(analysis_state.evidence_mean_case_speedup),
            current_mean_case_speedup=self.best_mean_case_speedup,
            cumulative_diff_path=cumulative_diff.path,
            cumulative_diff_error=cumulative_diff.error,
        )
        self._active_analysis_context = context
        return context
