# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator result writeback: settle finished tasks into SharedState and rebuild that state on resume."""

from __future__ import annotations
import logging as _logging
import time
from ..actions.executors._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS
from ..actions.executors.benchmark_result import is_valid_measurement
from ..phases.machine_state import LifecycleMark, record_lifecycle_event
from ..state._shared_state.attempt_audit import _AUDIT_ACTIONS
from ..state.task_registry import Task
from ._baseline_failures import BaselineFailureMixin
from ._baseline_promotion import BaselinePromotionMixin
from ._current_best_lift import CurrentBestLiftMixin
from ._explore_promotion import ExplorePromotionMixin
from ._geak_revalidation import GeakRevalidation
from ._integrate_promotion import IntegratePromotionMixin
from ._keep_patch_column import AgentKeepStagingMixin
from ._launch_identity import LaunchIdentityMixin
from ._profile_roofline_promotion import ProfileRooflineMixin
from ._promote_outcome import _PromoteOutcome
from ._resume_consistency import ResumeConsistencyMixin
from ._resume_recovery import ResumeRecoveryMixin
from ._resume_replay import ResumeReplayMixin
from ._validation_watermark import ValidationMixin
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinator import Coordinator

log = _logging.getLogger(__name__)


class WritebackCollaborator(
    AgentKeepStagingMixin,
    BaselineFailureMixin,
    BaselinePromotionMixin,
    CurrentBestLiftMixin,
    ExplorePromotionMixin,
    IntegratePromotionMixin,
    LaunchIdentityMixin,
    ProfileRooflineMixin,
    ResumeConsistencyMixin,
    ResumeRecoveryMixin,
    ResumeReplayMixin,
    ValidationMixin,
    GeakRevalidation,
):
    """Handles result writeback: promotion of measured results, KEEP/REVERT settlement, and resume replay."""

    def __init__(self, coordinator: "Coordinator") -> None:
        super().__init__(coordinator)
        self._lifecycle_last_save: float = 0.0
        self._lifecycle_save_min_interval_s: float = 2.0
        self._resumed_from: dict[str, Any] = {}

    def emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event.

        Terminal events (END/ERROR) flush immediately; non-terminal markers are
        debounced within ``_lifecycle_save_min_interval_s``.

        Args:
            step: The machine step name (resolved to a human label downstream).
            status: The lifecycle status (e.g. START / END / ERROR / ENTER).
            artifacts: Optional mapping of produced artifact paths.
            detail: Optional free-text detail.
            duration_s: Optional elapsed seconds for the step.
        """
        record_lifecycle_event(
            self.shared_state,
            LifecycleMark(
                step=step,
                status=status,
                artifacts=artifacts,
                detail=detail,
                duration_s=duration_s,
            ),
        )
        terminal = status in ("END", "ERROR")
        now = time.monotonic()
        if terminal or (now - self._lifecycle_last_save >= self._lifecycle_save_min_interval_s):
            self.shared_state.save(self.session_dir)
            self._lifecycle_last_save = now

    def is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        """Decide whether a settled task result should be promoted.

        Per-kind rules: baseline/profile require a valid measurement, sweep
        requires ``status == "succeeded"``, ``replay_warm_recipe`` always routes
        through promotion (it owns its own failure bookkeeping), and everything
        else is promotable unless ``status == "failed"``.

        Args:
            task_kind (str): The task's kind.
            result (dict[str, Any]): The task result payload.

        Returns:
            bool: ``True`` when the result should go through
                :meth:`promote_to_shared_state`.
        """
        if task_kind == "baseline":
            # A baseline whose accuracy eval failed measured throughput but must
            # not anchor; route it to handle_unpromotable_result for enablement.
            # One refused for a cold prefix cache measured too, but must not anchor
            # either; handle_unpromotable_result stops the run on it.
            if bool(result.get("baseline_eval_failed")) or result.get("error_class") == PREFIX_CACHE_OFF_ERROR_CLASS:
                return False
            return is_valid_measurement(result)
        if task_kind == "profile":
            measurement_status = result.get("measurement_status")
            if measurement_status is not None and str(measurement_status) != "succeeded":
                return False
            # Magpie marks a profiled replay diagnostic_only, so a native AgentX profile
            # is never a valid measurement. The trace is what a profile task produces, and
            # ``ProfileExecutor`` already accepts the round on this field.
            if result.get("round_succeeded") is True:
                return True
            return is_valid_measurement(result)
        # replay_warm_recipe always routes through promote_warm_replay (owns its own failure bookkeeping).
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def record_intervention_for_task(
        self,
        task: "Task",
        result: Any,
        verdict: Verdict,
    ) -> None:
        """Log a settled task's change_type into SharedState.intervention_mix.

        explore → ``config`` when adopted, else ``config_attempt``;
        integrate_patch → ``code_patch`` when adopted, else ``code_patch_attempt``.

        Args:
            task: The settled task whose kind selects the intervention class.
            result: The task result dict; non-dict results are ignored.
            verdict: The settlement verdict.
        """
        if not isinstance(result, dict):
            return
        adopted = verdict is Verdict.ADOPTED
        kind = (task.kind or "").strip()
        if kind == "explore":
            best = result.get("best_variant")
            delta_pct = best.get("gain_pct") if adopted and isinstance(best, dict) else None
            self.shared_state.record_intervention(
                change_type="config" if adopted else "config_attempt",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            if not str(result.get("status") or "").strip():
                return
            self.shared_state.record_intervention(
                change_type="code_patch" if adopted else "code_patch_attempt",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    async def promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> _PromoteOutcome:
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid).

        Args:
            task_kind: The settled task's kind, selecting the promote branch.
            result: The task result dict.
            task: The originating task, used for audit fingerprints and
                pending-roofline gating.

        Returns:
            The promotion outcome; its ``verdict`` is the settlement verdict.
        """
        # ``integrate_patch`` settles "apply_failed" / "reverted", which promote,
        # so a promoted result is still the only record of why a patch failed.
        error_class = str(result.get("error_class") or "").strip()
        if error_class and task is not None:
            self.shared_state.record_action_failure(
                action=task_kind,
                task_id=task.task_id,
                result=result,
            )
        # ``replay_warm_recipe`` is mirrored by _promote_replay_warm_recipe
        # instead: its executor settles on "succeeded" and the keep decision is
        # only reached further down this call, so mirroring it here published
        # every replay as discarded -- including the ones that went on to be
        # pushed onto the stack.
        handler_name = self._PROMOTE_HANDLERS.get(task_kind)
        if handler_name is None:
            return _PromoteOutcome(verdict=Verdict.RECORDED)
        outcome = _PromoteOutcome(verdict=Verdict.RECORDED)
        await getattr(self, handler_name)(result, task, outcome)
        # sweep / conc_sweep already recorded + saved + returned via their handler.
        if outcome.early_return:
            return outcome
        # Audit trail: one succeeded-attempt record with branch-supplied decision/extras.
        if outcome.audit_decision is not None and task_kind in _AUDIT_ACTIONS:
            self.shared_state.record_action_attempt(
                action=task_kind,
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision=outcome.audit_decision,
                result=result,
                extras=outcome.audit_extras,
            )
            outcome.changed = True
        if outcome.changed:
            self.shared_state.save(self.session_dir)
            self._drain_agent_keep_outbox()
        return outcome

    _PROMOTE_HANDLERS: dict[str, str] = {
        "baseline": "_promote_baseline",
        "replay_warm_recipe": "_promote_replay_warm_recipe",
        "profile": "_promote_profile",
        "roofline": "_promote_roofline",
        "explore": "_promote_explore",
        "integrate_patch": "_promote_integrate_patch",
        "conc_sweep": "_promote_conc_sweep",
    }
