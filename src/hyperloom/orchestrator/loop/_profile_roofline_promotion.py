# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting a profile or roofline result: the trace it adopts and the anchor it moves."""

from __future__ import annotations

import logging

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ._promote_outcome import _PromoteOutcome
from ..actions.executors.profile import profile_trace_path
from ..state.task_registry import Task

log = logging.getLogger(__name__)


class ProfileRooflineMixin:
    """Promoting the two discovery measurements."""

    async def _promote_replay_warm_recipe(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Separate promote path so replay doesn't overwrite baseline_tput/current_best."""
        outcome.verdict = self._coord.phase_prelude.promote_warm_replay(result, task=task)
        # PRELUDE initial roofline was deferred while replay ran.
        await self._coord.phase_prelude.maybe_enqueue_prelude_initial_analysis_after_baseline()

    def _record_failed_profile(self, trace_path: str) -> None:
        self.shared_state.last_profile_status = "failed"
        self.shared_state.last_profile_workload = {}
        if not trace_path:
            self.shared_state.last_profile_trace = ""
            self.shared_state.last_profile_launch_evidence_path = ""
        self.shared_state.last_profile_args = ""
        self.shared_state.last_profile_workload_action = ""

    def _adopt_profile_trace(self, result: dict, task: "Task | None", trace_path: str) -> None:
        self.shared_state.last_profile_trace = trace_path
        self.shared_state.last_profile_launch_evidence_path = str(result.get("launch_evidence_path") or "")
        self.shared_state.last_profile_status = "succeeded"
        profile_args = ""
        if task is not None:
            # Tag the workload with the arm it measured, so a later same-arm check can trust it.
            task_params = task.params or {}
            profile_args = str(task_params.get("base_extra_args") or "")
            self.shared_state.record_profile_workload(
                task_params,
                arm=("baseline" if str(task_params.get("reason") or "") == "prelude_initial" else ""),
            )
        else:
            self.shared_state.last_profile_workload = self.shared_state.current_profile_workload_context()
            self.shared_state.last_profile_workload_action = str(
                (self.shared_state.current_best or {}).get("action") or ""
            )
        self.shared_state.last_profile_args = str(
            self.shared_state.last_profile_workload.get("server_args") or profile_args
        )
        self.shared_state.last_trace_analyze = {}

    async def _promote_profile(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a profile result: trace path / status, optional current_best, roofline anchor."""
        trace_path = profile_trace_path(result)
        profile_status = str(result.get("status") or "")
        measurement_status = str(result.get("measurement_status") or profile_status)
        if profile_status == "failed" or result.get("error_class") == "no_trace_files":
            outcome.verdict = Verdict.FAILED
            self._record_failed_profile(trace_path)
        elif trace_path:
            self._adopt_profile_trace(result, task, trace_path)
        # Host-side rewrite evidence is independent of the trace: it answers
        # "which host-side work is redundant", which no kernel timeline can, and
        # a run whose trace was unusable can still have produced good evidence.
        # So promote it on its own, outside the trace_path branch.
        from ..actions.executors.framework_change._framework_rewrite_evidence import promote_evidence_path

        promote_evidence_path(self.shared_state, result)
        # On a successful profile, re-anchor last_roofline_tput and clear the pending field.
        if measurement_status == "succeeded":
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
        outcome.changed = True

    def _anchor_watermark_on_roofline(self, task: "Task | None") -> None:
        # Prefer the executor's last_trace_analyze snapshot over the result dict.
        last_ta = self.shared_state.last_trace_analyze or {}
        self.shared_state.roofline_failure_streak = 0
        if last_ta.get("has_analysis_md"):
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
        else:
            log.warning(
                "roofline %s produced no analysis; leaving the watermark anchor at %.4f so a real one can still be taken",
                task.task_id if task else "?",
                float(self.shared_state.last_roofline_tput or 0.0),
            )

    async def _promote_roofline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a roofline result: failure streak reset and roofline anchor (reads last_trace_analyze).

        The roofline executor settles ``succeeded`` or ``recovered`` here; a ``failed`` roofline is never promoted and
        is charged on the unpromotable path.
        """
        # The composite roofline action runs profile + trace_analyze atomically;
        # its executor writes last_profile_* + last_trace_analyze, so there is only the anchor to move here.
        self._anchor_watermark_on_roofline(task)
        outcome.changed = True
        # Clear the pending pointer (matched by task id).
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
