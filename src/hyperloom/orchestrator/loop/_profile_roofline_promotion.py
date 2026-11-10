# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting a profile or roofline result: the trace it adopts and the anchor it moves."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ._promote_outcome import _PromoteOutcome
from ..state.task_registry import Task

log = logging.getLogger(__name__)


def _profile_trace_path(result: Mapping[str, Any]) -> Any:
    if result.get("trace_input_ready") is False:
        return None
    return result.get("main_trace_path") or (result.get("trace_files") or [None])[0]


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

    def _record_failed_profile(self, trace_path: Any) -> None:
        self.shared_state.last_profile_status = "failed"
        self.shared_state.last_profile_workload = {}
        if not trace_path:
            self.shared_state.last_profile_trace = ""
            self.shared_state.last_profile_launch_evidence_path = ""
        self.shared_state.last_profile_args = ""
        self.shared_state.last_profile_workload_action = ""

    def _adopt_profile_trace(self, result: dict, task: "Task | None", trace_path: Any) -> str:
        self.shared_state.last_profile_trace = str(trace_path)
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
        return profile_args

    async def _promote_profile(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a profile result: trace path / status, optional current_best, roofline anchor."""
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # Defensive skipped arm: audit as skipped + drop the gate.
        if str(result.get("status") or "") == "skipped":
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
        else:
            audit_decision = "promoted"
            audit_extras = {
                "trace_path": None,
                "profile_args": None,
                "output_throughput": result.get("output_throughput"),
            }
        trace_path = _profile_trace_path(result)
        profile_status = str(result.get("status") or "")
        measurement_status = str(result.get("measurement_status") or profile_status)
        audit_extras["measurement_status"] = measurement_status
        audit_extras["trace_capture_status"] = result.get("trace_capture_status")
        audit_extras["trace_capture_reason"] = (result.get("trace_capture") or {}).get("reason")
        if profile_status == "failed" or result.get("error_class") == "no_trace_files":
            outcome.verdict = Verdict.FAILED
            self._record_failed_profile(trace_path)
        elif trace_path:
            audit_extras["trace_path"] = str(trace_path)
            audit_extras["profile_args"] = self._adopt_profile_trace(result, task, trace_path)
        # Host-side rewrite evidence is independent of the trace: it answers
        # "which host-side work is redundant", which no kernel timeline can, and
        # a run whose trace was unusable can still have produced good evidence.
        # So promote it on its own, outside the trace_path branch.
        from ..actions.executors._framework_rewrite_evidence import promote_evidence_path

        evidence_path = promote_evidence_path(self.shared_state, result)
        if evidence_path:
            audit_extras["framework_rewrite_evidence"] = evidence_path
            audit_extras["framework_rewrite_candidate_count"] = result.get("framework_rewrite_candidate_count")
        # On a successful profile, re-anchor last_roofline_tput and clear the pending field.
        if measurement_status == "succeeded":
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    def _anchor_watermark_on_roofline(self, result: dict, task: "Task | None") -> dict[str, Any]:
        # Prefer the executor's last_trace_analyze snapshot over the result dict.
        last_ta = self.shared_state.last_trace_analyze or {}
        self.shared_state.roofline_failure_streak = 0
        if str(last_ta.get("analysis_md_text") or ""):
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
        else:
            log.warning(
                "roofline %s produced no analysis; leaving the watermark anchor at %.4f so a real one can still be taken",
                task.task_id if task else "?",
                float(self.shared_state.last_roofline_tput or 0.0),
            )
        snapshot_id = last_ta.get("roofline_snapshot_id")
        return {
            "snapshot_id": result.get("snapshot_id") if snapshot_id is None else snapshot_id,
            "last_profile_trace": (self.shared_state.last_profile_trace or result.get("last_profile_trace")),
            "analysis_md_path": (last_ta.get("analysis_md_path") or result.get("analysis_md_path")),
            "profile_workspace": result.get("profile_workspace"),
            "degraded": bool(result.get("degraded", False)),
        }

    def _charge_roofline_failure(self, result: dict, task: "Task | None") -> dict[str, Any]:
        self.shared_state.roofline_failure_streak += 1
        log.warning(
            "Auto-roofline %s failed (reason=%s phase=%s "
            "error_class=%s); continuing in degraded mode "
            "(specialists / explore proceed without a fresh "
            "analysis_md). No retry, no fallback.",
            task.task_id if task else "?",
            str((task.params or {}).get("reason") or "") if task is not None else "",
            result.get("phase"),
            result.get("error_class"),
        )
        return {
            "phase": result.get("phase"),
            "error_class": result.get("error_class"),
            "error": result.get("error"),
        }

    async def _promote_roofline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a roofline result: audit + failure streak + roofline anchor (reads last_trace_analyze)."""
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # The composite roofline action runs profile + trace_analyze atomically;
        # its executor writes last_profile_* + last_trace_analyze, so here we just record the audit row.
        status = str(result.get("status") or "")
        if status == "skipped":
            # Defensive arm: clean no-op, no streak/watermark touch.
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
        elif status in {"succeeded", "recovered"}:
            audit_decision = "promoted"
            audit_extras = self._anchor_watermark_on_roofline(result, task)
        else:
            outcome.verdict = Verdict.FAILED
            audit_decision = "discarded"
            audit_extras = self._charge_roofline_failure(result, task)
        # Clear the pending pointer (matched by task id).
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras
