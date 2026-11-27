# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Action failures, kernel integrate verdicts, tick exceptions and crash counts."""

from __future__ import annotations

import time
import traceback
from typing import Any
from hyperloom.common.env_safety import redact_secret_values
from hyperloom_common.timeutil import now_iso
from ..failure_evidence import tail_excerpt


# Upper bound on retained crash timestamps (trailing-window rate needs only the recent tail).
_CRASH_TIMESTAMP_CAP: int = 200


# Global ``last_action_failures`` rolling-log cap.
_DEFAULT_LAST_FAILURES = 30


# Integration faults (environment / apply / bench crashes) are distinct from a genuine gate REVERT; a fault means the
# patch was never fairly measured, so it gets its own small retry budget instead of burning the REVERT quota.
_INTEGRATE_FAULT_ERROR_CLASSES = frozenset(
    {
        "missing_integration_inputs",
        "patch_not_applied",
        "apply_failed",
        # The served context cannot host an eval request, so the accuracy gate can never return a verdict under this
        # configuration.
        "eval_context_too_small",
        "mn_server_restart_failed_post_patch",
        "rebaseline_exception",
        "cpp_itfs_rebuild_not_verified",
        "framework_script_mismatch",
        "bench_exception",
        "subtask_exception",
        "handler_exception",
        "subprocess_timeout",
    }
)


class _ActionRecordsMixin:
    """Records how actions and runs failed or crashed."""

    def record_tick_exception(
        self,
        *,
        tick: int,
        stage: str,
        exc: BaseException,
        agent: str = "",
    ) -> dict[str, Any]:
        """Persist a compact Coordinator exception summary for postmortems."""
        entry = {
            "tick": int(tick or 0),
            "ts": now_iso(),
            "stage": str(stage or ""),
            "agent": str(agent or ""),
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[:12000],
        }
        self.last_tick_exception = entry
        return entry

    def increment_crash_count(self, by: int = 1) -> int:
        """Increment the cumulative crash counter and record crash times."""
        self.crash_count += by
        now = time.time()
        for _ in range(max(1, int(by))):
            self.crash_timestamps.append(now)
        if len(self.crash_timestamps) > _CRASH_TIMESTAMP_CAP:
            del self.crash_timestamps[:-_CRASH_TIMESTAMP_CAP]
        return self.crash_count

    def recent_crash_count(self, *, window_sec: float, now: float | None = None) -> int:
        """Count crashes recorded within the trailing ``window_sec`` seconds."""
        ref = time.time() if now is None else now
        cutoff = ref - float(window_sec)
        return sum(1 for t in self.crash_timestamps if t >= cutoff)

    @staticmethod
    def _is_integrate_fault(result: dict[str, Any]) -> bool:
        """True when an integrate result is an integration *fault*, not a verdict."""
        status = str(result.get("status") or "").strip().lower()
        if status == "failed":
            return True
        err_class = str(result.get("error_class") or "").strip()
        return err_class in _INTEGRATE_FAULT_ERROR_CLASSES

    def record_kernel_integrate_result(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Record an integrate result and capture the gate's verdict on the KERNEL timeline.

        Every integrate settlement passes through here, so the verdict is captured once rather than at each caller.
        """
        from ...kernel import kernel_decisions as _m

        entry = _m.record_kernel_integrate_result(self, result)
        self._record_integrate_verdict_on_timeline(entry)
        return entry

    def _record_integrate_verdict_on_timeline(self, entry: dict[str, Any] | None) -> None:
        """Mirror one settled integrate verdict onto the KERNEL timeline."""
        if not isinstance(entry, dict):
            return
        from hyperloom.inference_optimizer.breakdown.recorder import kernel_event_rows
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import IntegrateVerdict

        last = (entry.get("attempts") or [{}])[-1]
        last = last if isinstance(last, dict) else {}
        rejected = entry.get("rejected")
        cycle = int(getattr(self, "macro_cycle", 0) or 0)
        integration_id = str(entry.get("integration_id") or "")
        kernel_event_rows.record_integrate_verdict(
            macro_cycle=cycle,
            verdict=IntegrateVerdict(
                integration_id=integration_id,
                kernel_id=str(entry.get("kernel_id") or ""),
                decision=str(entry.get("last_decision") or ""),
                status=str(entry.get("last_status") or ""),
                attempt_count=entry.get("attempt_count"),
                fault_count=entry.get("fault_count"),
                gain_pct=entry.get("best_gain_pct"),
                accuracy_pass=last.get("accuracy_pass"),
                validation_tier=str(last.get("validation_tier") or ""),
                patch_path=str(entry.get("patch_path") or ""),
                target_file=str(entry.get("target_file") or ""),
                error_class=str(entry.get("last_error_class") or ""),
                rejected_reason=str(rejected.get("reason") or "") if isinstance(rejected, dict) else "",
                retryable=bool(entry.get("retryable")),
                settled_at=str(entry.get("updated_at") or ""),
                extra_server_args=str(entry.get("extra_server_args") or ""),
                basis=str(entry.get("basis") or ""),
                alignment_status=str(entry.get("alignment_status") or ""),
                # Absent means attributable; a writer that cannot pin the gain on this kernel says so.
                gain_attributed=bool(entry.get("validated", True)),
            ),
        )
        kernel_event_rows.record_integrate_axis(
            macro_cycle=cycle, integration_id=integration_id, graded_objective=str(last.get("graded_objective") or "")
        )

    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 1200) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` chars; None for falsy inputs (renderer shows ``err=(none)``)."""
        if value is None:
            return None
        text = redact_secret_values(str(value))
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[:limit]

    def _common_result_fields(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build the failure/diagnostic fields ``last_baseline`` and the failure log share."""
        return {
            "error_class": (str(result.get("error_class")) if result.get("error_class") else None),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "stderr_tail": tail_excerpt(result.get("error"), limit=1000),
            "stderr_log_path": (str(result.get("stderr_log_path")) if result.get("stderr_log_path") else None),
            "workspace": (str(result.get("workspace")) if result.get("workspace") else None),
            "raw_result_path": (str(result.get("raw_result_path")) if result.get("raw_result_path") else None),
            "reported_success": result.get("reported_success"),
            "variant_name": (str(result.get("variant_name")) if result.get("variant_name") else None),
            "failure_id": result.get("failure_id"),
        }

    def record_action_failure(
        self,
        *,
        action: str,
        task_id: str,
        result: dict[str, Any] | None,
        max_history: int = _DEFAULT_LAST_FAILURES,
    ) -> dict[str, Any]:
        """Append one rich failure record to :attr:`last_action_failures` for self-correction; invoked for EVERY unpromotable task kind."""
        result = result or {}
        entry: dict[str, Any] = {
            "ts": now_iso(),
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            **self._common_result_fields(result),
        }
        history = list(self.last_action_failures or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        self.last_action_failures = history
        return entry
