# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import os
from ..kernel.request_handlers import get_handler
from ..policy.gate import (
    PolicyDenied,
)
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)

import logging as _logging
log = _logging.getLogger(__name__)


class GatingCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    # Execution order guard
    def _target_analysis_baseline_exists(self) -> bool:
        """True iff target_analysis produced ``target_baseline.json`` (file existence is a sufficient gate signal).

        Returns:
            ``True`` when the target baseline file exists (or the helper is
            unavailable, treated as done); ``False`` otherwise.
        """
        try:
            from hyperloom.inference_optimizer.session_paths import target_baseline_json

            return target_baseline_json(self.session_dir).exists()
        except ImportError:
            # Missing helper -> treat the gate as satisfied (legacy/partial build).
            log.debug("_target_analysis_baseline_exists: helper unavailable", exc_info=True)
            return True

    def _kernel_opt_keep_pending(self) -> str:
        """Return the next kernel_id awaiting integrate, or "" if none (delegates to SharedState.next_pending_keep_kernel_id).

        Returns:
            The next kernel id pending integrate, or ``""`` when none.
        """
        return self.shared_state.next_pending_keep_kernel_id()

    def _sequence_denial_for_action(
        self,
        action_name: str,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts before baseline. Only invariant: nothing runs until baseline_tput > 0 (a data-dependency).

        Args:
            action_name: The proposed/delegated action name.

        Returns:
            A :class:`PolicyDenied` when the action must wait for baseline, else
            ``None``.
        """
        action = str(action_name or "").strip()
        sequence_actions = {
            "target_analysis",
            "baseline",
            "profile",
            "roofline",
            "sweep",
            "report",
            "integrate",
            "explore",
        }
        if action not in sequence_actions:
            return None
        if self.shared_state.stop_reason:
            return None
        if self.shared_state.baseline_tput <= 0 and action not in {"baseline", "target_analysis"}:
            return PolicyDenied(
                f"action={action!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` until baseline_tput > 0",
            )
        return None

    def _sequence_denial_for_request(
        self,
        target_agent: str,
        kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip the baseline prerequisite (invariant: nothing kernel-side runs before baseline_tput > 0).

        Args:
            target_agent: The request's target agent; only ``"kernel_agent"`` is
                gated.
            kind: The kernel request kind; ``trace_analyze`` and unknown kinds
                are exempt.

        Returns:
            A :class:`PolicyDenied` when the kernel request must wait for
            baseline, else ``None``.
        """
        target = str(target_agent or "").strip()
        req_kind = str(kind or "").strip()
        if target != "kernel_agent" or self.shared_state.stop_reason:
            return None
        if req_kind == "trace_analyze":
            return None
        if get_handler(req_kind) is None:
            return None
        if self.shared_state.baseline_tput <= 0:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` before kernel requests",
            )
        return None

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        """Report whether GEMM tuning is disabled via the env escape hatch.

        Returns:
            bool: ``True`` when ``INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING`` is set.
        """
        return os.environ.get(
            "INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Decide whether GEMM tuning must run before kernel_opt.

        When using forge-gemm-tune backend: eligible for any framework
        (sglang/vllm) and any precision with a MoE model or FP8 dense.
        When using GEAK backend: only FP8 + SGLang (legacy behavior).

        Returns:
            bool: ``True`` when GEMM tuning should run before source-level
                ``kernel_opt``.
        """
        if self._skip_gemm_tuning():
            return False
        ss = self.shared_state
        precision = str(getattr(ss, "precision", "") or "").strip().lower()
        framework = str(getattr(ss, "framework", "") or "").strip().lower()

        from ..kernel.request_handlers import _resolve_gemm_tuning_backend

        backend = _resolve_gemm_tuning_backend({})

        if backend == "forge":
            # forge-gemm-tune handles any precision (bf16/fp16/fp8/fp4/mxfp4),
            # dense or MoE, on sglang/vllm. Real e2e KEEPs span all of these —
            # including bf16 *dense* (+11.1%) — so we must NOT pre-filter on
            # precision/MoE here, or a category that can optimize gets silently
            # blocked. Gate only on a supported framework and let forge itself
            # return no_improvement when a shape can't be beaten.
            eligible = framework in ("sglang", "vllm", "vllm-aiter")
        else:
            # GEAK: legacy FP8 + SGLang only.
            eligible = (precision == "fp8" and framework == "sglang")

        if not eligible:
            return False
        last = getattr(ss, "last_gemm_tuning", {}) or {}
        status = str(last.get("status") or "").strip().lower()
        if self._bf16_dense_gemm_fallback_pending():
            return True
        return status not in {
            "ok",
            "succeeded",
            "success",
            "complete",
            "completed",
            "skipped",
            "failed",
        }
