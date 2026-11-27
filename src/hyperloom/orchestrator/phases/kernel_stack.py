# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-stack handler: draining the pending KEEP integrates a macro-cycle leaves for CYCLE_END entry."""

from __future__ import annotations
import logging as _logging
from typing import Any
from ..kernel import kernel_decisions
from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)


class KernelStackPhase(CoordinatorCollaborator):
    """Lands the kernel KEEPs still queued for integration when a macro-cycle ends."""

    def _reject_undispatchable_integrate(self, pending: dict[str, Any]) -> None:
        """Retire a queued integrate whose dispatch raised, so the drain loop cannot deadlock on it."""
        state = self.shared_state
        kid = str(pending.get("kernel_id") or "")
        if state.rejected_kernel_ids is None:
            state.rejected_kernel_ids = []
        pending_task_key = str(pending.get("task_key") or "")
        if not pending_task_key and kid not in state.rejected_kernel_ids:
            state.rejected_kernel_ids.append(kid)
        queued = (state.pending_kernel_integrations or {}).get(str(pending.get("integration_id") or ""))
        if isinstance(queued, dict):
            queued["status"] = "dispatch_failed"

    async def drain_pending_keep_integrates(self) -> None:
        """Drain pending KEEP integrates inherited from KERNEL so sweep measures full current_best. Cap 10; a dispatch failure flips the queued record to ``dispatch_failed``, and only records with no ``task_key`` are also appended to ``rejected_kernel_ids``."""
        from ..kernel.request_handlers import integrate_handler

        state = self.shared_state
        drained = 0
        max_drain = 10
        while drained < max_drain:
            pending_records = kernel_decisions.pending_kernel_integration_records(state)
            if not pending_records:
                break
            pending = pending_records[0]
            kid = str(pending.get("kernel_id") or "")
            integration_id = str(pending.get("integration_id") or "")
            log.info(
                "CYCLE_END entry: draining pending KEEP integrate for kernel_id=%s integration_id=%s (drained %d so far)",
                kid,
                integration_id,
                drained,
            )
            try:
                result = await integrate_handler(
                    {
                        "kernel_id": kid,
                        "integration_id": integration_id,
                        "task_group_key": str(pending.get("task_group_key") or ""),
                        "identity_route": str(pending.get("identity_route") or ""),
                        "base_tput": float((state.current_best or {}).get("tput") or state.baseline_tput or 0.0),
                    },
                    session_dir=self.session_dir,
                )
                if isinstance(result, dict) and result.get("status") != "skipped":
                    state.record_kernel_integrate_result(result)
                    if str(result.get("decision") or "").upper() == "KEEP":
                        await self._coord.writeback.record_integrate_keep(result)
            except Exception as exc:  # isolation boundary: one pending KEEP integrate, marked rejected
                log.exception(
                    "CYCLE_END entry: integrate(%s) raised %r; marking rejected to prevent drain loop deadlock",
                    kid,
                    exc,
                )
                self._reject_undispatchable_integrate(pending)
            state.save(self.session_dir)
            drained += 1
        if drained >= max_drain:
            log.warning(
                "CYCLE_END entry: drain cap (%d) reached; remaining pending KEEPs stay queued",
                max_drain,
            )
