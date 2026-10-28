# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply specialist patches to live framework roots and KEEP or REVERT by benchmark."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

from ._grid_runner import DEFAULT_KEEP_THRESHOLD_PCT, SessionDirField
from ._integrate_apply import ApplyStageMixin
from ._integrate_attempt import IntegrateAttempt
from ._integrate_bench import BenchStageMixin
from ._integrate_enablement_gate import EnablementGateMixin
from ._integrate_perf_gate import PerfGateMixin
from ._integrate_resolve import ResolveStageMixin
from ._integrate_vocabulary import KEEP_STATUSES, RESULT_BENCHED
from ._patch_git import restore_pending_integrate


log = logging.getLogger(__name__)


class IntegratePatchExecutor(
    ApplyStageMixin,
    BenchStageMixin,
    EnablementGateMixin,
    PerfGateMixin,
    ResolveStageMixin,
):
    """ActionRunner for the ``integrate_patch`` action (PR-A4)."""

    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the integrate-patch executor.

        Args:
            session_dir (Path | str | None): Session output directory;
                auto-resolved when ``None``.
            default_config_path (Path | str | None): Fallback benchmark
                config path, if any.
            keep_threshold_pct (float): Minimum gain to KEEP a patch.
                Defaults to :data:`DEFAULT_KEEP_THRESHOLD_PCT`.
        """
        self.session_dir = session_dir
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Apply a specialist's patches/config changes and benchmark them."""
        attempt = IntegrateAttempt(task_id=ctx.task.task_id)
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}

        early = await self._stage_resolve(attempt, params, extra)
        if early is not None:
            return early

        # Provision an attempt-scoped runtime AFTER the Critic gate (in
        # _stage_resolve) and BEFORE any patch apply / setup replay.
        provision_early = await self._stage_provision_attempt_runtime(attempt, params, attempt.specialist_task_id)
        if provision_early is not None:
            return provision_early

        await self._stage_localize_source(attempt, params)

        # The apply and the gate both mutate the framework tree behind the
        # operator's auto-stash, and both cross awaits while it is on the stack --
        # the apply stage writes a KB record on each of its failure verdicts. So
        # the guard spans both: whichever stage was running, the candidate is
        # taken back out and the stash handed back, and the stop is re-raised
        # rather than graded. The attempt records mutation as it happens, even
        # when a stage cannot return because it was cancelled.
        try:
            result = await self._stage_apply(attempt, params, extra)
            if result is None:
                result = {**await self._stage_gate(attempt, params), RESULT_BENCHED: True}
        except BaseException:
            try:
                self._finish_attempt(attempt, {"status": "cancelled"})
            except (OSError, AttributeError) as exc:
                log.error("integrate_patch: could not persist cancelled attempt recovery: %s", exc)
            raise
        return self._finish_attempt(attempt, result)

    def _finish_attempt(self, attempt: IntegrateAttempt, result: dict[str, Any]) -> dict[str, Any]:
        """Discharge the attempt's restore obligations exactly once before returning."""
        applied_only = result.get("status") == "applied_no_bench"
        accepted = result.get("status") in KEEP_STATUSES
        summary = (
            restore_pending_integrate(attempt.pending, keep=accepted or applied_only)
            if attempt.pending
            else {"failed": [], "artifacts_reverted": []}
        )
        if applied_only and attempt.pending and not summary["failed"]:
            recovery = attempt.pending["recovery"]
            recovery["phase"] = "applied"
            if recovery.get("stash_oid"):
                recovery["phase"] = "applied_with_restored_stash"
        state = attempt.shared_state
        if state is not None:
            state.save(self.session_dir)
        if summary["failed"]:
            if state is not None and hasattr(state, "set_stop_reason"):
                # Not environment_fault: the host is fine, the tree this attempt
                # patched is not, and the report for environment_fault tells the
                # operator to look at the install instead.
                state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
                state.save(self.session_dir)
            return {
                **result,
                "status": "failed",
                "error_class": "integrate_restore_incomplete",
                "recovery_errors": summary["failed"],
            }
        if not accepted and not applied_only:
            result["patches_reverted"] = [str(patch) for patch in attempt.applied]
            if attempt.applied_artifacts:
                result["artifacts_reverted"] = summary["artifacts_reverted"]
            root = attempt.attempt_venv_root
            if root:
                self._gc_attempt_dir(Path(root).parent)
        return result


__all__ = ["IntegratePatchExecutor"]
