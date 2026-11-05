# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_ControllerIntegrationMixin``: E2E integration of the Rewrite Controller's patches for the KERNEL phase."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable

from .kernel_round_budget import controller_round_stop, integrate_within_phase, round_stop

log = logging.getLogger("hyperloom.orchestrator.phases.kernel")


class _ControllerIntegrationMixin:
    """Seal the campaign baselines, integrate published Controller patches, and validate the sub-threshold bundle."""

    def _seal_campaign_baselines(self) -> dict[str, object]:
        try:
            from ..kernel.campaign_baseline import seal_campaign_baseline

            baselines = seal_campaign_baseline(
                self.shared_state,
                session_id=str(self.shared_state.session_id or self.session_dir.name),
                macro_cycle=self.shared_state.macro_cycle,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            log.exception("KERNEL entry: sealing the campaign baseline failed")
            return {}
        if baselines:
            log.info(
                "KERNEL entry: sealed campaign baselines %s",
                {repo: baseline.commit for repo, baseline in baselines.items()},
            )
        return baselines

    async def _integrate_unintegrated_attempts(self) -> bool:
        """E2E the patches earlier attempts of this cycle published before their host was killed; True on any KEEP."""
        from hyperloom.inference_optimizer.session.session_paths import unintegrated_forge_attempt_dirs

        kept = False
        for attempt_dir in unintegrated_forge_attempt_dirs(self.session_dir, self.shared_state.macro_cycle):
            patches_root = attempt_dir / "result" / "patches"
            log.info("KERNEL entry: integrating the patches %s published before it was killed", attempt_dir.name)
            result: dict[str, Any] = {
                "run_id": f"forge-cycle-{self.shared_state.macro_cycle}-{attempt_dir.name}",
                "patch_count": sum(not patch.name.startswith(".") for patch in patches_root.iterdir()),
                "patches_root": str(patches_root),
            }
            # No Controller timeout held a round back for these, so each one has to fit the phase.
            await self._integrate_controller_output(
                result,
                attempt_dir,
                admit_round=lambda rounds_run: controller_round_stop(self.shared_state, rounds_run + 1),
            )
            self._file_controller_integration_failure(result)
            kept = kept or result["integration"]["kept_count"] > 0
        return kept

    async def _integrate_controller_output(
        self,
        result: dict[str, Any],
        output_dir: Path,
        admit_round: Callable[[int], Any] | None = None,
    ) -> None:
        if int(result.get("patch_count") or 0) <= 0:
            result["integration"] = {
                "status": "not_run",
                "reason": "Controller published no patches",
                "kept_count": 0,
                "reverted_count": 0,
                "skipped_count": 0,
            }
            return
        from ..kernel.controller_patch_integration import KeepSession, integrate_controller_patches
        from ..kernel.subthreshold_bundle import defer_patch_win

        session = KeepSession(
            session_dir=self.session_dir,
            shared_state=self.shared_state,
            record_keep=self._coord.writeback.record_integrate_keep,
        )
        try:
            integration = await integrate_controller_patches(
                patches_root=str(result.get("patches_root") or output_dir / "result" / "patches"),
                session=session,
                admit_round=admit_round or (lambda rounds_run: controller_round_stop(self.shared_state, rounds_run)),
                on_unkept=lambda publication, validation: defer_patch_win(session, publication, validation),
            )
            result["integration"] = integration.to_dict()
        except Exception as error:  # isolation boundary: the Controller patch integration step, recorded as failed
            log.exception("KERNEL entry: Controller patch integration failed")
            result["integration"] = {"status": "failed", "reason": str(error), "kept_count": 0}

    def _file_controller_integration_failure(self, result: dict[str, Any]) -> None:
        integration = result["integration"]
        skipped = [
            str(row.get("reason") or "")
            for row in integration.get("results") or []
            if isinstance(row, dict) and str(row.get("status") or "").startswith("skipped")
        ]
        status = str(integration.get("status") or "")
        if status not in {"failed", "no_patch_admitted"} and not skipped:
            return
        cycle = self.shared_state.macro_cycle
        self.shared_state.record_action_failure(
            action="kernel_rewrite_controller",
            task_id=str(result.get("run_id") or f"forge-cycle-{cycle}"),
            result={
                "error_class": status or "patches_skipped",
                "error": "; ".join(x for x in ([str(integration.get("reason") or "")] + skipped) if x)[:800],
            },
        )

    async def _validate_subthreshold_bundle(self) -> None:
        from ..kernel.controller_patch_integration import KeepSession
        from ..kernel.subthreshold_bundle import validate_subthreshold_bundle

        await validate_subthreshold_bundle(
            KeepSession(self.session_dir, self.shared_state, self._coord.writeback.record_integrate_keep),
            measure=lambda payload: integrate_within_phase(
                payload, state=self.shared_state, session_dir=self.session_dir
            ),
            round_stop=lambda: round_stop(self.shared_state),
        )
