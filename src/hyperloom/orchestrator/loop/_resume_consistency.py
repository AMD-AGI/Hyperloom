# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one-shot resume audit of stack and current_best, and what it re-enqueues."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.inferencex_checkout import has_benchmark_lib

from ..actions.executors.explore import STACK_REVALIDATE_SOURCE
from ._promote_outcome import IntegrateRecoveryIncomplete
from .resume_source_layers import install_kept_source_layers
from ..state.task_registry import TaskSpec

log = logging.getLogger(__name__)


class ResumeConsistencyMixin:
    """Auditing a resumed run for consistency."""

    async def _resume_consistency_pass(self) -> dict[str, Any]:
        """One-shot resume audit + recovery for stack/current_best consistency.

        Recovers half-applied / orphaned KEEPs through the same lift the live
        path uses, then compensates the validation watermark by enqueuing a
        single full-stack end-to-end rebench. Idempotent — only runs on a
        resumed session and every recovery step dedupes, so a second pass is a
        no-op.
        """
        if not self._resumed_from.get("is_resume"):
            return {"skipped": True, "reason": "not_resume"}
        state = self.shared_state
        report: dict[str, Any] = {
            "skipped": False,
            "fixes": [],
            "warnings": [],
        }
        active_inferencex = str(state.active_inferencex_path or "").strip()
        if active_inferencex:
            if has_benchmark_lib(active_inferencex):
                os.environ["INFERENCEX_PATH"] = active_inferencex
            elif Path(active_inferencex).is_dir():
                # Not an InferenceX tree: dropped, so the checkout this launch resolved stays the one every round uses.
                report["warnings"].append({"kind": "active_inferencex_checkout_invalid", "path": active_inferencex})
                state.active_inferencex_path = ""
            else:
                report["warnings"].append({"kind": "active_inferencex_checkout_missing", "path": active_inferencex})
                state.set_stop_reason("active_inferencex_checkout_missing")
        # Loading the state file is the one boundary the persisted
        # ``optimization_stack`` crosses, so bind its rows to their members here:
        # a row that cannot name what it integrated makes every later read of the
        # stack a guess, and every reader below inherits that guess.
        stack_phase = self._coord.phase_kernel_stack
        try:
            stack_phase.stack_resolved_kernel_ids()
        except ValueError as exc:
            stack_phase.halt_stack_recovery(exc)
        # (0) Interrupted stack attempt: its members may still be applied to the
        # framework tree, and everything this pass or the leg benchmarks before
        # SWEEP entry would measure them. SWEEP entry settles through the same
        # recovery.
        await self._resume_recover_interrupted_stack(report)
        # (1) Half-applied integrate window: replay the
        # missing stack append or roll back the partial patch BEFORE anything
        # reads the stack, so the rest of the pass sees the recovered truth.
        await self._resume_recover_pending_integrate(report)
        if state.pending_integrate:
            state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
            state.save(self.session_dir)
            raise IntegrateRecoveryIncomplete(
                "pending integrate recovery is incomplete; refusing resume before measurement"
            )
        # (1b) In-flight targeted build: an off-loop compile cannot survive a
        # coordinator restart, so kill the orphan group, GC its attempt dir,
        # sweep its jit locks, fail the row, and clear the sentinel.
        await self._resume_recover_pending_targeted_build(report)
        # (1c) Combined PRELUDE replay: no benchmark verdict survived the
        # restart, so restore both Recipe and Kernel trees before continuing.
        await self._resume_recover_pending_warm_replay(report)
        # (1d) Orphaned revalidation tasks: if enablement_validation_pending is set but the tracked revalidation task
        # is already terminal, unstick the window so a fresh revalidation can be enqueued -- closed and charged to the
        # stall counter, or reopened uncharged when the run cancelled the row.
        await self._resume_recover_pending_revalidation(report)
        # (2) Replay orphaned KEEPs that crashed before the append, then (2b) put the kept source patches on the tree.
        await self._resume_recover_orphaned_keeps(report)
        if halt := install_kept_source_layers(state, self.session_dir, report):
            raise IntegrateRecoveryIncomplete(halt)

        # (3) A config with no stack behind it cannot be reproduced.
        if state.current_best and not state.optimization_stack:
            report["warnings"].append({"kind": "current_best_without_stack"})

        self._publish_recovered_keeps(report)
        await self._compensate_validation_watermark(report)
        try:
            state.save(self.session_dir)
        except Exception:
            log.exception("Coordinator: resume consistency save failed")
        await self.bus.record_observation("coordinator", "observation", {"kind": "resume_consistency", **report})
        return report

    async def enqueue_internal_stack_rebench(
        self,
        *,
        reason: str,
        idempotency_key: str = "stack-revalidate",
    ) -> dict[str, Any]:
        """Enqueue one full-stack end-to-end rebench of the cumulative config.

        Builds a single-variant ``explore`` task from ``current_best``'s launch
        args/envs, benched against ``baseline_tput`` so the measured delta
        becomes the validated cumulative gain. Tagged ``source=stack_revalidate``
        so ``promote_to_shared_state`` reconciles ``cumulative_gain_validated_stack_len``
        from the measured throughput.

        Args:
            reason: Human-readable reason stamped on the task params.
            idempotency_key: Key for the native stack rebench.

        Returns:
            A summary ``{"task_id", "existing"}`` or ``{"skipped", "reason"}``.
        """
        benchmark_script = self.shared_state.accepted_baseline_script()
        # The grid is frozen from ``current_best`` now; a lift before the result lands makes it a measurement of an
        # older Recipe.
        recipe_generation = int(self.shared_state.working_recipe_generation or 0)
        launch = self.current_best_launch_config()
        args = launch["extra_server_args"]
        envs = launch["extra_envs"]
        overlay = launch["final_overlay"]
        cb_now = self.shared_state.current_best
        cb_remove = cb_now.get("remove_args")
        cb_unset = cb_now.get("unset_envs")
        cb_replace = str(cb_now.get("args_mode") or "").strip().lower() == "replace"
        if not (args or envs or cb_remove or cb_unset or cb_replace):
            return {"skipped": True, "reason": "empty_config"}
        params: dict[str, Any] = {
            "source": STACK_REVALIDATE_SOURCE,
            "reason": reason,
            "recipe_generation": recipe_generation,
            "grid": [
                {
                    "name": STACK_REVALIDATE_SOURCE,
                    "extra_args": args,
                    "extra_envs": dict(envs),
                    # Carry the overlay so an authored-kernel native stack rebuild
                    # loads the built kernels (inert when empty).
                    "overlay_pythonpath": overlay,
                    "provenance": STACK_REVALIDATE_SOURCE,
                    "note": "full-stack end-to-end revalidation",
                }
            ],
            # The rebench reproduces the whole stack, so its gain is cumulative-vs-baseline.
            "base_tput": float(self.shared_state.baseline_tput or 0.0),
        }
        if cb_remove:
            params["base_remove_args"] = [cb_remove] if isinstance(cb_remove, str) else list(cb_remove or [])
        if cb_unset:
            params["base_unset_envs"] = [cb_unset] if isinstance(cb_unset, str) else list(cb_unset or [])
        if cb_replace:
            params["base_args_mode"] = "replace"
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("explore")
        self._coord.proposals.inject_explore_runtime_params(params)
        task, existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        return {"task_id": task.task_id, "existing": bool(existing)}

    async def _resume_reenter_kernel_if_needed(self) -> None:
        """Idempotently re-fire the KERNEL_AGENT entry hook on resume.

        Phase-entry side effects (the ``kernel_agent`` task that carries the
        phase's work) are bound to a phase *transition* via
        ``_on_phase_entered``; a resume only restores ``phase`` from state.json
        and never re-enters the current phase. The resume reaps the task the
        dead leg held, so the next tick sees no agent in flight and leaves
        KERNEL as settled -- the whole round is silently lost.

        The decision is driven purely by whether THIS phase's round finished:

          * finished -> the phase machine exits KERNEL on the next tick with no
            re-run (GEAK: a ``geak`` completion record on the current phase row;
            Forge: this macro cycle's rewrite controller reported terminal);
          * not finished -> re-enter ``on_enter_kernel``, which reuses a live
            ``kernel_agent`` row or queues a replacement for the reaped one. A
            GEAK re-entry promotes an existing OK ``result.json`` and re-runs the
            e2e only when there is nothing to recover.

        No-op unless resumed while parked in ``KERNEL_AGENT``.
        """
        from ..phases.machine_state import PHASE_KERNEL_AGENT, kernel_controller_settled

        if not self._resumed_from.get("is_resume"):
            return
        state = self.shared_state
        if (state.phase or "").strip().upper() != PHASE_KERNEL_AGENT:
            return
        if not self._coord.phase_machine.kernel_enabled():
            return
        if self._coord.phase_kernel.geak_enabled():
            history = state.phase_history or []
            row = history[-1] if history else {}
            evidence = row.get("evidence") if isinstance(row, dict) else {}
            finished = isinstance(evidence, dict) and isinstance(evidence.get("geak"), dict)
        else:
            finished = kernel_controller_settled(state)
        if finished:
            log.info("resume: KERNEL round already finished this phase; machine will exit on next tick.")
            return
        log.info("resume: re-entering KERNEL (its round did not finish before the leg stopped).")
        from hyperloom.orchestrator.phases.machine import Transition

        await self._coord.phase_kernel.on_enter_kernel(
            Transition(from_phase="resume", to_phase="KERNEL_AGENT", reason="resume", evidence={})
        )
