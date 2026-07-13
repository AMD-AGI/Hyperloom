# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SWEEP phase handler: auto-enqueue of the concurrency-sweep task."""

from __future__ import annotations
import logging as _logging
from typing import Any
from ..state.shared_state import SharedState
from ..state.task_registry import Task
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class SweepPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _on_enter_sweep(self, *, from_phase: str) -> None:
        """Auto-enqueue a ``conc_sweep`` task on SWEEP entry.

        The historical full workload ``sweep`` remains available as a manual
        executor, but the automatic phase path now runs the baseline-vs-current
        concurrency curve directly.

        Args:
            from_phase: The phase being left, used only for logging.
        """
        state = self.shared_state
        # Drain pending KEEP integrates from prior KERNEL so sweep measures full current_best.
        if getattr(state, "has_keep_pending_integrate", False):
            await self._drain_pending_keep_integrates()
        # Always attempt stack validation for positive NEEDS_REVIEW kernels,
        # regardless of whether there were pending KEEPs to drain.
        await self._maybe_validate_positive_needs_review_stack()
        if not getattr(state, "conc_sweep_enabled", False):
            log.info(
                "SWEEP entry (from=%s): conc_sweep disabled; no automatic workload sweep is enqueued.",
                from_phase or "<unknown>",
            )
            self._record_phase_entry_evidence(
                auto_conc_sweep_skipped="disabled",
            )
            return
        try:
            task = await self._enqueue_internal_conc_sweep_task(
                reason="phase_entry",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "SWEEP entry hook: failed to enqueue auto-conc-sweep: %r",
                exc,
            )
            self._record_phase_entry_evidence(auto_conc_sweep_error=repr(exc)[:240])
            return
        if task is None:
            self._record_phase_entry_evidence(auto_conc_sweep_error="enqueue_returned_none")
            return
        log.info(
            "SWEEP entry (from=%s): auto-enqueued conc_sweep task=%s (concs=%s total_budget_sec=%s)",
            from_phase or "<unknown>",
            task.task_id,
            task.params.get("concs") or [],
            task.params.get("total_budget_sec"),
        )
        self._record_phase_entry_evidence(
            auto_conc_sweep_enqueued=True,
            auto_conc_sweep_task_id=task.task_id,
            auto_conc_sweep_concs=list(task.params.get("concs") or []),
        )

    async def _enqueue_internal_conc_sweep_task(
        self,
        *,
        reason: str,
    ) -> Task | None:
        """Build + enqueue a Coordinator-internal ``conc_sweep`` task (caller checks conc_sweep_enabled). Idempotency key + PolicyGate singleton ensure ≤1 per SWEEP; returns None on error.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            The created (or existing) ``conc_sweep`` task, or ``None`` on
            enqueue error.
        """
        state = self.shared_state
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "concs": list(state.conc_sweep_concs or []),
            "variant_timeout_sec": int(state.conc_sweep_variant_timeout_sec or 0),
            "total_budget_sec": int(state.conc_sweep_total_budget_sec or 0),
        }
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="conc_sweep",
                params=params,
                idempotency_key=f"internal-conc_sweep-{reason}{self._cycle_idem_suffix()}",
                # lease_ttl matches total_budget_sec so a multi-hour conc_sweep doesn't expire mid-flight.
                lease_ttl_sec=int(state.conc_sweep_total_budget_sec or 9000),
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "conc_sweep: failed to enqueue internal task: %r",
                exc,
            )
            return None
        if was_existing:
            log.info(
                "internal-conc_sweep task already exists (idempotent: task_id=%s, state=%s)",
                task.task_id,
                task.state,
            )
        else:
            log.info(
                "internal-conc_sweep task enqueued (task_id=%s reason=%s concs=%s total_budget_sec=%s)",
                task.task_id,
                reason,
                params["concs"],
                params["total_budget_sec"],
            )
        # Stamp evidence so PolicyGate's conc_sweep_phase_singleton denies later LLM conc_sweep.
        self._record_phase_entry_evidence(auto_conc_sweep_task_id=task.task_id)
        return task

    async def _enqueue_internal_sweep_task(
        self,
        *,
        reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``sweep`` task. Grid priority: warm_start_recipe.sweep_grid then SKILL.md defaults. Idempotency key internal-sweep-<reason>.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            The created (or existing idempotent) ``sweep`` :class:`Task`.
        """
        state = self.shared_state
        grid_params = self._build_sweep_params_from_recipe(state)
        params: dict[str, Any] = {
            "source": grid_params["source"],
            "reason": str(reason),
            "conc_values": list(grid_params["conc_values"]),
            "isl_osl_configs": list(grid_params["isl_osl_configs"]),
            "num_prompts_factor": int(grid_params["num_prompts_factor"]),
        }
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        # GEAK-owned KERNEL: hand the e2e result to the sweep so it reuses
        # GEAK's bench_e2e.sh + overlay instead of relaunching via Magpie.
        ps_result = getattr(state, "geak_result", None) or {}
        if isinstance(ps_result, dict) and ps_result.get("status") == "ok" \
                and ps_result.get("bench_script"):
            params["geak_result"] = ps_result
        cb = state.current_best or {}
        if isinstance(cb, dict):
            cb_args = str(cb.get("extra_server_args") or "")
            if cb_args:
                params["base_extra_args"] = cb_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            # Mirror baseline's benchmark_script so re-launch uses the same wrapper.
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="sweep",
            params=params,
            idempotency_key=f"internal-sweep-{reason}{self._cycle_idem_suffix()}",
        )
        if was_existing:
            log.info(
                "internal-sweep task already exists (idempotent: task_id=%s, state=%s)",
                task.task_id,
                task.state,
            )
        return task

    @staticmethod
    def _build_sweep_params_from_recipe(state: SharedState) -> dict[str, Any]:
        """Pick a sweep grid: warm_start_recipe.sweep_grid takes precedence over SKILL.md defaults; per-field fallback. Returns source/conc_values/isl_osl_configs/num_prompts_factor.

        Args:
            state: The session SharedState whose ``warm_start_recipe`` may carry
                a ``sweep_grid`` override.

        Returns:
            A dict with ``source`` (``"cortex_recipe"`` or
            ``"skill_md_default"``), ``conc_values``, ``isl_osl_configs`` and
            ``num_prompts_factor``.
        """
        from ..actions.executors.sweep import (
            DEFAULT_CONC_VALUES,
            DEFAULT_ISL_OSL,
            DEFAULT_NUM_PROMPTS_FACTOR,
        )

        recipe = getattr(state, "warm_start_recipe", None)
        sweep_grid = None
        if isinstance(recipe, dict):
            sg = recipe.get("sweep_grid")
            if isinstance(sg, dict):
                sweep_grid = sg

        def _coerce_int_list(value: Any) -> list[int] | None:
            """Coerce a recipe value into a non-empty list of ints.

            Args:
                value (Any): The raw recipe field (expected: list of ints).

            Returns:
                list[int] | None: The coerced ints, or ``None`` if ``value`` is
                    not a non-empty all-int list.
            """
            if not isinstance(value, list) or not value:
                return None
            out: list[int] = []
            for v in value:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    return None
            return out if out else None

        def _coerce_isl_osl_list(value: Any) -> list[str] | None:
            """Coerce a recipe value into a list of ``"<ISL>:<OSL>"`` strings.

            Accepts either ``"<ISL>:<OSL>"`` strings or ``[isl, osl]`` pairs.

            Args:
                value (Any): The raw recipe field.

            Returns:
                list[str] | None: Normalized ISL:OSL strings, or ``None`` if the
                    value is not a recognisable non-empty list.
            """
            if not isinstance(value, list) or not value:
                return None
            out: list[str] = []
            for v in value:
                # Accept either "<ISL>:<OSL>" strings or [isl, osl] pairs.
                if isinstance(v, str) and ":" in v:
                    out.append(v)
                    continue
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, str)) for x in v):
                    out.append(f"{int(v[0])}:{int(v[1])}")
                    continue
                return None
            return out if out else None

        conc_values: list[int] = list(DEFAULT_CONC_VALUES)
        isl_osl_configs: list[str] = list(DEFAULT_ISL_OSL)
        num_prompts_factor: int = int(DEFAULT_NUM_PROMPTS_FACTOR)
        used_recipe = False

        if sweep_grid is not None:
            cv = _coerce_int_list(sweep_grid.get("conc_values"))
            if cv is not None:
                conc_values = cv
                used_recipe = True
            io = _coerce_isl_osl_list(sweep_grid.get("isl_osl_configs"))
            if io is not None:
                isl_osl_configs = io
                used_recipe = True
            npf_raw = sweep_grid.get("num_prompts_factor")
            try:
                npf = int(npf_raw) if npf_raw is not None else None
            except (TypeError, ValueError):
                npf = None
            if npf is not None and npf > 0:
                num_prompts_factor = npf
                used_recipe = True
            if not used_recipe:
                log.warning(
                    "sweep recipe present but unusable (no recognisable fields); falling back to SKILL.md defaults"
                )

        return {
            "source": "cortex_recipe" if used_recipe else "skill_md_default",
            "conc_values": conc_values,
            "isl_osl_configs": isl_osl_configs,
            "num_prompts_factor": num_prompts_factor,
        }
