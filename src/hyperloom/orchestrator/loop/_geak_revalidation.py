# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GEAK 2b revalidation: the rebench it dispatches, and the ruling that rebench earns.

A GEAK end-to-end win is accepted on GEAK's own measurement, so the orchestrator
re-measures it on its own harness before any of it reaches ``current_best``. This
module owns that round: the params it is dispatched with, how its result is read,
and what each reading is allowed to publish.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..actions.executors._accuracy_gate import EVAL_KIND_ACCURACY_UNAVAILABLE
from ..actions.executors.explore import STACK_REVALIDATE_SOURCE
from ..collaborator import CoordinatorCollaborator
from ..kernel.geak_config import (
    _accepted_config_as_variant,
    _accepted_config_controls,
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    _geak_overlay_digest,
    _geak_result_has_material,
    _geak_revalidation_decision,
    _normalize_geak_overlay_dir,
    geak_overlay_is_loadable,
    geak_spec_name,
)
from ..state.shared_state import resolve_graded_comparison, stack_base_params
from ..state.task_registry import Task

if TYPE_CHECKING:
    from .writeback import _PromoteOutcome

log = _logging.getLogger(__name__)


# Minimum over-baseline gain a same-harness revalidation must show to count as "engaged"; detects a collapse back to
# ~baseline.
_MIN_KERNEL_ENGAGED_GAIN_PCT: float = 2.0


class _GeakRebench(NamedTuple):
    """A GEAK 2b rebench: what it measured, and what it measured against.

    ``current_best`` and ``candidate`` are read once, before the ruling can
    promote and move them, so every arm of the ruling judges the same state the
    rebench was measured against.
    """

    variant: Mapping[str, Any]
    measurement: dict[str, Any]
    measured: float
    measured_ok: bool
    got_hash: str
    current_best: Mapping[str, Any]
    current_best_tput: Any
    candidate: Mapping[str, Any]


class _OverlayReading(NamedTuple):
    """Whether the overlay a 2b rebench expected was in the measurement."""

    expected: str
    loaded: bool
    digest: str
    decision: str


class _RebenchDecision(NamedTuple):
    """What a 2b rebench decided, and the native refusal behind it if any."""

    decision: str
    native_rejection: str


class _ReplayRuling(NamedTuple):
    """How a GEAK-harness replay ruled a revalidation it was handed."""

    decision: str
    promoted: bool


class _MaterialVerdict(NamedTuple):
    """Whether a validated GEAK rebench has material product behind it."""

    decision: str
    rejection: str | None = None


class GeakRevalidation(CoordinatorCollaborator):
    """Dispatches the GEAK 2b rebench and rules on what it measured."""

    def geak_rebench_params(self, *, reason: str) -> dict[str, Any]:
        """Build canonical revalidation params, or explain why a product cannot be deployed.

        Native source products use reviewed integration; configuration and overlay
        products use the existing explore executor. ``reason`` is recorded in the task.
        """
        benchmark_script = self.shared_state.accepted_baseline_script()
        recipe_generation = int(self.shared_state.working_recipe_generation or 0)
        ps = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        ps_cfg = ps.get("accepted_config") or {}
        ps_overlay = _normalize_geak_overlay_dir(str(ps.get("final_overlay") or "").strip())
        ps_admissible = str(ps.get("status") or "") == "ok" or _geak_has_accepted_kernel(ps)
        try:
            ps_controls = _accepted_config_controls(
                ps_cfg, inherited_remove_args=self.current_best_launch_config()["remove_args"]
            )
            ps_flags, ps_envs = _accepted_config_as_variant(ps_cfg)
            ps_has_material = ps_admissible and _geak_result_has_material(ps)
        except ValueError as exc:
            self._coord.phase_kernel.reject_geak_promotion(
                ps, measured_tput=0.0, current_best_tput=0.0, reason=str(exc)
            )
            self.shared_state.save(self.session_dir)
            return {"skipped": True, "reason": "geak_invalid_config"}
        if not (
            ps_admissible
            and (
                ps_cfg.get("flags")
                or ps_cfg.get("env")
                or "env_map" in ps_cfg
                or ps_controls
                or ps_overlay
                or ps_has_material
            )
        ):
            return {"skipped": True, "reason": "geak_no_material"}

        from ..actions.executors._proposal_identity import effective_fingerprint

        from ..phases.geak_native_revalidation import source_rebench_params

        source_params = source_rebench_params(
            self._coord,
            reason=reason,
            result=ps,
            launch={"extra_server_args": ps_flags, "extra_envs": dict(ps_envs), **ps_controls},
            overlay=ps_overlay,
        )
        if source_params is not None:
            return source_params

        overlay_requested = bool(ps_overlay)
        if ps_overlay and not geak_overlay_is_loadable(ps_overlay):
            log.warning(
                "geak 2b: overlay %r is not loadable (no sitecustomize.py); "
                "revalidating the config WITHOUT the authored kernel",
                ps_overlay,
            )
            ps_overlay = ""
        if not (ps_flags or ps_envs or ps_controls or ps_overlay):
            if not ps_has_material:
                return {"skipped": True, "reason": "geak_no_material"}
            return {
                "skipped": True,
                "reason": "geak_overlay_unloadable" if overlay_requested else "geak_material_requires_harness",
                "fallback": "geak_harness",
            }
        base_params = stack_base_params({**self.shared_state.current_best, **self.current_best_launch_config()})
        base_params = {
            key: value
            for key, value in base_params.items()
            if key not in {"base_remove_args", "base_unset_envs", "base_args_mode"} or value
        }
        if ps_controls.get("args_mode") == "replace":
            base_params["base_extra_args"] = ""
            base_params["base_args_mode"] = "replace"
            base_params.pop("base_remove_args", None)
        expected_cfg_hash = effective_fingerprint(
            ps_flags,
            ps_envs,
            controls=ps_controls,
            base_remove_args=base_params.get("base_remove_args"),
            base_unset_envs=base_params.get("base_unset_envs"),
            base_args_mode=base_params.get("base_args_mode"),
        )
        expected_overlay_digest = _geak_overlay_digest(ps_overlay)
        ps_kernels = [geak_spec_name(k) for k in _geak_accepted_kernel_specs(ps)]
        params_ps: dict[str, Any] = {
            "source": STACK_REVALIDATE_SOURCE,
            "reason": reason,
            "recipe_generation": recipe_generation,
            "geak_fallback": True,
            "expected_cfg_hash": expected_cfg_hash,
            "expected_overlay": ps_overlay,
            "expected_overlay_digest": expected_overlay_digest,
            "grid": [
                {
                    "name": "geak_revalidate",
                    "extra_args": ps_flags,
                    "extra_envs": dict(ps_envs),
                    **ps_controls,
                    "overlay_pythonpath": ps_overlay,
                    "provenance": "geak_revalidate",
                    "accepted_kernels": ps_kernels if ps_overlay else [],
                    "note": "same-harness config-identity revalidation of the geak e2e win",
                }
            ],
            "base_tput": float(self.shared_state.baseline_tput or 0.0),
            **base_params,
        }
        if self.shared_state.baseline_config_path:
            params_ps["config_path"] = self.shared_state.baseline_config_path
        if benchmark_script:
            params_ps["benchmark_script"] = benchmark_script
        self._coord.proposals.inject_explore_runtime_params(params_ps)
        return params_ps

    async def _rule_on_geak_rebench(
        self,
        result: dict,
        *,
        task: "Task",
        outcome: _PromoteOutcome,
    ) -> None:
        """Rule on a GEAK (2b) revalidation rebench and publish what it decided.

        A GEAK revalidation must assert config identity and that the optimization
        engaged before it may stamp ``validated``, else the round replays through
        the GEAK harness (2a). That ruling is the whole of such a round, so the
        verdict is written onto ``outcome`` here rather than handed back to the
        winner-accept path a revalidation never reaches.
        """
        # Set by the validated promotion and by the harness replay; a ruling that
        # reaches neither leaves the round unpromoted.
        promoted = False
        rebench = self._read_geak_rebench(result)
        ruling = self._geak_rebench_decision(result, task=task, rebench=rebench)
        decision, native_rejection = ruling.decision, ruling.native_rejection
        # ``expected_cfg_hash`` fingerprints (args, envs) only, so it
        # cannot see the overlay drop out between dispatch and launch —
        # ``run_grid`` skips an overlay whose dir has gone away and logs
        # a warning, and the run then measures plain flags while the
        # credit still reads as a kernel win. Re-check the overlay's own
        # identity here; a miss is inconclusive, not validated.
        overlay = self._overlay_reading(task, decision=decision)
        expected_overlay, overlay_loaded = overlay.expected, overlay.loaded
        got_digest, decision = overlay.digest, overlay.decision
        if native_rejection:
            decision = "no_promote"
        if decision == "validated":
            verdict = self._material_verdict(rebench)
            decision = verdict.decision
            if verdict.rejection is not None:
                native_rejection = verdict.rejection

        pending = self.shared_state.geak_pending or {}
        pending_status = str(pending.get("status") or "") if isinstance(pending, dict) else ""
        if decision == "validated":
            # Only an overlay that was dispatched AND still matches its
            # manifest proves a kernel was in the measurement.
            promoted = self._promote_validated_rebench(
                rebench, overlay_loaded=bool(expected_overlay) and overlay_loaded
            )
            if not promoted:
                decision = "no_promote"
        if decision == "no_material":
            await self._drop_immaterial_rebench(rebench)
        elif decision in {"no_promote", "accuracy_drop", EVAL_KIND_ACCURACY_UNAVAILABLE, "intvty_regression"}:
            await self._refuse_rebench(
                result, task=task, rebench=rebench, decision=decision, native_rejection=native_rejection
            )
        elif decision == "fallback":
            replay = await self._replay_via_geak_harness(result, task=task, rebench=rebench)
            decision, promoted = replay.decision, replay.promoted
        expected_hash = str((task.params or {}).get("expected_cfg_hash") or "")
        self._record_geak_rebench_timeline(
            task=task,
            decision=decision,
            pending_status=pending_status,
            measured=rebench.measured,
            config_matched=(rebench.got_hash == expected_hash) if expected_hash else None,
            overlay_loaded=overlay_loaded if expected_overlay else None,
            got_hash=rebench.got_hash,
            got_overlay_digest=got_digest,
        )
        self.shared_state.record_action_attempt(
            action="explore",
            task_id=task.task_id,
            status=str(result.get("status") or "succeeded"),
            decision="promoted" if promoted else "no_promote" if decision == "no_promote" else "discarded",
            result=result,
            extras={"revalidation_decision": decision, "output_throughput": rebench.measured},
        )
        outcome.verdict = Verdict.ADOPTED if promoted else Verdict.REFUSED
        outcome.changed = True

    def _read_geak_rebench(self, result: dict) -> _GeakRebench:
        """Read a 2b rebench round: the variant measured, and what it measured against.

        A variant carries its measurement under one of three keys depending on how
        it was produced, and a measurement with no throughput of its own falls back
        to the round's -- so a rebench is never graded against a missing number.
        ``measured_ok`` judges the round's own throughput, not the rebench's: a
        round that measured nothing says nothing, whatever the variant carries.
        """
        from hyperloom.common.perf_metric import output_tput_of

        best_winner = result.get("best_variant")
        winners = result.get("winners") or []
        round_tput = result.get("output_throughput")
        rebench_variant = (
            best_winner
            if isinstance(best_winner, Mapping)
            else next((winner for winner in winners if isinstance(winner, Mapping)), {})
        )
        rebench_measurement = rebench_variant.get("bench_result")
        if not isinstance(rebench_measurement, Mapping):
            rebench_measurement = rebench_variant.get("measurement")
        if not isinstance(rebench_measurement, Mapping):
            rebench_measurement = rebench_variant
        rebench_measurement = dict(rebench_measurement)
        if "output_throughput" not in rebench_measurement and "tput" not in rebench_measurement:
            rebench_measurement["output_throughput"] = round_tput
        current_best = self.shared_state.current_best
        return _GeakRebench(
            variant=rebench_variant,
            measurement=rebench_measurement,
            measured=output_tput_of(rebench_measurement),
            measured_ok=isinstance(round_tput, (int, float)) and round_tput > 0,
            got_hash=str(rebench_variant.get("fingerprint") or ""),
            current_best=current_best,
            current_best_tput=current_best.get("tput"),
            candidate=self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {},
        )

    @staticmethod
    def _overlay_reading(
        task: "Task",
        *,
        decision: str,
    ) -> _OverlayReading:
        """Read whether the kernel overlay a 2b rebench expected survived the run.

        An overlay that no longer loads, or whose digest moved, means the kernel
        was not in the measurement -- which sends an otherwise validated round to
        the harness replay rather than letting it stamp a kernel gain.
        """
        expected_overlay = str((task.params or {}).get("expected_overlay") or "")
        overlay_loaded = True
        got_digest = ""
        if expected_overlay:
            expected_digest = str((task.params or {}).get("expected_overlay_digest") or "")
            got_digest = _geak_overlay_digest(expected_overlay)
            overlay_loaded = geak_overlay_is_loadable(expected_overlay) and (got_digest == expected_digest)
            if not overlay_loaded and decision == "validated":
                log.warning(
                    "geak 2b: overlay %r did not survive the run "
                    "(loadable=%r digest expected=%r got=%r) -> 2a fallback",
                    expected_overlay,
                    geak_overlay_is_loadable(expected_overlay),
                    expected_digest,
                    got_digest,
                )
                decision = "fallback"
        return _OverlayReading(expected=expected_overlay, loaded=overlay_loaded, digest=got_digest, decision=decision)

    def _geak_rebench_decision(
        self,
        result: dict,
        *,
        task: "Task",
        rebench: _GeakRebench,
    ) -> _RebenchDecision:
        """Read what a GEAK 2b rebench decided about the stack it re-measured.

        Config identity and an engaged optimization are both required before a
        rebench may read as ``validated``; an incomparable grade or a losing
        objective sends the round to the harness replay instead.
        """
        measured, got_hash = rebench.measured, rebench.got_hash
        expected_hash = str((task.params or {}).get("expected_cfg_hash") or "")
        baseline_grade = resolve_graded_comparison(self.shared_state, rebench.measurement, against_baseline=True)
        current_grade = resolve_graded_comparison(self.shared_state, rebench.measurement)
        decision = _geak_revalidation_decision(
            measured=baseline_grade.candidate,
            baseline=baseline_grade.reference,
            got_hash=got_hash,
            expected_hash=expected_hash,
            min_engaged_gain_pct=_MIN_KERNEL_ENGAGED_GAIN_PCT,
            current_best=current_grade.reference,
        )
        if (
            measured <= 0
            or not baseline_grade.comparable
            or not current_grade.comparable
            or baseline_grade.objective != current_grade.objective
            or str(result.get("status") or "succeeded") not in {"succeeded", "ok"}
        ):
            decision = "fallback"
        elif (not expected_hash or got_hash == expected_hash) and any(
            grade.graded_on_intvty and grade.verdict != "KEEP" for grade in (baseline_grade, current_grade)
        ):
            decision = "no_promote"
        native_rejection = next(
            (
                str(entry.get("reason") or "native_gate_rejected")
                for entry in result.get("per_variant_outcomes") or []
                if isinstance(entry, Mapping)
                and entry.get("outcome") == "REVERT"
                and (
                    any(
                        isinstance(gate, Mapping)
                        and gate.get("gate") in {"graded_axes", "accuracy"}
                        and gate.get("passed") is False
                        for gate in entry.get("gates") or []
                    )
                    or (
                        "gates" not in entry
                        and entry.get("reason")
                        in {"accuracy_drop", EVAL_KIND_ACCURACY_UNAVAILABLE, "intvty_regression"}
                    )
                )
                and (not expected_hash or entry.get("fingerprint") == expected_hash)
            ),
            "",
        )
        if (
            measured > 0
            and got_hash == expected_hash
            and (not baseline_grade.comparable or not current_grade.comparable)
        ):
            decision = "no_promote"
        return _RebenchDecision(decision=decision, native_rejection=native_rejection)

    async def _drop_immaterial_rebench(self, rebench: _GeakRebench) -> None:
        """Record a rebench that beat current_best with no GEAK product behind it.

        Same-config measurement noise: the headline, the stack and the gain are
        left alone and the candidate is cleared.
        """
        candidate, measured = rebench.candidate, rebench.measured
        current_best_tput = rebench.current_best_tput
        log.info(
            "geak 2b rebench beat current_best but GEAK shipped no "
            "material product (measured=%r current_best=%r) -> "
            "no_material drop",
            measured,
            current_best_tput,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "geak_no_material",
                "measured_tput": float(measured),
                "current_best_tput": (
                    float(current_best_tput) if isinstance(current_best_tput, (int, float)) else None
                ),
                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
            },
        )
        # Stamp the drop on geak_result (always, so an empty {} is
        # distinguishable from never-populated on resume/debug and
        # acts as a tombstone against KERNEL crash-recovery
        # re-enqueue) and reject any provisional KEEP in
        # kernel_journey so a session audit does not read a dropped
        # candidate as an accepted kernel (no-op when the journey
        # has no KEEP / the file is absent).
        ps_stamped = dict(candidate) if isinstance(candidate, dict) else {}
        ps_stamped["revalidation_status"] = "no_material"
        self.shared_state.geak_result = ps_stamped
        self._record_geak_rebench_conclusion(final_status="no_material")
        self._coord.phase_kernel.reject_geak_kernel_journey(
            ps_stamped,
            measured_tput=float(measured),
            current_best_tput=(float(current_best_tput) if isinstance(current_best_tput, (int, float)) else 0.0),
            provenance="geak_no_material",
            rejection_reason="geak_no_material_product",
        )
        self.shared_state.geak_pending = {}

    async def _refuse_rebench(
        self,
        result: dict,
        *,
        task: "Task",
        rebench: _GeakRebench,
        decision: str,
        native_rejection: str,
    ) -> None:
        """Record a revalidation the rebench refused, and why it was refused."""
        measured, measured_ok = rebench.measured, rebench.measured_ok
        candidate, current_best_tput = rebench.candidate, rebench.current_best_tput
        log.info(
            "geak 2b rebench rejected (decision=%s measured=%r current_best=%r)",
            decision,
            measured,
            current_best_tput,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": f"geak_{decision}",
                "measured_tput": float(measured) if measured_ok else None,
                "current_best_tput": (
                    float(current_best_tput) if isinstance(current_best_tput, (int, float)) else None
                ),
                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
            },
        )
        # Persist the closed verdict so a later KERNEL entry does
        # not recover stale result.json and re-enqueue this already
        # adjudicated candidate (#1240).
        self._coord.phase_kernel.reject_geak_promotion(
            candidate,
            measured_tput=float(measured) if measured_ok else 0.0,
            current_best_tput=float(current_best_tput or 0.0),
            reason=native_rejection or "graded_comparison_rejected",
            attempt_id=str(task.idempotency_key or task.task_id),
        )
        result["status"] = "no_promote"

    def _promote_validated_rebench(self, rebench: _GeakRebench, *, overlay_loaded: bool) -> bool:
        """Write the headline a validated orchestrator-harness rebench earns.

        Lifts current_best + optimization_stack + the validated gain and clears
        ``geak_pending``. The candidate's own throughput fields give way to the
        rebench's, so the headline reports what was just measured rather than what
        the candidate claimed when it was accepted.

        Returns:
            Whether the promotion landed.
        """
        measurement, variant = rebench.measurement, rebench.variant
        promotion_result = dict(rebench.candidate)
        for key in (
            "output_throughput",
            "input_throughput",
            "total_throughput",
            "total_token_throughput",
            "e2e_norm_intvty_p90",
            "tpot_p90_ms",
            "submission_valid",
            "submission_invalid_reasons",
        ):
            promotion_result.pop(key, None)
            if key in measurement:
                promotion_result[key] = measurement[key]
        return self._coord.phase_kernel.promote_geak_from_candidate(
            promotion_result,
            measured_tput=float(rebench.measured),
            provenance="geak_orch_harness_validated",
            overlay_loaded=overlay_loaded,
            measurement_provenance={**variant, **measurement},
        )

    def _material_verdict(self, rebench: _GeakRebench) -> _MaterialVerdict:
        """Rule on whether a validated GEAK rebench has material product behind it.

        A rebench that beats current_best is only a KERNEL gain when
        GEAK actually produced something. Without a material product
        (kernel/head/overlay/patch or a config delta vs the pre-KERNEL
        best) the win is same-config measurement noise; drop it.
        An empty geak_result cannot be judged by the helper, so
        disambiguate here: a pre-existing ``geak_e2e`` stack entry
        means this is a resume revalidation of an already-material win
        (let it through); otherwise there is no material to validate.
        """
        pending_summary, current_best = rebench.candidate, rebench.current_best
        stack_now = self.shared_state.optimization_stack or []
        has_prior_geak_e2e = any(isinstance(e, dict) and e.get("action") == "geak_e2e" for e in stack_now)
        # Escape hatch first: a pre-existing geak_e2e stack entry is
        # an already-proven win, so this is a resume revalidation.
        # It must short-circuit the material check regardless of
        # geak_result (which is persisted and thus non-empty on
        # resume) — by now current_best already carries the GEAK
        # accepted_config, so the fingerprint would match and be
        # mis-judged no_material, reverting a real win. Only apply
        # the material gate on the FIRST validation (no prior entry),
        # where current_best is still the pre-KERNEL config.
        if not has_prior_geak_e2e:
            if not pending_summary:
                return _MaterialVerdict(decision="no_material")
            try:
                _accepted_config_as_variant(pending_summary.get("accepted_config"))
                has_material = _geak_result_has_material(
                    pending_summary,
                    prev_best_flags=str(current_best.get("extra_server_args") or ""),
                    prev_best_envs=current_best.get("extra_envs") or {},
                    prev_best_controls={**current_best, **self.current_best_launch_config()},
                )
            except ValueError as exc:
                return _MaterialVerdict(
                    decision="no_promote",
                    rejection=f"invalid_accepted_config: {exc}",
                )
            else:
                if not has_material:
                    return _MaterialVerdict(decision="no_material")
        return _MaterialVerdict(decision="validated")

    def _record_geak_rebench_timeline(
        self,
        *,
        task: Any,
        decision: str,
        pending_status: str,
        measured: Any,
        config_matched: bool | None,
        overlay_loaded: bool | None,
        got_hash: str,
        got_overlay_digest: str,
    ) -> None:
        """Record one GEAK rebench attempt on the kernel timeline event.

        The attempt is recorded whatever the verdict, including one the slot
        ownership check goes on to reject: a measured rebench that was dropped
        as orphaned or late is exactly the case that is hard to reconstruct
        afterwards.

        Args:
            task: The settled rebench task.
            decision: The final verdict.
            pending_status: The candidate slot's status.
            measured: The throughput the attempt measured.
            config_matched: Whether the config fingerprint survived the run.
            overlay_loaded: Whether the overlay survived the run.
            got_hash: The fingerprint the run reported.
            got_overlay_digest: The overlay digest observed after the run.
        """
        recorder = self._coord.phase_kernel.timeline()
        if recorder is None:
            return
        params = task.params or {}
        recorder.record_geak_rebench_attempt(
            attempt_id=str(task.idempotency_key or task.task_id or ""),
            source_ref=None,
            idempotency_key=str(task.idempotency_key or ""),
            task_id=str(task.task_id or ""),
            dispatched_at=None,
            settled_at=None,
            base_tput=params.get("base_tput"),
            measured_tput=measured,
            decision=decision,
            decision_reason=str(params.get("reason") or ""),
            status=pending_status,
            engagement={
                "config_matched": config_matched,
                "overlay_loaded": overlay_loaded,
                "expected_cfg_hash": params.get("expected_cfg_hash"),
                "observed_cfg_hash": got_hash,
                "expected_overlay_digest": params.get("expected_overlay_digest"),
                "observed_overlay_digest": got_overlay_digest,
            },
        )

    def _record_geak_rebench_conclusion(
        self,
        *,
        final_status: str,
        final_error_class: str = "",
        final_error: str = "",
    ) -> None:
        """Record the terminal revalidation verdict on the kernel timeline event.

        The points that stamp a closed verdict onto ``geak_result`` also release
        the candidate slot, so the verdict has nowhere else to be read from
        afterwards. The recorder declines silently when the kernel event has
        already closed, and the close-out's ``geak_candidate`` carries it then.
        """
        recorder = self._coord.phase_kernel.timeline()
        if recorder is None:
            return
        recorder.record_geak_rebench_conclusion(
            final_status=final_status,
            final_error_class=final_error_class,
            final_error=final_error,
        )
