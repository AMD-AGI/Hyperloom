# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ruling on a GEAK end-to-end win: the 2b rebench, and the 2a harness replay behind it."""

from __future__ import annotations

import logging as _logging
from typing import Any, Mapping, NamedTuple

from hyperloom.common.env import env_int
from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_FAILED,
    REBENCH_FALLBACK,
    REBENCH_FALLBACK_FAILED,
    REBENCH_NO_MATERIAL,
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
)
from hyperloom.common.perf_metric import GradedComparison
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import GeakRebenchAttempt
from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..actions.executors._accuracy_gate import EVAL_KIND_ACCURACY_UNAVAILABLE, accuracy_passed
from ..actions.executors.explore import STACK_REVALIDATE_SOURCE
from ..collaborator import CoordinatorCollaborator
from ..kernel.geak_config import (
    _accepted_config_as_variant,
    _accepted_config_controls,
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    _geak_overlay_digest,
    _geak_result_has_artifact,
    _geak_revalidation_decision,
    _geak_sweep_measured_tput,
    _normalize_geak_overlay_dir,
    geak_overlay_is_loadable,
    geak_spec_name,
)
from ..state.shared_state import ActionAttempt, resolve_graded_comparison, stack_base_params
from ..state.task_registry import Task

log = _logging.getLogger(__name__)


class _GeakRebench(NamedTuple):
    variant: Mapping[str, Any]
    measurement: dict[str, Any]
    measured: float
    measured_ok: bool
    got_hash: str
    current_best: Mapping[str, Any]
    current_best_tput: Any
    candidate: Mapping[str, Any]


class _OverlayReading(NamedTuple):
    expected: str
    loaded: bool
    digest: str
    decision: str


class _RebenchDecision(NamedTuple):
    decision: str
    native_rejection: str


class _ReplayRuling(NamedTuple):
    decision: str
    promoted: bool


class _RebenchEngagement(NamedTuple):
    config_matched: bool | None
    overlay_loaded: bool | None
    observed_cfg_hash: str
    observed_overlay_digest: str


class _GeakProduct(NamedTuple):
    flags: str
    envs: Mapping[str, Any]
    controls: dict[str, Any]
    overlay: str
    has_artifact: bool


def _geak_rebench_grid_row(accepted: Mapping[str, Any], product: _GeakProduct, overlay: str) -> dict[str, Any]:
    kernels = [geak_spec_name(spec) for spec in _geak_accepted_kernel_specs(accepted)]
    return {
        "name": "geak_revalidate",
        "extra_args": product.flags,
        "extra_envs": dict(product.envs),
        **product.controls,
        "overlay_pythonpath": overlay,
        "provenance": "geak_revalidate",
        "accepted_kernels": kernels if overlay else [],
        "note": "same-harness config-identity revalidation of the geak e2e win",
    }


def _grades_can_rule(
    result: Mapping[str, Any],
    measured: float,
    baseline_grade: GradedComparison,
    current_grade: GradedComparison,
) -> bool:
    if measured <= 0 or str(result.get("status") or "succeeded") not in {"succeeded", "ok"}:
        return False
    return (
        baseline_grade.comparable and current_grade.comparable and baseline_grade.objective == current_grade.objective
    )


def _gate_refused(entry: Mapping[str, Any]) -> bool:
    if "gates" not in entry:
        return entry.get("reason") in {"accuracy_drop", EVAL_KIND_ACCURACY_UNAVAILABLE, "intvty_regression"}
    return any(
        isinstance(gate, Mapping) and gate.get("gate") in {"graded_axes", "accuracy"} and gate.get("passed") is False
        for gate in entry.get("gates") or []
    )


def _native_rejection_reason(result: Mapping[str, Any], expected_hash: str) -> str:
    return next(
        (
            str(entry.get("reason") or "native_gate_rejected")
            for entry in result.get("per_variant_outcomes") or []
            if isinstance(entry, Mapping)
            and entry.get("outcome") == "REVERT"
            and _gate_refused(entry)
            and (not expected_hash or entry.get("fingerprint") == expected_hash)
        ),
        "",
    )


class _ReplayRegime(NamedTuple):
    conc: int
    isl: int
    osl: int


class _ReplayOverlay(NamedTuple):
    path: str
    digest_before: str


def _promotion_measurement(res: Mapping[str, Any]) -> Mapping[str, Any]:
    measurement = res.get("promotion_measurement")
    return measurement if isinstance(measurement, Mapping) else {}


def _replay_regime(accepted: Mapping[str, Any]) -> _ReplayRegime:
    regimes = accepted.get("validated_regimes") or []
    first = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
    try:
        return _ReplayRegime(
            conc=int(first.get("conc") or 64),
            isl=int(first.get("isl") or 1024),
            osl=int(first.get("osl") or 1024),
        )
    except (TypeError, ValueError):
        return _ReplayRegime(conc=64, isl=1024, osl=1024)


def _overlay_survived_replay(overlay: _ReplayOverlay) -> bool:
    if not overlay.path or not geak_overlay_is_loadable(overlay.path):
        return False
    return not overlay.digest_before or _geak_overlay_digest(overlay.path) == overlay.digest_before


class GeakRevalidation(CoordinatorCollaborator):
    def geak_rebench_params(self, *, reason: str) -> dict[str, Any]:
        from ..phases.geak_native_revalidation import source_rebench_params

        accepted = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        try:
            product = self._accepted_geak_product(accepted)
        except ValueError as exc:
            self._coord.phase_kernel.reject_geak_promotion(
                accepted, measured_tput=0.0, current_best_tput=0.0, reason=str(exc)
            )
            self.shared_state.save(self.session_dir)
            return {"skipped": True, "reason": "geak_invalid_config"}
        if product is None:
            return {"skipped": True, "reason": "geak_no_material"}
        source_params = source_rebench_params(
            self._coord,
            reason=reason,
            result=accepted,
            launch={"extra_server_args": product.flags, "extra_envs": dict(product.envs), **product.controls},
            overlay=product.overlay,
        )
        if source_params is not None:
            return source_params
        return self._geak_explore_rebench_params(accepted, product, reason=reason)

    def settle_geak_no_material(self) -> None:
        """Close a GEAK candidate that adds nothing to ``current_best``, revoking its provisional journey KEEPs."""
        state = self.shared_state
        state.geak_result = {**state.geak_result, "revalidation_status": REBENCH_NO_MATERIAL}
        self._coord.phase_kernel.reject_geak_kernel_journey(
            state.geak_result,
            measured_tput=0.0,
            current_best_tput=float(state.current_best.get("tput") or 0.0),
            provenance="geak_no_material",
            rejection_reason="geak_no_material_product",
        )
        self._record_geak_rebench_conclusion(final_status=REBENCH_NO_MATERIAL)
        state.geak_pending = {}

    def _accepted_geak_product(self, accepted: Mapping[str, Any]) -> _GeakProduct | None:
        config = accepted.get("accepted_config") or {}
        admissible = str(accepted.get("status") or "") == "ok" or _geak_has_accepted_kernel(accepted)
        controls = _accepted_config_controls(
            config, inherited_remove_args=self.current_best_launch_config()["remove_args"]
        )
        flags, envs = _accepted_config_as_variant(config)
        overlay = _normalize_geak_overlay_dir(str(accepted.get("final_overlay") or "").strip())
        has_artifact = _geak_result_has_artifact(accepted)
        deployable = config.get("flags") or config.get("env") or "env_map" in config or controls or has_artifact
        if not (admissible and deployable):
            return None
        return _GeakProduct(flags=flags, envs=envs, controls=controls, overlay=overlay, has_artifact=has_artifact)

    def _geak_explore_rebench_params(
        self,
        accepted: Mapping[str, Any],
        product: _GeakProduct,
        *,
        reason: str,
    ) -> dict[str, Any]:
        from ..actions.executors._proposal_identity import BaseControls, effective_fingerprint

        overlay = product.overlay
        if overlay and not geak_overlay_is_loadable(overlay):
            log.warning(
                "geak 2b: overlay %r is not loadable (no sitecustomize.py); "
                "revalidating the config WITHOUT the authored kernel",
                overlay,
            )
            overlay = ""
        if not overlay and self._geak_launch_restates_current_best(product.flags, product.envs, product.controls):
            if product.overlay:
                return {"skipped": True, "reason": "geak_overlay_unloadable", "fallback": "geak_harness"}
            if product.has_artifact:
                return {"skipped": True, "reason": "geak_material_requires_harness", "fallback": "geak_harness"}
            return {"skipped": True, "reason": "geak_no_material"}
        base_params = self._geak_rebench_base_params(product.controls)
        params: dict[str, Any] = {
            "source": STACK_REVALIDATE_SOURCE,
            "reason": reason,
            "recipe_generation": int(self.shared_state.working_recipe_generation or 0),
            "geak_fallback": True,
            "expected_cfg_hash": effective_fingerprint(
                product.flags,
                product.envs,
                controls=product.controls,
                base=BaseControls.of_params(base_params),
            ),
            "expected_overlay": overlay,
            "expected_overlay_digest": _geak_overlay_digest(overlay),
            "grid": [_geak_rebench_grid_row(accepted, product, overlay)],
            "base_tput": float(self.shared_state.baseline_tput or 0.0),
            **base_params,
        }
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        benchmark_script = self.shared_state.accepted_baseline_script()
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        self._coord.proposals.inject_explore_runtime_params(params)
        return params

    def _geak_launch_restates_current_best(
        self, flags: str, envs: Mapping[str, str], controls: Mapping[str, Any]
    ) -> bool:
        """Whether GEAK's accepted launch, applied over ``current_best``, launches what ``current_best`` runs.

        Both launches are composed over ``current_best``'s complete observed engine flags, so a GEAK config that
        spells the incumbent launch out in full compares equal to the shorter delta ``current_best`` stores. Removal
        lists count as identity: a removal whose effect the composed flags cannot show still changes the launch.
        """
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
        from hyperloom.inference_optimizer.grid_server_args import compose_server_args

        launch = self.current_best_launch_config()
        complete_flags = self.build_env_spec()["config"]["server_launch_flags"]
        replacing = controls.get("args_mode") == "replace"
        added_removals = controls.get("remove_args", [])
        candidate_removals = (
            added_removals if replacing else list(dict.fromkeys(launch["remove_args"] + added_removals))
        )
        current_args = compose_server_args(
            inherited_args=complete_flags,
            base_extra_args=launch["extra_server_args"],
            remove_args=launch["remove_args"],
            args_mode=launch["args_mode"],
        )
        candidate_args = compose_server_args(
            inherited_args=complete_flags,
            base_extra_args="" if replacing else launch["extra_server_args"],
            variant_extra_args=flags,
            remove_args=candidate_removals,
            args_mode="replace" if replacing else launch["args_mode"],
        )
        unset = sorted(set(launch["unset_envs"]) | set(controls.get("unset_envs", [])))
        candidate_envs = {name: value for name, value in launch["extra_envs"].items() if name not in unset}
        candidate_envs.update(envs)
        return canonical_fingerprint(
            candidate_args, candidate_envs, remove_args=candidate_removals, unset_envs=unset
        ) == canonical_fingerprint(
            current_args, launch["extra_envs"], remove_args=launch["remove_args"], unset_envs=launch["unset_envs"]
        )

    def _geak_rebench_base_params(self, controls: Mapping[str, Any]) -> dict[str, Any]:
        base = stack_base_params({**self.shared_state.current_best, **self.current_best_launch_config()})
        base = {
            key: value
            for key, value in base.items()
            if key not in {"base_remove_args", "base_unset_envs", "base_args_mode"} or value
        }
        if controls.get("args_mode") == "replace":
            base["base_extra_args"] = ""
            base["base_args_mode"] = "replace"
            base.pop("base_remove_args", None)
        return base

    async def _rule_on_geak_rebench(self, result: dict, *, task: "Task") -> Verdict:
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
            decision = REBENCH_NO_PROMOTE

        pending_status = self._geak_candidate_slot_status()
        if decision == REBENCH_VALIDATED:
            # Only an overlay that was dispatched AND still matches its
            # manifest proves a kernel was in the measurement.
            promoted = self._promote_validated_rebench(
                rebench, overlay_loaded=bool(expected_overlay) and overlay_loaded
            )
            if not promoted:
                decision = REBENCH_NO_PROMOTE
        if decision in {REBENCH_NO_PROMOTE, "accuracy_drop", EVAL_KIND_ACCURACY_UNAVAILABLE, "intvty_regression"}:
            await self._refuse_rebench(
                result, task=task, rebench=rebench, decision=decision, native_rejection=native_rejection
            )
        elif decision == REBENCH_FALLBACK:
            replay = await self._replay_via_geak_harness(result, task=task, rebench=rebench)
            decision, promoted = replay.decision, replay.promoted
        expected_hash = str((task.params or {}).get("expected_cfg_hash") or "")
        self._record_geak_rebench_timeline(
            task=task,
            decision=decision,
            pending_status=pending_status,
            measured=rebench.measured,
            engagement=_RebenchEngagement(
                config_matched=(rebench.got_hash == expected_hash) if expected_hash else None,
                overlay_loaded=overlay_loaded if expected_overlay else None,
                observed_cfg_hash=rebench.got_hash,
                observed_overlay_digest=got_digest,
            ),
        )
        self.shared_state.record_action_attempt(
            action="explore",
            attempt=ActionAttempt(
                task_id=task.task_id,
                status=str(result.get("status") or "succeeded"),
                decision="promoted" if promoted else "no_promote" if decision == REBENCH_NO_PROMOTE else "discarded",
                result=result,
                extras={"revalidation_decision": decision, "output_throughput": rebench.measured},
            ),
        )
        return Verdict.ADOPTED if promoted else Verdict.REFUSED

    def _geak_candidate_slot_status(self) -> str:
        pending = self.shared_state.geak_pending or {}
        return str(pending.get("status") or "") if isinstance(pending, dict) else ""

    def _read_geak_rebench(self, result: dict) -> _GeakRebench:
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
        expected_overlay = str((task.params or {}).get("expected_overlay") or "")
        overlay_loaded = True
        got_digest = ""
        if expected_overlay:
            expected_digest = str((task.params or {}).get("expected_overlay_digest") or "")
            got_digest = _geak_overlay_digest(expected_overlay)
            overlay_loaded = geak_overlay_is_loadable(expected_overlay) and (got_digest == expected_digest)
            if not overlay_loaded and decision == REBENCH_VALIDATED:
                log.warning(
                    "geak 2b: overlay %r did not survive the run "
                    "(loadable=%r digest expected=%r got=%r) -> 2a fallback",
                    expected_overlay,
                    geak_overlay_is_loadable(expected_overlay),
                    expected_digest,
                    got_digest,
                )
                decision = REBENCH_FALLBACK
        return _OverlayReading(expected=expected_overlay, loaded=overlay_loaded, digest=got_digest, decision=decision)

    def _geak_rebench_decision(
        self,
        result: dict,
        *,
        task: "Task",
        rebench: _GeakRebench,
    ) -> _RebenchDecision:
        measured, got_hash = rebench.measured, rebench.got_hash
        expected_hash = str((task.params or {}).get("expected_cfg_hash") or "")
        baseline_grade = resolve_graded_comparison(self.shared_state, rebench.measurement, against_baseline=True)
        current_grade = resolve_graded_comparison(self.shared_state, rebench.measurement)
        decision = _geak_revalidation_decision(
            measured=baseline_grade.candidate,
            baseline=baseline_grade.reference,
            got_hash=got_hash,
            expected_hash=expected_hash,
            current_best=current_grade.reference,
        )
        if not _grades_can_rule(result, measured, baseline_grade, current_grade):
            decision = REBENCH_FALLBACK
        elif (not expected_hash or got_hash == expected_hash) and any(
            grade.graded_on_intvty and grade.verdict != "KEEP" for grade in (baseline_grade, current_grade)
        ):
            decision = REBENCH_NO_PROMOTE
        native_rejection = _native_rejection_reason(result, expected_hash)
        if (
            measured > 0
            and got_hash == expected_hash
            and (not baseline_grade.comparable or not current_grade.comparable)
        ):
            decision = REBENCH_NO_PROMOTE
        return _RebenchDecision(decision=decision, native_rejection=native_rejection)

    async def _refuse_rebench(
        self,
        result: dict,
        *,
        task: "Task",
        rebench: _GeakRebench,
        decision: str,
        native_rejection: str,
    ) -> None:
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
        result["status"] = REBENCH_NO_PROMOTE

    def _promote_validated_rebench(self, rebench: _GeakRebench, *, overlay_loaded: bool) -> bool:
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

    def _record_geak_rebench_timeline(
        self,
        *,
        task: Any,
        decision: str,
        pending_status: str,
        measured: Any,
        engagement: _RebenchEngagement,
    ) -> None:
        recorder = self._coord.phase_kernel.timeline()
        if recorder is None:
            return
        params = task.params or {}
        recorder.record_geak_rebench_attempt(
            GeakRebenchAttempt(
                attempt_id=str(task.idempotency_key or task.task_id or ""),
                idempotency_key=str(task.idempotency_key or ""),
                task_id=str(task.task_id or ""),
                base_tput=params.get("base_tput"),
                measured_tput=measured,
                decision=decision,
                decision_reason=str(params.get("reason") or ""),
                status=pending_status,
                engagement={
                    "config_matched": engagement.config_matched,
                    "overlay_loaded": engagement.overlay_loaded,
                    "expected_cfg_hash": params.get("expected_cfg_hash"),
                    "observed_cfg_hash": engagement.observed_cfg_hash,
                    "expected_overlay_digest": params.get("expected_overlay_digest"),
                    "observed_overlay_digest": engagement.observed_overlay_digest,
                },
            )
        )

    def _close_failed_geak_revalidation(self, result_payload: Mapping[str, Any]) -> None:
        geak_result = {
            **(self.shared_state.geak_result or {}),
            "revalidation_status": REBENCH_FAILED,
            "revalidation_error_class": str(result_payload.get("error_class") or ""),
            "revalidation_error": str(result_payload.get("error") or result_payload.get("reason") or "")[:500],
        }
        self.shared_state.geak_result = geak_result
        self._record_geak_rebench_conclusion(
            final_status=REBENCH_FAILED,
            final_error_class=geak_result["revalidation_error_class"],
            final_error=geak_result["revalidation_error"],
        )
        # The settled verdict and its diagnostics live on ``geak_result``; ``geak_pending`` only holds live work.
        self.shared_state.geak_pending = {}

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

    async def validate_geak_via_geak_harness(self, *, reason: str) -> dict[str, Any]:
        from hyperloom.common.perf_metric import agentx_enabled, is_agentx_mode

        mode = str(self.shared_state.benchmark_mode or "").strip()
        agentx = is_agentx_mode(mode) if mode else agentx_enabled()
        if agentx:
            return {
                "validated": False,
                "status": "incomparable",
                "reason": "geak_harness_unsupported_canonical_workload",
            }
        accepted = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        if str(accepted.get("status") or "") != "ok" and not _geak_has_accepted_kernel(accepted):
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        # The overlay's identity is captured before the replay so it can be
        # compared after; see :func:`_overlay_survived_replay`.
        overlay_path = _normalize_geak_overlay_dir(str(accepted.get("final_overlay") or "").strip())
        overlay = _ReplayOverlay(
            path=overlay_path,
            digest_before=_geak_overlay_digest(overlay_path) if overlay_path else "",
        )
        res = await self._replay_through_geak(accepted, mode)
        accuracy_failure = self._replay_accuracy_failure(_promotion_measurement(res))
        if accuracy_failure:
            self._refuse_replay_on_accuracy(accepted, res, accuracy_failure)
            return {"validated": False, "status": REBENCH_NO_PROMOTE, "reason": accuracy_failure}
        if str(res.get("status") or "") == "succeeded":
            return self._promote_replayed_geak_win(accepted, res, overlay, reason=reason)
        if res.get("error"):
            reason = str(res["error"])
        log.warning(
            "geak 2a fallback did not validate (status=%r reason=%s)",
            res.get("status"),
            reason,
        )
        return {"validated": False, "status": res.get("status"), "reason": reason}

    async def _replay_through_geak(self, accepted: Mapping[str, Any], mode: str) -> dict[str, Any]:
        from hyperloom.common.jsonio import read_json
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir
        from ..actions.executors._geak_sweep import GeakSweepSpec, sweep_via_geak
        from ..actions.executors._workload_envs import geak_metric_axis

        regime = _replay_regime(accepted)
        return await sweep_via_geak(
            result=accepted,
            handoff=read_json(self.session_dir / "geak" / "handoff.json", default={}, require_dict=True),
            env_spec=self.build_env_spec(),
            # A kernel-lane re-benchmark, not a dispatched action: it borrows the
            # ``integrate`` workspace namespace under a task id that names it,
            # as the stack re-validation and the GEAK integrate rebench do.
            output_root=unique_runs_dir(self.session_dir, "integrate", "revalidate_geak"),
            spec=GeakSweepSpec(
                conc_values=[regime.conc],
                isl_osl_configs=[f"{regime.isl}:{regime.osl}"],
                variant_timeout_sec=env_int("SWEEP_VARIANT_TIMEOUT_SEC", default=2400),
                repeats=3,
                # The axis the handoff sent GEAK, so the replay measures what GEAK accepted on.
                metric_axis=geak_metric_axis(benchmark_mode=mode, grading=self.shared_state.grading or None),
            ),
        )

    def _replay_accuracy_failure(self, measurement: Mapping[str, Any]) -> str:
        baseline_accuracy = float(self.shared_state.baseline_accuracy or 0.0)
        if baseline_accuracy <= 0:
            return ""
        replay_accuracy = measurement.get("accuracy")
        if (
            isinstance(replay_accuracy, bool)
            or not isinstance(replay_accuracy, (int, float))
            or not 0.0 <= replay_accuracy <= 1.0
        ):
            return EVAL_KIND_ACCURACY_UNAVAILABLE
        if not accuracy_passed(baseline_accuracy, float(replay_accuracy)):
            return "accuracy_drop"
        return ""

    def _refuse_replay_on_accuracy(
        self,
        accepted: Mapping[str, Any],
        res: Mapping[str, Any],
        failure: str,
    ) -> None:
        self._coord.phase_kernel.reject_geak_promotion(
            {
                **accepted,
                "fallback_result": res,
                "failure_reason": failure,
                "baseline_accuracy": float(self.shared_state.baseline_accuracy or 0.0),
            },
            measured_tput=_geak_sweep_measured_tput(res) or 0.0,
            current_best_tput=float((self.shared_state.current_best or {}).get("tput") or 0.0),
            reason=failure,
        )
        self.shared_state.save(self.session_dir)

    def _promote_replayed_geak_win(
        self,
        accepted: Mapping[str, Any],
        res: Mapping[str, Any],
        overlay: _ReplayOverlay,
        *,
        reason: str,
    ) -> dict[str, Any]:
        measured = _geak_sweep_measured_tput(res)
        if measured is None:
            log.warning("geak 2a: succeeded sweep but no measurable throughput; candidate stays pending")
            return {"validated": False, "status": res.get("status"), "reason": reason}
        overlay_loaded = _overlay_survived_replay(overlay)
        if overlay.path and not overlay_loaded:
            log.warning(
                "geak 2a: overlay %r is not loadable evidence "
                "(loadable=%r digest before=%r after=%r) -> gain credited "
                "to config, not to a kernel",
                overlay.path,
                geak_overlay_is_loadable(overlay.path),
                overlay.digest_before,
                _geak_overlay_digest(overlay.path),
            )
        promoted = self._coord.phase_kernel.promote_geak_from_candidate(
            accepted,
            measured_tput=measured,
            provenance="geak_same_harness_geak",
            overlay_loaded=overlay_loaded,
            measurement_provenance=_promotion_measurement(res) or res,
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("geak 2a: SharedState.save failed")
        if not promoted:
            return {
                "validated": False,
                "status": REBENCH_NO_PROMOTE,
                "reason": str(self.shared_state.geak_result.get("revalidation_error") or "promotion_rejected"),
            }
        return {"validated": True, "gain": float(self.shared_state.cumulative_gain_validated), "reason": reason}

    async def _replay_via_geak_harness(
        self,
        result: dict,
        *,
        task: Task,
        rebench: _GeakRebench,
    ) -> _ReplayRuling:
        measured, got_hash = rebench.measured, rebench.got_hash
        promoted = False
        decision = REBENCH_FALLBACK
        log.warning(
            "geak 2b revalidation inconclusive (measured=%r got_hash=%r expected=%r) -> GEAK-harness 2a fallback",
            measured,
            got_hash,
            (task.params or {}).get("expected_cfg_hash"),
        )
        fallback_result: dict[str, Any]
        try:
            fallback_result = await self.validate_geak_via_geak_harness(reason="2b_inconclusive")
        except Exception as exc:  # isolation boundary: one GEAK candidate's harness fallback
            log.exception("geak 2a GEAK-harness fallback failed")
            fallback_result = {
                "validated": False,
                "reason": repr(exc),
            }
        if not bool(fallback_result.get("validated")):
            from ..phases.geak_rebench import INCOMPARABLE_REVALIDATION

            geak_result = dict(self.shared_state.geak_result) if isinstance(self.shared_state.geak_result, dict) else {}
            # 2a reports a refusal it will repeat for this workload
            # as a typed status. Persist it beside the verdict: the
            # reason text is for the report, and a later KERNEL entry
            # needs to know the replay is settled, not merely broken.
            refusal = str(fallback_result.get("status") or "")
            geak_result["revalidation_status"] = (
                REBENCH_NO_PROMOTE if refusal == REBENCH_NO_PROMOTE else REBENCH_FALLBACK_FAILED
            )
            geak_result["revalidation_error_class"] = refusal
            geak_result["revalidation_error"] = str(
                fallback_result.get("reason") or refusal or "GEAK harness fallback did not validate"
            )[:500]
            self.shared_state.geak_result = geak_result
            self._record_geak_rebench_conclusion(
                final_status=geak_result["revalidation_status"],
                final_error=str(geak_result.get("revalidation_error") or ""),
            )
            self.shared_state.geak_pending = {}
            result["status"] = refusal if refusal in {INCOMPARABLE_REVALIDATION, REBENCH_NO_PROMOTE} else "failed"
        else:
            promoted = True
            decision = REBENCH_VALIDATED
        return _ReplayRuling(decision=decision, promoted=promoted)
