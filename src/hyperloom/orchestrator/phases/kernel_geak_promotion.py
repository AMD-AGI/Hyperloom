# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_GeakPromotionMixin``: GEAK candidate revalidation, promotion, journey and adoption records for the KERNEL phase."""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import CancelledError as FuturesCancelledError
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from . import geak_rebench as _geak_rebench
from hyperloom.common.coerce import first_str, to_float
from hyperloom.common import gain_math
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_FALLBACK_FAILED, REBENCH_NO_PROMOTE
from hyperloom.common.perf_metric import axis_of, graded_axes_of
from hyperloom.orchestrator.lever import LEVER_CONFIG, LEVER_KERNEL
from hyperloom.inference_optimizer.breakdown.recorder import tool_versions
from ..actions.executors._workload_envs import GEAK_METRIC_OUTPUT
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    kernel_event_id,
    record_geak_attempts,
    reject_geak_attempts,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import (
    GeakDelegation,
    GeakProduct,
    IntegrateVerdict,
)
from ..state.task_registry import Task
from ..kernel.geak_config import (
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    geak_acceptance_specs,
    geak_spec_name,
    _accepted_config_as_variant,
    _accepted_config_controls,
)

log = logging.getLogger(__name__)

# |measurement_divergence_pct| above this (GEAK vs orchestrator, same config) is logged as a measurement-mismatch
# warning at geak promote.
_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT: float = 3.0


def _bench_envs(bench: Mapping[str, Any]) -> dict[str, Any]:
    envs = bench.get("envs")
    return dict(envs) if isinstance(envs, dict) else {}


def _covered_step(kind: str, params: dict[str, Any], *, idempotency_key: str) -> Task:
    """An unpersisted task for a step run under the ``kernel_agent`` task's lanes."""
    return Task(task_id=uuid.uuid4().hex, kind=kind, state="running", params=params, idempotency_key=idempotency_key)


def _geak_decline_status(decline_reason: Any) -> str:
    """Map a 2b decline reason to the status left on ``geak_pending``."""
    reason = str(decline_reason or "").strip().lower()
    return "overlay_unloadable" if reason == "geak_overlay_unloadable" else "rebench_declined"


def _record_geak_integration(entry: dict[str, Any], *, kernel_id: str, macro_cycle: int) -> None:
    """Mirror one GEAK adoption onto the kernel event as an integrate row.

    GEAK adopts by writing the per-kernel ledger directly rather than through
    the integrate queue, so without this the timeline holds no gate row for a
    GEAK adoption at all -- and the basis the gain was measured on, along with
    whether it could be pinned on this one kernel, lives only on that ledger.

    The integration id is synthesized from the kernel, because the queue that
    would have minted one was never involved. It keys the row, so it carries
    no ``:``: that is the fragment key's own separator, and a row whose key
    contains one is dropped on the way to the event.

    Args:
        entry (dict[str, Any]): The per-kernel ledger entry just written.
        kernel_id (str): The kernel the adoption is for.
        macro_cycle (int): The cycle the adoption settled in.
    """
    if not kernel_id:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_integrate_verdict

    record_integrate_verdict(
        macro_cycle=macro_cycle,
        verdict=IntegrateVerdict(
            integration_id=f"geak-{kernel_id}",
            kernel_id=kernel_id,
            decision=str(entry.get("last_decision") or ""),
            status=str(entry.get("last_status") or ""),
            attempt_count=entry.get("attempt_count"),
            gain_pct=entry.get("best_gain_pct"),
            basis=str(entry.get("basis") or ""),
            alignment_status=str(entry.get("alignment_status") or ""),
            gain_attributed=bool(entry.get("validated", True)),
            settled_at=str(entry.get("updated_at") or ""),
        ),
    )


def geak_absent_backends(eval_dir: str) -> dict[str, str]:
    """``{backend: probe}`` for the backends GEAK's environment report lists as absent.

    GEAK records them in ``<eval_dir>/env_report.json`` and deep in its final
    report; they are provisioning gaps, and nothing else carries them to the
    optimizer's log or reports.
    """
    if not eval_dir:
        return {}
    try:
        report = json.loads((Path(eval_dir) / "env_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    absent = report.get("absent_backends") if isinstance(report, dict) else None
    if not isinstance(absent, dict):
        return {}
    return {
        str(name): str(entry.get("probe") or "") if isinstance(entry, dict) else str(entry or "")
        for name, entry in absent.items()
    }


@dataclass(frozen=True)
class GeakRunnerFacts:
    """How the GEAK runner process ran, as the timeline and the phase evidence report it.

    These are properties of the dispatch, not of the product GEAK shipped, so
    they travel beside ``result.json``'s content and never merge into it: the
    persisted ``geak_result`` has to stay byte-comparable with the file
    :func:`geak_candidate_matches` reads back on the next KERNEL entry.
    """

    started_at: str = ""
    duration_sec: float | None = None
    runner_timeout_sec: int | None = None
    kill_timeout_sec: int | None = None


class _GeakPromotionMixin:
    async def _revalidate_geak_candidate(self, *, reason: str, handoff_cards: list[int] | None) -> None:
        """Measure the GEAK candidate on the orchestrator harness and settle its verdict.

        The 2b rebench runs as a step of the ``kernel_agent`` task, under the
        lanes it holds, and its result goes through the same promotion a
        dispatched explore would. A candidate the grid cannot carry goes to the
        GEAK-harness replay (2a) instead. ``handoff_cards`` are drained only
        when a measurement follows.
        """
        state = self.shared_state
        params = self._coord.writeback.geak_rebench_params(reason=reason)
        from .geak_native_revalidation import ORIGIN, finish_invalid_source, revalidate_source

        if params.get("reason") == "geak_native_source_invalid":
            finish_invalid_source(self._coord, str(params["error"]))
            return
        skip_reason = params.get("reason") if params.get("skipped") else None
        if skip_reason == "geak_invalid_config":
            return
        if skip_reason == "geak_no_material":
            self._coord.writeback.settle_geak_no_material()
            state.save(self.session_dir)
            return
        await self._wait_for_gpus_before_rebench(handoff_cards)
        if params.get("origin") == ORIGIN:
            await revalidate_source(self._coord, params)
            return
        if skip_reason is not None:
            await self._revalidate_on_geak_harness(decline_reason=str(skip_reason))
            return
        task = _covered_step("explore", params, idempotency_key=f"geak-revalidate-c{int(state.macro_cycle or 0)}")
        try:
            result = await self.sub.execute_covered(task)
        except FuturesCancelledError as exc:
            state.geak_pending = {
                **(state.geak_pending or {}),
                "status": "rebench_cancelled",
                "revalidation_error": str(exc)[:500],
            }
            state.save(self.session_dir)
            raise
        if self._coord.writeback.is_promotable_result("explore", result):
            await self._coord.writeback.promote_to_shared_state("explore", result, task=task)
        else:
            await self._coord.writeback.handle_unpromotable_result(task, result)

    async def _revalidate_on_geak_harness(self, *, decline_reason: str) -> None:
        """Replay the candidate through GEAK's own harness (2a); record the decline when it does not validate."""
        state = self.shared_state
        log.warning("geak: 2b declined (%s); validating through the GEAK harness instead", decline_reason)
        fb = await self._coord.writeback.validate_geak_via_geak_harness(reason=decline_reason)
        # Both are verdicts 2a has already recorded.
        if fb.get("validated") or fb.get("status") == REBENCH_NO_PROMOTE:
            return
        error = str(fb.get("reason") or decline_reason)[:500]
        state.geak_pending = {
            **(state.geak_pending or {}),
            "status": _geak_decline_status(decline_reason),
            "revalidation_error": error,
        }
        if (
            not _geak_rebench.geak_harness_replays_workload(state)
            and fb.get("status") == _geak_rebench.INCOMPARABLE_REVALIDATION
        ):
            state.geak_result = {
                **state.geak_result,
                "revalidation_status": REBENCH_FALLBACK_FAILED,
                "revalidation_error_class": _geak_rebench.INCOMPARABLE_REVALIDATION,
                "revalidation_error": error,
                # This refusal is reusable only while its overlay remains unloadable.
                "revalidation_blocked_overlay": str(state.geak_result.get("final_overlay") or ""),
            }
        state.save(self.session_dir)

    def _geak_win_already_recorded(self) -> bool:
        """Whether a GEAK e2e win is already in this session's state."""
        return any(
            isinstance(item, dict) and item.get("action") == "geak_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    def _record_geak_claim_and_product(
        self, result: dict[str, Any], accepted_flags: str, parsed_envs: dict[str, str]
    ) -> None:
        recorder = self.timeline()
        if recorder is None:
            return
        recorder.record_geak_claim(self.shared_state.geak_pending, specs=geak_acceptance_specs(result))
        recorder.record_geak_product(
            GeakProduct(
                accepted_flags=accepted_flags,
                accepted_envs=dict(parsed_envs),
                accepted_config=result.get("accepted_config"),
                final_overlay=result.get("final_overlay") or "",
                final_launch_script=result.get("final_launch_script") or "",
                bench_script=result.get("bench_script") or "",
                final_patch=result.get("final_patch") or "",
            )
        )

    def _record_geak_candidate(self, result: dict[str, Any]) -> bool:
        """Validate the return and record any measured claim without a headline gain."""
        if not isinstance(result, dict):
            return False
        result.setdefault("kernel_event_id", kernel_event_id(int(self.shared_state.macro_cycle or 0)))
        try:
            accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False
        if result.get("status") != "ok" and not _geak_has_accepted_kernel(result):
            return True
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        if new_tput <= 0:
            return True
        basis_facts = result.get("baseline_basis") or {}
        self.shared_state.geak_pending = self._geak_pending_record(
            result, basis_facts, accepted_flags=accepted_flags, parsed_envs=parsed_envs, new_tput=new_tput
        )
        self._record_geak_claim_and_product(result, accepted_flags, parsed_envs)
        # Surface a large cross-harness measurement divergence as a warning only.
        divergence = to_float(basis_facts.get("measurement_divergence_pct"), 0.0)
        if abs(divergence) > _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT:
            log.warning(
                "geak candidate: large cross-harness measurement divergence %.2f%% (|.|>%.1f%%) - candidate held "
                "out of headline until a main-flow rebench validates it",
                divergence,
                _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
            )
        return True

    def _geak_pending_record(
        self,
        result: dict[str, Any],
        basis_facts: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
        new_tput: float,
    ) -> dict[str, Any]:
        from .geak_native_revalidation import matching_review

        basis = str(result.get("metric_basis") or GEAK_METRIC_OUTPUT[1])
        base = (
            float(self.shared_state.baseline_tput or 0.0)
            if basis == GEAK_METRIC_OUTPUT[1]
            else axis_of(self.shared_state.baseline_perf, basis)
        )
        # synthetic isl/osl at face value reads as roughly +175% with nothing
        # optimized. The downstream rebench would eventually reject it, but the
        # number would already be recorded, so refuse to compute it here.
        # An ABSENT verdict means an older GEAK that only ever ran the synthetic
        # path, which stays comparable and keeps today's value exactly.
        comparability = basis_facts.get("workload_comparability") or {}
        comparable = comparability.get("comparable") is not False
        am = result.get("alignment_metrics") or {}
        return {
            "status": "awaiting_rebench",
            # Audit-only self-reported numbers (not the headline until rebench).
            "self_reported_tput": new_tput,
            "self_reported_speedup": result.get("throughput_speedup"),
            "self_reported_gain_pct": gain_math.incremental_gain_pct(new_tput, base) if comparable else None,
            # Why the gain above is a number or None, so a reader never has to
            # guess whether an absent gain means "no gain" or "not comparable".
            "workload_comparability": comparability or None,
            "self_reported_basis": result.get("final_throughput_basis"),
            # Reproducible config the rebench launches from.
            "accepted_flags": accepted_flags,
            "accepted_envs": dict(parsed_envs),
            **_accepted_config_controls(result.get("accepted_config")),
            # Carry the kernels and the basis they were judged on into the
            # pending record, so a later promotion can name what it adopted
            # without re-reading result.json.
            "accepted_kernels": result.get("accepted_kernels") or [],
            "geak_status": str(result.get("status") or ""),
            "baseline_alignment_status": str((result.get("baseline_alignment") or {}).get("status") or ""),
            "final_overlay": result.get("final_overlay") or "",
            "final_launch_script": result.get("final_launch_script"),
            "bench_script": result.get("bench_script"),
            "eval_dir": result.get("eval_dir"),
            # GEAK's own within-harness speedups, for the report's audit cross-check.
            "alignment": {
                "hot_geak_speedup": am.get("hot_geak_speedup"),
                "cold_geak_speedup": am.get("cold_geak_speedup"),
                "hot_speedup": am.get("hot_speedup"),
                "cold_speedup": am.get("cold_speedup"),
                "final_basis": am.get("final_basis") or result.get("final_throughput_basis"),
                "geak_throughput_speedup": result.get("throughput_speedup"),
            },
            "ts": datetime.now(timezone.utc).isoformat(),
            **matching_review(self.shared_state, result),
        }

    @staticmethod
    def _geak_stack_entry_extra(result: dict[str, Any], *, overlay_loaded: bool | None) -> dict[str, Any]:
        """Build the ``geak_e2e`` stack entry, carrying only kernels proven to have run."""
        proven = overlay_loaded is True
        return {
            "backend": "geak",
            "accepted_kernels": (result.get("accepted_kernels") or []) if proven else [],
            "accepted_heads": (result.get("accepted_heads") or []) if proven else [],
            "report_path": result.get("report_path"),
            "source": "geak_e2e",
            "overlay_loaded": overlay_loaded,
        }

    def _geak_promotion_measurement(
        self,
        result: dict[str, Any],
        graded_measurement: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
        kernel_proven: bool,
    ) -> dict[str, Any]:
        measurement = {
            "name": "geak_e2e",
            "candidate_extra_server_args": accepted_flags,
            "extra_server_args": accepted_flags,
            "extra_envs": dict(parsed_envs),
            **_accepted_config_controls(result.get("accepted_config")),
            "final_overlay": result.get("final_overlay") or "",
            "source_phase": "KERNEL_AGENT",
            "lever_kind": LEVER_KERNEL if kernel_proven else LEVER_CONFIG,
            "ttft_mean_ms": result.get("ttft_ms"),
            "tpot_mean_ms": result.get("tpot_ms"),
            **graded_axes_of(graded_measurement),
            "workspace": result.get("eval_dir"),
        }
        if graded_measurement is result:
            return measurement
        for key in (
            "extra_server_args",
            "effective_extra_server_args",
            "extra_envs",
            "candidate_extra_server_args",
            "candidate_extra_envs",
            "recipe_delta",
            "remove_args",
            "unset_envs",
            "args_mode",
            "fingerprint",
        ):
            if key in graded_measurement:
                measurement[key] = graded_measurement[key]
        for key in (
            "accuracy",
            "gpu_power_avg_w",
            "gpu_power_by_gpu_w",
            "launch_evidence",
            "launch_evidence_path",
            "server_log_path",
            "workspace",
            "single_workspace",
        ):
            value = graded_measurement.get(key)
            if value not in (None, "", {}):
                measurement[key] = value
        return measurement

    def _compose_geak_launch_recipe(
        self,
        measurement: dict[str, Any],
        accepted_config: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
    ) -> None:
        from hyperloom.common.coerce import to_str_list
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
        from hyperloom.inference_optimizer.framework_registry import server_args_env_name
        from hyperloom.inference_optimizer.grid_server_args import compose_server_args, remove_server_args

        cb_now = self.shared_state.current_best
        accepted_controls = _accepted_config_controls(accepted_config)
        prior_controls = _accepted_config_controls(cb_now)
        launch_controls = {**prior_controls, **accepted_controls}
        for key in ("remove_args", "unset_envs"):
            values = list(dict.fromkeys(to_str_list(prior_controls.get(key)) + to_str_list(accepted_controls.get(key))))
            if values:
                launch_controls[key] = values
        complete = accepted_controls.get("args_mode") == "replace"
        if complete and "remove_args" in accepted_config:
            launch_controls["remove_args"] = to_str_list(accepted_config["remove_args"])
        if not launch_controls:
            return
        inherited_args = ""
        if not complete and prior_controls.get("args_mode") != "replace":
            recipe_envs = _bench_envs(self._recipe_benchmark(str(self.shared_state.baseline_config_path or "")))
            inherited_args = str(recipe_envs.get(server_args_env_name(self.shared_state.framework)) or "")
        measurement["extra_server_args"] = compose_server_args(
            inherited_args=inherited_args,
            base_extra_args="" if complete else cb_now.get("extra_server_args"),
            variant_extra_args=accepted_flags,
            remove_args=launch_controls.get("remove_args"),
            args_mode="replace" if complete else "append",
        )
        if not complete and launch_controls.get("remove_args"):
            # A legacy delta can re-enable a removed flag. The retained
            # snapshot is complete, so stale removals would prune it again.
            accepted_identity = canonical_fingerprint(accepted_flags, {})
            launch_controls["remove_args"] = [
                spec
                for spec in launch_controls["remove_args"]
                if canonical_fingerprint(remove_server_args(accepted_flags, [spec]), {}) == accepted_identity
            ]
        launch_envs = dict(cb_now.get("extra_envs") or {})
        for key in accepted_controls.get("unset_envs", []):
            launch_envs.pop(key, None)
        launch_envs.update(parsed_envs)
        measurement["extra_envs"] = launch_envs
        measurement.update(launch_controls)
        measurement["args_mode"] = "replace"

    def _record_geak_lift_attribution(
        self,
        result: dict[str, Any],
        *,
        measured: float,
        provenance: str,
        overlay_loaded: bool | None,
        pre_geak_tput: float,
    ) -> None:
        """Attribute the lifted GEAK result to the kernels that can carry it."""
        base = float(self.shared_state.baseline_tput or 0.0)
        # ``pre_geak_tput`` is read before the lift, which overwrites ``current_best`` with ``measured``: the journey
        # rejection and the route-level residual both measure from where the session stood before GEAK ran.
        self.record_geak_adopted_kernels(
            result,
            measured_tput=measured,
            baseline_tput=base,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
        )
        if overlay_loaded is not True:
            # The journey is replayed before the main-flow rebench and can therefore contain GEAK-internal KEEPs for
            # kernels that were not present in the configuration that produced ``measured``.
            self.reject_geak_kernel_journey(
                result,
                measured_tput=measured,
                current_best_tput=pre_geak_tput if pre_geak_tput > 0 else base,
                provenance=provenance,
                rejection_reason="overlay_not_proven_loaded",
            )

    def promote_geak_from_candidate(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        provenance: str = "geak_e2e_promote",
        overlay_loaded: bool | None = None,
        measurement_provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Promote a measured GEAK candidate and return whether the lift succeeded."""
        if not isinstance(result, dict):
            return False
        measured = to_float(measured_tput, 0.0)
        if measured <= 0:
            return False
        # Read before the lift overwrites ``current_best``; see :meth:`_record_geak_lift_attribution`.
        pre_geak_tput = to_float(self.shared_state.current_best.get("tput"), 0.0)
        try:
            accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False

        # The lever is stamped here, not guessed from the task kind: GEAK promotes on a proven kernel overlay OR on a
        # config/env-only win, and only this site holds the overlay proof.
        entry_extra = self._geak_stack_entry_extra(result, overlay_loaded=overlay_loaded)
        graded_measurement = measurement_provenance if isinstance(measurement_provenance, Mapping) else result
        promotion_measurement = self._geak_promotion_measurement(
            result,
            graded_measurement,
            accepted_flags=accepted_flags,
            parsed_envs=parsed_envs,
            kernel_proven=bool(entry_extra.get("accepted_kernels") or entry_extra.get("accepted_heads")),
        )
        if graded_measurement is result or "extra_server_args" not in graded_measurement:
            self._compose_geak_launch_recipe(
                promotion_measurement,
                result.get("accepted_config") or {},
                accepted_flags=accepted_flags,
                parsed_envs=parsed_envs,
            )
        if not self._coord.writeback.lift_to_current_best(
            "geak_e2e", measured, promotion_measurement, entry_extra=entry_extra
        ):
            self.reject_geak_promotion(
                result,
                measured_tput=measured,
                current_best_tput=pre_geak_tput,
                reason="graded_comparison_rejected",
            )
            return False
        self._record_geak_lift_attribution(
            result,
            measured=measured,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
            pre_geak_tput=pre_geak_tput,
        )
        self._coord.writeback.validate(measured, graded_measurement, source="geak_e2e_promote")
        self.shared_state.geak_pending = {}
        return True

    @staticmethod
    def _geak_journey_path(result: dict[str, Any]) -> str:
        """Resolve the journey file for a GEAK result."""
        if not isinstance(result, dict):
            return ""
        path = str(result.get("kernel_journey_path") or "")
        if not path:
            eval_dir = str(result.get("eval_dir") or "")
            if eval_dir:
                path = str(Path(eval_dir) / "kernel_journey.json")
        return path if path and Path(path).is_file() else ""

    @classmethod
    def _load_geak_journey(cls, result: dict[str, Any]) -> dict[str, Any]:
        """Read the journey file, or return ``{}`` when it is unusable."""
        path = cls._geak_journey_path(result)
        if not path:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.debug("geak journey read failed", exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    def record_geak_adopted_kernels(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        baseline_tput: float,
        provenance: str,
        overlay_loaded: bool | None,
    ) -> None:
        """Write one adoption row per accepted GEAK kernel."""
        from .geak_adoption import AdoptionMeasurement, adoption_entry

        if not isinstance(result, dict):
            return
        # Both acceptance lanes, ``env`` selections excluded and alias twins collapsed.
        specs = _geak_accepted_kernel_specs(result)
        if not specs:
            return
        rows = [{"kernel_id": geak_spec_name(k), "spec": k} for k in specs]

        rebench_gain = gain_math.gain_pct(measured_tput, baseline_tput)
        # One kernel, overlay proven loaded, one measured number: the gain is attributable.
        attributable = bool(overlay_loaded) and len(rows) == 1
        am = result.get("alignment_metrics") or {}
        basis = str(am.get("final_basis") or result.get("final_throughput_basis") or "")
        alignment_status = str((result.get("baseline_alignment") or {}).get("status") or "")
        ts = datetime.now(timezone.utc).isoformat()
        ledger = self.shared_state.kernel_integrate_attempts
        if not isinstance(ledger, dict):
            return
        cycle = int(self.shared_state.macro_cycle or 0)
        measurement = AdoptionMeasurement(
            measured_tput=measured_tput,
            rebench_gain=rebench_gain,
            attributable=attributable,
            provenance=provenance,
            ts=ts,
            cycle=cycle,
            overlay_loaded=overlay_loaded,
            basis=basis,
            alignment_status=alignment_status,
        )
        for row in rows:
            kid = row["kernel_id"]
            spec = row["spec"]
            entry = adoption_entry(
                ledger.get(kid) or {},
                kid=kid,
                spec=spec,
                measurement=measurement,
            )
            ledger[kid] = entry
            # GEAK adopts by writing this ledger directly rather than through
            # the integrate queue, so without this the kernel timeline holds no
            # gate row for a GEAK adoption at all -- and the basis the gain was
            # measured on lives only here.
            _record_geak_integration(
                entry,
                kernel_id=kid,
                macro_cycle=cycle,
            )
        self.shared_state.kernel_integrate_attempts = ledger
        log.info(
            "geak: recorded %d adopted kernel(s) in the per-kernel ledger (overlay_loaded=%r attributable=%r gain=%r)",
            len(rows),
            overlay_loaded,
            attributable,
            rebench_gain if attributable else None,
        )

    def _record_geak_measurement(self, result: dict[str, Any]) -> None:
        """Record the latency and parity GEAK's harness measured for this run.

        Called where ``geak_result`` is set rather than beside the candidate
        record, because a run that measured a latency but accepted nothing
        never reaches the candidate path and would otherwise report none of it.
        """
        if not isinstance(result, dict) or not result:
            return
        recorder = self.timeline()
        if recorder is None:
            return
        recorder.record_geak_measurement(result)

    def _record_geak_delegation_timeline(
        self,
        result: dict[str, Any],
        *,
        handoff: dict[str, Any],
        facts: GeakRunnerFacts,
        recovered_from_disk: bool = False,
    ) -> None:
        """Record the delegated GEAK runner's terminal state."""
        eval_dir = str(result.get("eval_dir") or handoff.get("eval_dir") or "")
        absent = geak_absent_backends(eval_dir)
        if absent:
            result["absent_backends"] = absent
            log.warning(
                "GEAK could not use these backends (provisioning gaps, not measured no-wins): %s",
                "; ".join(f"{name}: {probe}" for name, probe in absent.items()),
            )
        recorder = self.timeline()
        if recorder is None:
            return
        versions = result.get("versions")
        versions = versions if isinstance(versions, dict) else {}
        recorder.record_geak_delegation(
            GeakDelegation(
                runner_status=first_str(result.get("status"), "unknown"),
                started_at=facts.started_at or first_str(result.get("started_at")),
                ended_at=str(result.get("ended_at") or datetime.now(timezone.utc).isoformat()),
                duration_sec=facts.duration_sec if facts.duration_sec is not None else result.get("duration_sec"),
                error_class=first_str(result.get("error_class")),
                error=first_str(result.get("error")),
                returncode=result.get("returncode"),
                runner_timeout_sec=(
                    facts.runner_timeout_sec if facts.runner_timeout_sec is not None else result.get("runner_timeout_s")
                ),
                kill_timeout_sec=(
                    facts.kill_timeout_sec if facts.kill_timeout_sec is not None else result.get("kill_timeout_s")
                ),
                exp_root=str(result.get("exp_root") or handoff.get("exp_root") or ""),
                eval_dir=eval_dir,
                report_path=first_str(result.get("report_path")),
                versions=versions,
                recovered_from_disk=recovered_from_disk,
                stages_reached=result.get("stages_reached"),
            )
        )

    def _record_geak_kernel_journey(self, result: dict[str, Any]) -> None:
        """Record what GEAK-e2e's ``kernel_journey.json`` says about its run.

        Two facts come out of the file. The attempts themselves go onto the
        kernel timeline event, which is what the breakdown reads. The tool
        builds are the other: GEAK reports the build of every tool its run went
        through, and this journey is the only place they reach the optimizer at
        all, so they are recorded even though nothing else in the file is.

        A missing or unreadable journey file records nothing and returns.
        """
        journey = self._load_geak_journey(result)
        if not journey:
            return

        record_geak_attempts(
            event=str(result.get("kernel_event_id") or kernel_event_id(int(self.shared_state.macro_cycle or 0))),
            journey=journey,
        )

        for tool, meta in (journey.get("versions") or {}).items():
            if not isinstance(meta, dict):
                continue
            tool_versions.record_tool_version(
                self.session_dir,
                tool=str(tool),
                root=str(meta.get("root_dir") or "") or None,
                version=str(meta.get("version") or meta.get("commit") or "") or None,
            )

    def reject_geak_kernel_journey(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        current_best_tput: float,
        provenance: str,
        rejection_reason: str = "rebench_did_not_beat_current_best",
    ) -> None:
        """Revoke the persisted provisional GEAK KEEPs after a final rebench."""
        reject_geak_attempts(
            event=str(result.get("kernel_event_id") or kernel_event_id(int(self.shared_state.macro_cycle or 0))),
            measured_tput=measured_tput,
            current_best_tput=current_best_tput,
            provenance=provenance,
            rejection_reason=rejection_reason,
        )
