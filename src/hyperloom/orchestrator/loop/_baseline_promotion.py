# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting a baseline result: the anchor it sets, the lane it closes, the chain it starts."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Mapping

from hyperloom.common.coerce import to_float
from hyperloom.common.env import is_truthy
from hyperloom.common.perf_metric import constraint_fields, stamp_output_per_gpu
from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
import yaml

from ..actions.executors._accuracy_gate import (
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    ENABLEMENT_REVALIDATION_REASON,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
    accuracy_meets_floor,
)
from ..state.round_store import FAILED
from ._promote_outcome import _PromoteOutcome
from ..state.task_registry import Task

log = logging.getLogger(__name__)


def _baseline_perf_axes(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    axes: dict[str, Any] = {
        "total_throughput": snapshot["total_throughput"],
        "e2e_norm_intvty_p90": snapshot["e2e_norm_intvty_p90"],
    }
    for name in ("input_throughput", "tpot_p90_ms", "e2e_norm_intvty_p50", "duration_seconds", "request_error_rate"):
        if snapshot.get(name) is not None:
            axes[name] = snapshot[name]
    return axes


# task.params fields fingerprinted by the self-loop guard.
_BASELINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "benchmark_script",
    "result_dir",
    "extra_server_args",
    "extra_envs",
    "model_path",
    "gpu_type",
    "config_path",
)

#: Server flags that assert a workload-shape field, each with what it asserts.
_WORKLOAD_ARG_FLAGS: dict[str, dict[str, bool]] = {
    "--enable-chunked-prefill": {"chunked_prefill_enabled": True},
    "--disable-chunked-prefill": {"chunked_prefill_enabled": False},
    "--enable-torch-compile": {"enable_torch_compile": True},
}

#: Server flags whose following token carries the value of a workload-shape count.
_WORKLOAD_ARG_COUNTS: dict[str, str] = {
    "--max-running-requests": "max_running_requests",
    "--max-num-seqs": "max_num_seqs",
}


def _parse_baseline_workload_extra(yaml_path: str) -> dict[str, Any]:
    """Extract KB workload-tag fields from a baseline-materialized Magpie YAML."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    out: dict[str, Any] = {}
    bm = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bm, dict):
        return out
    for key in ("workload_mode", "quant_scheme"):
        value = bm.get(key)
        if value not in (None, "", 0):
            out[key] = value
    envs = bm.get("envs") if isinstance(bm.get("envs"), dict) else {}
    declared = envs.get(server_args_env_name(bm.get("framework")))
    tokens = declared.split() if isinstance(declared, str) else []
    for index, token in enumerate(tokens):
        out.update(_WORKLOAD_ARG_FLAGS.get(token, {}))
        count_field = _WORKLOAD_ARG_COUNTS.get(token)
        if count_field is None or index + 1 >= len(tokens):
            continue
        try:
            out[count_field] = int(tokens[index + 1])
        except ValueError:
            continue
    # Torch compile may also be a separate env var.
    if "enable_torch_compile" not in out:
        tc_env = envs.get("ENABLE_TORCH_COMPILE")
        if isinstance(tc_env, str):
            out["enable_torch_compile"] = is_truthy(tc_env)
    return out


def _preserves_measured_eval_trigger(state: Any, result_payload: Mapping[str, Any]) -> bool:
    if result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY) != EVAL_KIND_ACCURACY_UNAVAILABLE:
        return False
    if to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY)) is not None:
        return False
    stored_kind = str(state.enablement.baseline_eval_kind or "")
    return bool(stored_kind) and stored_kind != EVAL_KIND_ACCURACY_UNAVAILABLE


def _adopt_eval_trigger(state: Any, result_payload: Mapping[str, Any]) -> None:
    enablement = state.enablement
    cfg = result_payload.get("materialized_config")
    if isinstance(cfg, str) and cfg:
        enablement.probe_config_path = cfg
    fingerprint = result_payload.get(BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY)
    if isinstance(fingerprint, str) and fingerprint:
        enablement.eval_contract_fingerprint = fingerprint
    incoming_kind = result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY)
    if isinstance(incoming_kind, str) and incoming_kind:
        enablement.baseline_eval_kind = incoming_kind
    observed = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY))
    if observed is not None:
        enablement.observed_accuracy = float(observed)
    enablement.observed_task = str(result_payload.get("accuracy_task") or "")
    enablement.observed_metric = str(result_payload.get("accuracy_metric") or "")
    evidence = str(result_payload.get(BASELINE_EVAL_EVIDENCE_KEY) or "")
    if evidence:
        enablement.baseline_eval_evidence = evidence[:4000]
        enablement.record_launch_observation(
            log=evidence, observation_path=str(result_payload.get("boot_observation_path") or "")
        )


class BaselinePromotionMixin:
    """Promoting a baseline measurement into the run's anchor."""

    @staticmethod
    def baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
        """Project ``params`` to the keys that determine baseline behavior."""
        params = params or {}
        out: dict[str, Any] = {}
        for key in _BASELINE_FINGERPRINT_KEYS:
            if key == "extra_envs":
                envs = params.get(key) or {}
                if isinstance(envs, dict):
                    out[key] = sorted([str(k), str(v)] for k, v in envs.items())
                else:
                    out[key] = None
                continue
            value = params.get(key)
            out[key] = None if value is None else str(value)
        return out

    def _should_run_prelude_bootstrap(self, tput: Any) -> bool:
        """Whether to enqueue the post-baseline PRELUDE bootstrap chain.

        Returns ``False`` when there is no positive baseline throughput, when a
        roofline task is already pending, or when a stop is already pending
        (e.g. the baseline accuracy test produced no result ->
        ``baseline_accuracy_failed``). In the stop case the run is about to halt
        at the Coordinator's end-of-tick check, so no new bootstrap work (warm
        replay / roofline / scout / static recon) must be enqueued or dispatched
        in the meantime.

        Args:
            tput: The promoted baseline throughput.

        Returns:
            bool: True only when the bootstrap chain should run.
        """
        if not (isinstance(tput, (int, float)) and tput > 0):
            return False
        if (self.shared_state.auto_roofline_pending_task_id or "").strip():
            return False
        if (self.shared_state.stop_reason or "").strip():
            return False
        return True

    def _anchor_held_reason(self, result: dict, *, is_revalidation: bool) -> str:
        """Why a baseline result does not replace the anchor, or ``""`` when it does.

        The anchor is the best measurement of the unmodified stack. A later, lower re-baseline must not replace it, or
        downstream gains end up measured against a different reference than the one they claim. A revalidation is
        exempt: the enablement patch changed the stack, so the prior anchor no longer describes anything reproducible.
        """
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and tput > 0):
            return "no_measurement"
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        if prior_anchor <= 0.0 or is_revalidation:
            return ""
        # A stamped cumulative gain is measured against this anchor; moving it afterwards leaves every stamp
        # describing a reference that no longer exists, which the report then classifies as `inconsistent`.
        if self.shared_state.has_validated_gain_stamp:
            return "anchor_load_bearing"
        # A hot pass measured against a recorded COLD anchor is a correction, not a regression, so it lands even when
        # the number is lower. The two are not comparable -- that is what the cold marker says -- and a cold figure
        # can read higher than the hot one that replaces it whenever the "cold" pass was not really cold: weights in
        # page cache and a JIT cache a prior run populated leave it paying none of the startup its depressed
        # reputation assumes. Without this the session cannot escape the marker. PRELUDE refuses to finish while it
        # is set, the retry that would clear it is rejected for measuring lower, and the run re-measures whole
        # baseline rounds until the clock kills it.
        hot_pass_ran = result.get("measure_round_runtime_sec")
        corrects_a_cold_anchor = (
            bool(self.shared_state.baseline_measure_round_dropped)
            and isinstance(hot_pass_ran, (int, float))
            and hot_pass_ran > 0
        )
        if float(tput) > prior_anchor or corrects_a_cold_anchor:
            return ""
        return "below_anchor"

    async def _promote_baseline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a baseline result: anchor tput / accuracy / config and bootstrap PRELUDE."""
        stamp_output_per_gpu(result, self.shared_state.tp)
        tput = result.get("output_throughput")
        measured_ok = isinstance(tput, (int, float)) and tput > 0
        promoting_tid = str(getattr(task, "task_id", "") or "").strip()
        task_params = getattr(task, "params", None) or {}
        tracked_tid = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
        is_revalidation = bool(
            task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON
            or (tracked_tid and tracked_tid == promoting_tid)
        )
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        held_reason = self._anchor_held_reason(result, is_revalidation=is_revalidation)
        anchor_accepted = not held_reason
        if measured_ok:
            self._anchor_baseline_tput(
                result, held_reason=held_reason, prior_anchor=prior_anchor, promoting_tid=promoting_tid
            )
            self.shared_state.baseline_failure_streak = 0
            # A genuine baseline is what revalidates an enablement KEEP.
            await self._settle_pending_revalidation(
                result, is_revalidation=is_revalidation, promoting_tid=promoting_tid
            )
        if anchor_accepted:
            self._promote_anchor_run_facts(result, task_params)
            # Once the stack carries a validated layer, current_best belongs to
            # the stack top and a baseline must not reset it to the bare
            # reference config.
            if not (self.shared_state.optimization_stack or []):
                self._anchor_current_best(result)
            outcome.verdict = Verdict.ADOPTED
            outcome.audit_decision = "promoted"
        elif measured_ok:
            outcome.verdict = Verdict.REVERTED
            outcome.audit_decision = "no_promote"
        else:
            outcome.verdict = Verdict.FAILED
            outcome.audit_decision = "discarded"
        audit_extras = self._baseline_audit_extras(result, task_params, held_reason, prior_anchor)
        # seed the gaps[] ledger from baseline.
        await self._coord.gap_refresh.refresh_gaps(
            reason="baseline_done", workload_id=self._coord.recipe_journal.workload_canonical_id()
        )
        if self.shared_state.baseline_tput > 0:
            await self.drain_queued_baselines(reason="baseline_established")
        # Only the run that first establishes the anchor bootstraps; a later
        # re-baseline must not re-fire replay, scout or recon.
        if prior_anchor <= 0.0 and self._should_run_prelude_bootstrap(tput):
            await self._bootstrap_prelude_after_baseline(tput)
        outcome.audit_extras = audit_extras

    def _anchor_baseline_tput(
        self,
        result: dict,
        *,
        held_reason: str,
        prior_anchor: float,
        promoting_tid: str,
    ) -> None:
        measured = float(result.get("output_throughput"))
        if held_reason:
            log.info(
                "baseline anchor: keeping %.1f; re-baseline measured %.1f (task=%s reason=%s)",
                prior_anchor,
                measured,
                promoting_tid,
                held_reason,
            )
            return
        self.shared_state.baseline_tput = measured

    async def _settle_pending_revalidation(
        self,
        result: dict,
        *,
        is_revalidation: bool,
        promoting_tid: str,
    ) -> None:
        enablement = self.shared_state.enablement
        tracked_tid = str(getattr(enablement, "revalidation_task_id", "") or "").strip()
        if not bool(getattr(enablement, "validation_pending", False)):
            await self._coord.enablement_lane.on_baseline_promoted()
            return
        if not is_revalidation:
            log.info(
                "enablement revalidation pending: unrelated baseline promoted "
                "(task=%s tracked=%s); not consuming pending state",
                promoting_tid,
                tracked_tid,
            )
            enablement.origin = ""
            enablement.pending = False
            return
        accuracy = result.get("accuracy")
        floor = float(getattr(enablement, "accuracy_floor", 0.0) or 0.0)
        generation = int(getattr(enablement, "revalidation_generation", 0) or 0)
        if accuracy_meets_floor(accuracy, floor):
            await self._close_enablement_lane(generation, accuracy, floor, promoting_tid)
            return
        await self._rearm_enablement_after_subfloor(generation, accuracy, floor, promoting_tid)

    async def _close_enablement_lane(
        self,
        generation: int,
        accuracy: Any,
        floor: float,
        promoting_tid: str,
    ) -> None:
        enablement_event.record_revalidation_outcome(
            enablement_event.RevalidationOutcome(
                generation=generation,
                promoted=True,
                task_id=str(promoting_tid or ""),
                accuracy=accuracy,
                accuracy_floor=floor,
            ),
        )
        # This promote is the lane's terminal: the KEEP that preceded it was
        # provisional, so the round that landed it did not close the lane.
        await self._coord.enablement_lane.close_booted(reason="revalidation promoted")

    async def _rearm_enablement_after_subfloor(
        self,
        generation: int,
        accuracy: Any,
        floor: float,
        promoting_tid: str,
    ) -> None:
        log.warning(
            "enablement revalidation: accuracy %.4f below floor %.4f; rearming",
            accuracy if isinstance(accuracy, (int, float)) else float("nan"),
            floor,
        )
        enablement = self.shared_state.enablement
        enablement.validation_pending = False
        enablement.revalidation_task_id = ""
        await self._coord.enablement_lane.settle_enablement_round(FAILED, reason="revalidation_below_floor")
        enablement_event.record_revalidation_outcome(
            enablement_event.RevalidationOutcome(
                generation=generation,
                promoted=False,
                task_id=str(promoting_tid or ""),
                accuracy=accuracy,
                accuracy_floor=floor,
                reason="accuracy below floor",
            ),
        )

    def _promote_anchor_run_facts(self, result: dict, task_params: Mapping[str, Any]) -> None:
        from hyperloom.common.perf_metric import perf_snapshot_from_mapping

        state = self.shared_state
        state.baseline_benchmark_script = str(task_params.get("benchmark_script") or "").strip()
        state.baseline_perf = perf_snapshot_from_mapping(result) or {}
        accuracy = result.get("accuracy")
        if isinstance(accuracy, (int, float)):
            state.baseline_accuracy = float(accuracy)
        # Persist the materialized YAML so downstream tasks reuse the exact workload contract.
        materialized = result.get("materialized_config")
        if isinstance(materialized, str) and materialized:
            state.baseline_config_path = materialized
            # Workload-shape extras off the YAML feed the lesson and pitfall attrs.
            parsed = _parse_baseline_workload_extra(materialized)
            if parsed:
                state.baseline_workload_extra = parsed
        # ExploreExecutor derives the overtime kill deadline from this wall-clock.
        runtime_sec = result.get("subprocess_runtime_sec")
        if isinstance(runtime_sec, (int, float)) and runtime_sec > 0:
            state.baseline_runtime_sec = float(runtime_sec)
        # Whether this baseline had to keep its cold figure because the budget
        # could not fund a hot pass and a variant to use it. Carried on the
        # session because the decision it drives is the session's: PRELUDE routes
        # to CLOSE rather than optimizing against a denominator that was never
        # the baseline.
        state.baseline_measure_round_dropped = bool(result.get("measure_round_dropped"))
        # Present only on the double-run baseline path; else explore uses the cold anchor.
        warm_runtime = result.get("measure_round_runtime_sec")
        state.baseline_warm_runtime_sec = (
            float(warm_runtime) if isinstance(warm_runtime, (int, float)) and warm_runtime > 0 else 0.0
        )
        post_ready = result.get("post_ready_runtime_sec")
        state.baseline_post_ready_runtime_sec = (
            float(post_ready) if isinstance(post_ready, (int, float)) and post_ready > 0 else 0.0
        )

    def _anchor_current_best(self, result: dict) -> None:
        tput = result.get("output_throughput")
        warmup_anchor = result.get("warmup_round_tput")
        measured = float(tput) if isinstance(tput, (int, float)) else None
        anchor_tput = float(self.shared_state.baseline_tput or 0.0)
        current_best = {
            "action": "baseline",
            "tput": anchor_tput if anchor_tput > 0 else measured,
            "hot_tput": measured,
            "cold_tput": (
                float(warmup_anchor) if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0 else None
            ),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            **constraint_fields(result),
            "tpot_mean_ms": result.get("tpot_mean_ms"),
            "input_throughput": result.get("input_throughput"),
            "total_throughput": result.get("total_token_throughput"),
            "tpot_p90_ms": result.get("tpot_p90_ms"),
            "e2e_norm_intvty_p90": result.get("e2e_norm_intvty_p90"),
            "e2e_norm_intvty_p50": result.get("e2e_norm_intvty_p50"),
            "duration_seconds": result.get("duration_seconds"),
            "request_error_rate": result.get("request_error_rate"),
            "output_tput_per_gpu": result.get("output_tput_per_gpu"),
            "workspace": result.get("workspace"),
            **_baseline_perf_axes(self.shared_state.baseline_perf),
        }
        self.shared_state.current_best = current_best
        # Reads the current_best just assigned, so it has to follow it.
        self._stamp_current_best_measurement(result)
        from hyperloom.orchestrator.measurement.baseline_budget import baseline_budget_stop_reason

        if stop_reason := baseline_budget_stop_reason(self.shared_state, result):
            self.shared_state.set_stop_reason(stop_reason)

    def _baseline_audit_extras(
        self,
        result: dict,
        task_params: Mapping[str, Any],
        held_reason: str,
        prior_anchor: float,
    ) -> dict[str, Any]:
        tput = result.get("output_throughput")
        measured = float(tput) if isinstance(tput, (int, float)) else None
        extras: dict[str, Any] = {
            "materialized_config": result.get("materialized_config"),
            "accuracy": result.get("accuracy"),
            "baseline_tput": measured,
            # Stamp the canonical params fingerprint for the self-loop denial helper.
            "fingerprint": self.baseline_params_fingerprint(task_params),
            "is_revalidation": bool(task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON),
            "enablement_succeeded": bool(getattr(self.shared_state.enablement, "succeeded", False)),
            "enablement_accuracy_floor": float(getattr(self.shared_state.enablement, "accuracy_floor", 0.0) or 0.0),
        }
        if held_reason and measured is not None and measured > 0:
            extras["anchor_kept_tput"] = prior_anchor
            extras["anchor_held"] = {"reason": held_reason, "prior": prior_anchor, "measured": measured}
        # Present only when the probe cut a runaway eval short; explains a ~0 accuracy.
        if result.get("eval_probe"):
            extras["eval_probe"] = result["eval_probe"]
        return extras

    async def _bootstrap_prelude_after_baseline(self, tput: Any) -> None:
        # History injection fires regardless of --no-warm-replay.
        self._coord.phase_prelude.inject_warm_recipe_history_into_ledger()
        # Warm-recipe replay, anchored on the hot baseline_tput contract.
        await self._coord.phase_prelude.maybe_enqueue_warm_replay(
            baseline_tput=float(self.shared_state.baseline_tput or tput),
        )
        # Auto-analysis (roofline / profile); may defer.
        await self._coord.phase_prelude.maybe_enqueue_prelude_initial_analysis_after_baseline(
            baseline_tput=float(tput),
        )
        # Research scout and static recon: parallel, read-only, CPU-only. Recon
        # seeds bridge candidates as gaps[] before the optimisation phase starts.
        await self._coord.phase_internal.maybe_enqueue_prelude_research_scout()
        await self._coord.phase_internal.maybe_enqueue_prelude_static_recon()

    async def drain_queued_baselines(self, *, reason: str) -> list[str]:
        """Cancel redundant queued baselines, preserving enablement revalidation.

        Args:
            reason: Stamped onto the cancellation history and the observation.

        Returns:
            The cancelled task ids (empty when the queue held none).
        """
        spared = await self._enablement_revalidation_task_ids()
        try:
            cancelled = await self.tasks.cancel_family(
                ["baseline"],
                reason=reason,
                exclude_task_ids=spared,
            )
        except (sqlite3.Error, ValueError):
            log.exception("baseline drain: cancel_family failed")
            return []
        if not cancelled:
            return []
        log.info(
            "baseline drain: cancelled %d queued baseline task(s) (reason=%s, spared=%d)",
            len(cancelled),
            reason,
            len(spared),
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "baseline_drain",
                "reason": reason,
                "cancelled_task_ids": cancelled,
                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
            },
        )
        return cancelled

    async def _enablement_revalidation_task_ids(self) -> set[str]:
        """Return queued enablement-revalidation baseline task IDs."""
        spared: set[str] = set()
        tracked = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
        if tracked:
            spared.add(tracked)
        for task in await self.tasks.queued():
            if str(getattr(task, "kind", "") or "") != "baseline":
                continue
            if (getattr(task, "params", None) or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
                spared.add(str(getattr(task, "task_id", "") or ""))
        return {t for t in spared if t}
