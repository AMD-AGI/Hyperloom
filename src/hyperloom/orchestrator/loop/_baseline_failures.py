# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a failed baseline costs the run: the streak, the stop reason, the lane it reopens."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from hyperloom.common.coerce import to_float
from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.breakdown.stop_reasons import AGENTX_PREFLIGHT_STOP_REASON

from ..actions.executors._accuracy_gate import (
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    ENABLEMENT_REVALIDATION_REASON,
)
from ..actions.executors._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS
from ..actions.executors._ray_serving import RAY_INFEASIBLE_MARKER
from ..actions.executors._subprocess_kill import AGENTX_PREFLIGHT_ERROR_CLASS
from ..actions.stop_attribution import stopped_by_the_run_class
from ..bringup import ARGV_INVALID
from ..bus.message_bus import Message
from ..phases.machine_state import PHASE_ENABLEMENT, PHASE_PRELUDE
from ..state._shared_state.attempt_audit import _AUDIT_ACTIONS
from ..state.failure_evidence import MAGPIE_NONZERO_INVALID_MEASUREMENT
from ..state.shared_state import ActionAttempt
from ..state.round_store import ABANDONED, FAILED
from ..state.task_registry import Task
from ._baseline_promotion import _adopt_eval_trigger, _preserves_measured_eval_trigger

log = logging.getLogger(__name__)


# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline failures.
_BASELINE_MAX_TOTAL_FAILURES: int = 3

_BASELINE_FAILED_STOP_REASON = "baseline_failed"

#: Baseline failures a retry cannot change, each mapped to the stop reason it ends the run with on its FIRST
#: occurrence, with no streak.
_STOPS_ON_FIRST_OCCURRENCE: dict[str, str] = {
    # A retry recomposes the same argv the installed parser already refused.
    ARGV_INVALID: ARGV_INVALID,
    # The server booted and measured a cold cache. The next attempt launches the same server and no framework patch
    # changes the operator's flag; ``_authorable_launch_log`` keeps it out of enablement.
    PREFIX_CACHE_OFF_ERROR_CLASS: _BASELINE_FAILED_STOP_REASON,
    # The round served its whole protocol and the harness around it still refused the result. The next
    # attempt runs the same harness over the same workload, so it is refused the same way.
    MAGPIE_NONZERO_INVALID_MEASUREMENT: _BASELINE_FAILED_STOP_REASON,
}


def _is_ray_cluster_infeasible(result_payload: Mapping[str, Any]) -> bool:
    """Whether a baseline failed because the Ray cluster can never place its round."""
    return RAY_INFEASIBLE_MARKER in str(result_payload.get("error") or "")


def _authorable_launch_log(result_payload: dict[str, Any] | None, *, err_class: str) -> str:
    """The failed boot's text the next enablement round authors against, or ``""``.

    A failure that never reached a framework error is no wall: as
    integrate_patch's "before" boot it passes any patch that starts the server.
    """
    if not isinstance(result_payload, dict) or _is_ray_cluster_infeasible(result_payload):
        return ""
    if err_class in (
        "fast_exit_arg_error",
        AGENTX_PREFLIGHT_ERROR_CLASS,
        PREFIX_CACHE_OFF_ERROR_CLASS,
    ) or stopped_by_the_run_class(err_class):
        return ""
    parts: list[str] = []
    for key in ("error", "stderr", "log_tail", "log_excerpt", "traceback", "reason"):
        val = result_payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, (list, tuple)):
            joined = "\n".join(str(x) for x in val if str(x).strip())
            if joined.strip():
                parts.append(joined.strip())
    return "\n".join(parts).strip()


class BaselineFailureMixin:
    """Recording a baseline failure and charging it against the run."""

    def _persist_eval_failure(self, result_payload: dict[str, Any]) -> None:
        """Persist an eval-rooted baseline failure so enablement can re-run it."""
        state = self.shared_state
        was_validation_pending = bool(state.enablement.validation_pending)
        stored_kind = str(state.enablement.baseline_eval_kind or "")
        preserve_measured_trigger = _preserves_measured_eval_trigger(state, result_payload)
        state.enablement.origin = "eval"
        state.enablement.pending = True
        # A failed revalidation reopens the authoring loop; its cost is charged
        # against the round ledger by the caller that settles the round.
        if was_validation_pending:
            state.enablement.validation_pending = False
            enablement_event.record_revalidation_outcome(
                enablement_event.RevalidationOutcome(
                    generation=int(state.enablement.revalidation_generation or 0),
                    promoted=False,
                    task_id=str(state.enablement.revalidation_task_id or ""),
                    accuracy=result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY),
                    accuracy_floor=state.enablement.accuracy_floor,
                    error_class=str(result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY) or ""),
                    reason="revalidation eval failed",
                ),
            )
        floor = to_float(result_payload.get(BASELINE_EVAL_ACCURACY_FLOOR_KEY))
        if floor is not None:
            state.enablement.accuracy_floor = float(floor)
        if preserve_measured_trigger:
            log.info(
                "enablement: keeping measured trigger kind=%s (accuracy=%s task=%s); "
                "an eval-less baseline reported accuracy_unavailable and must not "
                "overwrite it",
                stored_kind,
                state.enablement.observed_accuracy,
                state.enablement.observed_task,
            )
        else:
            _adopt_eval_trigger(state, result_payload)
        self._record_enablement_eval_trigger()

    def _record_enablement_eval_trigger(self) -> None:
        """Open the enablement lane's event on the eval failure that triggered it.

        Recorded here rather than derived at export: ``origin`` is cleared when
        the lane succeeds, so the fields it reads do not all survive the run.
        Only the first trigger is kept, which is what stops an eval-less
        re-baseline downgrading a measured trigger to an empty one.
        """
        state = self.shared_state
        enablement_event.record_trigger(
            enablement_event.LaneTrigger(
                origin=enablement_event.ORIGIN_EVAL,
                mode=str(state.enablement_mode or ""),
                kind=str(state.enablement.baseline_eval_kind or ""),
                evidence=str(state.enablement.baseline_eval_evidence or ""),
                observed_accuracy=state.enablement.observed_accuracy,
                accuracy_floor=state.enablement.accuracy_floor,
                observed_task=state.enablement.observed_task,
                observed_metric=state.enablement.observed_metric,
                eval_contract_fingerprint=state.enablement.eval_contract_fingerprint,
                probe_config_path=state.enablement.probe_config_path,
            ),
        )

    async def _reopen_revalidation_window(self) -> None:
        """Leave an enablement revalidation window open for a round the run stopped.

        A round the run stopped measured nothing, so the window stays open and
        no observation is charged.

        The generation advances for the same reason opening a window does: the
        next enqueue's idempotency key must not resolve to the row the run
        stopped. The tracked id goes with it, since the row it names is spent and
        the next enqueue records its own.

        Two callers reach the same round by different routes: the writeback, when
        the reaped row's result is routed, and the resume recovery, when the row
        was cancelled at dispatch and so produced no result to route at all.
        """
        state = self.shared_state
        state.enablement.revalidation_generation = int(state.enablement.revalidation_generation or 0) + 1
        state.enablement.revalidation_task_id = ""
        await self._coord.enablement_lane.settle_enablement_round(ABANDONED, reason="revalidation_stopped_by_the_run")

    async def _record_revalidation_not_promoted(
        self,
        *,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
        stopped_by_the_run: bool,
    ) -> None:
        """Close out an enablement revalidation baseline that did not promote.

        A genuine failure -- boot, OOM, timeout, eval -- closes the revalidation
        window, reopens the authoring loop, and charges its observation to the
        round ledger.

        A round the run stopped is none of those things; what it gets instead, and
        why, is :meth:`_reopen_revalidation_window`.

        ``result_payload`` is read for the launch/traceback text, ``err_class`` for the
        log line, and ``stopped_by_the_run`` says the round, not the baseline, ended.
        """
        state = self.shared_state
        generation = int(state.enablement.revalidation_generation or 0)
        floor = state.enablement.accuracy_floor
        tid = str(task.task_id or "")
        if stopped_by_the_run:
            await self._reopen_revalidation_window()
            enablement_event.record_revalidation_stopped(generation=generation, task_id=tid, accuracy_floor=floor)
        else:
            state.enablement.revalidation_task_id = ""
            state.enablement.validation_pending = False
            await self._coord.enablement_lane.settle_enablement_round(FAILED, reason=err_class or "revalidation_failed")
            enablement_event.record_revalidation_outcome(
                enablement_event.RevalidationOutcome(
                    generation=generation,
                    promoted=False,
                    task_id=tid,
                    accuracy_floor=floor,
                    error_class=str(err_class or ""),
                    reason="revalidation baseline failed",
                ),
            )
        launch_log = _authorable_launch_log(result_payload, err_class=str(err_class or ""))
        if launch_log:
            observed = str(result_payload.get("boot_observation_path") or "")
            state.enablement.record_launch_observation(log=launch_log, observation_path=observed)
        log.warning(
            "enablement revalidation task %s %s (error_class=%s); pending=%s rearm=%s",
            task.task_id,
            "was stopped by the run" if stopped_by_the_run else "failed",
            err_class,
            bool(state.enablement.validation_pending),
            not bool(state.stop_reason),
        )

    def _generic_baseline_failure_stops(
        self,
        task: Task,
        err_class: str,
        result_payload: Mapping[str, Any],
        *,
        retries_suppressed: bool,
    ) -> bool:
        """Count a baseline failure of no specially handled class; whether it ends the run.

        Args:
            task: The failed baseline task.
            err_class: The failure's ``error_class``.
            result_payload: The failed result.
            retries_suppressed: A pending eval or an open enablement lane owns the
                retries, so the streak alone does not stop the run.

        Returns:
            ``True`` when the run should stop as ``baseline_failed``.
        """
        # A process that exited with the same output as last time (modulo numbers)
        # will do so again: stop at two instead of spending a third attempt. The
        # executor decides which classes are worth signing and signs them from the
        # process's whole output rather than the 2000-character ``error`` tail, so a
        # signature is present exactly when it identifies the failure.
        signature = str(result_payload.get("output_signature") or "")
        repeated = bool(
            signature
            and self.shared_state.baseline_failure_streak >= 1
            and signature == self.shared_state.baseline_last_failure_signature
        )
        self.shared_state.baseline_failure_streak += 1
        self.shared_state.baseline_last_failure_signature = signature
        streak = self.shared_state.baseline_failure_streak
        if _is_ray_cluster_infeasible(result_payload):
            # The Ray cluster cannot place the round, and the runtime has already
            # repaired what it safely could (a lone local head without
            # serving_slot). A retry meets the same cluster and no framework patch
            # changes it, so stop on the first one, like the AgentX preflight.
            # Waiting for a second would not be bounded: with no launch log to
            # author against, the enablement lane this failure opens never
            # schedules another baseline.
            log.error(
                "baseline %s failed because the Ray cluster cannot place it; stopping the run: %s",
                task.task_id,
                result_payload.get("error") or err_class,
            )
            return True
        return not retries_suppressed and (streak >= 3 or (streak >= 2 and repeated))

    def _arms_baseline_failure_gates(self, err_class: str) -> bool:
        """Whether a failed baseline goes through the streak and stop gates.

        Only while no baseline has succeeded yet, except a cache-off refusal:
        a re-baseline refused for it stops the run as the first baseline does,
        since the next round would launch the same server.
        """
        return self.shared_state.baseline_tput <= 0 or err_class == PREFIX_CACHE_OFF_ERROR_CLASS

    async def handle_unpromotable_result(
        self,
        task: Task,
        result: dict[str, Any] | None,
    ) -> None:
        """Record a failed / unpromotable task result into SharedState: append to last_action_failures (+ a failed attempts row for _AUDIT_ACTIONS) and apply the baseline failure_streak/stop_reason gates.

        Args:
            task: The failed/unpromotable task.
            result: The task result payload; ``None`` is treated as an empty
                result.
        """
        result_payload = dict(result or {})
        if task.kind == "conc_sweep" and not result_payload.get("status"):
            result_payload["status"] = "failed"
        if task.kind == "explore" and bool((task.params or {}).get("geak_fallback")):
            self._close_failed_geak_revalidation(result_payload)
        # Per-action audit (failed attempt) for the in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint for the self-loop denial helper.
            if task.kind == "baseline":
                audit_extras["fingerprint"] = self.baseline_params_fingerprint(task.params)
            self.shared_state.record_action_attempt(
                action=task.kind,
                attempt=ActionAttempt(
                    task_id=task.task_id,
                    status="failed",
                    decision="no_promote",
                    result=result_payload,
                    extras=audit_extras,
                ),
            )
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        if task.kind == "conc_sweep":
            # No audit attempt here: ``record_action_attempt`` returns early for
            # any kind outside ``_AUDIT_ACTIONS``, which conc_sweep is not in.
            # The conc_sweep event carries these facts instead.
            self.shared_state.record_conc_sweep(result_payload)
        # An upstream-PR candidate task that settles failed/empty never reaches
        # the promote branch that writes the terminal progress row; stamp
        # no_result_failed so the pump does not re-select it every tick.
        self._coord.phase_framework.record_unpromoted_candidate(task, result_payload)
        baseline_event_payload: dict[str, Any] | None = None
        err_class = result_payload.get("error_class", "")
        # Only arm the streaks and stop gates while no baseline has succeeded yet.
        if task.kind == "baseline" and self._arms_baseline_failure_gates(err_class):
            baseline_event_payload = await self._charge_baseline_failure(task, result_payload, err_class)
        # Mirror the promote-path roofline failure handling: bump streak, clear gate, warn.
        if task.kind == "roofline":
            self.shared_state.roofline_failure_streak += 1
            if self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
            log.warning(
                "Auto-roofline %s failed (reason=%s phase=%s "
                "error_class=%s); continuing in degraded mode "
                "(specialists / explore proceed without a fresh "
                "analysis_md). No retry, no fallback.",
                task.task_id,
                str((task.params or {}).get("reason") or ""),
                result_payload.get("phase"),
                result_payload.get("error_class"),
            )
        elif task.kind == "profile" and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            # With enable_roofline off the initial analysis is a profile; leaving its id set would hold PRELUDE.
            self.shared_state.auto_roofline_pending_task_id = ""
        self.shared_state.save(self.session_dir)
        if baseline_event_payload is not None:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "event",
                    baseline_event_payload,
                )
            )

    def _tracks_open_revalidation(self, task: Task) -> bool:
        enablement = self.shared_state.enablement
        if not bool(getattr(enablement, "validation_pending", False)):
            return False
        if (task.params or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
            return True
        tracked = str(getattr(enablement, "revalidation_task_id", "") or "").strip()
        return bool(tracked) and tracked == str(task.task_id or "")

    async def _charge_baseline_failure(
        self,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
    ) -> dict[str, Any]:
        from ..actions.executors._accuracy_gate import eval_enablement_allowed
        from ..actions.executors._multi_node_env import is_multi_node

        stopped_by_the_run = stopped_by_the_run_class(err_class) is not None
        eval_failed = bool(result_payload.get(BASELINE_EVAL_FAILED_KEY))
        if self._tracks_open_revalidation(task):
            await self._record_revalidation_not_promoted(
                task=task,
                result_payload=result_payload,
                err_class=err_class,
                stopped_by_the_run=stopped_by_the_run,
            )
        # Single-node eval-pending failure: throughput measured fine and the eval
        # is expected to re-run under enablement, so do not spend the
        # baseline_failed budget yet. Multi-node keeps the strict backstop, and
        # so does a session that never admitted the eval lane -- nothing would
        # re-run the eval, so holding the budget just stalls the run. Baseline
        # failures are also the ENABLEMENT phase's own input, and its attempt cap
        # bounds them -- including a PRELUDE retry that settles before the transition.
        eval_pending_suppress = eval_failed and not is_multi_node() and eval_enablement_allowed(self.shared_state)
        phase = str(self.shared_state.phase or "").strip().upper()
        in_enablement = phase == PHASE_ENABLEMENT or (
            phase == PHASE_PRELUDE and self._coord.enablement_lane.enablement_admitted()
        )
        retries_suppressed = bool(eval_pending_suppress or in_enablement)
        if eval_failed:
            self._persist_eval_failure(result_payload)
        self._stop_the_run_on_baseline_failure(task, result_payload, err_class, retries_suppressed=retries_suppressed)
        # Combined backstop: count ALL baseline failures so mixed error_classes
        # that split the per-class streaks still fast-fail.
        if not stopped_by_the_run:
            self.shared_state.baseline_total_failures += 1
        if (
            self.shared_state.baseline_total_failures >= _BASELINE_MAX_TOTAL_FAILURES
            and not self.shared_state.stop_reason
            and not retries_suppressed
        ):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)
        self._arm_eager_baseline_fallback(task, err_class)
        self._stash_enablement_launch_log(result_payload, err_class)
        return {
            "kind": "baseline_not_promoted",
            "task_id": task.task_id,
            "failure_streak": self.shared_state.baseline_failure_streak,
            "arg_error_streak": self.shared_state.baseline_arg_error_streak,
            "stop_reason": self.shared_state.stop_reason,
            "result_status": result_payload.get("status"),
            "error_class": err_class,
            "baseline_eval_failed": eval_failed,
        }

    def _stop_the_run_on_baseline_failure(
        self,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
        *,
        retries_suppressed: bool,
    ) -> None:
        if stopped_by_the_run_class(err_class) is not None:
            log.warning(
                "baseline %s was stopped by the run (%s); the failure streak stays at %d "
                "because nothing about the baseline was measured",
                task.task_id,
                err_class,
                self.shared_state.baseline_failure_streak,
            )
        elif err_class in _STOPS_ON_FIRST_OCCURRENCE:
            self.shared_state.set_stop_reason(_STOPS_ON_FIRST_OCCURRENCE[err_class])
        elif err_class == "fast_exit_arg_error":
            self.shared_state.baseline_arg_error_streak += 1
            if self.shared_state.baseline_arg_error_streak >= 2:
                self.shared_state.set_stop_reason("baseline_arg_error")
        elif err_class == AGENTX_PREFLIGHT_ERROR_CLASS:
            # The delegated client must pass its environment preflight before framework optimization.
            log.error(
                "baseline %s failed the agentic client preflight; setup must "
                "resolve it before resuming. This is an environment gap, not something a "
                "framework patch can close: %s",
                task.task_id,
                result_payload.get("error") or err_class,
            )
            self.shared_state.set_stop_reason(AGENTX_PREFLIGHT_STOP_REASON)
        # ``baseline_arg_error_streak`` is deliberately left alone on every other
        # class: a boot that failed some other way is no evidence the arguments
        # were fixed.
        elif self._generic_baseline_failure_stops(
            task, err_class, result_payload, retries_suppressed=retries_suppressed
        ):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)

    def _arm_eager_baseline_fallback(self, task: Task, err_class: str) -> None:
        if err_class != "cuda_graph_capture_failed" or self.shared_state.baseline_eager_fallback:
            return
        self.shared_state.baseline_eager_fallback = True
        log.warning(
            "baseline %s hit cuda-graph capture failure; arming "
            "disable-cuda-graph fallback for the next baseline retry",
            task.task_id,
        )

    def _stash_enablement_launch_log(self, result_payload: dict[str, Any], err_class: str) -> None:
        launch_log = _authorable_launch_log(result_payload, err_class=err_class)
        if not launch_log:
            return
        self.shared_state.enablement.record_launch_observation(
            log=launch_log, observation_path=str(result_payload.get("boot_observation_path") or "")
        )
        if str(self.shared_state.enablement.origin or ""):
            return
        enablement_event.record_trigger(
            enablement_event.LaneTrigger(
                origin=enablement_event.ORIGIN_BOOT,
                mode=str(self.shared_state.enablement_mode or ""),
                kind=str(err_class or ""),
                evidence=launch_log,
            ),
        )
