# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a failed baseline costs the run: the streak, the stop reason, the lane it reopens."""

from __future__ import annotations

import logging
from typing import Any

from inference_suite.contracts import STEPS_BEFORE_SPAWN, FailedStep

from hyperloom_common.coerce import to_float
from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.breakdown.stop_reasons import AGENTX_PREFLIGHT_STOP_REASON, HOST_GPUS_UNAVAILABLE

from ..actions.executors._accuracy_gate import (
    ACCURACY_DEFINITION_DIGEST_KEY,
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    ENABLEMENT_REVALIDATION_REASON,
)
from ..actions.executors.workload._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS
from ..actions.executors import failed_step_error_class
from ..actions.executors._subprocess_kill import AGENTX_PREFLIGHT_ERROR_CLASS
from ..actions.stop_attribution import stopped_by_the_run_class
from ..bringup import ARGV_INVALID
from ..phases.machine_state import PHASE_ENABLEMENT, PHASE_PRELUDE
from ..state.failure_evidence import ROUND_NONZERO_INVALID_MEASUREMENT
from ..state.shared_state import BaselineAttempt
from ..state.round_store import ABANDONED, FAILED
from ..state.task_registry import Task
from ._baseline_promotion import _adopt_eval_trigger, _preserves_measured_eval_trigger

log = logging.getLogger(__name__)


# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline failures.
_BASELINE_MAX_TOTAL_FAILURES: int = 3

_BASELINE_FAILED_STOP_REASON = "baseline_failed"

#: inference_suite refused the round before it spawned a server: the operator's environment or config, never a
#: framework error an enablement round could author against.
_REFUSED_BEFORE_SPAWN_CLASSES: frozenset[str] = frozenset(failed_step_error_class(step) for step in STEPS_BEFORE_SPAWN)
#: Refusals before spawn that the next attempt can clear by itself (a port or a held server freed meanwhile), so they
#: count toward the streak instead of stopping the run at once.
_TRANSIENT_BEFORE_SPAWN_CLASSES: frozenset[str] = frozenset(
    failed_step_error_class(step) for step in (FailedStep.PORT_IN_USE, FailedStep.REUSE_NOT_OWNED)
)

#: Baseline failures a retry cannot change, each mapped to the stop reason it ends the run with on its FIRST
#: occurrence, with no streak.
_STOPS_ON_FIRST_OCCURRENCE: dict[str, str] = {
    **dict.fromkeys(_REFUSED_BEFORE_SPAWN_CLASSES - _TRANSIENT_BEFORE_SPAWN_CLASSES, _BASELINE_FAILED_STOP_REASON),
    # Refused before spawn, or by Hyperloom's own argv check: a retry recomposes the same argv.
    ARGV_INVALID: ARGV_INVALID,
    # The server booted and measured a cold cache. The next attempt launches the same server and no framework patch
    # changes the operator's flag; ``_authorable_launch_log`` keeps it out of enablement.
    PREFIX_CACHE_OFF_ERROR_CLASS: _BASELINE_FAILED_STOP_REASON,
    # The harness refused the result after the round served its whole protocol. The next attempt runs the same
    # harness over the same workload, so it is refused the same way. ``client_failed`` stays on the streak: the suite
    # names any nonzero client exit so, a transient refusal of a download included.
    ROUND_NONZERO_INVALID_MEASUREMENT: _BASELINE_FAILED_STOP_REASON,
    # Refused before spawn: the host lacks the idle GPUs the round needs. Only the operator can free them.
    HOST_GPUS_UNAVAILABLE: HOST_GPUS_UNAVAILABLE,
}


def _authorable_launch_log(result_payload: dict[str, Any] | None, *, err_class: str) -> str:
    """The failed boot's text the next enablement round authors against, or ``""``.

    A failure that never reached a framework error is no wall: as
    integrate_patch's "before" boot it passes any patch that starts the server.
    """
    if not isinstance(result_payload, dict):
        return ""
    if (
        err_class in (AGENTX_PREFLIGHT_ERROR_CLASS, PREFIX_CACHE_OFF_ERROR_CLASS)
        or err_class in _REFUSED_BEFORE_SPAWN_CLASSES
        or stopped_by_the_run_class(err_class)
    ):
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
                "enablement: keeping measured trigger kind=%s; an eval-less baseline reported "
                "accuracy_unavailable and must not overwrite it",
                stored_kind,
            )
        else:
            _adopt_eval_trigger(state, result_payload)
        # Recorded here rather than derived at export: ``origin`` is cleared when the lane succeeds. The recorder
        # keeps the first trigger, which is what stops an eval-less re-baseline downgrading a measured one.
        enablement_event.record_trigger(
            enablement_event.LaneTrigger(
                origin=enablement_event.ORIGIN_EVAL,
                mode=str(state.enablement_mode or ""),
                kind=result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY),
                evidence=result_payload.get(BASELINE_EVAL_EVIDENCE_KEY),
                observed_accuracy=result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY),
                accuracy_floor=state.enablement.accuracy_floor,
                observed_task=result_payload.get("accuracy_task"),
                observed_metric=result_payload.get("accuracy_metric"),
                eval_definition_digest=result_payload.get(ACCURACY_DEFINITION_DIGEST_KEY),
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
        await self._coord.enablement_lane.settle_enablement_round(ABANDONED)

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
            await self._coord.enablement_lane.settle_enablement_round(FAILED)
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

    def _generic_baseline_failure_stops(self, *, retries_suppressed: bool) -> bool:
        """Count a baseline failure of no specially handled class; whether it ends the run.

        Args:
            retries_suppressed: A pending eval or an open enablement lane owns the
                retries, so the streak alone does not stop the run.

        Returns:
            ``True`` when the run should stop as ``baseline_failed``.
        """
        self.shared_state.baseline_failure_streak += 1
        self.shared_state.baseline_failures_this_leg += 1
        streak = self.shared_state.baseline_failure_streak
        return not retries_suppressed and streak >= 3

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
        """Record a failed / unpromotable task result into SharedState: append to last_action_failures (and, for a baseline, replace last_baseline) and apply the baseline failure_streak/stop_reason gates.

        Args:
            task: The failed/unpromotable task.
            result: The task result payload; ``None`` is treated as an empty
                result.
        """
        result_payload = dict(result or {})
        if task.kind == "explore" and bool((task.params or {}).get("geak_fallback")):
            self._close_failed_geak_revalidation(result_payload)
        if task.kind == "baseline":
            self.shared_state.record_baseline_attempt(
                BaselineAttempt(
                    task_id=task.task_id,
                    status="failed",
                    decision="no_promote",
                    result=result_payload,
                    extras={"fingerprint": self.baseline_params_fingerprint(task.params)},
                )
            )
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        # An upstream-PR candidate task that settles failed/empty never reaches
        # the promote branch that writes the terminal progress row; stamp
        # no_result_failed so the pump does not re-select it every tick.
        self._coord.phase_framework.record_unpromoted_candidate(task, result_payload)
        err_class = result_payload.get("error_class", "")
        # Only arm the streaks and stop gates while no baseline has succeeded yet.
        if task.kind == "baseline" and self._arms_baseline_failure_gates(err_class):
            await self._charge_baseline_failure(task, result_payload, err_class)
        # A failed roofline is charged here and nowhere else: bump streak, clear gate, warn.
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
    ) -> None:
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
        # Combined backstop: count ALL baseline failures, the classes the streak
        # does not count included, so a mix of classes still fast-fails.
        if not stopped_by_the_run:
            self.shared_state.baseline_total_failures += 1
        if (
            self.shared_state.baseline_total_failures >= _BASELINE_MAX_TOTAL_FAILURES
            and not self.shared_state.stop_reason
            and not retries_suppressed
        ):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)
        self._stash_enablement_launch_log(result_payload, err_class)

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
            log.error(
                "baseline %s failed with %s, which a retry cannot change; stopping the run: %s",
                task.task_id,
                err_class,
                result_payload.get("error") or err_class,
            )
            self.shared_state.set_stop_reason(_STOPS_ON_FIRST_OCCURRENCE[err_class])
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
        elif self._generic_baseline_failure_stops(retries_suppressed=retries_suppressed):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)

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
