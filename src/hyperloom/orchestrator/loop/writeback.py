# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Any
from hyperloom.common.coerce import to_float
from hyperloom.common.payload_aliases import read_extra_server_args
from ..state.optimization_journal import (
    Journal,
    classify_change_kind,
    operation_kind_for,
)
from hyperloom.inference_optimizer.protocol.intent import Intent
from ..bus.message_bus import Message
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _MIN_KERNEL_ENGAGED_GAIN_PCT,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _geak_revalidation_decision,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)
from ..policy.gate import (
    PolicyDenied,
)
from ..state.task_registry import Task
from ..actions.executors.benchmark_result import is_valid_measurement

from .coordinator import (
    _AUDIT_ACTIONS,
    _BASELINE_MAX_TOTAL_FAILURES,
    _DEFAULT_RESUME_DRIFT_FLOOR_PCT,
    _SEVERITY_CRASH,
    _SEVERITY_REGRESS,
    _extract_enablement_launch_log,
)
import logging as _logging
log = _logging.getLogger(__name__)


class WritebackCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    # REQUEST / RESPONSE (Plan A)
    # ------------------------------------------------------------------
    def _emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event.

        Best-effort by design: operator-facing logging must never break the
        orchestration loop, so any failure is swallowed at debug level.

        Args:
            step: The machine step name (resolved to a human label downstream).
            status: The lifecycle status (e.g. START / END / ERROR / ENTER).
            artifacts: Optional mapping of produced artifact paths.
            detail: Optional free-text detail.
            duration_s: Optional elapsed seconds for the step.
        """
        try:
            self.shared_state.record_lifecycle_event(
                step=step,
                status=status,
                artifacts=artifacts,
                detail=detail,
                duration_s=duration_s,
            )
            # Terminal events (END/ERROR) carry the produced artifact paths an
            # operator is waiting on — always flush them. Non-terminal markers
            # (START / phase ENTER) are debounced: skip the write if we flushed
            # within the last ``_lifecycle_save_min_interval_s`` seconds, since
            # the next terminal event (or a later marker past the window) will
            # persist the coalesced tail anyway.
            terminal = status in ("END", "ERROR")
            now = time.monotonic()
            if terminal or (now - self._lifecycle_last_save >= self._lifecycle_save_min_interval_s):
                self.shared_state.save(self.session_dir)
                self._coord._lifecycle_last_save = now
        except Exception:  # noqa: BLE001 — defensive
            log.debug(
                "Coordinator: lifecycle emit failed (step=%s status=%s)",
                step,
                status,
                exc_info=True,
            )

    # Bookkeeping
    async def _record_policy_denied(
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
    ) -> None:
        """Record a PolicyGate denial and apply escalation side effects.

        Publishes a ``policy_denied`` observation, records the denial streak,
        auto-prunes the action family at streak >= 5, and sets the
        ``policy_loop`` stop reason at streak >= 10.

        Args:
            source (str): The agent whose intent was denied.
            intent (Intent): The denied intent.
            denied (PolicyDenied): The denial carrying rule / hint / reason.
            action_name (str | None): Explicit action name override; falls back
                to ``intent.payload['action_name']``.
        """
        # Surface every PolicyGate denial in the standard process log (not just
        # on the bus) so security rejections — including the newly-gated
        # framework_source_root / CORE-field / path-containment checks — are
        # observable in ops logs. Denials are exceptional, so this is not noisy.
        log.warning(
            "PolicyGate denied intent: source=%s type=%s rule=%s reason=%s",
            source,
            intent.type.value,
            denied.rule,
            str(denied),
        )
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                source,
                "observation",
                {
                    "kind": "policy_denied",
                    "intent_type": intent.type.value,
                    "rule": denied.rule,
                    "hint": denied.hint,
                    "reason": str(denied),
                },
                priority=0,
            )
        )
        resolved_action = action_name or str((intent.payload or {}).get("action_name") or "")
        # Streak counter is a fact for LLM self-correction only; system no longer auto-prunes or stops on it (long-run continuity over loop stop-loss).
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        """Append a broadcast observation message to the bus.

        Args:
            source (str): The agent recording the observation.
            topic (str): The bus topic to publish under.
            payload (dict): The observation payload.
        """
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    def _record_kernel_opt_partial(self, result: dict[str, Any]) -> None:
        """Streaming callback for ``_run_optimization_batch`` sub-attempts: write each per-kernel entry to kernel_opt_attempts immediately so the next-tick prompt is accurate mid-batch.

        Args:
            result: One sub-attempt's per-kernel result dict.
        """
        try:
            self.shared_state.record_kernel_opt(result)
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            # Never let a per-sub-attempt hiccup poison the gather; the post-gather record_kernel_opt picks it up.
            log.exception(
                "_record_kernel_opt_partial failed for kernel_id=%s",
                (result or {}).get("kernel_id") if isinstance(result, dict) else None,
            )

    async def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Appends a deduped ``integrate`` entry to the optimization stack, mirrors
        the gain into the per-entry gain ledger, updates ``current_best`` and
        ``cumulative_gain`` / ``cumulative_gain_validated``, and fires a
        watermark roofline when the gain crosses the threshold. No-op when the
        result lacks a positive ``new_tput``.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        cb = self.shared_state.current_best or {}
        # Read result via the compat helper (handles legacy extra_sglang_args); cb is migrated at load time.
        extra_args = (
            read_extra_server_args(result) or (str(cb.get("extra_server_args") or "") if isinstance(cb, dict) else "")
        ).strip()
        apply_result = result.get("apply_result") or {}
        backup_manifest = apply_result.get("manifest_path") if isinstance(apply_result, dict) else None
        if not backup_manifest and isinstance(apply_result, dict):
            stack_applies = apply_result.get("stack_apply_results")
            if isinstance(stack_applies, list):
                for applied in stack_applies:
                    if isinstance(applied, dict) and applied.get("manifest_path"):
                        backup_manifest = applied.get("manifest_path")
                        break
        entry = {
            "action": "integrate",
            "kernel_id": result.get("kernel_id"),
            "patch_path": result.get("patch_path"),
            "target_file": result.get("target_file"),
            "backup_manifest": backup_manifest,
            "gain_pct": result.get("gain_pct"),
            "tput": float(new_tput),
            "workspace": result.get("workspace"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        stack_kernel_ids = result.get("stack_kernel_ids")
        if isinstance(stack_kernel_ids, list) and stack_kernel_ids:
            entry["stack_kernel_ids"] = [str(kid) for kid in stack_kernel_ids if str(kid)]
        integrate_gap_cid = str(result.get("gap_canonical_id") or "").strip()
        if integrate_gap_cid:
            entry["gap_canonical_id"] = integrate_gap_cid
        key = (entry["kernel_id"], entry["patch_path"], entry["target_file"])
        existing = {
            (item.get("kernel_id"), item.get("patch_path"), item.get("target_file"))
            for item in self.shared_state.optimization_stack
            if isinstance(item, dict) and item.get("action") == "integrate"
        }
        if key not in existing:
            self.shared_state.optimization_stack.append(entry)
            # Mirror into gain_per_stack_entry so breakdown attribution works without re-walking the event log.
            self.shared_state.append_stack_gain_entry(
                action="integrate",
                variant_name=entry.get("kernel_id"),
                new_tput=new_tput,
                extra_server_args=extra_args,
                ts=entry["ts"],
            )

        self.shared_state.current_best = {
            "action": "integrate",
            "tput": float(new_tput),
            "kernel_id": result.get("kernel_id"),
            "extra_server_args": extra_args,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            "e2el_mean_ms": result.get("e2el_mean_ms"),
            "tpot_mean_ms": result.get("tpot_mean_ms"),
            "workspace": result.get("workspace"),
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
            )
            # Integrate KEEP is already rebench-validated: promote into cumulative_gain_validated + watermark.
            validated_gain = (
                (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
            )
            self.shared_state.cumulative_gain_validated = float(validated_gain)
            self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
            self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
            await self._maybe_enqueue_watermark_roofline(
                reason="integrate_keep_watermark",
            )

    # Dispatcher (pulls queued tasks → SubAgentRunner)
    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        """Decide whether a settled task result should be promoted.

        Per-kind rules: baseline/profile require a valid measurement, sweep
        requires ``status == "succeeded"``, ``replay_warm_recipe`` always routes
        through promotion (it owns its own failure bookkeeping), and everything
        else is promotable unless ``status == "failed"``.

        Args:
            task_kind (str): The task's kind.
            result (dict[str, Any]): The task result payload.

        Returns:
            bool: ``True`` when the result should go through
                :meth:`_promote_to_shared_state`.
        """
        if not isinstance(result, dict):
            return False
        if task_kind in ("baseline", "profile"):
            return is_valid_measurement(result)
        if task_kind == "sweep":
            return result.get("status") == "succeeded"
        # replay_warm_recipe ALWAYS routes through _promote_warm_replay (owns succeeded/drift/FAILED + clears in_flight); else PRELUDE blocks forever.
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self,
        task: "Task",
        result: Any,
    ) -> None:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._record_intervention_for_task(task, result)

    async def _handle_unpromotable_result(
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
        any_changed = False
        # Per-action audit (failed attempt) for the 6 in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint so the self-loop denial helper detects "same params failed twice".
            if task.kind == "baseline":
                audit_extras["fingerprint"] = _baseline_params_fingerprint(task.params)
            self.shared_state.record_action_attempt(
                action=task.kind,
                task_id=task.task_id,
                status="failed",
                decision="no_promote",
                result=result_payload,
                extras=audit_extras,
            )
            any_changed = True
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        any_changed = True
        if task.kind == "conc_sweep":
            self.shared_state.record_action_attempt(
                action="conc_sweep",
                task_id=task.task_id,
                status=str(result_payload.get("status") or "failed"),
                decision="discarded",
                result=result_payload,
                extras={
                    "was_skipped": bool(result_payload.get("was_skipped", False)),
                    "skip_reason": result_payload.get("skip_reason"),
                    "budget_exhausted": bool(result_payload.get("budget_exhausted", False)),
                    "total_budget_sec": result_payload.get("total_budget_sec"),
                    "elapsed_sec": result_payload.get("elapsed_sec"),
                    "best_speedup": ((result_payload.get("summary") or {}).get("best_speedup")),
                    "best_conc": ((result_payload.get("summary") or {}).get("best_conc")),
                    "successful_pairs": ((result_payload.get("summary") or {}).get("successful_pairs")),
                    "report_path": result_payload.get("report_json_path"),
                },
            )
            self.shared_state.record_conc_sweep(result_payload)
        # FRAMEWORK apply/bench silent failure: a
        # framework_agent task that settles ``status="failed"`` (or empty) never
        # reaches the promote branch that writes the terminal progress row, so
        # without stamping here the candidate stays "unprocessed" and the pump
        # re-selects it every tick until the budget cap. Stamp no_result_failed.
        if task.kind == "framework_agent":
            cand = (task.params or {}).get("candidate")
            cand_id = self._framework_candidate_key(cand if isinstance(cand, dict) else None)
            if cand_id:
                self._stamp_framework_progress(
                    candidate_id=cand_id,
                    batch_id=str((task.params or {}).get("batch_id") or ""),
                    status="no_result_failed",
                    kept=False,
                    rationale=str(result_payload.get("reason") or result_payload.get("error") or "")[:500],
                    provenance="executor",
                    extra={"status": str(result_payload.get("status") or "")},
                )
        # Baseline-specific gates: streak counter + stop_reason + baseline_not_promoted event.
        # Fast arg errors (fast_exit_arg_error) get their own streak so
        # they don't burn the slow-baseline retry budget on deterministic
        # failures that the same params will never fix.
        baseline_event_payload: dict[str, Any] | None = None
        # Intentional: only arm/streak while no baseline has succeeded yet
        # (tput <= 0). On resume with an existing baseline we never re-arm the
        # eager fallback. Scope is baseline-only; explore/sweep do not benefit.
        if task.kind == "baseline" and self.shared_state.baseline_tput <= 0:
            err_class = result_payload.get("error_class", "")
            # Enablement-aware backstop suppression: while a serial enablement is
            # actively engaged, baseline boots re-fail *on purpose* — each round
            # clears gap #n and the next boot stops at a new, deeper gap #(n+1).
            # Those crashes are progress, so the ``baseline_failed`` fast-fail
            # (streak / total) must NOT fire here; the honest ``enablement_stalled``
            # cap (consecutive NO-progress rounds in _maybe_rearm_enablement) is
            # the correct fast-fail in this regime. Engaged = a progressing patch
            # already stacked OR a specialist currently dispatched/attempting.
            # ``fast_exit_arg_error`` is deterministic (a bad CLI arg the same
            # params never fix) and stays gated on its own streak regardless.
            enablement_engaged = bool(
                (getattr(self.shared_state, "enablement_kept_patches", None) or [])
                or getattr(self.shared_state, "enablement_dispatched", False)
                or int(getattr(self.shared_state, "enablement_attempts", 0) or 0) > 0
            )
            if err_class == "fast_exit_arg_error":
                self.shared_state.baseline_arg_error_streak += 1
                if self.shared_state.baseline_arg_error_streak >= 2:
                    self.shared_state.set_stop_reason("baseline_arg_error")
            else:
                self.shared_state.baseline_failure_streak += 1
                self.shared_state.baseline_arg_error_streak = 0
                if self.shared_state.baseline_failure_streak >= 3 and not enablement_engaged:
                    self.shared_state.set_stop_reason("baseline_failed")
            # Combined backstop: mixed error_classes split the per-class
            # streaks above so neither reaches its threshold and the session
            # burns the whole budget -> time_exhausted. Count ALL baseline
            # failures and fast-fail at the same 3-failure intent.
            self.shared_state.baseline_total_failures += 1
            if (
                self.shared_state.baseline_total_failures
                >= _BASELINE_MAX_TOTAL_FAILURES
                and not self.shared_state.stop_reason
                and not enablement_engaged
            ):
                self.shared_state.set_stop_reason("baseline_failed")
            # One-shot eager fallback: a (non-OOM) cuda-graph capture failure is
            # often recoverable by disabling cuda-graph capture. Arm it once.
            if err_class == "cuda_graph_capture_failed" and not self.shared_state.baseline_eager_fallback:
                self.shared_state.baseline_eager_fallback = True
                log.warning(
                    "baseline %s hit cuda-graph capture failure; arming "
                    "disable-cuda-graph fallback for the next baseline retry",
                    task.task_id,
                )
            # Stash the launch/traceback text for the FRAMEWORK pump to classify
            # and dispatch an enablement_specialist. Fast arg errors are excluded.
            if err_class != "fast_exit_arg_error":
                launch_log = _extract_enablement_launch_log(result_payload)
                if launch_log:
                    self.shared_state.enablement_launch_log = launch_log
            baseline_event_payload = {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "arg_error_streak": self.shared_state.baseline_arg_error_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": result_payload.get("status"),
                "error_class": err_class,
            }
            any_changed = True
        # Mirror the promote-path roofline failure handling: bump streak, clear auto-roofline gate, emit operator warning.
        if task.kind == "roofline":
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            if self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
            any_changed = True
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
        if any_changed:
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

    # Fact-write dispatcher (KEEP / REVERT entry point): route terminal results to journal + KB fact-write helpers.
    def _source_session_id(self) -> str:
        """Return the hyperloom-local session id used as source_session_id on KB fact writes.

        NOT a KB-side session id; prefers cortex_session_id, falls back to session_dir.name.

        Returns:
            The hyperloom-local session id (cortex_session_id when set, else
            ``session_dir.name``).
        """
        return str(getattr(self.shared_state, "cortex_session_id", "") or "") or self.session_dir.name

    async def _fact_write_hook(
        self,
        *,
        task: "Task",
        result: Any,
        kept: bool,
    ) -> None:
        """Per-task fact-write entry point (per_variant for explore grids, else per-task); best-effort, never raises.

        Args:
            task: The completed task being recorded.
            result: The task's :class:`SubAgentResult` (or result dict).
            kept: Whether the task's result was KEEP-promoted.
        """
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        source_session_id = self._source_session_id()
        per_variant = result_dict.get("per_variant_outcomes")
        if task.kind == "explore" and isinstance(per_variant, list) and per_variant:
            for vo in per_variant:
                try:
                    self._record_fact_per_variant(
                        task=task,
                        source_session_id=source_session_id,
                        variant_outcome=vo,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "fact-write per-variant failed (task=%s)",
                        task.task_id,
                    )
        else:
            try:
                self._record_fact_per_task(
                    task=task,
                    source_session_id=source_session_id,
                    result_dict=result_dict,
                    kept=kept,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "fact-write per-task failed (task=%s)",
                    task.task_id,
                )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; never crash on save
            log.exception("fact-write SharedState.save failed")

    def _ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume).

        Returns:
            The per-session :class:`Journal` instance (created on first call,
            with the baseline backfilled on subsequent calls).
        """
        existing = getattr(self, "_journal", None)
        if existing is None:
            ss = self.shared_state
            self._coord._journal = Journal.load_or_create(
                self.session_dir,
                session_id=str(getattr(ss, "cortex_session_id", "") or "")
                or str(getattr(ss, "session_id", "") or "")
                or self.session_dir.name,
                model=str(getattr(ss, "model_name", "") or ""),
                hardware=str(getattr(ss, "gpu_type", "") or ""),
                framework=str(getattr(ss, "framework", "") or ""),
                baseline_throughput=float(getattr(ss, "baseline_tput", 0.0) or 0.0),
            )
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0))
        return self._journal

    def _pitfall_severity_for(
        self,
        result_dict: dict[str, Any] | None,
    ) -> str | None:
        """Decide whether a failed result warrants a pitfall row (Threshold-B): crash/oom/hang → SEVERITY_CRASH; gain_pct ≤ -5% → SEVERITY_REGRESS; else None.

        Args:
            result_dict: The failed task's result dict; non-dict yields ``None``.

        Returns:
            The pitfall severity (``SEVERITY_CRASH`` / ``SEVERITY_REGRESS``), or
            ``None`` when no pitfall is warranted.
        """
        if not isinstance(result_dict, dict):
            return None
        error_class = str(result_dict.get("error_class") or "").lower()
        # ``detokenizer_stall`` is a hang in all but name (server ready, no
        # generation progress); record it as a crash-severity pitfall so the
        # offending variant config is remembered and not re-proposed, instead
        # of burning explore budget on the same stall again.
        if error_class in ("crash", "oom", "hang", "detokenizer_stall"):
            return _SEVERITY_CRASH
        status = str(result_dict.get("status") or "").lower()
        if status in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        gain = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain) if gain is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        if gain_pct is not None and gain_pct <= self.PITFALL_REGRESS_THRESHOLD_PCT:
            return _SEVERITY_REGRESS
        return None

    def _journal_entry_phase(self) -> str:
        """Return the current phase label for journal entries.

        Returns:
            str: The uppercased phase name, or ``"UNKNOWN"`` when unset.
        """
        return str(getattr(self.shared_state, "phase", "") or "").strip().upper() or "UNKNOWN"

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._record_fact_per_task(task=task, source_session_id=source_session_id, result_dict=result_dict, kept=kept)

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        gain_pct: float | None = None,  # optional measured gain, forwarded to the statement builder
        severity: str | None = None,
    ) -> str:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._build_statement(change=change, kind=kind, gain_pct=gain_pct, severity=severity)

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
    ) -> dict[str, Any]:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        from .result_recorder import ResultRecorder as _RR
        return _RR._build_measured_impact(gain_pct=gain_pct, throughput_after=throughput_after, stack_depth=stack_depth, measured_at=measured_at)

    @staticmethod
    def _predicted_gain(
        *sources: dict[str, Any] | None,
    ) -> float | None:
        """First parseable ``predicted_gain_pct`` across ordered sources.

        Sources are checked in order (e.g. variant_outcome → variant attrs →
        task params); a non-zero prediction wins. Returns ``None`` when none
        carry a usable value so the journal row stays ``predicted``-free for
        unpredicted (default-grid) changes rather than recording a fake 0.
        """
        for src in sources:
            if not isinstance(src, dict):
                continue
            val = to_float(src.get("predicted_gain_pct"))
            if val is not None and val != 0.0:
                return val
        return None

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._record_fact_per_variant(task=task, source_session_id=source_session_id, variant_outcome=variant_outcome)

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._collect_workload_tags()

    def _build_kernel_optimizations_from_state(self) -> list[dict[str, Any]]:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._build_kernel_optimizations_from_state()

    def _collect_attempt_provenance(
        self,
    ) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
        """Map proven optimizations to their research-hint origin from the gaps[] attempts ledger; returns (kept_sources by name/kernel, kept_by_gap by canonical_id, reverted_rows). Fail-soft.

        Returns:
            A ``(kept_sources, kept_by_gap, reverted_rows)`` tuple: KEEP'd
            provenance keyed by variant/kernel name, KEEP'd provenance keyed by
            gap canonical_id, and reverted-attempt rows.
        """
        kept_sources: dict[str, str] = {}
        kept_by_gap: dict[str, str] = {}
        reverted_rows: list[dict[str, Any]] = []
        gaps = getattr(self.shared_state, "gaps", []) or []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            provenance = str(gap.get("provenance") or "").strip()
            canonical = str(gap.get("canonical_id") or "").strip()
            for attempt in gap.get("attempts") or []:
                if not isinstance(attempt, dict):
                    continue
                variant = str(attempt.get("variant_name") or "").strip()
                kernel = str(attempt.get("kernel_id") or "").strip()
                outcome = str(attempt.get("outcome") or "").strip().upper()
                if outcome == "KEEP" and provenance:
                    if variant:
                        kept_sources.setdefault(variant, provenance)
                    if kernel:
                        kept_sources.setdefault(kernel, provenance)
                    if canonical:
                        kept_by_gap.setdefault(canonical, provenance)
                elif outcome == "REVERT" and (variant or kernel):
                    row: dict[str, Any] = {
                        "name": variant or kernel,
                        "reason": "reverted",
                        "gain_pct": attempt.get("gain_pct"),
                    }
                    if provenance:
                        row["source"] = provenance
                    reverted_rows.append(row)
        return kept_sources, kept_by_gap, reverted_rows

    def _build_recipe_attrs_from_state(self) -> dict[str, Any]:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder._build_recipe_attrs_from_state()

    def cortex_finalize_recipe_and_journal(self) -> None:
        """Thin forwarding shim — implementation in :class:`ResultRecorder`."""
        return self.recorder.cortex_finalize_recipe_and_journal()

    def _lift_to_current_best(
        self,
        task_kind: str,
        best_tput: float,
        bv: dict[str, Any],
        *,
        gap_canonical_id: str = "",
    ) -> None:
        """Update SharedState.current_best + recompute cumulative_gain; gap_canonical_id (when known) is stamped onto the stack entry so provenance resolves by gap id not name.

        Args:
            task_kind: The action kind that produced the winner (stamped on the
                stack entry / current_best).
            best_tput: The winning variant's measured throughput.
            bv: The winning variant dict (args, envs, metrics, provenance).
            gap_canonical_id: When known, stamped onto the stack entry so
                provenance resolves by gap id rather than name.
        """
        previous = self.shared_state.current_best or {}
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        base_args = ""
        if isinstance(previous, dict):
            base_args = str(previous.get("extra_server_args") or "").strip()
        candidate_args = ""
        if isinstance(bv, dict):
            candidate_args = str(bv.get("candidate_extra_server_args") or bv.get("extra_server_args") or "").strip()
        full_args = ""
        if isinstance(bv, dict):
            full_args = str(bv.get("extra_server_args") or bv.get("extra_sglang_args") or "").strip()
        # Build cumulative launch args without double-stacking; helper dedupes repeated --flag pairs (last wins).
        full_args = _merge_cumulative_extra_sglang_args(
            base_args,
            candidate_args,
            full_args,
        )

        variant_name = bv.get("name") if isinstance(bv, dict) else None
        if candidate_args or variant_name:
            existing = {
                (str(e.get("action")), str(e.get("variant_name")))
                for e in self.shared_state.optimization_stack
                if isinstance(e, dict)
            }
            key = (task_kind, str(variant_name or ""))
            if key not in existing:
                stack_entry: dict[str, Any] = {
                    "action": task_kind,
                    "variant_name": variant_name,
                    "candidate_extra_server_args": candidate_args,
                    "extra_envs": (dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}),
                    "tput": float(best_tput),
                    "workspace": (bv.get("workspace") if isinstance(bv, dict) else None),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                if gap_canonical_id:
                    stack_entry["gap_canonical_id"] = gap_canonical_id
                # Stamp the variant's stable join key (and source) so breakdown
                # attribution can map this explore gain back to its specialist
                # provenance via explore_search.winners_history. Without it the
                # phase_breakdown.explore.by_domain join always misses and every
                # gain collapses into ``default_grid``.
                fp_val = ""
                prov_val = ""
                if isinstance(bv, dict):
                    fp_val = str(bv.get("fingerprint") or "").strip()
                    if not fp_val:
                        from ..actions.executors._canonical_fingerprint import (
                            canonical_fingerprint,
                        )

                        fp_val = canonical_fingerprint(
                            candidate_args or full_args,
                            dict(bv.get("extra_envs") or {}),
                        )
                    prov_val = str(bv.get("provenance") or "").strip()
                if fp_val:
                    stack_entry["fingerprint"] = fp_val
                if prov_val:
                    stack_entry["provenance"] = prov_val
                # Stable filter label for "what kind of optimization" (backend /
                # param / env), so the stack can be sliced like the timeline.
                _stack_envs = dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
                stack_entry["operation_kind"] = operation_kind_for(
                    task_kind,
                    classify_change_kind(
                        task_kind,
                        {"extra_server_args": candidate_args, "extra_envs": _stack_envs},
                    ),
                )
                _stack_scope = str(bv.get("scope") or "").strip() if isinstance(bv, dict) else ""
                if _stack_scope:
                    stack_entry["scope"] = _stack_scope
                self.shared_state.optimization_stack.append(stack_entry)
                # Mirror append into gain_per_stack_entry so the two parallel lists stay index-aligned.
                self.shared_state.append_stack_gain_entry(
                    action=task_kind,
                    variant_name=variant_name,
                    new_tput=best_tput,
                    extra_server_args=full_args,
                )

        self.shared_state.current_best = {
            "action": task_kind,
            "tput": float(best_tput),
            "variant_name": variant_name,
            "extra_server_args": full_args,
            "extra_envs": (dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}),
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms") if isinstance(bv, dict) else None,
            "e2el_mean_ms": bv.get("e2el_mean_ms") if isinstance(bv, dict) else None,
            "tpot_mean_ms": bv.get("tpot_mean_ms") if isinstance(bv, dict) else None,
            "workspace": bv.get("workspace") if isinstance(bv, dict) else None,
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(best_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
            )

    async def _promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid).

        Args:
            task_kind: The settled task's kind, selecting the promote branch.
            result: The task result dict; non-dict results are ignored.
            task: The originating task, used for audit fingerprints and
                pending-roofline gating.
        """
        if not isinstance(result, dict):
            return
        changed = False
        # Audit-trail bookkeeping: each branch sets audit_decision/extras; record_action_attempt runs once after.
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        if task_kind == "baseline":
            tput = result.get("output_throughput")
            warmup_anchor = result.get("warmup_round_tput")
            if isinstance(tput, (int, float)) and tput > 0:
                # Baseline's conclusion contract is the hot measure round:
                # BaselineExecutor already discards the cold first round and
                # returns the second round as ``output_throughput``. Keep the
                # discarded value only as an audit field so leaderboard/report
                # gain math never mixes cold-before with hot-after.
                if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0:
                    self.shared_state.baseline_tput = float(tput)
                    self.shared_state.baseline_cold_tput = float(warmup_anchor)
                    self.shared_state.baseline_hot_tput = float(tput)
                    log.info(
                        "baseline anchor: using hot measure tput %.1f as "
                        "baseline_tput (discarded cold warmup %.1f kept as "
                        "baseline_cold_tput)",
                        float(tput),
                        float(warmup_anchor),
                    )
                else:
                    self.shared_state.baseline_tput = float(tput)
                self.shared_state.baseline_failure_streak = 0
                self.shared_state.baseline_arg_error_streak = 0
                changed = True
            acc = result.get("accuracy")
            if isinstance(acc, (int, float)):
                self.shared_state.baseline_accuracy = float(acc)
                changed = True
            # Persist the materialized YAML so downstream tasks reuse the exact workload contract baseline ran.
            materialized = result.get("materialized_config")
            if isinstance(materialized, str) and materialized:
                self.shared_state.baseline_config_path = materialized
                changed = True
                # parse workload-shape extras from the YAML for lesson/pitfall attrs. Best-effort.
                try:
                    parsed = _parse_baseline_workload_extra(materialized)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "baseline workload extra parsing failed for %s",
                        materialized,
                    )
                    parsed = {}
                if parsed:
                    self.shared_state.baseline_workload_extra = parsed
            # Promote baseline wall-clock so ExploreExecutor derives the per-variant overtime kill deadline.
            runtime_sec_raw = result.get("subprocess_runtime_sec")
            if isinstance(runtime_sec_raw, (int, float)) and runtime_sec_raw > 0:
                self.shared_state.baseline_runtime_sec = float(runtime_sec_raw)
                changed = True
            # Promote the WARM measure-round wall-clock (client-only, no
            # boot) as the anchor for the explore decision-round overtime kill.
            # Present only on the double-run baseline path; absent on the
            # single-round path (then explore falls back to the cold anchor).
            warm_runtime_raw = result.get("measure_round_runtime_sec")
            if isinstance(warm_runtime_raw, (int, float)) and warm_runtime_raw > 0:
                self.shared_state.baseline_warm_runtime_sec = float(warm_runtime_raw)
                changed = True
            elif float(getattr(self.shared_state, "baseline_warm_runtime_sec", 0.0) or 0.0) != 0.0:
                self.shared_state.baseline_warm_runtime_sec = 0.0
                changed = True
            # current_best.tput follows the same hot baseline contract.
            # run_grid/explore/integrate_patch measure optimization candidates
            # on the same warm second-round basis when lifecycle reuse is
            # available, so the numerator and denominator stay aligned.
            anchor_tput = float(self.shared_state.baseline_tput or 0.0)
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": (anchor_tput if anchor_tput > 0 else (float(tput) if isinstance(tput, (int, float)) else None)),
                "hot_tput": (float(tput) if isinstance(tput, (int, float)) else None),
                "cold_tput": (
                    float(warmup_anchor)
                    if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0
                    else None
                ),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "tpot_mean_ms": result.get("tpot_mean_ms"),
                "workspace": result.get("workspace"),
            }
            changed = True
            audit_decision = "promoted" if isinstance(tput, (int, float)) and tput > 0 else "discarded"
            audit_extras = {
                "materialized_config": result.get("materialized_config"),
                "accuracy": result.get("accuracy"),
                "baseline_tput": (float(tput) if isinstance(tput, (int, float)) else None),
                # Stamp canonical params fingerprint so the self-loop denial helper compares run-vs-proposed (_baseline_params_fingerprint).
                "fingerprint": _baseline_params_fingerprint(task.params if task is not None else None),
            }
            # seed the gaps[] ledger from baseline (best-effort).
            await self._refresh_gaps(reason="baseline_done")
            # Standalone baseline-arm roofline ceiling (pure CPU, no GPU/trace):
            # backs up the snapshot ceiling so the frontend still has data when
            # the roofline (profile + trace_analyze) step later fails.
            if isinstance(tput, (int, float)) and tput > 0:
                try:
                    self.shared_state.record_baseline_roofline_ceiling()
                except Exception as exc:  # noqa: BLE001 — best-effort backup
                    log.warning(
                        "baseline roofline-ceiling backup failed: %r", exc,
                    )
            # PRELUDE bootstrap (post-baseline), ordering mandatory: (1) inject warm-recipe history, (2) warm-replay, (3) auto-analysis (deferred while replay in_flight, same GPU/port), (4) research scout.
            if (
                isinstance(tput, (int, float))
                and tput > 0
                and not (self.shared_state.auto_roofline_pending_task_id or "").strip()
            ):
                # History injection (fires regardless of --no-warm-replay).
                try:
                    self._inject_warm_recipe_history_into_ledger()
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.exception(
                        "PRELUDE: warm-recipe history injection failed: %r",
                        exc,
                    )
                # Warm-recipe replay. Anchor replay gain on the hot
                # baseline_tput contract; candidate replays also return their
                # hot measure round.
                try:
                    await self._maybe_enqueue_warm_replay(
                        baseline_tput=float(self.shared_state.baseline_tput or tput),
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.exception(
                        "PRELUDE: failed to enqueue warm-replay task: %r",
                        exc,
                    )
                # Auto-analysis (roofline / profile); may defer.
                await self._maybe_enqueue_prelude_initial_analysis_after_baseline(
                    baseline_tput=float(tput),
                )
                # Research scout (parallel, read-only, CPU-only).
                await self._maybe_enqueue_prelude_research_scout()
                # Static-recon (parallel, read-only, CPU-only): grep
                # the framework source for un-bridged capability switches and
                # seed bridge candidates as gaps[] before EXPLORE starts.
                await self._maybe_enqueue_prelude_static_recon()
        elif task_kind == "replay_warm_recipe":
            # separate promote path so replay doesn't overwrite baseline_tput/current_best via the baseline branch.
            try:
                self._promote_warm_replay(result, task=task)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay promote failed")
            # PRELUDE initial roofline was deferred while replay ran.
            await self._maybe_enqueue_prelude_initial_analysis_after_baseline()
        elif task_kind == "profile":
            # atom profiles natively now, so this skipped arm is defensive; audit as skipped + drop the gate.
            if str(result.get("status") or "") == "skipped":
                audit_decision = "skipped"
                audit_extras = {
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                    self.shared_state.auto_roofline_pending_task_id = ""
                    changed = True
            else:
                audit_decision = "promoted"
                audit_extras = {
                    "trace_path": None,
                    "profile_args": None,
                    "output_throughput": result.get("output_throughput"),
                }
            # Surface ProfileExecutor's trace path so Orch passes a real path to trace_analyze.
            trace_path = result.get("main_trace_path") or (result.get("trace_files") or [None])[0]
            profile_status = str(result.get("status") or "")
            if profile_status == "failed" or result.get("error_class") == "no_trace_files":
                self.shared_state.last_profile_status = "failed"
                if not trace_path:
                    self.shared_state.last_profile_trace = ""
                changed = True
            elif trace_path:
                self.shared_state.last_profile_trace = str(trace_path)
                self.shared_state.last_profile_status = "succeeded"
                # Record the server config in effect for this trace so Orch can decide whether to re-profile.
                profile_args = ""
                if task is not None:
                    profile_args = str((task.params or {}).get("base_extra_args") or "")
                self.shared_state.last_profile_args = profile_args
                # New trace invalidates the stale trace_analyze cache.
                self.shared_state.last_trace_analyze = {}
                changed = True
                audit_extras["trace_path"] = str(trace_path)
                audit_extras["profile_args"] = profile_args
            # profile result may include a tput; promote into current_best on the same +1% rule as the grid path.
            tput = result.get("output_throughput")
            cb = self.shared_state.current_best or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            cur_best = (
                float(cb_tput)
                if isinstance(cb_tput, (int, float)) and cb_tput > 0
                else float(self.shared_state.baseline_tput or 0.0)
            )
            if (
                isinstance(tput, (int, float))
                and tput > 0
                and cur_best > 0
                and (tput - cur_best) / cur_best * 100.0 >= 1.0
            ):
                self.shared_state.current_best = {
                    "action": "profile",
                    "tput": float(tput),
                    "ttft_mean_ms": result.get("ttft_mean_ms"),
                    "e2el_mean_ms": result.get("e2el_mean_ms"),
                    "tpot_mean_ms": result.get("tpot_mean_ms"),
                    "workspace": result.get("workspace"),
                }
                if self.shared_state.baseline_tput > 0:
                    self.shared_state.cumulative_gain = (
                        (float(tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
                    )
                changed = True
            # On a successful profile, mirror the roofline-branch watermark handling: re-anchor
            # last_roofline_tput on the projected tput and clear the pending field for THIS task id.
            if profile_status == "succeeded":
                anchor_tput = self._current_tput_from_validated_gain()
                if anchor_tput > 0:
                    self.shared_state.last_roofline_tput = float(anchor_tput)
                    changed = True
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif task_kind == "roofline":
            # The composite roofline action runs profile + trace_analyze atomically and
            # its executor already writes last_profile_* + last_trace_analyze; here we just record the audit row.
            status = str(result.get("status") or "")
            if status == "skipped":
                # Defensive arm (atom profiles natively now): clean no-op, no streak/watermark touch.
                audit_decision = "skipped"
                audit_extras = {
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                # Still clear the pending pointer so the watermark check can re-arm.
                if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                    self.shared_state.auto_roofline_pending_task_id = ""
                    changed = True
            elif status == "succeeded":
                audit_decision = "promoted"
                # prefer the executor's published last_trace_analyze snapshot over the result dict for the audit row.
                _last_ta = self.shared_state.last_trace_analyze or {}
                audit_extras = {
                    "snapshot_id": (
                        _last_ta.get("roofline_snapshot_id")
                        if _last_ta.get("roofline_snapshot_id") is not None
                        else result.get("snapshot_id")
                    ),
                    "last_profile_trace": (self.shared_state.last_profile_trace or result.get("last_profile_trace")),
                    "analysis_md_path": (_last_ta.get("analysis_md_path") or result.get("analysis_md_path")),
                    "profile_workspace": result.get("profile_workspace"),
                    "degraded": bool(result.get("degraded", False)),
                }
                # reset the roofline failure streak on a successful snapshot (prompt-visibility only).
                if hasattr(self.shared_state, "roofline_failure_streak"):
                    self.shared_state.roofline_failure_streak = 0
                # Re-anchor the 10% watermark step on the projected current tput.
                anchor_tput = self._current_tput_from_validated_gain()
                if anchor_tput > 0:
                    self.shared_state.last_roofline_tput = float(anchor_tput)
                changed = True
            else:
                audit_decision = "discarded"
                audit_extras = {
                    "phase": result.get("phase"),
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                # bump the failure streak (mirrors the audit ledger on SharedState for prompt renderers).
                if hasattr(self.shared_state, "roofline_failure_streak"):
                    self.shared_state.roofline_failure_streak += 1
                changed = True
                log.warning(
                    "Auto-roofline %s failed (reason=%s phase=%s "
                    "error_class=%s); continuing in degraded mode "
                    "(specialists / explore proceed without a fresh "
                    "analysis_md). No retry, no fallback.",
                    task.task_id if task else "?",
                    str((task.params or {}).get("reason") or "") if task is not None else "",
                    result.get("phase"),
                    result.get("error_class"),
                )
            # Clear the pending pointer (matched by task id so an unrelated roofline can't clear another's anchor).
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif task_kind == "explore":
            # explore is the merged grid runner; the executor already did per-variant KEEP/REVERT + rebench,
            # so winners are authoritative. Coordinator is single-writer for explore_search.accepted +
            # current_best + optimization_stack and does NOT re-threshold.
            # 1. Apply the executor's ledger increment.
            update = result.get("explore_search_update")
            if isinstance(update, dict):
                self.shared_state.apply_explore_search_update(update)
                changed = True
            # 2. Search-space expansion bookkeeping (honoured defensively when an update is present).
            disc_update = result.get("discovered_flags_update")
            if isinstance(disc_update, dict):
                self.shared_state.record_discovered_flags(
                    framework=str(disc_update.get("framework") or ""),
                    backend_flags=disc_update.get("backend_flags"),
                    param_flags=disc_update.get("param_flags"),
                    source_path=str(disc_update.get("source_path") or ""),
                )
                err = disc_update.get("discovery_error")
                if err:
                    self.shared_state.discovered_flags_error = str(err)
                changed = True
            # 3. Per-winner record_explore_accepted — Coordinator is sole writer of explore_search.accepted.
            winners = result.get("winners") or []
            round_id = str(result.get("round_id") or "")
            best_winner = result.get("best_variant")
            best_tput = result.get("output_throughput")
            promoted = False
            # A post-resume revalidation task (full-stack ``resume_stack_revalidate``
            # or env-gated current_best ``resume_reverify_best``) confirms the
            # EXISTING cumulative stack rather than adding a variant, so it never
            # "promotes". Reconcile the validation watermark + clear the
            # ``resume_pending_revalidation`` flag from the measured tput — but
            # ONLY when the rebench actually produced a valid measurement, so a
            # failed/empty rebench leaves the flag set and reports keep warning.
            _revalidate_sources = {"resume_stack_revalidate", "resume_reverify_best"}
            if task is not None and str((task.params or {}).get("source") or "") in _revalidate_sources:
                measured = result.get("output_throughput")
                measured_ok = isinstance(measured, (int, float)) and measured > 0
                # A GEAK revalidation (2b) must not blindly stamp validated
                # from the measured tput: assert the ran config's identity + that
                # the optimization actually engaged, else replay via the GEAK
                # harness (2a). Generic (native) revalidations keep the original
                # unconditional watermark reconciliation below.
                if bool((task.params or {}).get("geak_fallback")):
                    got_hash = ""
                    if isinstance(best_winner, dict):
                        got_hash = str(best_winner.get("fingerprint") or "")
                    if not got_hash and isinstance(winners, list) and winners and isinstance(winners[0], dict):
                        got_hash = str(winners[0].get("fingerprint") or "")
                    decision = _geak_revalidation_decision(
                        measured=measured,
                        baseline=self.shared_state.baseline_tput,
                        got_hash=got_hash,
                        expected_hash=str((task.params or {}).get("expected_cfg_hash") or ""),
                        min_engaged_gain_pct=_MIN_KERNEL_ENGAGED_GAIN_PCT,
                    )
                    if decision == "validated":
                        if self._geak_legacy_promote():
                            # Legacy: current_best/stack were written up front by
                            # the provisional promote; here we only stamp the
                            # same-harness validated watermark.
                            self.shared_state.cumulative_gain_validated = (
                                (float(measured) - self.shared_state.baseline_tput)
                                / self.shared_state.baseline_tput
                                * 100.0
                            )
                            self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
                            self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
                            self.shared_state.cumulative_gain_provenance = "geak_orch_harness_validated"
                            self.shared_state.resume_pending_revalidation = False
                        else:
                            # Rebench-first: THIS is where the headline is first
                            # written - from the measured orchestrator-harness
                            # rebench. Lifts current_best + optimization_stack +
                            # the validated gain and clears geak_pending.
                            ps = (
                                self.shared_state.geak_result
                                if isinstance(
                                    getattr(self.shared_state, "geak_result", None), dict
                                )
                                else {}
                            )
                            self._promote_geak_from_candidate(
                                ps,
                                measured_tput=float(measured),
                                provenance="geak_orch_harness_validated",
                            )
                    else:
                        # 2b inconclusive (config-identity or engagement) -> GEAK
                        # harness replay (2a). Leaves pending flag set; 2a clears
                        # it on success. Best-effort so a fallback failure never
                        # crashes the reactor (provisional gain + warning remain).
                        log.warning(
                            "geak 2b revalidation inconclusive "
                            "(measured=%r got_hash=%r expected=%r) -> GEAK-harness 2a fallback",
                            measured, got_hash, (task.params or {}).get("expected_cfg_hash"),
                        )
                        try:
                            await self._validate_geak_via_geak_harness(reason="2b_inconclusive")
                        except Exception:  # noqa: BLE001 - defensive
                            log.exception("geak 2a GEAK-harness fallback failed")
                    changed = True
                else:
                    if measured_ok and self.shared_state.baseline_tput > 0:
                        self.shared_state.cumulative_gain_validated = (
                            (float(measured) - self.shared_state.baseline_tput)
                            / self.shared_state.baseline_tput
                            * 100.0
                        )
                        self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
                        self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
                        cb_rec = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
                        recorded = cb_rec.get("tput")
                        try:
                            floor = float(
                                os.environ.get("INFERENCE_OPTIMIZER_RESUME_DRIFT_FLOOR", "").strip()
                                or _DEFAULT_RESUME_DRIFT_FLOOR_PCT
                            )
                        except (TypeError, ValueError):
                            floor = _DEFAULT_RESUME_DRIFT_FLOOR_PCT
                        if (
                            isinstance(recorded, (int, float))
                            and recorded > 0
                            and float(measured) < float(recorded) * floor / 100.0
                        ):
                            await self._record_observation(
                                "coordinator",
                                "observation",
                                {
                                    "kind": "current_best_drift",
                                    "severity": "high",
                                    "measured_tput": float(measured),
                                    "recorded_tput": float(recorded),
                                    "floor_pct": floor,
                                },
                            )
                    if measured_ok:
                        self.shared_state.resume_pending_revalidation = False
                    changed = True
            if isinstance(winners, list) and winners:
                for winner in winners:
                    if not isinstance(winner, dict):
                        continue
                    accepted = dict(winner)
                    accepted.setdefault("accepted_at_round", round_id)
                    accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
                    self.shared_state.record_explore_accepted(accepted)
                    # Per-anchor coverage (point 1): a specialist-provenance KEEP
                    # zeroes that domain's rounds_since_last_keep counter.
                    prov = str(accepted.get("provenance") or "")
                    if prov.startswith("specialist:"):
                        try:
                            self.shared_state.note_domain_keep(prov.split(":", 1)[1].strip())
                        except Exception:  # noqa: BLE001 — defensive
                            log.exception(
                                "depth: note_domain_keep failed for provenance=%r",
                                prov,
                            )
                    changed = True
                # 4. Lift the best winner into current_best / optimization_stack (best_tput is post-rebench).
                if isinstance(best_winner, dict) and isinstance(best_tput, (int, float)) and best_tput > 0:
                    explore_gap_cid = (
                        str((task.params or {}).get("gap_canonical_id") or "").strip() if task is not None else ""
                    )
                    self._lift_to_current_best(
                        "explore",
                        float(best_tput),
                        best_winner,
                        gap_canonical_id=explore_gap_cid,
                    )
                    promoted = True
                    changed = True
            try:
                self.shared_state.note_explore_outcome(promoted=promoted)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("depth: note_explore_outcome failed")
            if promoted:
                # explore inlines the per-KEEP rebench, so promote it into cumulative_gain_validated +
                # advance validated_stack_len so the long-run #4 unvalidated-stack guard clears immediately.
                if self.shared_state.baseline_tput > 0 and isinstance(best_tput, (int, float)) and best_tput > 0:
                    validated_gain = (
                        (float(best_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
                    )
                    self.shared_state.cumulative_gain_validated = float(validated_gain)
                    self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
                    self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
                    # Watermark refresh: enqueue a fresh roofline once projected tput crosses +10% over the last.
                    await self._maybe_enqueue_watermark_roofline(
                        reason="explore_keep_watermark",
                    )
            else:
                changed = True
            audit_decision = "promoted" if promoted else "discarded"
            audit_extras = {
                "round_id": round_id,
                "winners_count": (len(winners) if isinstance(winners, list) else 0),
                "losers_count": len(result.get("losers") or []),
                "skipped_dup_count": len(result.get("skipped_dup") or []),
                "best_variant_name": (best_winner.get("name") if isinstance(best_winner, dict) else None),
                "best_gain_pct_vs_base": result.get("best_gain_pct"),
                "output_throughput": best_tput,
                "keep_unstable_count": len(result.get("keep_unstable_in_stack") or []),
                "explore_grid_exhausted": bool(result.get("explore_grid_exhausted")),
            }
        elif task_kind == "integrate_patch":
            status = str(result.get("status") or "")
            new_tput = result.get("output_throughput")
            kept_flag = status == "kept" and isinstance(new_tput, (int, float)) and float(new_tput) > 0
            if kept_flag:
                specialist_task_id = str(result.get("specialist_task_id") or "")
                lift = {
                    "name": specialist_task_id or "integrate_patch_keep",
                    "candidate_extra_server_args": "",
                    "extra_envs": dict(result.get("config_changes_applied") or {}),
                    "tput": float(new_tput),
                    "workspace": result.get("workspace"),
                    "provenance": "integrate_patch",
                    "scope": "source_patch",
                    # Durable source-layer handles so current_best stays
                    # relaunchable (and reproducible in the GEAK baseline)
                    # regardless of later git hygiene on the shared live tree.
                    "source_snapshot": result.get("source_snapshot") or "",
                    "framework_root": result.get("framework_root") or "",
                    "base_sha": result.get("base_sha") or "",
                }
                self._lift_to_current_best("integrate_patch", float(new_tput), lift)
                if self.shared_state.baseline_tput > 0:
                    validated_gain = (
                        (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
                    )
                    self.shared_state.cumulative_gain_validated = float(validated_gain)
                    self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
                    self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
                    self.shared_state.resume_pending_revalidation = False
                    await self._maybe_enqueue_watermark_roofline(
                        reason="integrate_keep_watermark",
                    )
                changed = True
            # Clear long-run #4 sentinel after the task outcome has been
            # observed. On a crash before this point, resume sees it and audits.
            if isinstance(getattr(self.shared_state, "pending_integrate", None), dict):
                pending = self.shared_state.pending_integrate
                if not pending or str(pending.get("task_id") or "") in {
                    "",
                    str(getattr(task, "task_id", "") or ""),
                }:
                    self.shared_state.pending_integrate = {}
                    changed = True
            audit_decision = "promoted" if kept_flag else "discarded"
            audit_extras = {
                "status": status,
                "specialist_task_id": result.get("specialist_task_id"),
                "output_throughput": new_tput,
                "delta_pct": result.get("delta_pct"),
                "accuracy_pass": result.get("accuracy_pass"),
                "patches_applied": result.get("patches_applied") or [],
                "patches_reverted": result.get("patches_reverted") or [],
            }
        elif task_kind == "framework_agent":
            # FRAMEWORK per-candidate result: append a progress row, update the batch max-gain stat, and on
            # KEEP lift to current_best + optimization_stack + cumulative_gain_validated + watermark roofline.
            status = str(result.get("status") or "")
            candidate = result.get("candidate") or {}
            cand_id = self._framework_candidate_key(candidate if isinstance(candidate, dict) else None)
            # Silent apply/bench failure: the executor returned a promotable-
            # looking result (status != "failed") but with no candidate / no
            # status (empty result dict). Recover the candidate key from the
            # task params and coerce the status so the row is a real terminal
            # verdict the pump can dedup on, not a blank row keyed on "".
            if not cand_id and task is not None:
                task_cand = (getattr(task, "params", None) or {}).get("candidate")
                cand_id = self._framework_candidate_key(task_cand if isinstance(task_cand, dict) else None)
            if not status:
                status = "no_result_failed"
            batch_id = str(
                result.get("batch_id")
                or candidate.get("batch_id")
                or ((getattr(task, "params", None) or {}).get("batch_id") if task is not None else "")
                or ""
            )
            delta_pct = result.get("delta_pct")
            new_tput = result.get("output_throughput")
            kept_flag = status == "kept"
            progress_entry = {
                "candidate_id": cand_id,
                "pr_url": str(candidate.get("pr_url") or ""),
                "status": status,
                "pre_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                "post_tput": float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
                "gain_pct": float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0,
                "kept": kept_flag,
                "batch_id": batch_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if not isinstance(self.shared_state.framework_agent_phase_progress, list):
                self.shared_state.framework_agent_phase_progress = []
            self.shared_state.framework_agent_phase_progress.append(progress_entry)
            try:
                from ..framework.artifacts import write_decision_json

                write_decision_json(
                    self.session_dir,
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status=status,
                    kept=kept_flag,
                    provenance="raw_diff",
                    reason=str(result.get("reason") or ""),
                    gain_pct=(float(delta_pct) if isinstance(delta_pct, (int, float)) else None),
                    accuracy_pass=result.get("accuracy_pass"),
                    extra={"workspace": str(result.get("workspace") or "")},
                )
            except Exception:  # noqa: BLE001 — observability is best-effort
                log.debug("FRAMEWORK: executor decision.json write failed", exc_info=True)
            # Update batch max-gain rolling stat (for the plateau judge).
            batches = getattr(self.shared_state, "framework_agent_batches", None) or []
            if isinstance(batches, list) and batches:
                for entry in reversed(batches):
                    if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                        prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                        gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
                        if gain > prev:
                            entry["max_gain_pct_observed_in_batch"] = gain
                        break
            changed = True
            if kept_flag and isinstance(new_tput, (int, float)) and new_tput > 0:
                lift = {
                    "name": f"framework:{cand_id}",
                    "variant_name": cand_id,
                    "candidate_extra_server_args": "",
                    "extra_envs": {},
                    "workspace": result.get("workspace"),
                }
                self._lift_to_current_best("framework", float(new_tput), lift)
                if self.shared_state.baseline_tput > 0:
                    validated_gain = (
                        (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
                    )
                    self.shared_state.cumulative_gain_validated = float(validated_gain)
                    self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
                    self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
                    await self._maybe_enqueue_watermark_roofline(
                        reason="framework_keep_watermark",
                    )
            audit_decision = "promoted" if kept_flag else "discarded"
            audit_extras = {
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "status": status,
                "delta_pct": delta_pct,
                "output_throughput": new_tput,
                "kept": kept_flag,
            }
        elif task_kind == "sweep":
            pareto = result.get("pareto_front") or []
            self.shared_state.record_action_attempt(
                action="sweep",
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision="discarded",
                result=result,
                extras={
                    "grid_size": result.get("grid_size"),
                    "best_overall": result.get("best_overall"),
                    "best_for_each_conc": result.get("best_for_each_conc"),
                    "pareto_front_size": (len(pareto) if isinstance(pareto, list) else None),
                },
            )
            self.shared_state.record_sweep(result)
            # Sweep is discovery-only (never promotes) and MUST NOT mutate params_no_promote_streak.
            self.shared_state.save(self.session_dir)
            # SWEEP post-hook: chain conc_sweep after a succeeded sweep when opted in (best-effort, non-blocking).
            if getattr(self.shared_state, "conc_sweep_enabled", False) and result.get("status") == "succeeded":
                try:
                    await self._enqueue_internal_conc_sweep_task(
                        reason="post_sweep",
                    )
                except Exception:  # noqa: BLE001 — never block SWEEP->CLOSE
                    log.exception("conc_sweep: post-sweep enqueue raised (non-fatal)")
            return
        elif task_kind == "conc_sweep":
            self.shared_state.record_action_attempt(
                action="conc_sweep",
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status=str(result.get("status") or "succeeded"),
                decision="discarded",
                result=result,
                extras={
                    "was_skipped": bool(result.get("was_skipped", False)),
                    "skip_reason": result.get("skip_reason"),
                    "budget_exhausted": bool(result.get("budget_exhausted", False)),
                    "total_budget_sec": result.get("total_budget_sec"),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "best_speedup": ((result.get("summary") or {}).get("best_speedup")),
                    "best_conc": ((result.get("summary") or {}).get("best_conc")),
                    "successful_pairs": ((result.get("summary") or {}).get("successful_pairs")),
                    "report_path": result.get("report_json_path"),
                },
            )
            # Write last_conc_sweep so exit_normal_sweep can fire conc_sweep_done without budget exhaustion.
            self.shared_state.record_conc_sweep(result)
            self.shared_state.save(self.session_dir)
            return
        # Audit trail (kernel-parity): one succeeded-attempt record with branch-supplied decision/extras.
        if audit_decision is not None and task_kind in _AUDIT_ACTIONS:
            self.shared_state.record_action_attempt(
                action=task_kind,
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision=audit_decision,
                result=result,
                extras=audit_extras,
            )
            changed = True
        if changed:
            self.shared_state.save(self.session_dir)
