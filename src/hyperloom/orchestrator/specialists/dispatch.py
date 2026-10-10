# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist dispatch helpers: warmup, auto-retry, wave fan-out, stalled-domain forcing, and round-entry construction."""

from __future__ import annotations

import asyncio
import logging as _logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from hyperloom.common.env import env_flag
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

from ..collaborator import CoordinatorCollaborator
from ..measurement.runtime_findings import render_runtime_findings
from ..phases import machine_state as _phase_state
from ..policy.gate import (
    PolicyDenied,
    validate_freeform_wave_task,
)
from hyperloom.inference_optimizer.trace.trajectory_trace import EVENT_TASK_RETRY, record_event, trajectory_scope
from .runner import SpecialistFailureType, specialist_patch_preflight_error

if TYPE_CHECKING:
    from ..loop.coordinator import Coordinator
    from ..loop.sub_agent_runner import SubAgentResult
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)

__all__ = ["SpecialistDispatchCollaborator"]

_SOURCE_PATCH_FAMILY = "source_patch"

# Bounded transient-failure auto-retry for specialist dispatches (infra-only).
SPECIALIST_AUTO_RETRY_MAX: int = 2

# Hard-trigger thresholds: optimisation rounds a domain may go without a specialist dispatch / a KEEP before the
# Coordinator force-dispatches one.
FORCE_STALLED_SPECIALIST_ROUNDS: int = 8
FORCE_STALLED_KEEP_ROUNDS: int = 12


class SpecialistDispatchCollaborator(CoordinatorCollaborator):
    """Specialist dispatch: warmup, auto-retry, wave fan-out, stalled-domain forcing, and round-entry construction."""

    def __init__(self, coordinator: "Coordinator", proposal_scorer: Any = None) -> None:
        super().__init__(coordinator)
        # Advisory only: scores proposals, never gates them.
        self._proposal_scorer = proposal_scorer

    async def warm_specialist_params(self, params: dict[str, Any]) -> None:
        """Fill specialist task params with KnowledgePlane data before enqueue (mutates in place); missing fields stay empty.

        Args:
            params: The specialist task params dict mutated in place with PR
                feed, warm-start, hardware/workload and gap/roofline context.
        """
        state = self.shared_state
        plane = self.knowledge_plane

        from .domains import normalize_dispatch_tags
        from .profile import resolve_specialist_profile

        # Bench-capable specialists run a real serving + benchmark loop, so
        # default needs_gpu to route them through the gpu_specialist_pool.
        if resolve_specialist_profile(params).reserves_benchmark_lane:
            params.setdefault("needs_gpu", True)

        domain = str(params.get("domain") or "").strip()
        normalize_dispatch_tags(params)

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(plane is not None and getattr(plane, "pr_monitor_enabled", True))

        params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls + lessons from T0 anchor.
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)
        if state.warm_start_lessons and "warm_start_lessons" not in params:
            params["warm_start_lessons"] = list(state.warm_start_lessons)
        # runtime framework/version for version-mismatch annotation.
        if "framework" not in params:
            fw = str(getattr(state, "framework", "") or "").strip()
            if fw:
                params["framework"] = fw
        if "framework_version" not in params:
            fp_meta = getattr(state, "stack_fingerprint_meta", None) or {}
            if isinstance(fp_meta, dict):
                fw = str(params.get("framework") or getattr(state, "framework", "") or "").lower()
                if fw in ("sglang", "vllm"):
                    v = str(fp_meta.get(fw) or "").strip()
                    if v and v != "unknown":
                        params["framework_version"] = v

        # Local-source navigation hint.
        if "framework_source_roots" not in params:
            try:
                from hyperloom.inference_optimizer.framework_paths import (
                    resolve_framework_tree,
                    resolve_kernel_search_roots,
                )

                roots = resolve_kernel_search_roots()
                if roots:
                    params["framework_source_roots"] = list(roots)
                tree = resolve_framework_tree(str(getattr(state, "framework", "") or ""))
                if tree:
                    params["session_framework_tree"] = tree
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "specialist warmup: framework_source_roots lookup failed: %r",
                    exc,
                )

        # Hardware + workload hints from SharedState; else dataclass defaults win.
        params.setdefault("gpu_type", state.gpu_type or "")
        # Active server framework name.
        if getattr(state, "framework", "") or "":
            params.setdefault("framework", str(state.framework))
        if int(getattr(state, "tp", 0) or 0) > 0:
            params.setdefault("tp", int(state.tp))
        if getattr(state, "precision", "") or "":
            params.setdefault("precision", str(state.precision))
        if int(getattr(state, "conc", 0) or 0) > 0:
            params.setdefault("conc", int(state.conc))
        if int(getattr(state, "isl", 0) or 0) > 0:
            params.setdefault("isl", int(state.isl))
        if int(getattr(state, "osl", 0) or 0) > 0:
            params.setdefault("osl", int(state.osl))
        if int(getattr(state, "max_model_len", 0) or 0) > 0:
            params.setdefault("max_model_len", int(state.max_model_len))
        # The mode selects the specialist's workload block; the corpus shape
        # supplies its numbers.
        if getattr(state, "benchmark_mode", "") or "":
            params.setdefault("benchmark_mode", str(state.benchmark_mode))
        if getattr(state, "agentx_corpus_shape", None):
            params.setdefault("agentx_corpus_shape", dict(state.agentx_corpus_shape))
        if isinstance(getattr(state, "grading", None), dict) and state.grading:
            params.setdefault("agentx_grading", dict(state.grading))
        if getattr(state, "agentx_backend", ""):
            params.setdefault("agentx_backend", str(state.agentx_backend))

        # Advisory model_arch profile via arch_notes carrier (prompt-context only).
        if "arch_notes" not in params:
            from ..state._shared_state.render import render_model_arch_compact

            _arch_notes = render_model_arch_compact(getattr(state, "model_arch", None))
            if _arch_notes:
                params["arch_notes"] = _arch_notes

        if domain == "static_recon_specialist" and "model_info" not in params:
            _minfo = getattr(state, "model_info", None)
            if isinstance(_minfo, dict) and _minfo:
                params["model_info"] = dict(_minfo)

        # Checklist-derived focus directories; a caller that named its own keeps it.
        if "source_hint_directories" not in params:
            from ..knowledge import static_recon_checklist as _src_recon

            _dirs = _src_recon.source_hint_directories_for(
                model_class=str(getattr(state, "model_class", "") or ""),
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                precision=_src_recon.workload_precision(state),
            )
            if _dirs:
                params["source_hint_directories"] = list(_dirs)

        if "target_gap_notes" not in params:
            _gap_notes = self._coord.conversation.target_gap_advisory_block()
            if _gap_notes:
                params["target_gap_notes"] = _gap_notes

        if "research_hints" not in params:
            try:
                from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

                _hints_block = _research_hints.summarise_for_prompt(
                    self.session_dir,
                )
            except Exception:
                log.exception("Coordinator: specialist research hints failed")
                _hints_block = ""
            if _hints_block:
                params["research_hints"] = _hints_block

        # Fill gap-specific anchors from the gaps[] ledger.
        gap_cid = str(params.get("gap_canonical_id") or "").strip() or str(params.get("gap") or "").strip()
        if gap_cid:
            gap = state.find_gap(gap_cid)
            if gap is not None:
                if not params.get("gap_symptom"):
                    params["gap_symptom"] = str(gap.get("symptom") or "")
                if not params.get("gap_layer"):
                    params["gap_layer"] = str(gap.get("layer") or "")
                if not params.get("domain"):
                    # LLM omitted domain → gap's domain_hint wins.
                    hint = str(gap.get("domain_hint") or "")
                    if hint:
                        params["domain"] = hint
                evidence = params.get("gap_evidence")
                if not isinstance(evidence, dict) or not evidence:
                    attempts = list(gap.get("attempts") or [])[-5:]
                    if attempts:
                        params["gap_evidence"] = {
                            "recent_attempts": attempts,
                            "severity": str(gap.get("severity") or ""),
                        }

        if "baseline_tput" not in params:
            _bt = float(getattr(state, "baseline_tput", 0.0) or 0.0)
            if _bt > 0:
                params["baseline_tput"] = _bt
        if "current_tput" not in params:
            cb = getattr(state, "current_best", None)
            _ct = float((cb.get("tput") if isinstance(cb, dict) else 0) or 0.0)
            if _ct > 0:
                params["current_tput"] = _ct
        if "cumulative_gain_validated" not in params:
            _cgv = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
            if _cgv != 0:
                params["cumulative_gain_validated"] = _cgv
        if "keep_threshold_pct" not in params:
            params["keep_threshold_pct"] = _phase_state.resolve_keep_threshold(state)
        if "applied_stack" not in params:
            _stack = list(getattr(state, "optimization_stack", None) or [])
            if _stack:
                params["applied_stack"] = [
                    {"variant_name": str(e.get("variant_name") or ""), "gain_pct": float(e.get("gain_pct") or 0.0)}
                    for e in _stack
                    if isinstance(e, dict)
                ]

        # Pack bottleneck signals into roofline_evidence for the specialist.
        # Hot kernels alone are enough: a trace whose quality gate withheld
        # analysis.md still names where device time goes.
        last_ta = getattr(state, "last_trace_analyze", None) or {}
        has_evidence = isinstance(last_ta, dict) and bool(
            last_ta.get("analysis_md_text") or last_ta.get("hot_kernels_top15")
        )
        if has_evidence and "roofline_evidence" not in params:
            from hyperloom.inference_optimizer.roofline_snapshot import extract_workload_summary

            analysis_path = str(last_ta.get("analysis_md_path") or "")
            executive_summary: dict[str, Any] = {}
            if analysis_path:
                try:
                    executive_summary = extract_workload_summary(analysis_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "specialist warmup: extract_workload_summary(%s) failed: %r",
                        analysis_path,
                        exc,
                    )
                    executive_summary = {}
            hot_kernels = list(last_ta.get("hot_kernels_top15") or [])[:8]
            params["roofline_evidence"] = {
                "analysis_md_path": analysis_path,
                "roofline_snapshot_id": last_ta.get("roofline_snapshot_id"),
                "executive_summary": executive_summary,
                "hot_kernels_top15": hot_kernels,
            }

        measurement = getattr(state, "current_best_measurement", None)
        if measurement and "runtime_findings" not in params:
            params["runtime_findings"] = await asyncio.to_thread(render_runtime_findings, measurement)

        await self._warm_experience_kb(params)

    async def _warm_experience_kb(self, params: dict[str, Any]) -> None:
        """Inject this dispatch's Experience KB block into a FRAMEWORK_AGENT specialist and record the injection."""
        state = self.shared_state
        if "kb_read_id" in params:
            return
        if str(getattr(state, "phase", "") or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        integration = self._coord.experience_kb
        if integration is None:
            return
        evidence = await asyncio.to_thread(integration.read_for_specialist, state, params)
        # The exposure travels with everything this specialist authors, under the keys orchestration proposals use;
        # a read that matched nothing is recorded too, so it stays distinguishable from no read at all.
        if evidence.read_id:
            params["kb_read_id"] = evidence.read_id
            params["kb_rendered_refs"] = [dict(ref) for ref in evidence.rendered_refs]
        if evidence.status != "completed" or not evidence.prompt_block:
            return
        params["experience_kb_block"] = evidence.prompt_block
        state.record_experience_kb_injection(
            consumer="specialist",
            domain=str(params.get("domain") or ""),
            gap_canonical_id=str(params.get("gap_canonical_id") or ""),
            read_id=evidence.read_id,
            experience_ids=[str(ref.get("id") or "") for ref in evidence.rendered_refs],
            experiences=[dict(item) for item in evidence.experiences],
            prompt_block=evidence.prompt_block,
        )

    async def maybe_auto_retry_specialist(
        self,
        task: "Task",
        result: "SubAgentResult",
    ) -> bool:
        """Re-enqueue a fresh specialist task on a transient infra failure.

        Returns ``True`` when a retry was scheduled (the caller must then skip
        this attempt's delegated_result + bookkeeping). Only infra failures
        (timeout / crash / stale-heartbeat, per ``classify_specialist_failure``)
        are retried, capped at :data:`SPECIALIST_AUTO_RETRY_MAX`; the failure
        reason is injected into the retry prompt. Disabled when
        ``INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY`` is set to ``0``.

        Args:
            task: The specialist task whose attempt just failed.
            result: The sub-agent result classified for infra-failure
                eligibility.

        Returns:
            ``True`` when a retry was scheduled (caller must skip this
            attempt's bookkeeping); ``False`` otherwise.
        """
        if not env_flag("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", default=True):
            return False
        try:
            cap = int(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX",
                    str(SPECIALIST_AUTO_RETRY_MAX),
                )
            )
        except (TypeError, ValueError):
            cap = SPECIALIST_AUTO_RETRY_MAX
        if cap <= 0:
            return False
        from .runner import classify_specialist_failure

        result_dict = result.result if isinstance(result.result, dict) else {}
        runner_status = str(result_dict.get("runner_status") or "")
        # The specialist executor never raises, so the reason lives in the
        # result envelope rather than on SubAgentResult.
        error = str(result.error or result_dict.get("error") or "")
        ftype, retry_eligible = classify_specialist_failure(runner_status, error)
        if not retry_eligible:
            return False
        params = task.params or {}
        attempt = int(params.get("_auto_retry_attempt", 0) or 0)
        if attempt >= cap:
            await self._record_specialist_retry_exhausted(
                task=task,
                ftype=ftype,
                error=error,
                attempts_used=attempt,
                cap=cap,
                detail="retry cap reached",
            )
            return False
        next_attempt = attempt + 1

        retry_params = dict(params)
        retry_params["_auto_retry_attempt"] = next_attempt
        retry_params["_auto_retry_reason"] = f"{ftype.value}: {error}"[:300]

        # Mirror handle_delegate lane/ttl resolution so the retry task holds the
        # same pools as the original and cannot run concurrently with serving.
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("specialist")
        from .profile import requires_gpu, specialist_lanes

        lanes = list(specialist_lanes(retry_params, list(lanes)))
        if requires_gpu(retry_params):
            ttl = self._coord.dispatcher.gpu_lease_ttl_sec(int(ttl or 0), params=retry_params)

        # Stable base key across attempts: strip any prior ``-autoretryN`` suffix.
        base_key = str(task.idempotency_key or task.task_id or "")
        if "-autoretry" in base_key:
            head, _, tail = base_key.rpartition("-autoretry")
            if tail.isdigit():
                base_key = head
        retry_key = f"{base_key}-autoretry{next_attempt}"

        with trajectory_scope(parent_span_id=task.task_id):
            new_task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=retry_params,
                idempotency_key=retry_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        if was_existing:
            # Retry slot already taken: let normal bookkeeping record this attempt.
            await self._record_specialist_retry_exhausted(
                task=task,
                ftype=ftype,
                error=error,
                attempts_used=attempt,
                cap=cap,
                detail="retry slot already taken",
            )
            return False
        record_event(
            EVENT_TASK_RETRY,
            task_id=task.task_id,
            parent_span_id=task.task_id,
            attributes={
                "name": "specialist",
                "retry_task_id": new_task.task_id,
                "attempt": next_attempt,
                "max_attempts": cap,
                "failure_type": ftype.value,
                "reason": error[:200],
            },
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_auto_retry",
                "task_id": task.task_id,
                "retry_task_id": new_task.task_id,
                "attempt": next_attempt,
                "max_attempts": cap,
                "failure_type": ftype.value,
                "reason": error[:200],
            },
        )
        log.info(
            "specialist auto-retry: task=%s failure=%s attempt=%d/%d re-enqueued as %s",
            task.task_id,
            ftype.value,
            next_attempt,
            cap,
            new_task.task_id,
        )
        return True

    async def _record_specialist_retry_exhausted(
        self,
        *,
        task: "Task",
        ftype: SpecialistFailureType,
        error: str,
        attempts_used: int,
        cap: int,
        detail: str,
    ) -> None:
        """Broadcast that an infra-failed specialist is being abandoned.

        Args:
            task: The specialist task whose final attempt failed.
            ftype: The classified failure type.
            error: The failure reason carried by the attempt.
            attempts_used: Retry attempts already spent.
            cap: Configured retry ceiling.
            detail: Why no further retry was scheduled.
        """
        params = task.params or {}
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_auto_retry_exhausted",
                "task_id": task.task_id,
                "domain": str(params.get("domain") or ""),
                "gap_canonical_id": str(params.get("gap_canonical_id") or ""),
                "attempts_used": attempts_used,
                "max_attempts": cap,
                "failure_type": ftype.value,
                "reason": error[:200],
                "detail": detail,
            },
        )
        log.warning(
            "specialist auto-retry exhausted: task=%s failure=%s attempts=%d/%d (%s)",
            task.task_id,
            ftype.value,
            attempts_used,
            cap,
            detail,
        )

    async def fan_out_specialist_wave(
        self,
        source: str,
        intent: Intent,
        params: dict[str, Any],
    ) -> None:
        """Fan a specialist delegate carrying ``params.tasks=[...]`` into N
        standard free-form specialist dispatches (scope=freeform, mode=research
        defaults). Each fanned task is re-dispatched through the
        normal ``handle_delegate`` path. Per-task idempotency keys derive from
        the wave key. Each entry must pass the same structural checks as
        :func:`validate_freeform_wave_task` (the PolicyGate runs these first).

        Args:
            source: The agent issuing the wave delegate.
            intent: The originating specialist DELEGATE intent.
            params: The delegate params carrying the ``tasks`` list to fan out.
        """
        tasks = params.get("tasks") or []
        shared = {k: v for k, v in params.items() if k != "tasks"}
        base_key = str(intent.payload.get("idempotency_key") or "").strip()
        pending: list[Intent] = []
        for idx, task in enumerate(tasks):
            desc = validate_freeform_wave_task(task, index=idx)
            sub_params = dict(shared)
            sub_params["scope"] = "freeform"
            sub_params["task_description"] = desc
            summary = str(task.get("task_summary") or "").strip()
            if summary:
                sub_params["task_summary"] = summary
            for carry in (
                "mode",
                "bench",
                "model",
                "priority",
                "timeout_minutes",
                "max_turns",
            ):
                if isinstance(task, dict) and carry in task:
                    sub_params[carry] = task[carry]
            sub_params.setdefault("mode", "research")
            sub_payload = dict(intent.payload)
            sub_payload["params"] = sub_params
            if base_key:
                sub_payload["idempotency_key"] = f"{base_key}-w{idx}"
            else:
                sub_payload.pop("idempotency_key", None)
            sub_intent = Intent(type=intent.type, payload=sub_payload)
            try:
                self.policy.validate_intent(source, sub_intent)
            except PolicyDenied as denied:
                await self._coord.router.record_policy_denied(source, sub_intent, denied)
                raise
            pending.append(sub_intent)
        for sub_intent in pending:
            await self._coord.router.handle_delegate(source, sub_intent)

    async def maybe_force_stalled_domain_specialist(self) -> None:
        """Force-dispatch a domain specialist for a domain untouched for too many
        config-arm rounds that still has an open gap in the gaps[] ledger.

        A real scheduling event (a domain delegate routed through PolicyGate +
        warmup + the GPU specialist pool). Idempotent per
        ``(anchor, round, macro_cycle)``; a domain with a specialist already
        queued or running is skipped, and the dispatcher zeroes the per-anchor
        counter when the forced specialist spawns. At most one forced dispatch
        per tick.

        Note:
            Side-effecting: may dispatch a domain specialist via
            ``handle_intent``. Returns nothing.
        """
        state = self.shared_state
        if str(getattr(state, "phase", "") or "").upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return None
        if not bool(getattr(state, "force_stalled_specialist_enabled", True)):
            return None
        spec_thr = max(1, int(getattr(state, "force_stalled_specialist_rounds", 0) or FORCE_STALLED_SPECIALIST_ROUNDS))
        keep_thr = max(1, int(getattr(state, "force_stalled_keep_rounds", 0) or FORCE_STALLED_KEEP_ROUNDS))
        stalled = state.stalled_domains(
            specialist_threshold=spec_thr,
            keep_threshold=keep_thr,
        )
        if not stalled:
            return None

        from .domains import domain_for_tag

        busy_domains = {
            str((t.params or {}).get("domain") or "")
            for t in (*await self.tasks.queued(), *await self.tasks.running())
            if t.kind == "specialist"
        }
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        for anchor in stalled:
            gap_cid = state.best_gap_for_anchor(anchor)
            if not gap_cid:
                continue
            dom = domain_for_tag(anchor)
            if dom is None or dom.key in busy_domains:
                continue
            params: dict[str, Any] = {
                "domain": dom.key,
                "tags": [anchor],
                "gap_canonical_id": gap_cid,
                "scope": "domain",
                "source": "coordinator_internal",
                "reason": f"stalled_domain_force:{anchor}",
            }
            from .profile import MODE_PATCH, resolve_specialist_profile

            is_source_patch = resolve_specialist_profile(params, domain=dom).mode == MODE_PATCH
            if is_source_patch and state.is_pruned(_SOURCE_PATCH_FAMILY):
                continue
            idempotency_key = f"forced-stalled-{anchor}-round{round_id}{self._coord.dispatcher.cycle_idem_suffix()}"
            lookup = getattr(self.tasks, "find_by_idempotency_key", None)
            if callable(lookup):
                existing = await lookup(idempotency_key)
                if existing is not None:
                    continue
            await self.warm_specialist_params(params)
            if is_source_patch:
                preflight_error = specialist_patch_preflight_error(
                    params,
                    framework_repo_path=str(getattr(state, "framework_repo_path", "") or ""),
                )
                if preflight_error:
                    if state.add_pruned_family(_SOURCE_PATCH_FAMILY):
                        state.record_action_failure(
                            action="specialist",
                            task_id=idempotency_key,
                            result={
                                "error_class": preflight_error,
                                "error": preflight_error,
                            },
                        )
                        try:
                            state.save(self.session_dir)
                        except Exception:
                            log.exception("stalled-domain force: source-patch prune save failed")
                        log.error(
                            "stalled-domain force: pruned %s after deterministic failure: %s",
                            _SOURCE_PATCH_FAMILY,
                            preflight_error,
                        )
                    continue
            intent = Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": params,
                    "idempotency_key": idempotency_key,
                },
            )
            await self._coord.router.handle_intent("orchestration", intent)
            try:
                state.save(self.session_dir)
            except Exception:
                log.exception("stalled-domain force: state save failed")
            log.info(
                "stalled-domain force: dispatched domain=%s anchor=%s gap=%s round=%d (spec_thr=%d keep_thr=%d)",
                dom.key,
                anchor,
                gap_cid,
                round_id,
                spec_thr,
                keep_thr,
            )
            # One forced dispatch per tick.
            return None
        return None

    def build_specialist_round_entry(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
        source: str,
        run_error: str = "",
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a SharedState.specialist_rounds[] row; round_id defaults to task_id for idempotent overwrite.

        Args:
            task: The completed specialist task.
            done_payload: The specialist done payload (proposal_set, domain,
                tags, summary, etc.).
            source: The emitting agent string, recorded on the row.
            run_error: Dispatch failure text when no valid payload was produced.

        Returns:
            A specialist-round row dict suitable for
            ``SharedState.record_specialist_round``.
        """
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        task_params = task.params or {}
        round_id = str(task_params.get("round_id") or task.task_id)
        source_phase = (
            str(
                task_params.get("source_phase")
                or done_payload.get("source_phase")
                or getattr(getattr(self, "shared_state", None), "phase", "")
                or ""
            )
            .strip()
            .upper()
        )
        from .domains import normalize_dispatch_tags

        # Knowledge-domain tags; reported tags win over dispatch params.
        tags = normalize_dispatch_tags(done_payload)
        if not tags:
            tags = normalize_dispatch_tags(task.params or {})
        entry: dict[str, Any] = {
            "round_id": round_id,
            "task_id": task.task_id,
            "source": source or "coordinator",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "domain": str(done_payload.get("domain") or task_params.get("domain") or ""),
            "tags": list(tags),
            "gap_canonical_id": str(done_payload.get("gap_canonical_id") or task_params.get("gap_canonical_id") or ""),
            "proposals_total": len(proposals),
            "proposal_set": list(proposals),
            "summary": str(done_payload.get("summary") or "")[:480],
            "reason": str(run_error or done_payload.get("reason") or "")[:480],
            "confidence": done_payload.get("confidence"),
            "new_findings": list(done_payload.get("new_findings") or []),
            "residual_questions": list(done_payload.get("residual_questions") or []),
        }
        for key in (
            "task_kind",
            "scope",
            "proposal_msg_id",
            "framework_agent_candidate_id",
            "framework_batch_id",
            "reauthor_attempt",
            "apply_retry_attempt",
            "kb_read_id",
            "kb_rendered_refs",
        ):
            value = done_payload.get(key)
            if value in (None, "", [], {}):
                value = task_params.get(key)
            if value not in (None, "", [], {}):
                entry[key] = value
        for key in ("candidate_discovery", "framework_agent_authoring"):
            if bool(done_payload.get(key) or task_params.get(key)):
                entry[key] = True
        if run_error:
            entry["status"] = "failed"
            entry["error"] = str(run_error)[:1000]
            entry["run_error"] = str(run_error)[:1000]
        elif done_payload.get("status") not in (None, ""):
            entry["status"] = str(done_payload.get("status"))
        if source_phase:
            entry["source_phase"] = source_phase
        gpu_ids = done_payload.get("allocated_gpu_ids") or []
        if isinstance(gpu_ids, list) and gpu_ids:
            entry["allocated_gpu_ids"] = [
                int(g) for g in gpu_ids if isinstance(g, (int, str)) and str(g).strip().lstrip("-").isdigit()
            ]
        specialist_notes = done_payload.get("_specialist_notes") or []
        if isinstance(specialist_notes, list) and specialist_notes:
            entry["notes"] = [str(n) for n in specialist_notes]
        return entry

    def _record_specialist_round_product(self, *, task: Task, round_entry: dict[str, Any]) -> None:
        """Record what a specialist round came back with, on the event that owns it.

        The FRAMEWORK arm's dispatch already has a run row on the framework
        event keyed by this same task id, so the product merges onto that. Every
        other round merges onto the action row its dispatching phase opened.
        """
        product = {
            "summary": round_entry.get("summary") or "",
            "proposals_total": round_entry.get("proposals_total"),
            "empty": round_entry.get("empty"),
            "confidence": round_entry.get("confidence"),
            "new_findings": round_entry.get("new_findings") or [],
            "residual_questions": round_entry.get("residual_questions") or [],
            "notes": round_entry.get("notes") or [],
            "ensemble_scores": round_entry.get("ensemble_scores") or {},
        }
        source_phase = str(round_entry.get("source_phase") or "").strip().upper()
        recorder = self._coord.phase_framework.timeline()
        if recorder is not None and source_phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            recorder.record_run(str(task.task_id or ""), **product)
            return
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import phase_event

            phase_event.record_specialist_round(
                task_id=str(task.task_id or ""),
                phase=source_phase or str(self.shared_state.phase or ""),
                macro_cycle=int(self.shared_state.macro_cycle or 0),
                round_id=str(round_entry.get("round_id") or ""),
                domain=round_entry.get("domain") or "",
                gap_canonical_id=round_entry.get("gap_canonical_id") or "",
                reason=round_entry.get("reason") or "",
                source=round_entry.get("source") or "",
                tags=round_entry.get("tags") or [],
                **product,
            )
        except Exception:
            log.debug("specialist bookkeeping: phase round product record failed", exc_info=True)

    async def record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
        run_error: str = "",
    ) -> None:
        """Common bookkeeping for any specialist task termination (dispatcher loop + intent routing); idempotent on round_id, failures logged not raised.

        Args:
            task: The terminated specialist task.
            done_payload: The specialist's done payload (proposal_set, domain,
                summary, etc.).
            source: The emitting agent string (``specialist:<task_id>``).
            run_error: Dispatch failure text when the specialist produced no
                usable payload.
        """
        task_params = task.params or {}
        domain = str(done_payload.get("domain") or task_params.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = len(proposals) == 0

        round_entry = self.build_specialist_round_entry(
            task=task,
            done_payload=done_payload,
            source=source,
            run_error=run_error,
        )
        # Specialist notes reach the prompt only through this task's one inbox
        # line; ``last_action_failures`` is rendered every SEED turn.
        ungrounded = done_payload.get("patches_ungrounded")
        if isinstance(ungrounded, list) and ungrounded:
            self.shared_state.record_action_failure(
                action="specialist",
                task_id=task.task_id,
                result={
                    "error_class": "patch_targets_ungrounded",
                    "error": "; ".join(str(d) for d in ungrounded[:4]),
                },
            )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing.
        if self._proposal_scorer is not None and proposals:
            scores = await self._proposal_scorer.score(
                gap={
                    "domain": domain,
                    "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                    "gap_symptom": task_params.get("gap_symptom"),
                    "gap_evidence": task_params.get("gap_evidence"),
                    "summary": done_payload.get("summary", ""),
                },
                proposals=proposals,
                task_id=task.task_id,
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
            )
            if scores and (scores.get("models") or scores.get("errors")):
                round_entry["ensemble_scores"] = scores
                input_err = (scores.get("errors") or {}).get("input")
                if input_err and not scores.get("models"):
                    log.warning(
                        "specialist bookkeeping: proposal scoring skipped for task=%s: %s",
                        task.task_id,
                        input_err,
                    )
        self.shared_state.record_specialist_round(round_entry)
        self._record_specialist_round_product(task=task, round_entry=round_entry)

        # Per-anchor coverage ledger: every specialist completion is
        # one "round" — tick all anchors.
        self.shared_state.bump_domain_round_counters()

        # Persist so a resume picks up the bookkeeping without re-running the specialist.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )

        # Harvest specialist findings (hints, gap seeds, PR dedup) from any domain.
        if done_payload.get("new_findings"):
            await self._harvest_specialist_findings(done_payload)

        # Consume static-recon bridge candidates into gaps[] so the
        # freeform specialist picks them up with a precise mandate.
        if domain == "static_recon_specialist":
            self._coord.phase_internal.consume_static_recon(done_payload)

        # Aggregate research evidence from any research domain that
        # self-reports a ``research`` block, so FRAMEWORK / explore lanes
        # reuse the session-wide seen-set. Idempotent for research_scout
        # (already harvested above).
        self._aggregate_research_evidence(done_payload)

        # Refresh the gaps ledger after a specialist round closes; record the verdict as a gap attempt.
        gap_cid = str(done_payload.get("gap_canonical_id") or "").strip()
        if gap_cid:
            self.shared_state.append_gap_attempt(
                gap_cid,
                {
                    "action": "specialist",
                    "variant_name": domain,
                    "outcome": "EMPTY" if is_empty else "PROPOSALS",
                    "proposals_total": len(proposals),
                },
            )
        await self._coord.gap_refresh.refresh_gaps(
            reason="specialist_done", workload_id=self._coord.recipe_journal.workload_canonical_id()
        )
        if bool((task.params or {}).get("enablement")) and isinstance(done_payload.get("needs_targeted_build"), dict):
            await self._coord.enablement_build.maybe_enqueue_specialist_requested_build(
                task_id=str(task.task_id or ""),
                payload=done_payload,
            )
        # Push specialist-authored patches to the Critic so integrate_patch can pass.
        await self._coord.phase_framework.maybe_autosubmit_specialist_patches(
            task=task,
            done_payload=done_payload,
        )
        # Relaxed FRAMEWORK rule: a config-lever deliverable (no source patch,
        # but a proposal_set of serving flags / env vars) is routed through the
        # same integrate_patch gate via its config_changes channel.
        await self._coord.phase_framework.maybe_autosubmit_framework_config(
            task=task,
            done_payload=done_payload,
        )

    def _aggregate_research_evidence(self, done_payload: dict[str, Any]) -> None:
        """Aggregate research evidence (PR ids / diffs / NVIDIA refs) into the
        session-wide seen-set, de-duped across the session.

        Applies to every domain that self-reports a ``research`` block
        (candidate discovery + research_scout), so FRAMEWORK / explore lanes
        do not re-fetch the same references.
        """
        block = done_payload.get("research")
        if not isinstance(block, dict):
            return
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        if not pr_ids:
            return
        added = self.shared_state.register_seen_pr_ids(pr_ids)
        if added:
            log.info(
                "depth: aggregated %d new research reference(s) into seen-set",
                added,
            )

    async def _harvest_specialist_findings(self, done_payload: dict[str, Any]) -> None:
        """Persist top-level specialist findings and re-seed Orchestration.

        Any ``competitor_target`` numbers emitted are intentionally ignored here:
        measured competitor baselines are sourced from InferenceX, not authored
        by specialists, so LLM-written numbers must never be persisted as a
        consumable target.

        Args:
            done_payload: The completed specialist task payload.
        """
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        hints = done_payload.get("new_findings") or []
        if not isinstance(hints, list):
            hints = []
        added, dropped = _research_hints.append_hints(
            self.session_dir,
            hints,
        )
        if dropped:
            log.info(
                "research-scout: dropped %d sourceless hint(s)",
                dropped,
            )
        # Share inspected PR ids with the FRAMEWORK dedup set.
        pr_ids: list[Any] = []
        for hint in hints:
            if isinstance(hint, dict) and hint.get("source"):
                pr_ids.append(hint["source"])
        proposals = done_payload.get("proposal_set") or []
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                for key in ("pr_evidence", "source_evidence"):
                    refs = proposal.get(key)
                    if isinstance(refs, list):
                        pr_ids.extend(refs)
        self.shared_state.register_seen_pr_ids(pr_ids)
        # Seed high-priority hints as gaps[] so the config arm tries them early.
        self._coord.gap_refresh.seed_gaps_from_research_hints()
        log.info(
            "specialist findings harvested: hints_added=%d seen_pr_ids=%d",
            added,
            len(self.shared_state.research_scout_seen_pr_ids or []),
        )
