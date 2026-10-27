# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist dispatch helpers: warmup, auto-retry, wave fan-out, stalled-domain forcing, and round-entry construction."""

from __future__ import annotations

import asyncio
import csv
import logging as _logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from hyperloom.common.env import env_flag
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

from ..collaborator import CoordinatorCollaborator
from ..measurement.runtime_findings import render_runtime_findings
from ..state.shared_state import KbInjection
from ..state.task_registry import TaskSpec
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

#: ``(specialist_done key, failure class)`` for what the patch-safety gate dropped at admission: each reaches the
#: durable failure log, which every SEED turn renders.
_ADMISSION_DROPS: tuple[tuple[str, str], ...] = (
    ("patches_ungrounded", "patch_targets_ungrounded"),
    ("proposals_env_ungrounded", "proposal_env_file_missing"),
)

#: Domains whose focus reads the target's ``config.json`` summary from ``model_info``.
_MODEL_INFO_DOMAINS: frozenset[str] = frozenset({"static_recon_specialist", "speculative_decoding_specialist"})

#: ``(SharedState field, param key, conversion)`` for the values a specialist is warmed with. The converted value
#: decides: an empty string, an empty container and a zero are not exported, because an absent key is how the
#: specialist reads "this session has none".
_STATE_PARAMS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("model_name", "model_name", str),
    ("model_path", "model_path", str),
    ("warm_start_recipe", "warm_start_recipe", dict),
    ("warm_start_pitfalls", "warm_start_pitfalls", list),
    ("warm_start_lessons", "warm_start_lessons", list),
    ("framework", "framework", str.strip),
    ("tp", "tp", int),
    ("precision", "precision", str),
    ("conc", "conc", int),
    ("isl", "isl", int),
    ("osl", "osl", int),
    ("max_model_len", "max_model_len", int),
    # The mode selects the specialist's workload block; the corpus shape supplies its numbers.
    ("benchmark_mode", "benchmark_mode", str),
    ("agentx_corpus_shape", "agentx_corpus_shape", dict),
    ("grading", "agentx_grading", dict),
    ("agentx_backend", "agentx_backend", str),
    ("baseline_tput", "baseline_tput", float),
    ("cumulative_gain_validated", "cumulative_gain_validated", float),
)

#: Optional round-row fields, taken from the done payload or else the dispatch params; an empty value is not carried.
_ROUND_CARRIED_KEYS: tuple[str, ...] = (
    "task_kind",
    "scope",
    "proposal_msg_id",
    "framework_agent_candidate_id",
    "framework_batch_id",
    "reauthor_attempt",
    "apply_retry_attempt",
    "kb_read_id",
    "kb_rendered_refs",
)

# Bounded transient-failure auto-retry for specialist dispatches (infra-only).
SPECIALIST_AUTO_RETRY_MAX: int = 2

# Hard-trigger thresholds: optimisation rounds a domain may go without a specialist dispatch / a KEEP before the
# Coordinator force-dispatches one.
FORCE_STALLED_SPECIALIST_ROUNDS: int = 8
FORCE_STALLED_KEEP_ROUNDS: int = 12

# Gap id for the custom dispatch when the ledger has no gap yet.
CUSTOM_SESSION_GAP_ID: str = "gap.custom.session"


class ClassifiedFailure(NamedTuple):
    """A specialist attempt's classified failure type and the reason it carried."""

    ftype: SpecialistFailureType
    error: str


def _round_entry_extras(
    done_payload: dict[str, Any], task_params: dict[str, Any], *, run_error: str, source_phase: str
) -> dict[str, Any]:
    """The optional specialist-round fields: carried keys, lane flags, status, phase, GPUs and notes."""
    extras: dict[str, Any] = {}
    for key in _ROUND_CARRIED_KEYS:
        value = done_payload.get(key)
        if value in (None, "", [], {}):
            value = task_params.get(key)
        if value not in (None, "", [], {}):
            extras[key] = value
    for key in ("candidate_discovery", "framework_agent_authoring"):
        if done_payload.get(key) or task_params.get(key):
            extras[key] = True
    if run_error:
        extras["status"] = "failed"
        extras["error"] = extras["run_error"] = str(run_error)[:1000]
    elif done_payload.get("status") not in (None, ""):
        extras["status"] = str(done_payload.get("status"))
    if source_phase:
        extras["source_phase"] = source_phase
    gpu_ids = done_payload.get("allocated_gpu_ids")
    if isinstance(gpu_ids, list) and gpu_ids:
        extras["allocated_gpu_ids"] = [
            int(g) for g in gpu_ids if isinstance(g, (int, str)) and str(g).strip().lstrip("-").isdigit()
        ]
    specialist_notes = done_payload.get("_specialist_notes")
    if isinstance(specialist_notes, list) and specialist_notes:
        extras["notes"] = list(map(str, specialist_notes))
    return extras


class SpecialistDispatchCollaborator(CoordinatorCollaborator):
    """Specialist dispatch: warmup, auto-retry, wave fan-out, stalled-domain forcing, and round-entry construction."""

    def __init__(self, coordinator: "Coordinator", proposal_scorer: Any = None) -> None:
        super().__init__(coordinator)
        # Advisory only: scores proposals, never gates them.
        self._proposal_scorer = proposal_scorer
        # Key of the custom dispatch admission last refused; it is not requested again until the key changes.
        self.refused_custom_key = ""

    def refuse_patch_without_source(self, params: dict[str, Any], *, task_id: str) -> bool:
        """Whether a warmed patch specialist must be refused because no git source tree backs it.

        A refusal ``last_action_failures`` already lists is not recorded again: the force dispatch retries every tick.
        """
        state = self.shared_state
        error = specialist_patch_preflight_error(params, framework_repo_path=state.framework_repo_path)
        if not error:
            return False
        if not any(
            row.get("action") == "specialist" and row.get("error_class") == error for row in state.last_action_failures
        ):
            state.record_action_failure(
                action="specialist", task_id=task_id, result={"error_class": error, "error": error}
            )
            log.error("specialist: refused a source-patch dispatch: %s", error)
        return True

    async def warm_specialist_params(self, params: dict[str, Any]) -> None:
        """Fill specialist task params with KnowledgePlane data before enqueue (mutates in place); missing fields stay empty.

        Args:
            params: The specialist task params dict mutated in place with PR
                feed, warm-start, hardware/workload and gap/roofline context.
        """
        state = self.shared_state
        plane = self.knowledge_plane

        from .domains import dispatch_domain, gap_backfilled_domain, normalize_dispatch_tags
        from .profile import resolve_specialist_profile

        # Bench-capable specialists run a real serving + benchmark loop, so
        # default needs_gpu to route them through the gpu_specialist_pool.
        if resolve_specialist_profile(params).reserves_benchmark_lane:
            params.setdefault("needs_gpu", True)

        if backfilled := gap_backfilled_domain(params, state.find_gap):
            params["domain"] = backfilled
        # Resolve as the runner does, so a tags-only or gap-backfilled dispatch is warmed for the domain it runs as.
        resolved_domain = dispatch_domain(params)
        domain = resolved_domain.key if resolved_domain is not None else ""
        normalize_dispatch_tags(params)

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(plane is not None and getattr(plane, "pr_monitor_enabled", True))

        params.setdefault("kb_subgraph", {})

        for attr, key, convert in _STATE_PARAMS:
            value = convert(getattr(state, attr))
            if value:
                params.setdefault(key, value)
        self._warm_framework_context(params)
        self._warm_prompt_hints(params, domain)
        self._warm_gap_anchors(params)
        self._warm_progress(params)

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
                except (ValueError, csv.Error) as exc:
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

    def _warm_framework_context(self, params: dict[str, Any]) -> None:
        """Warm the runtime framework version and the local framework source roots."""
        state = self.shared_state
        # Runs after the ``_STATE_PARAMS`` warmup, whose ``str.strip`` requires ``state.framework`` to be a str.
        # runtime framework version for version-mismatch annotation.
        if "framework_version" not in params:
            fp_meta = getattr(state, "stack_fingerprint_meta", None) or {}
            if isinstance(fp_meta, dict):
                fw = str(params.get("framework") or state.framework).lower()
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
                tree = resolve_framework_tree(state.framework)
                if tree:
                    params["session_framework_tree"] = tree
            except (KeyError, OSError) as exc:
                log.debug(
                    "specialist warmup: framework_source_roots lookup failed: %r",
                    exc,
                )

    def _warm_prompt_hints(self, params: dict[str, Any], domain: str) -> None:
        """Warm the hardware, model-architecture and advisory prompt hints ``domain`` reads."""
        state = self.shared_state
        # Hardware + workload hints from SharedState; else dataclass defaults win.
        params.setdefault("gpu_type", state.gpu_type or "")

        # Advisory model_arch profile via arch_notes carrier (prompt-context only).
        if "arch_notes" not in params:
            from ..state._shared_state.render import render_model_arch_compact

            _arch_notes = render_model_arch_compact(getattr(state, "model_arch", None))
            if _arch_notes:
                params["arch_notes"] = _arch_notes

        if domain in _MODEL_INFO_DOMAINS and "model_info" not in params:
            _minfo = getattr(state, "model_info", None)
            if isinstance(_minfo, dict) and _minfo:
                params["model_info"] = dict(_minfo)

        # Checklist-derived focus directories; a caller that named its own keeps it.
        if "source_hint_directories" not in params:
            from ..knowledge import static_recon_checklist as _src_recon

            _dirs = _src_recon.source_hint_directories_for(
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
            from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

            _hints_block = _research_hints.summarise_for_prompt(self.session_dir)
            if _hints_block:
                params["research_hints"] = _hints_block

    def _warm_gap_anchors(self, params: dict[str, Any]) -> None:
        """Warm the targeted gap's symptom, layer and recent attempts from the gaps[] ledger."""
        state = self.shared_state
        # Fill gap-specific anchors from the gaps[] ledger.
        gap_cid = str(params.get("gap_canonical_id") or "").strip() or str(params.get("gap") or "").strip()
        if gap_cid:
            gap = state.find_gap(gap_cid)
            if gap is not None:
                if not params.get("gap_symptom"):
                    params["gap_symptom"] = str(gap.get("symptom") or "")
                if not params.get("gap_layer"):
                    params["gap_layer"] = str(gap.get("layer") or "")
                evidence = params.get("gap_evidence")
                if not isinstance(evidence, dict) or not evidence:
                    attempts = list(gap.get("attempts") or [])[-5:]
                    if attempts:
                        params["gap_evidence"] = {
                            "recent_attempts": attempts,
                            "severity": str(gap.get("severity") or ""),
                        }

    def _warm_progress(self, params: dict[str, Any]) -> None:
        """Warm the session's current throughput, keep threshold and applied stack."""
        state = self.shared_state
        if "current_tput" not in params:
            _ct = float(state.current_best.get("tput") or 0.0)
            if _ct > 0:
                params["current_tput"] = _ct
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
            KbInjection(
                consumer="specialist",
                domain=str(params.get("domain") or ""),
                gap_canonical_id=str(params.get("gap_canonical_id") or ""),
                read_id=evidence.read_id,
                experience_ids=[str(ref.get("id") or "") for ref in evidence.rendered_refs],
                experiences=[dict(item) for item in evidence.experiences],
                prompt_block=evidence.prompt_block,
            )
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
                failure=ClassifiedFailure(ftype=ftype, error=error),
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
                TaskSpec(
                    kind="specialist",
                    params=retry_params,
                    idempotency_key=retry_key,
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="coordinator",
                )
            )
        if was_existing:
            # Retry slot already taken: let normal bookkeeping record this attempt.
            await self._record_specialist_retry_exhausted(
                task=task,
                failure=ClassifiedFailure(ftype=ftype, error=error),
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
        failure: ClassifiedFailure,
        attempts_used: int,
        cap: int,
        detail: str,
    ) -> None:
        """Broadcast that an infra-failed specialist is being abandoned.

        Args:
            task: The specialist task whose final attempt failed.
            failure: The classified failure type and the reason the attempt carried.
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
                "failure_type": failure.ftype.value,
                "reason": failure.error[:200],
                "detail": detail,
            },
        )
        log.warning(
            "specialist auto-retry exhausted: task=%s failure=%s attempts=%d/%d (%s)",
            task.task_id,
            failure.ftype.value,
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
        specialist rounds that still has an open gap in the gaps[] ledger.

        A real scheduling event (a domain delegate routed through PolicyGate +
        warmup + the GPU specialist pool). Idempotent per
        ``(anchor, round, macro_cycle)``; a domain with a specialist already
        queued or running is skipped. The dispatcher zeroes the anchor's
        dispatch counter when the forced specialist spawns; its keep counter
        clears only on a KEEP from that domain's specialist. At most one forced
        dispatch per tick.

        Note:
            Side-effecting: may dispatch a domain specialist via
            ``handle_intent``. Returns nothing.
        """
        state = self.shared_state
        if str(getattr(state, "phase", "") or "").upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        stalled = state.stalled_domains(
            specialist_threshold=FORCE_STALLED_SPECIALIST_ROUNDS,
            keep_threshold=FORCE_STALLED_KEEP_ROUNDS,
        )
        if not stalled:
            return

        from .domains import domain_for_tag
        from .in_flight import in_flight_domains

        busy_domains = in_flight_domains([*await self.tasks.queued(), *await self.tasks.running()])
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
            idempotency_key = f"forced-stalled-{anchor}-round{round_id}{self._coord.dispatcher.cycle_idem_suffix()}"
            if await self.tasks.find_by_idempotency_key(idempotency_key) is not None:
                continue
            await self.warm_specialist_params(params)
            if self.refuse_patch_without_source(params, task_id=idempotency_key):
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
            except OSError:
                log.exception("stalled-domain force: state save failed")
            log.info(
                "stalled-domain force: dispatched domain=%s anchor=%s gap=%s round=%d",
                dom.key,
                anchor,
                gap_cid,
                round_id,
            )
            # One forced dispatch per tick.
            return
        return

    async def ensure_custom_specialist(self) -> None:
        """Dispatch the operator-defined specialist once per macro cycle in FRAMEWORK_AGENT.

        Coordinator-owned: it bypasses the policy gate, which denies every custom delegate from Orchestration.
        """
        state = self.shared_state
        prompt = str(state.custom_specialist_prompt or "").strip()
        if not prompt or str(getattr(state, "phase", "") or "").upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        from .domains import CUSTOM_SPECIALIST_KEY, custom_specialist_key
        from .profile import MODE_RESEARCH, NO_LEASE_EXTENSION

        key = custom_specialist_key(int(state.macro_cycle or 0), prompt)
        if key == self.refused_custom_key or await self.tasks.find_by_idempotency_key(key) is not None:
            return
        from ..state._shared_state.phase_state import gap_actionability_key

        params: dict[str, Any] = {
            "domain": CUSTOM_SPECIALIST_KEY,
            "scope": "domain",
            "source": "coordinator_internal",
            "reason": "custom_specialist",
        }
        if state.custom_specialist_only:
            params[NO_LEASE_EXTENSION] = True
        gaps = [g for g in state.gaps if isinstance(g, dict) and str(g.get("canonical_id") or "").strip()]
        if gaps:
            params["gap_canonical_id"] = str(min(gaps, key=gap_actionability_key)["canonical_id"]).strip()
        else:
            params["gap_canonical_id"] = CUSTOM_SESSION_GAP_ID
            params["gap_symptom"] = "No profiled gap; follow the operator-defined focus."
            params["gap_layer"] = "operator-defined"
        await self.warm_specialist_params(params)

        research_reason = specialist_patch_preflight_error(params, framework_repo_path=state.framework_repo_path)
        if research_reason:
            params["mode"] = MODE_RESEARCH
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {"kind": "custom_specialist_research_only", "reason": research_reason},
            )
        intent = Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "specialist", "params": params, "idempotency_key": key},
        )
        await self._coord.router.handle_delegate("orchestration", intent)
        if await self.tasks.find_by_idempotency_key(key) is None:
            self.refused_custom_key = key
            log.warning("custom specialist: dispatch %s refused (see the policy denial); not retried this cycle", key)
            return
        log.info(
            "custom specialist: requested key=%s gap=%s mode=%s",
            key,
            params["gap_canonical_id"],
            params.get("mode") or "default",
        )

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
        proposals = done_payload.get("proposal_set")
        if not isinstance(proposals, list):
            proposals = []
        task_params = task.params or {}
        source_phase = (
            str(task_params.get("source_phase") or done_payload.get("source_phase") or self.shared_state.phase or "")
            .strip()
            .upper()
        )
        from .domains import normalize_dispatch_tags

        entry: dict[str, Any] = {
            "round_id": str(task_params.get("round_id") or task.task_id),
            "task_id": task.task_id,
            "source": source,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "domain": str(done_payload.get("domain") or task_params.get("domain") or ""),
            # Knowledge-domain tags; reported tags win over dispatch params.
            "tags": normalize_dispatch_tags(done_payload) or normalize_dispatch_tags(task_params),
            "gap_canonical_id": str(done_payload.get("gap_canonical_id") or task_params.get("gap_canonical_id") or ""),
            "proposals_total": len(proposals),
            "proposal_set": list(proposals),
            "summary": str(done_payload.get("summary") or "")[:480],
            "reason": str(run_error or done_payload.get("reason") or "")[:480],
            "confidence": done_payload.get("confidence"),
            "new_findings": list(done_payload.get("new_findings") or []),
            "residual_questions": list(done_payload.get("residual_questions") or []),
        }
        entry.update(_round_entry_extras(done_payload, task_params, run_error=run_error, source_phase=source_phase))
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
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import SpecialistRound

        phase_event.record_specialist_round(
            SpecialistRound(
                task_id=str(task.task_id or ""),
                round_id=str(round_entry.get("round_id") or ""),
                domain=round_entry.get("domain") or "",
                gap_canonical_id=round_entry.get("gap_canonical_id") or "",
                reason=round_entry.get("reason") or "",
                source=round_entry.get("source") or "",
                tags=round_entry.get("tags") or [],
                **product,
            )
        )

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
        round_entry = self.build_specialist_round_entry(
            task=task,
            done_payload=done_payload,
            source=source,
            run_error=run_error,
        )
        # Specialist notes reach the prompt only through this task's one inbox
        # line; ``last_action_failures`` is rendered every SEED turn.
        for dropped_key, error_class in _ADMISSION_DROPS:
            dropped = done_payload.get(dropped_key)
            if isinstance(dropped, list) and dropped:
                self.shared_state.record_action_failure(
                    action="specialist",
                    task_id=task.task_id,
                    result={"error_class": error_class, "error": "; ".join(str(d) for d in dropped[:4])},
                )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing.
        if self._proposal_scorer is not None and round_entry["proposal_set"]:
            scores = await self._proposal_scorer.score(
                gap={
                    "domain": round_entry["domain"].strip(),
                    "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                    "gap_symptom": task_params.get("gap_symptom"),
                    "gap_evidence": task_params.get("gap_evidence"),
                    "summary": done_payload.get("summary", ""),
                },
                proposals=round_entry["proposal_set"],
                task_id=task.task_id,
                tick=int(self.shared_state.tick or 0),
                phase=self.shared_state.phase or None,
            )
            if scores.get("models") or scores.get("errors"):
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
        except OSError:
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )
        await self._follow_up_specialist_round(task, done_payload, round_entry)

    async def _follow_up_specialist_round(
        self, task: Task, done_payload: dict[str, Any], round_entry: dict[str, Any]
    ) -> None:
        """Hand a recorded specialist round to the gap ledger and the phases that consume what it produced."""
        domain = round_entry["domain"].strip()
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
                    "outcome": "EMPTY" if not round_entry["proposals_total"] else "PROPOSALS",
                    "proposals_total": round_entry["proposals_total"],
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
