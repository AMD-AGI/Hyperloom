# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist dispatch helpers: warmup, auto-retry, wave fan-out, stalled-domain forcing, and round-entry construction."""

from __future__ import annotations

import logging as _logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from hyperloom.common.env import env_flag, is_truthy
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

from ..collaborator import CoordinatorCollaborator
from ..phases import machine_state as _phase_state
from ..policy.gate import (
    SPECIALIST_FROM_AGENT_PREFIX,
    PolicyDenied,
    validate_freeform_wave_task,
)
from .runner import SpecialistFailureType, specialist_patch_preflight_error

if TYPE_CHECKING:
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

    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
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
            _gap_notes = self._target_gap_advisory_block()
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

    async def _maybe_auto_retry_specialist(
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

        # Mirror _handle_delegate lane/ttl resolution so the retry task holds the
        # same pools as the original and cannot run concurrently with serving.
        lanes, ttl = self._registry_lanes_ttl("specialist")
        from .profile import resolve_specialist_profile, uses_whole_machine_gpu_lane

        if resolve_specialist_profile(retry_params).reserves_benchmark_lane:
            lanes = list(dict.fromkeys((*lanes, "benchmark_lane")))
        needs_gpu = is_truthy(retry_params.get("needs_gpu"))
        if not needs_gpu and uses_whole_machine_gpu_lane(retry_params):
            # bench specialist: ensure needs_gpu is set so gpu_research_lane is acquired.
            needs_gpu = True
        if needs_gpu:
            lanes = list(dict.fromkeys((*lanes, "gpu_research_lane")))
            ttl = self._gpu_lease_ttl_sec(
                int(ttl or 0),
                params=retry_params,
            )

        # Stable base key across attempts: strip any prior ``-autoretryN`` suffix.
        base_key = str(task.idempotency_key or task.task_id or "")
        if "-autoretry" in base_key:
            head, _, tail = base_key.rpartition("-autoretry")
            if tail.isdigit():
                base_key = head
        retry_key = f"{base_key}-autoretry{next_attempt}"

        new_task, was_existing = await self.tasks.create_or_return_existing(
            kind="specialist",
            params=retry_params,
            idempotency_key=retry_key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
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
        await self._record_observation(
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
        await self._record_observation(
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

    async def _fan_out_specialist_wave(
        self,
        source: str,
        intent: Intent,
        params: dict[str, Any],
    ) -> None:
        """Fan a specialist delegate carrying ``params.tasks=[...]`` into N
        standard free-form specialist dispatches (scope=freeform, lane=cpu,
        mode=research defaults). Each fanned task is re-dispatched through the
        normal ``_handle_delegate`` path. Per-task idempotency keys derive from
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
                "lane",
                "model",
                "priority",
                "timeout_minutes",
                "max_turns",
            ):
                if isinstance(task, dict) and carry in task:
                    sub_params[carry] = task[carry]
            sub_params.setdefault("mode", "research")
            sub_params.setdefault("lane", "cpu")
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
                await self._record_policy_denied(source, sub_intent, denied)
                raise
            pending.append(sub_intent)
        for sub_intent in pending:
            await self._handle_delegate(source, sub_intent)

    async def _maybe_force_stalled_domain_specialist(self) -> None:
        """Force-dispatch a domain specialist for a domain untouched for too many
        config-arm rounds that still has an open gap in the gaps[] ledger.

        A real scheduling event (a domain delegate routed through PolicyGate +
        warmup + the GPU specialist pool). Idempotent per
        ``(anchor, round, macro_cycle)`` and self-throttling (zeroes the
        per-anchor counter on dispatch). At most one forced dispatch per tick.

        Note:
            Side-effecting: may dispatch a domain specialist via
            ``_handle_intent`` and mutate per-anchor throttle counters on
            ``shared_state``. Returns nothing.
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

        round_id = int((state.explore_search or {}).get("cursor") or 0)
        for anchor in stalled:
            gap_cid = state.best_gap_for_anchor(anchor)
            if not gap_cid:
                continue
            dom = domain_for_tag(anchor)
            if dom is None:
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
            idempotency_key = f"forced-stalled-{anchor}-round{round_id}{self._cycle_idem_suffix()}"
            lookup = getattr(self.tasks, "find_by_idempotency_key", None)
            if callable(lookup):
                existing = await lookup(idempotency_key)
                if existing is not None:
                    continue
            await self._warm_specialist_params(params)
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
            # Zero the counter up-front so a slow enqueue can't re-fire next tick.
            state.note_specialist_dispatched(anchor)
            await self._handle_intent("orchestration", intent)
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

    def _build_specialist_round_entry(
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

    @staticmethod
    def _task_id_from_specialist_source(source: str) -> str:
        """Extract the task_id from a ``specialist:<task_id>`` source ("" when prefix is absent).

        Args:
            source: The from-agent string to parse.

        Returns:
            The task id when the specialist prefix is present, else ``""``.
        """
        if not source:
            return ""
        if source.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            return source[len(SPECIALIST_FROM_AGENT_PREFIX) :]
        return ""
