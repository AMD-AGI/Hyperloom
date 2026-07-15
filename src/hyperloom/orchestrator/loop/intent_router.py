# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Intent routing collaborator for :class:`Coordinator`.

:meth:`IntentRouter.handle_intent` validates an emitted intent through
``PolicyGate`` and dispatches it to the matching ``_handle_*`` method.

``IntentRouter`` holds a back-reference to its owning ``Coordinator`` and
delegates every attribute it does not define itself to that coordinator via
``__getattr__``, so handler bodies keep using ``self.shared_state`` /
``self.bus`` etc. ``Coordinator`` keeps thin forwarding shims that delegate
to the router.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from .coordinator_helpers import format_exc_brief, serialize_verdict_advisory
from ..bus.message_bus import Message
from ..policy.gate import PolicyDenied
from ..state.task_registry import Task
from ..kernel.request_handlers import get_handler

# ``Coordinator`` is intentionally NOT imported (avoids a module-level import
# cycle with coordinator.py); it is held as a back-reference and the annotation
# below is a deferred string.

log = __import__("logging").getLogger(__name__)


# IntentType -> the ``Coordinator`` handler method it dispatches to. Replaces the
# former 12-branch if/elif in :meth:`IntentRouter._handle_intent`; an unknown
# type falls through to the observation fallback (see the ``else`` branch there).
# ``SPECIALIST_DONE`` is a terminal specialist intent (R3 already validated); its
# handler only bookkeeps (defense-in-depth).
_INTENT_DISPATCH: dict[IntentType, str] = {
    IntentType.PROPOSE_ACTION: "_handle_propose_action",
    IntentType.REVIEW_VERDICT: "_handle_review_verdict",
    IntentType.DELEGATE: "_handle_delegate",
    IntentType.REQUEST: "_handle_request",
    IntentType.RESPONSE: "_handle_response",
    IntentType.KILL_TASK: "_handle_kill_task",
    IntentType.PRUNE_BRANCH: "_handle_prune_branch",
    IntentType.ESCALATE_STRATEGY_CHANGE: "_handle_escalate_strategy_change",
    IntentType.SEND_MESSAGE: "_handle_send_message",
    IntentType.ALERT: "_handle_alert",
    IntentType.UPDATE_STATE: "_handle_update_state",
    IntentType.SPECIALIST_DONE: "_handle_specialist_done",
}


class IntentRouter:
    """Validates and dispatches agent-emitted intents on behalf of a Coordinator."""

    def __init__(self, coordinator: "Coordinator") -> None:  # noqa: F821 - deferred ref, not imported to avoid an import cycle (see note above)
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        # Attributes not defined on the router resolve onto the coordinator.
        return getattr(object.__getattribute__(self, "_coord"), name)

    async def _handle_intent(self, source: str, intent: Intent) -> None:
        """Validate an emitted intent through PolicyGate, then route it.

        Runs the intent through :meth:`PolicyGate.validate_intent`; a
        :class:`PolicyDenied` is recorded and the intent dropped. Valid intents
        are dispatched to the matching ``_handle_*`` method by type, and the
        agent's message cursor is advanced to the latest sequence afterward.

        Args:
            source (str): The agent that emitted the intent.
            intent (Intent): The parsed intent to validate and route.
        """
        try:
            self.policy.validate_intent(source, intent)
        except PolicyDenied as denied:
            await self._record_policy_denied(source, intent, denied)
            return

        try:
            it = intent.type
            handler_name = _INTENT_DISPATCH.get(it)
            if handler_name is not None:
                await getattr(self._coord, handler_name)(source, intent)
            else:
                # Unknown / unhandled intent — record for replay.
                await self._record_observation(
                    source, "observation",
                    {"intent": it.value, "payload": intent.payload},
                )
            await self._cursor_advance_to_latest(source)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("intent handler for %s raised", source)
            self._record_coordinator_exception(
                stage="handle_intent",
                agent=source,
                exc=exc,
            )
            try:
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "handle_intent_exception",
                        "agent": source,
                        "intent_type": intent.type.value,
                        "error": format_exc_brief(exc, limit=500),
                    },
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to record handle_intent_exception observation")
            return

    async def _handle_propose_action(self, source: str, intent: Intent) -> None:
        """Gate a proposed action and enqueue it for Critic Review.

        Drops proposals for pruned families, applies the pending-roofline and
        execution-order denials, then publishes a ``proposal`` message and
        registers a :class:`PendingProposal` so the Critic gate can later
        return a verdict.

        Args:
            source (str): The agent proposing the action.
            intent (Intent): The PROPOSE_ACTION intent; ``payload`` carries
                ``action_name`` and optional ``params`` / ``predicted_gain_pct``.
        """
        action_name = intent.payload["action_name"]
        # Pruned families are advisory: proposal still queues with an advisory note.
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "proposal_pruned_advisory",
                    "from": source,
                    "action": action_name,
                    "hint": (
                        f"{action_name!r} is in pruned_families; if the "
                        "prune was speculative the LLM may pick this "
                        "action again, otherwise prefer another "
                        "phase-allowed action."
                    ),
                },
            )
        denied = self._sequence_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        msg = Message.new(
            source, "*", "proposal",
            {**intent.payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        from .coordinator import PendingProposal
        pending = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent=source,
            action_name=action_name,
            predicted_gain_pct=float(intent.payload.get("predicted_gain_pct", 0.0)),
            payload=dict(intent.payload),
        )
        self.state.pending_proposals[msg.msg_id] = pending

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        """Apply a Critic ``review_verdict`` to its target proposal; verdicts collapse by priority (approve > reject > needs_review).

        Args:
            source: The agent (Critic) emitting the verdict.
            intent: The REVIEW_VERDICT intent; payload carries
                ``target_proposal_msg_id`` and ``verdict``/``verdict_map``.
        """
        target = intent.payload["target_proposal_msg_id"]
        pending = self.state.pending_proposals.get(target)
        verdict_map = intent.payload.get("verdict_map")
        single_verdict = intent.payload.get("verdict")
        if pending is None:
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind":         "verdict_for_unknown_proposal",
                    "target":       target,
                    "verdict":      single_verdict or "",
                    "verdict_map":  bool(verdict_map),
                },
            )
            return
        verdict = str(single_verdict or "")
        if not verdict and isinstance(verdict_map, dict) and verdict_map:
            sub_verdicts = [
                str((entry or {}).get("verdict") or "").strip()
                for entry in verdict_map.values()
            ]
            verdict = (
                "approve" if "approve" in sub_verdicts
                else "reject" if "reject" in sub_verdicts
                else "advise" if "advise" in sub_verdicts
                else "needs_review"
            )
        await self._coord._handle_single_verdict(
            source=source,
            pending=pending,
            verdict=verdict,
            reasoning=str(intent.payload.get("reasoning") or ""),
            advisory=serialize_verdict_advisory(intent.payload),
        )

    async def _handle_single_verdict(
        self,
        *,
        source: str,
        pending: "PendingProposal",  # noqa: F821 - deferred ref; imported lazily in handlers to avoid import cycle.
        verdict: str,
        reasoning: str,
        advisory: dict[str, Any] | None = None,
    ) -> None:
        """Single-verdict handler (approve/advise materialises proposal as-is); mirrors integrate_patch/specialist verdicts onto specialist_patch_verdicts for PolicyGate.

        Args:
            source: The agent emitting the verdict.
            pending: The pending proposal the verdict targets.
            verdict: The collapsed verdict (approve / advise / reject / needs_review).
            reasoning: Free-text reasoning recorded with the verdict.
            advisory: Pre-serialised advisory fields (``required_evidence`` /
                ``risks`` / ``advice_text`` / ``alternative_action`` /
                ``notes`` / ``kb_evidence`` / ``packet_evidence``) to carry on
                the rebroadcast payload so the full Critic context reaches the
                orchestration inbox and downstream consumers.
        """
        pending.decided = True
        pending.verdict = verdict
        if pending.action_name == "framework_agent":
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind":            "framework_agent_verdict_received",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "candidate_id":    str((pending.payload or {}).get("framework_agent_candidate_id") or ""),
                    "verdict":         verdict,
                },
            )
        rebroadcast_payload: dict[str, Any] = {
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict":                verdict,
            "reasoning":              reasoning,
        }
        if advisory:
            rebroadcast_payload.update(advisory)
        await self.bus.append_and_seq(Message.new(
            source, pending.from_agent, "review_verdict",
            rebroadcast_payload,
            priority=0 if verdict == "reject" else 1,
            in_reply_to=pending.proposal_msg_id,
        ))
        # Mirror specialist / integrate_patch verdicts onto SharedState so
        # PolicyGate's integrate_patch gate can consult them on the next tick.
        try:
            pa_params = pending.payload.get("params") or {}
        except AttributeError:
            pa_params = {}
        sid_candidate = ""
        if pending.action_name == "integrate_patch":
            sid_candidate = str(
                pa_params.get("specialist_task_id") or ""
            ).strip()
        elif pending.action_name == "specialist":
            # Critic verdict on the specialist proposal counts as the verdict on its patches; task_id is the key.
            sid_candidate = str(pa_params.get("task_id") or "").strip()
        if sid_candidate and verdict:
            try:
                self.shared_state.record_specialist_patch_verdict(
                    sid_candidate, verdict,
                )
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — best-effort mirror
                log.exception(
                    "failed to mirror critic verdict for specialist task=%s",
                    sid_candidate,
                )
        # Both `approve` and `advise` mean "dispatch may proceed"; treat them
        # identically for materialization.
        if verdict in ("approve", "advise"):
            await self._materialize_approved_proposal(pending)
        elif verdict == "reject" and pending.action_name == "framework_agent":
            # Record the critic_denied row so the framework_agent pump advances.
            await self._coord._record_framework_agent_critic_denied(
                pending, reasoning,
            )
        elif verdict == "needs_review":
            await self._coord._maybe_reauthor_from_critic_feedback(
                pending, advisory,
            )

    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        """Validate and enqueue a delegated action as a TaskRegistry task.

        Drops pruned families and execution-order violations, re-routes
        ``explore`` grids through the Critic-review path, and otherwise
        materialises the delegated action (specialist, dynamic action, etc.)
        into a task with the appropriate lanes, tools and warmed params.

        Args:
            source (str): The agent issuing the delegation.
            intent (Intent): The DELEGATE intent; ``payload`` carries
                ``action_name`` and optional ``params``.
        """
        action_name = intent.payload["action_name"]
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "delegate_pruned_advisory",
                    "from": source,
                    "action": action_name,
                    "hint": (
                        f"{action_name!r} is in pruned_families; if the "
                        "prune was speculative the LLM may pick this "
                        "action again, otherwise prefer another "
                        "phase-allowed action."
                    ),
                },
            )
        denied = self._sequence_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(
                source, intent, denied, action_name=action_name,
            )
            return
        # delegate explore runs variants directly (no Critic pre-review).
        params = dict(intent.payload.get("params") or {})
        # idempotency_key is top-level per schema; strip a nested compat alias.
        nested_idempotency_key = params.pop("idempotency_key", None)
        # Plumb baseline's materialized YAML into grid-style tasks (delegator may override).
        if (
            action_name in ("sweep", "explore")
            and self.shared_state.baseline_config_path
        ):
            params.setdefault(
                "config_path", self.shared_state.baseline_config_path
            )
        # Parity with _materialize_approved_proposal: direct delegates need the same knobs.
        if action_name == "explore":
            self._inject_explore_runtime_params(params)
            # Inject base_tput tied to current_best (or baseline_tput).
            cb = getattr(self.shared_state, "current_best", None) or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else getattr(self.shared_state, "baseline_tput", 0.0)
            params.setdefault("base_tput", float(base or 0.0))
        # Wave sugar: a specialist delegate carrying params.tasks=[...] fans out
        # into N standard freeform specialist tasks, each dispatched through the
        # normal SpecialistRunner + TaskRegistry + lease + reap path.
        if action_name == "specialist" and isinstance(
            params.get("tasks"), list,
        ) and params["tasks"]:
            await self._fan_out_specialist_wave(source, intent, params)
            return
        # Specialist pre-dispatch warmup via KnowledgePlane.
        if action_name == "specialist":
            await self._warm_specialist_params(params)
        # Idempotency-key chain: top-level -> nested compat alias -> content-fingerprint auto-key.
        raw_key = intent.payload.get("idempotency_key") or nested_idempotency_key
        if not raw_key:
            content_fp = hashlib.sha1(
                json.dumps(params, sort_keys=True, default=str).encode()
            ).hexdigest()[:10]
            raw_key = (
                f"{source}:{action_name}:t{int(self.shared_state.tick or 0)}:"
                f"{content_fp}"
            )
        idempotency_key = str(raw_key)
        terminal_states = {
            "succeeded", "failed", "cancelled", "needs_manual_review",
        }
        task = None
        was_existing = False
        for attempt in range(6):
            idempotency_key = (
                str(raw_key) if attempt == 0 else f"{raw_key}-retry{attempt}"
            )
            lanes, ttl = self._registry_lanes_ttl(action_name)
            # Bench-enabled specialists serialize against the other GPU
            # benchmark/profile/server work via benchmark_lane (research_lane
            # alone conflicts with nothing).
            if action_name == "specialist":
                from ..specialists.profile import resolve_specialist_profile
                if resolve_specialist_profile(params).reserves_benchmark_lane:
                    lanes = tuple(dict.fromkeys((*lanes, "benchmark_lane")))
                # Any GPU-holding specialist serializes against serving via
                # gpu_research_lane. Its lane lease TTL comes from the agent wall
                # budget (iron law: kill <= gpu_lease TTL <= gpu_research_lane TTL).
                needs_gpu_raw = params.get("needs_gpu", False)
                needs_gpu = (
                    needs_gpu_raw.strip().lower() in ("1", "true", "yes", "on")
                    if isinstance(needs_gpu_raw, str)
                    else bool(needs_gpu_raw)
                )
                if needs_gpu:
                    lanes = tuple(dict.fromkeys((*lanes, "gpu_research_lane")))
                    try:
                        # Shared with the GPU-pool lease so the two TTLs never drift.
                        ttl = self._coord._gpu_lease_ttl_sec(int(ttl or 0))
                    except Exception:  # noqa: BLE001 — fall back to registry ttl
                        log.exception(
                            "WS2: failed to re-source gpu_research_lane TTL; "
                            "using registry default",
                        )
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=action_name,
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            if not was_existing:
                break
            if task.state not in terminal_states:
                hint = (
                    f"task {task.task_id} is still {task.state!r}; wait for the "
                    f"delegated_result event instead of re-emitting the same key."
                )
                await self._record_policy_denied(
                    source, intent,
                    PolicyDenied(
                        f"delegate{{action_name={action_name!r}}} duplicate "
                        f"idempotency_key={idempotency_key!r}",
                        rule="duplicate_idempotency_key_running",
                        hint=hint,
                    ),
                    action_name=action_name,
                )
                return
        else:
            hint = (
                f"task {task.task_id if task else '?'} terminated and could not "
                f"allocate a fresh idempotency_key after 5 retries"
            )
            await self._record_policy_denied(
                source, intent,
                PolicyDenied(
                    f"delegate{{action_name={action_name!r}}} duplicate "
                    f"idempotency_key exhausted retries for {raw_key!r}",
                    rule="duplicate_idempotency_key",
                    hint=hint,
                ),
                action_name=action_name,
            )
            return
        self.shared_state.reset_policy_denial_streak(action_name)
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    async def _handle_specialist_done(
        self, source: str, intent: Intent,
    ) -> None:
        """Handle a ``specialist_done`` intent (source ``specialist:<task_id>`` per Inv-5.3 / R3); bookkeeping in _record_specialist_result.

        Args:
            source: The emitting agent, expected as ``specialist:<task_id>``.
            intent: The SPECIALIST_DONE intent carrying the done payload.
        """
        payload = dict(intent.payload or {})
        task_id = self._task_id_from_specialist_source(source)
        task: Task | None = None
        if task_id:
            try:
                task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — TaskNotFound and friends
                task = None
        if task is None:
            # PolicyGate R3 should have caught this; log defensively but don't crash.
            log.warning(
                "specialist_done from source=%r references unknown "
                "task_id=%r; skipping bookkeeping (R3 should have "
                "denied; defense in depth)",
                source, task_id,
            )
            return
        await self._record_specialist_result(
            task=task, done_payload=payload, source=source,
        )

    async def _handle_request(self, source: str, intent: Intent) -> None:
        """Route a REQUEST intent to its target agent (Plan A: → kernel).

        Applies the kernel-request execution-order gate, records the request on
        the bus for the target reactor / replay, and auto-rejects requests whose
        target agent is not in the role registry (e.g. ``--no-kernel``) so the
        requester never hangs.

        Args:
            source (str): The agent issuing the request.
            intent (Intent): The REQUEST intent; ``payload`` carries
                ``target_agent`` and ``kind``.
        """
        from .coordinator import _lifecycle_paths
        target_agent = intent.payload["target_agent"]
        kind = intent.payload["kind"]
        denied = self._sequence_denial_for_request(target_agent, kind)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        # Always record the request on the bus for the kernel reactor / replay.
        request_msg = Message.new(
            source, target_agent, "request", dict(intent.payload), priority=1,
        )
        await self.bus.append_and_seq(request_msg)

        # Safety net: auto-reject when the target agent was removed (e.g. --no-kernel).
        if target_agent not in self.role_registry:
            await self.bus.append_and_seq(Message.new(
                target_agent, source, "response",
                {
                    "in_reply_to": request_msg.msg_id,
                    "kind": f"{kind}_done",
                    "status": "failed",
                    "result": {
                        "status": "failed",
                        "error_class": "agent_disabled",
                        "error": f"{target_agent} agent is disabled for this session",
                    },
                    "source": "coordinator_auto_reject",
                },
                in_reply_to=request_msg.msg_id, priority=1,
            ))
            return

        # Programmatic shortcut: run a registered kernel handler inline + emit RESPONSE without burning an LLM turn.
        if target_agent == "kernel_agent":
            handler = get_handler(kind)
            if handler is not None:
                params = intent.payload.get("params") or {}
                merged_payload = {**intent.payload, **params}
                # Force batch dispatch for run_optimization: inject candidates_path from last_trace_analyze. LLM value wins.
                if (
                    kind == "run_optimization"
                    and self.shared_state.last_trace_analyze
                    and not merged_payload.get("candidates_path")
                ):
                    cached_candidates_path = self.shared_state.last_trace_analyze.get(
                        "candidates_path"
                    )
                    if cached_candidates_path:
                        merged_payload["candidates_path"] = cached_candidates_path
                # Roofline data is read from the last_trace_analyze cache rather than auto-injected here.
                cache_hit_source = None
                cached_result = self._cached_kernel_request(kind, merged_payload)
                if cached_result is not None:
                    result = cached_result
                    cache_hit_source = "shared_state_cache"
                    # A cache hit never runs the handler; emit a single END
                    # (detail=cache_hit) so the lifecycle log records the step.
                    self._emit_lifecycle(
                        step=kind,
                        status="END",
                        artifacts=_lifecycle_paths(result),
                        detail="cache_hit",
                    )
                else:
                    rejected = (
                        self.shared_state.find_rejected_kernel_patch(merged_payload)
                        if kind == "integrate"
                        else None
                    )
                    if rejected is not None:
                        result = {
                            "status": "skipped",
                            "decision": "REVERT",
                            "error_class": "kernel_patch_rejected",
                            "error": "same kernel patch already exhausted E2E attempts",
                            "kernel_id": rejected.get("kernel_id"),
                            "patch_path": rejected.get("patch_path"),
                            "target_file": rejected.get("target_file"),
                            "extra_server_args": rejected.get("extra_server_args", ""),
                            "attempt_count": rejected.get("attempt_count"),
                            "best_gain_pct": rejected.get("best_gain_pct"),
                            "reason": rejected.get("reason"),
                        }
                        cache_hit_source = "shared_state_kernel_rejection"
                        # A short-circuited integrate never runs the handler;
                        # emit a lone END recording the rejection.
                        self._emit_lifecycle(
                            step=kind,
                            status="END",
                            artifacts=_lifecycle_paths(result),
                            detail="rejected",
                        )
                    else:
                        # Inject base_tput from current_best.tput when an integrate request omits it; operator value wins.
                        if (
                            kind == "integrate"
                            and not merged_payload.get("base_tput")
                        ):
                            cb_tput = (
                                self.shared_state.current_best or {}
                            ).get("tput")
                            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                                merged_payload["base_tput"] = float(cb_tput)

                        # Streaming-record callback for run_optimization batch: each sub-attempt writes immediately.
                        handler_kwargs: dict[str, Any] = {
                            "session_dir": self.session_dir,
                        }
                        if kind == "run_optimization":
                            handler_kwargs["record_partial"] = (
                                self._record_kernel_opt_partial
                            )
                        # Bracket the programmatic kernel step with START / END
                        # lifecycle events. ``kind`` is the machine step name;
                        # the human label is resolved from LIFECYCLE_STEP_LABELS.
                        _lc_t0 = time.monotonic()
                        self._emit_lifecycle(
                            step=kind,
                            status="START",
                            artifacts=_lifecycle_paths(merged_payload),
                        )
                        try:
                            result = await handler(
                                merged_payload,
                                **handler_kwargs,
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.exception(
                                "kernel_request_handler[%s] crashed for source=%s",
                                kind, source,
                            )
                            result = {
                                "status": "failed",
                                "error_class": "handler_exception",
                                "error": repr(exc),
                            }
                        _lc_status = (
                            "ERROR"
                            if str(result.get("status", "")).lower()
                            in ("failed", "error")
                            else "END"
                        )
                        _lc_detail = " ".join(
                            str(p) for p in (
                                result.get("decision"),
                                result.get("status"),
                                f"kernel={result.get('kernel_id')}"
                                if result.get("kernel_id") else "",
                            ) if p
                        )
                        self._emit_lifecycle(
                            step=kind,
                            status=_lc_status,
                            artifacts=_lifecycle_paths(result),
                            detail=_lc_detail,
                            duration_s=time.monotonic() - _lc_t0,
                        )
                await self.bus.append_and_seq(Message.new(
                    "kernel_agent", source, "response",
                    {
                        "in_reply_to": request_msg.msg_id,
                        "kind": f"{kind}_done",
                        "status": result.get("status", "ok"),
                        "result": result,
                        "source": cache_hit_source or "programmatic_handler",
                    },
                    in_reply_to=request_msg.msg_id, priority=1,
                ))
                # Cache trace_analyze output (successful runs only).
                if (
                    kind == "trace_analyze"
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_trace_analyze(merged_payload, result)
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState.
                if kind == "run_optimization":
                    # Batch mode already streamed each sub-result; re-recording would double-count.
                    if not bool(
                        isinstance(result, dict) and result.get("batch_mode")
                    ):
                        self.shared_state.record_kernel_opt(result)
                    self.shared_state.save(self.session_dir)
                    # Auto-enqueue integrate for KEEP'd kernels not yet integrated.
                    await self._auto_enqueue_pending_integrations()
                if kind == "run_gemm_tuning":
                    await self._handle_gemm_tuning_result(result)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    decision = str(result.get("decision", "")).upper()
                    if decision == "KEEP":
                        if isinstance(result, dict) and not result.get(
                            "gap_canonical_id"
                        ):
                            payload_gap = str(
                                merged_payload.get("gap_canonical_id") or ""
                            ).strip()
                            if payload_gap:
                                result["gap_canonical_id"] = payload_gap
                        await self._record_integrate_keep(result)
                    self.shared_state.save(self.session_dir)
                # Advance the kernel cursor past this request seq.
                await self.cursors.advance(
                    target_agent,
                    seq=request_msg.seq,
                    msg_id=request_msg.msg_id,
                )

    async def _handle_response(self, source: str, intent: Intent) -> None:
        """Route a RESPONSE intent back to the original requester.

        Looks up the request message referenced by ``in_reply_to`` to address
        the response, then publishes it on the bus.

        Args:
            source (str): The agent emitting the response.
            intent (Intent): The RESPONSE intent; ``payload`` carries
                ``in_reply_to``.
        """
        in_reply_to = intent.payload["in_reply_to"]
        # Locate the original requester so we can address the response.
        original = await self.bus.lookup_by_id(in_reply_to)
        target = original.from_agent if original else "*"
        await self.bus.append_and_seq(Message.new(
            source, target, "response",
            dict(intent.payload), in_reply_to=in_reply_to, priority=1,
        ))

    async def _handle_kill_task(self, source: str, intent: Intent) -> None:
        """Cancel a queued/running task in response to a kill_task intent.

        Records an observation for unknown task ids, transitions a
        queued/running task to ``cancelled``, and broadcasts a ``kill`` event.

        Args:
            source (str): The agent (typically robustness) issuing the kill.
            intent (Intent): The KILL_TASK intent; ``payload`` carries
                ``task_id`` and optional ``reason``.
        """
        task_id = intent.payload["task_id"]
        try:
            task = await self.tasks.get(task_id)
        except Exception:  # noqa: BLE001 — TaskNotFound
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "kill_task_unknown", "task_id": task_id, "source": source},
            )
            return
        if task.state in ("queued", "running"):
            await self.tasks.transition(
                task_id, "cancelled",
                evidence={"reason": intent.payload.get("reason"), "by": source},
            )
        await self.bus.append_and_seq(Message.new(
            source, "*", "kill",
            {"task_id": task_id, "reason": intent.payload.get("reason")},
        ))

    async def _handle_prune_branch(self, source: str, intent: Intent) -> None:
        """Prune an action family and cancel its in-flight tasks.

        Adds the family to the persistent pruned set, cancels any tasks in that
        family, and broadcasts a ``prune_branch`` event.

        Args:
            source (str): The agent issuing the prune.
            intent (Intent): The PRUNE_BRANCH intent; ``payload`` carries
                ``family`` and optional ``reason``.
        """
        family = intent.payload["family"]
        if self.shared_state.add_pruned_family(family):
            self.shared_state.save(self.session_dir)
        cancelled = await self.tasks.cancel_family([family])
        await self.bus.append_and_seq(Message.new(
            source, "*", "event",
            {"kind": "prune_branch", "family": family,
             "cancelled_task_ids": cancelled,
             "reason": intent.payload.get("reason")},
        ))

    async def _handle_escalate_strategy_change(self, source: str, intent: Intent) -> None:
        """Process ``escalate_strategy_change``: broadcast strategy_change, act on closed-vocab hints, drop unknown hints.

        Args:
            source: The agent issuing the escalation.
            intent: The ESCALATE_STRATEGY_CHANGE intent; ``payload`` may carry a
                closed-vocab ``next_action_hint``.
        """
        payload = dict(intent.payload or {})
        # Always emit the broadcast first.
        await self.bus.append_and_seq(Message.new(
            source, "*", "strategy_change",
            payload, priority=0,
        ))
        from ..phases.machine_state import (
            ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
            ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
            ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX,
            ESCALATE_HINT_SKIP_TO_CLOSE,
            PHASE_EXPLORE,
            PHASE_KERNEL_AGENT,
            apply_escalate_budget_bump,
            is_pause_specialist_hint,
            is_valid_escalate_hint,
        )
        hint = str(payload.get("next_action_hint") or "").strip()
        if not hint or not is_valid_escalate_hint(hint):
            return
        # Pre-enablement close guard: drop a premature ``skip_to_close`` while
        # the model is not yet runnable and let the enablement loop continue.
        if hint == ESCALATE_HINT_SKIP_TO_CLOSE and self.shared_state.enablement_close_guard_active():
            log.info(
                "escalate_strategy_change: dropping premature skip_to_close from %s "
                "(pre-enablement: baseline not established; enablement loop still active)",
                source,
            )
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "observation",
                {
                    "kind": "enablement_skip_to_close_suppressed",
                    "source": source,
                    "phase": (self.shared_state.phase or ""),
                },
            ))
            return
        # extend_*_budget mutates phase_budget_pct directly.
        now_ts = datetime.now(timezone.utc).isoformat()
        if hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct, phase=PHASE_EXPLORE,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        if hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct, phase=PHASE_KERNEL_AGENT,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # pause_specialist_<domain>: bump the per-domain empty-streak.
        if is_pause_specialist_hint(hint):
            domain = hint[len(ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX):]
            self.shared_state.bump_specialist_domain_empty_streak(
                domain, empty=True,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # skip_to_kernel / skip_to_close are deferred; next compute_next_phase picks them up.
        self.shared_state.set_pending_escalate_hint(hint)
        self.shared_state.save(self.session_dir)

    async def _handle_send_message(self, source: str, intent: Intent) -> None:
        """Publish a free-form message onto the bus.

        Soft-degrades an unknown topic to ``observation`` and
        routes to the requested recipient (defaulting to broadcast).

        Args:
            source (str): The sending agent.
            intent (Intent): The SEND_MESSAGE intent; ``payload`` may carry
                ``topic`` / ``to`` plus arbitrary message fields.
        """
        topic = intent.payload.get("topic", "observation")
        if topic not in __import__("hyperloom.orchestrator.bus.message_bus",
                                    fromlist=["TOPIC_ALLOWLIST"]).TOPIC_ALLOWLIST:
            # Soft-degrade unknown topic.
            topic = "observation"
        to_agent = intent.payload.get("to") or "*"
        await self.bus.append_and_seq(Message.new(
            source, to_agent, topic, {k: v for k, v in intent.payload.items() if k != "to"},
        ))

    async def _handle_alert(self, source: str, intent: Intent) -> None:
        """Broadcast an alert message, prioritized by severity.

        High-severity alerts are published at priority 0; everything else at
        priority 1.

        Args:
            source (str): The alerting agent.
            intent (Intent): The ALERT intent; ``payload`` may carry
                ``severity`` plus alert detail.
        """
        prio = 0 if intent.payload.get("severity") == "high" else 1
        await self.bus.append_and_seq(Message.new(
            source, "*", "alert", dict(intent.payload), priority=prio,
        ))

    async def _handle_update_state(self, source: str, intent: Intent) -> None:
        """Apply agent-requested SharedState changes and report the result.

        Applies the requested changes (core fields disallowed), persists when
        anything changed, and broadcasts an observation listing the applied vs
        rejected keys.

        Args:
            source (str): The agent requesting the state update.
            intent (Intent): The UPDATE_STATE intent; ``payload`` carries a
                ``changes`` dict.
        """
        # Apply to persistent SharedState (PolicyGate enforces core-field writes).
        applied = self.shared_state.apply_changes(
            intent.payload["changes"], allow_core=False,
        )
        if applied:
            self.shared_state.save(self.session_dir)
        await self.bus.append_and_seq(Message.new(
            source, "*", "observation",
            {"kind": "update_state", "changes": applied,
             "rejected": sorted(set(intent.payload["changes"]) - set(applied))},
        ))
