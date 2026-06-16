# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Intent routing collaborator extracted from :class:`Coordinator`.

The ``Coordinator`` God-object historically owned both the orchestration
spine (tick/run/dispatcher) *and* the per-intent handlers. This module hosts
the latter: :meth:`IntentRouter.handle_intent` validates an emitted intent
through ``PolicyGate`` and dispatches to the matching ``_handle_*`` method,
exactly as ``Coordinator._handle_intent`` did before.

Design (transitional collaborator)
----------------------------------
``IntentRouter`` holds a back-reference to its owning ``Coordinator`` and
delegates every attribute it does not define itself to that coordinator via
``__getattr__``. The handler bodies were moved here *verbatim* — they read and
call ``self.shared_state`` / ``self.bus`` / ``self._refresh_gaps(...)`` etc.,
which transparently resolve back onto the coordinator. This is safe because the
extracted handlers perform **no** ``self.<attr> = ...`` rebinding (verified by
AST before extraction): all coordinator state mutation happens through method
calls and mutable-object access, both of which route correctly through the
back-reference. The two-way ``coordinator <-> router`` reference is a known,
documented transitional coupling; later passes narrow the router's surface.

``Coordinator`` keeps thin forwarding shims (``_handle_intent`` etc.) so the
~45 existing tests that call ``coord._handle_intent(...)`` keep working unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from ..protocol.intent import Intent, IntentType, NoIntentEmitted
from .message_bus import Message
from .policy import PolicyDenied
from .task_registry import Task
from .kernel_request_handlers import get_handler

if TYPE_CHECKING:
    from .coordinator import Coordinator

log = __import__("logging").getLogger(__name__)


class IntentRouter:
    """Validates and dispatches agent-emitted intents on behalf of a Coordinator."""

    def __init__(self, coordinator: "Coordinator") -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        # Any attribute not defined on the router resolves onto the owning
        # coordinator. This lets the verbatim-moved handler bodies keep using
        # ``self.shared_state`` / ``self.bus`` / ``self._refresh_gaps`` etc.
        # ``__getattr__`` is only consulted on miss, so the handler methods
        # defined on this class take precedence.
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
            if it == IntentType.PROPOSE_ACTION:
                await self._coord._handle_propose_action(source, intent)
            elif it == IntentType.REVIEW_VERDICT:
                await self._coord._handle_review_verdict(source, intent)
            elif it == IntentType.DELEGATE:
                await self._coord._handle_delegate(source, intent)
            elif it == IntentType.REQUEST:
                await self._coord._handle_request(source, intent)
            elif it == IntentType.RESPONSE:
                await self._coord._handle_response(source, intent)
            elif it == IntentType.KILL_TASK:
                await self._coord._handle_kill_task(source, intent)
            elif it == IntentType.PRUNE_BRANCH:
                await self._coord._handle_prune_branch(source, intent)
            elif it == IntentType.FORCE_DISPATCH:
                await self._coord._handle_force_dispatch(source, intent)
            elif it == IntentType.ESCALATE_STRATEGY_CHANGE:
                await self._coord._handle_escalate_strategy_change(source, intent)
            elif it == IntentType.SEND_MESSAGE:
                await self._coord._handle_send_message(source, intent)
            elif it == IntentType.ALERT:
                await self._coord._handle_alert(source, intent)
            elif it == IntentType.UPDATE_STATE:
                await self._coord._handle_update_state(source, intent)
            elif it == IntentType.SPECIALIST_DONE:
                # Terminal specialist intent (R3 already validated); handler only bookkeeps. Defense-in-depth.
                await self._coord._handle_specialist_done(source, intent)
            else:
                # ASK_QUESTION / ANSWER / UPDATE_PERSONA — record for replay
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
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    },
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to record handle_intent_exception observation")
            return

    async def _handle_propose_action(self, source: str, intent: Intent) -> None:
        """Gate a proposed action and enqueue it for Critic Review.

        Drops proposals for pruned families, applies the pending-roofline and
        execution-order denials, then publishes a ``proposal`` message and
        registers a :class:`PendingProposal` so the Critic gate (§18) can later
        return a verdict.

        Args:
            source (str): The agent proposing the action.
            intent (Intent): The PROPOSE_ACTION intent; ``payload`` carries
                ``action_name`` and optional ``params`` / ``predicted_gain_pct``.
        """
        action_name = intent.payload["action_name"]
        # Pruned families are advisory: proposal still queues, but the inbox carries an advisory note.
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
        denied = self._sequence_denial_for_action(
            action_name,
            proposed_params=intent.payload.get("params"),
        )
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
        # KB hypothesize/verify retired; proposals enter the queue directly, facts written after task lands.
        self.state.pending_proposals[msg.msg_id] = pending

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        """Apply a Critic ``review_verdict`` to its target proposal; legacy verdict_map collapsed (approve > reject > needs_review)."""
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
                else "needs_review"
            )
        await self._coord._handle_single_verdict(
            source=source,
            pending=pending,
            verdict=verdict,
            reasoning=str(intent.payload.get("reasoning") or ""),
        )

    async def _handle_single_verdict(
        self,
        *,
        source: str,
        pending: "PendingProposal",
        verdict: str,
        reasoning: str,
    ) -> None:
        """Legacy v0.6 single-verdict handler (approve materialises proposal as-is); mirrors integrate_patch/specialist verdicts onto specialist_patch_verdicts for PolicyGate."""
        pending.decided = True
        pending.verdict = verdict
        await self.bus.append_and_seq(Message.new(
            source, pending.from_agent, "review_verdict",
            {
                "target_proposal_msg_id": pending.proposal_msg_id,
                "verdict":                verdict,
                "reasoning":              reasoning,
            },
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
            sid_candidate = str(pending.task_id or "").strip()
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
        if verdict == "approve":
            await self._materialize_approved_proposal(pending)

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
        denied = self._sequence_denial_for_action(
            action_name,
            proposed_params=intent.payload.get("params"),
        )
        if denied is not None:
            await self._record_policy_denied(
                source, intent, denied, action_name=action_name,
            )
            return
        # delegate explore runs variants directly (config/env grids are not source patches → no Critic pre-review).
        params = dict(intent.payload.get("params") or {})
        # idempotency_key is top-level per schema; treat a nested params value as a compat alias and strip it.
        nested_idempotency_key = params.pop("idempotency_key", None)
        # Plumb baseline's materialized YAML into grid-style tasks for the workload contract; setdefault lets delegator override.
        if (
            action_name in ("sweep", "explore")
            and self.shared_state.baseline_config_path
        ):
            params.setdefault(
                "config_path", self.shared_state.baseline_config_path
            )
        # Parity with _materialize_approved_proposal: direct delegates need the same operational knobs.
        if action_name == "explore":
            self._inject_explore_runtime_params(params)
            # Inject base_tput tied to current_best (or baseline_tput); else every variant lands FAILED.
            # Defensive getattr: lightweight state doubles in tests may omit current_best.
            cb = getattr(self.shared_state, "current_best", None) or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else getattr(self.shared_state, "baseline_tput", 0.0)
            params.setdefault("base_tput", float(base or 0.0))
        # Wave sugar: a specialist delegate carrying params.tasks=[...] fans
        # out into N standard freeform specialist tasks (scope=freeform,
        # lane=cpu, mode=research defaults), each dispatched through the
        # normal SpecialistRunner + TaskRegistry + lease + reap path. This
        # preserves the low-cost wide-net recon that the retired
        # dynamic_specialist channel provided.
        if action_name == "specialist" and isinstance(
            params.get("tasks"), list,
        ) and params["tasks"]:
            await self._fan_out_specialist_wave(source, intent, params)
            return
        # Specialist pre-dispatch warmup: warm external-knowledge sections via KnowledgePlane (setdefault fills gaps).
        if action_name == "specialist":
            await self._warm_specialist_params(params)
        # Idempotency-key chain: top-level → nested compat alias → content-fingerprint auto-key.
        # Terminal collisions retry with -retry<N> (up to 5); non-terminal collisions → policy_denied.
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
                from .specialist_profile import resolve_specialist_profile
                if resolve_specialist_profile(params).grants_bench_tool:
                    lanes = tuple(dict.fromkeys((*lanes, "benchmark_lane")))
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
        """Handle a ``specialist_done`` intent (source ``specialist:<task_id>`` per Inv-5.3 / R3); bookkeeping in _record_specialist_result."""
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
        # Always record the request on the bus so the kernel reactor (and tests/replay) can see it.
        request_msg = Message.new(
            source, target_agent, "request", dict(intent.payload), priority=1,
        )
        await self.bus.append_and_seq(request_msg)

        # Safety net: auto-reject when the target agent was removed (e.g. --no-kernel) so Orch doesn't hang.
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

        # Programmatic shortcut: run a registered kernel handler inline + emit RESPONSE so a deterministic shell-tool invocation doesn't burn an LLM turn (see kernel_request_handlers.py).
        if target_agent == "kernel":
            handler = get_handler(kind)
            if handler is not None:
                params = intent.payload.get("params") or {}
                merged_payload = {**intent.payload, **params}
                # Force batch dispatch for run_optimization: inject candidates_path from last_trace_analyze (else collapses to single-kernel run). LLM value wins.
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
                # Commit 1cd9f7d's roofline_json auto-inject is omitted here: Roofline-v2 caches under last_trace_analyze instead.
                cache_hit_source = None
                cached_result = self._cached_kernel_request(kind, merged_payload)
                if cached_result is not None:
                    result = cached_result
                    cache_hit_source = "shared_state_cache"
                    # #266: a cache hit produces a response but never runs the
                    # handler, so emit a single END (no paired START). Without
                    # this the lifecycle log would show no record at all for a
                    # cache-served step, leaving an operator unsure whether it
                    # ran. detail=cache_hit marks it as served-from-cache.
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
                        # #266: a short-circuited integrate (patch already
                        # exhausted) also never runs the handler; emit a lone
                        # END so the log records the step was resolved as a
                        # rejection rather than silently missing.
                        self._emit_lifecycle(
                            step=kind,
                            status="END",
                            artifacts=_lifecycle_paths(result),
                            detail="rejected",
                        )
                    else:
                        # Inject base_tput from current_best.tput when an integrate request omits it (else 2nd/3rd multi-KEEP integrate fails base_tput > 0); operator value wins.
                        if (
                            kind == "integrate"
                            and not merged_payload.get("base_tput")
                        ):
                            cb_tput = (
                                self.shared_state.current_best or {}
                            ).get("tput")
                            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                                merged_payload["base_tput"] = float(cb_tput)

                        # Streaming-record callback for run_optimization batch: each sub-attempt writes immediately (else a slow sibling starves a fast KEEP's integrate).
                        handler_kwargs: dict[str, Any] = {
                            "session_dir": self.session_dir,
                        }
                        if kind == "run_optimization":
                            handler_kwargs["record_partial"] = (
                                self._record_kernel_opt_partial
                            )
                        # #266: bracket the programmatic kernel step with
                        # START / END lifecycle events so operators see the
                        # step ran, how long it took, and where its outputs
                        # landed. ``kind`` is the machine step name
                        # (trace_analyze / run_optimization / integrate /
                        # run_gemm_tuning); the human label is resolved by
                        # SharedState from LIFECYCLE_STEP_LABELS.
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
                    "kernel", source, "response",
                    {
                        "in_reply_to": request_msg.msg_id,
                        "kind": f"{kind}_done",
                        "status": result.get("status", "ok"),
                        "result": result,
                        "source": cache_hit_source or "programmatic_handler",
                    },
                    in_reply_to=request_msg.msg_id, priority=1,
                ))
                # Cache trace_analyze output (successful runs only) to short-circuit identical next-tick requests.
                if (
                    kind == "trace_analyze"
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_trace_analyze(merged_payload, result)
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState so Orch sees decision/speedup next tick.
                if kind == "run_optimization":
                    # Batch mode already streamed each sub-result; re-recording would double-count. Cache hits lack batch_mode.
                    if not bool(
                        isinstance(result, dict) and result.get("batch_mode")
                    ):
                        self.shared_state.record_kernel_opt(result)
                    self.shared_state.save(self.session_dir)
                    # Auto-enqueue integrate for KEEP'd kernels that haven't
                    # been integrated yet (IR-3: integration is mandatory).
                    await self._auto_enqueue_pending_integrations()
                if kind == "run_gemm_tuning":
                    self.shared_state.record_gemm_tuning(result)
                    self.shared_state.save(self.session_dir)
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
                # Bug B: advance the kernel cursor past this request seq so the LLM kernel agent doesn't re-answer it next tick.
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

    async def _handle_force_dispatch(self, source: str, intent: Intent) -> None:
        """Handle a ``force_dispatch`` intent by emitting an event.

        Currently a P0-3 stub: it broadcasts a ``force_dispatch`` event;
        real dispatcher reordering arrives in P0-5 with the priority queue.

        Args:
            source: Identifier of the intent's originating agent.
            intent: The ``force_dispatch`` intent carrying ``task_id``.
        """
        # P0-3 stub: emit an event; real dispatcher reordering lands in P0-5 with the priority queue.
        await self.bus.append_and_seq(Message.new(
            source, "*", "event",
            {"kind": "force_dispatch", "task_id": intent.payload["task_id"],
             "reason": intent.payload.get("reason")},
        ))

    async def _handle_escalate_strategy_change(self, source: str, intent: Intent) -> None:
        """Process ``escalate_strategy_change`` (KB_design §3.8 §7.3 + §3.13 M7 §5.3); broadcasts strategy_change, acts on closed-vocab hints, drops unknown (Inv-8.2)."""
        payload = dict(intent.payload or {})
        # Always emit the broadcast first (back-compat with legacy contract tests).
        await self.bus.append_and_seq(Message.new(
            source, "*", "strategy_change",
            payload, priority=0,
        ))
        from .phase_state import (
            ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
            ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
            ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX,
            PHASE_EXPLORE,
            PHASE_KERNEL,
            apply_escalate_budget_bump,
            is_pause_specialist_hint,
            is_valid_escalate_hint,
        )
        hint = str(payload.get("next_action_hint") or "").strip()
        if not hint or not is_valid_escalate_hint(hint):
            return
        # extend_*_budget mutates phase_budget_pct directly (consulted every tick).
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
                self.shared_state.phase_budget_pct, phase=PHASE_KERNEL,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # pause_specialist_<domain>: bump the per-domain empty-streak so the next EXPLORE round skips it.
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

        Soft-degrades an unknown topic to ``observation`` per DESIGN §13.2 and
        routes to the requested recipient (defaulting to broadcast).

        Args:
            source (str): The sending agent.
            intent (Intent): The SEND_MESSAGE intent; ``payload`` may carry
                ``topic`` / ``to`` plus arbitrary message fields.
        """
        topic = intent.payload.get("topic", "observation")
        if topic not in __import__("inference_optimizer.orchestrator.message_bus",
                                    fromlist=["TOPIC_ALLOWLIST"]).TOPIC_ALLOWLIST:
            # Soft-degrade unknown topic per DESIGN §13.2.
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
        # Apply to persistent SharedState (PolicyGate already enforced that
        # the source role can't write CORE_STATE_FIELDS unless allowed).
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
