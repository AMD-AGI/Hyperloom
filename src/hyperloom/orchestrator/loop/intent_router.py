# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Intent routing collaborator for :class:`Coordinator`."""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from typing import Any, NamedTuple

from hyperloom.orchestrator.lever import patch_owner_phase
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom_common.timeutil import format_exc_brief
from hyperloom_common.coerce import to_int, to_mapping, to_unix
from ..actions.executors.integrate._integrate_vocabulary import PATCH_SOURCE_UPSTREAM_PR, resolve_patch_source
from ..bus.message_bus import Message
from ..policy.bringup_round import bringup_round_denial
from ..policy.denial import PolicyDenied
from ..specialists.wall_budget import grant_wall_budget_extension
from ..state.shared_state import inject_stack_base_params
from ..state.analysis_failures import remember_trace_failure
from ..state.experience_citations import merge_citations, normalize_citations, shown_ids
from ..state.task_registry import TERMINAL_STATES, IllegalTransition, TaskNotFound, confirmed_result
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_INTENT,
    record_event,
    trajectory_scope,
)
from ..kernel.request_handlers import KERNEL_REQUEST_HANDLERS
from ..collaborator import CoordinatorCollaborator
from .kernel_request import (
    KernelRequest,
    KernelRequestOutcome,
    remembered_outcome,
    run_kernel_handler,
)
from ._escalate_hint import escalate_hint_denial
from .message_delivery import publish_send_message
from .proposals import record_config_proposal

# ``Coordinator`` is intentionally NOT imported (avoids a module-level import cycle with coordinator.py); it is held
# as a back-reference and the annotation below is a deferred string.

log = __import__("logging").getLogger(__name__)

#: Reply source marking an answer the Coordinator wrote itself, refusing a request before any handler ran.
_AUTO_REJECT = "coordinator_auto_reject"

#: The ``prune_branch`` scope that retires the family for the rest of the run; every other scope only drains its queue.
_PRUNE_SCOPE_FAMILY = "family"


class DelegateRequest(NamedTuple):
    """A delegated action, with the task params being assembled for it."""

    source: str
    intent: Intent
    action_name: str
    params: dict[str, Any]


def _relayed_specialist_citations(state: Any) -> dict[str, list[dict[str, str]]]:
    """This cycle's specialist proposal citations, keyed by the change each proposal asks for."""
    from ..actions.executors._proposal_identity import content_fingerprint

    cycle = to_int(getattr(state, "macro_cycle", 0), default=0)
    by_change: dict[str, list[dict[str, str]]] = {}
    for entry in getattr(state, "specialist_rounds", None) or []:
        if not isinstance(entry, dict) or to_int(entry.get("cycle"), default=0) != cycle:
            continue
        for proposal in entry.get("proposal_set") or []:
            if isinstance(proposal, dict) and proposal.get("experience_citations"):
                change = content_fingerprint(proposal)
                by_change[change] = merge_citations(by_change.get(change, []), proposal["experience_citations"])
    return by_change


def _stamp_kb_exposure(
    router: Any,
    payload: dict[str, Any],
    *,
    source: str,
) -> None:
    """Attach the current orchestration read to proposals emitted by that tick, and the citations it supports.

    A grid row that asks for exactly the change a specialist proposed this cycle also carries that specialist's
    citations, which were checked against the read its own dispatch was shown.
    """
    if source != "orchestration":
        return
    evidence = router._coord.conversation.kb_last_read
    current = evidence is not None and int(getattr(evidence, "tick", -1)) == int(
        getattr(router.shared_state, "tick", 0) or 0
    )
    if current:
        payload["kb_read_id"] = str(getattr(evidence, "read_id", "") or "")
        payload["kb_rendered_refs"] = list(getattr(evidence, "rendered_refs", ()) or ())
    params = payload.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("grid"), list):
        return
    from ..actions.executors._proposal_identity import content_fingerprint

    shown = shown_ids(payload.get("kb_rendered_refs") if current else ())
    relayed = _relayed_specialist_citations(router.shared_state)
    grid: list[Any] = []
    for row in params["grid"]:
        if isinstance(row, dict):
            carried = relayed.get(content_fingerprint(row), [])
            if "experience_citations" in row or carried:
                own = normalize_citations(row.get("experience_citations"), shown)
                row = {**row, "experience_citations": merge_citations(own, carried)}
        grid.append(row)
    payload["params"] = {**params, "grid": grid}


class IntentRouter(CoordinatorCollaborator):
    """Validates and dispatches agent-emitted intents on behalf of a Coordinator."""

    def _resolve_primatune_mandate(self, params: dict[str, Any]) -> PolicyDenied | None:
        """Swap a queued predictor mandate in for the ``primatune_mandate_id`` that cites it.

        Side-effecting: rewrites ``task_description`` and the scope, mode,
        provenance and lever params in place. An id no round carries is refused
        rather than dispatched: the task_description beside it is the
        placeholder the prompt has orchestration write for the mandate's text.
        """
        from ..lever import LEVER_SOURCE_PATCH
        from ..predictor.mandate import find_mandate
        from ..predictor.rows import PROVENANCE

        mandate_id = str(params.get("primatune_mandate_id") or "").strip()
        if not mandate_id:
            return None
        mandate = find_mandate(self.shared_state, mandate_id)
        if not mandate:
            return PolicyDenied(
                f"delegate{{action_name='specialist'}} cites primatune_mandate_id={mandate_id!r}, "
                "which no predictor round carries",
                rule="unknown_primatune_mandate",
                hint=(
                    "Copy the id verbatim from the predictor's source-change mandate line, or dispatch the "
                    "specialist without primatune_mandate_id and with a task_description of its own."
                ),
            )
        params.update(
            task_description=mandate,
            scope="freeform",
            mode="patch",
            provenance=PROVENANCE,
            lever_kind=LEVER_SOURCE_PATCH,
        )
        # A domain would relabel the patch specialist:<domain> on its way to integrate_patch.
        params.pop("domain", None)
        params.pop("tags", None)
        return None

    async def _admit_specialist(self, source: str, intent: Intent, params: dict[str, Any]) -> bool:
        """Resolve a cited predictor mandate and stamp ownership; ``False`` once the dispatch is refused."""
        # Before the ownership stamp: a resolved mandate sets the lever the stamp would otherwise derive.
        refused = self._resolve_primatune_mandate(params)
        if refused is not None:
            await self.record_policy_denied(source, intent, refused, action_name="specialist")
            return False
        # Capture proposal ownership at dispatch.
        self._coord.specialist_dispatch.stamp_owner(params)
        return True

    def _consume_primatune_mandate(self, action_name: str, params: dict[str, Any], task_id: str) -> None:
        """Mark the predictor mandate a specialist dispatch cites as consumed by ``task_id``."""
        if action_name != "specialist":
            return
        from ..predictor.mandate import mark_mandate_consumed

        mark_mandate_consumed(self.shared_state, params.get("primatune_mandate_id"), task_id)

    async def _resolve_deliverable(self, params: dict[str, Any]) -> PolicyDenied | None:
        """Put the deliverable of the specialist an agent's integrate_patch names onto ``params``.

        The Coordinator queues a finished specialist's deliverable itself, so an agent's request for it is a repeat
        or a retry, and is the same task: the deliverable's own params, with whatever else the agent sent beside
        them. Refused for a specialist that has not ended with a result, or that left nothing to integrate.
        """
        if resolve_patch_source(params) == PATCH_SOURCE_UPSTREAM_PR:
            return PolicyDenied(
                "integrate_patch with patch_source='upstream_pr' fetches a diff no specialist wrote",
                rule="integrate_patch_upstream_pr",
                hint=(
                    "The FRAMEWORK_AGENT phase lands the upstream PRs it discovers itself. Name the "
                    "specialist_task_id of a finished specialist to land its deliverable."
                ),
            )
        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return None
        unfinished = PolicyDenied(
            f"integrate_patch: {specialist_task_id!r} is not a specialist that has ended with a result",
            rule="integrate_patch_specialist_unfinished",
            hint=(
                "Wait for the specialist's delegated_result event. The Coordinator queues the integrate_patch "
                "of what it delivers."
            ),
        )
        try:
            specialist = await self.tasks.get(specialist_task_id)
        except TaskNotFound:
            return unfinished
        outcome = confirmed_result(specialist) if specialist.state in TERMINAL_STATES else None
        if outcome is None:
            return unfinished
        deliverable = self._coord.phase_framework.specialist_deliverable(
            specialist, to_mapping(outcome.get("specialist_done"))
        )
        if deliverable is None:
            return PolicyDenied(
                f"integrate_patch: specialist {specialist_task_id!r} left nothing to integrate",
                rule="integrate_patch_no_deliverable",
                hint="Its specialist_done names no patch or artifact that exists in its workspace.",
            )
        refused = self._coord.proposals.deliverable_path_denial(deliverable.params)
        if refused is not None:
            return refused
        params.update(deliverable.params)
        return None

    async def _admit_integrate_patch(self, source: str, intent: Intent, params: dict[str, Any]) -> bool:
        """Resolve the deliverable an agent's integrate_patch lands and its owner; ``False`` once it is refused."""
        refused = await self._resolve_deliverable(params)
        if refused is not None:
            await self.record_policy_denied(source, intent, refused, action_name="integrate_patch")
            return False
        owner = patch_owner_phase(params)
        if not owner:
            await self._reject_missing_patch_owner(source, params)
            return False
        params["source_phase"] = owner
        return True

    async def handle_intent(self, source: str, intent: Intent) -> None:
        """Validate an emitted intent through PolicyGate, then route it under its trajectory event."""
        attributes = {"name": intent.type.value, "source": source}
        action_name = (intent.payload or {}).get("action_name")
        if isinstance(action_name, str) and action_name:
            attributes["action_name"] = action_name
        try:
            self.policy.validate_intent(source, intent)
        except PolicyDenied as denied:
            record_event(EVENT_INTENT, attributes={**attributes, "admitted": False, "denied": str(denied)[:200]})
            await self.record_policy_denied(source, intent, denied)
            return
        intent_span_id = record_event(EVENT_INTENT, attributes={**attributes, "admitted": True})
        with trajectory_scope(parent_span_id=intent_span_id):
            await self._route_intent(source, intent)

    async def _route_intent(self, source: str, intent: Intent) -> None:
        """Run the handler for an admitted intent; a handler failure is recorded, never raised."""
        try:
            handlers: dict[IntentType, Any] = {
                IntentType.DELEGATE: self.handle_delegate,
                IntentType.REQUEST: self._handle_request,
                IntentType.EXTEND_LEASE: self._handle_extend_lease,
                IntentType.PRUNE_BRANCH: self._handle_prune_branch,
                IntentType.ESCALATE_STRATEGY_CHANGE: self._handle_escalate_strategy_change,
                IntentType.SEND_MESSAGE: self._handle_send_message,
                IntentType.ALERT: self._handle_alert,
            }
            await handlers[intent.type](source, intent)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:  # isolation boundary: one routed intent
            log.exception("intent handler for %s raised", source)
            self._coord.record_exception(
                stage="handle_intent",
                agent=source,
                exc=exc,
            )
            try:
                await self.bus.record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "handle_intent_exception",
                        "agent": source,
                        "intent_type": intent.type.value,
                        "error": format_exc_brief(exc, limit=500),
                    },
                )
            except sqlite3.Error:
                log.exception("failed to record handle_intent_exception observation")
            return

    def _is_pruned_action(self, action_name: str) -> bool:
        """Whether ``prune_branch`` retired ``action_name`` itself or its catalogue family."""
        meta = self.action_registry.get(action_name)
        return self.shared_state.is_pruned(action_name) or (
            meta is not None and self.shared_state.is_pruned(meta.family)
        )

    async def _note_pruned_advisory(self, source: str, action_name: str) -> None:
        """Tell the delegator the action sits in a pruned family; the action is still admitted."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
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

    async def _reject_missing_patch_owner(self, source: str, params: dict[str, Any]) -> None:
        """Record that an integrate_patch delegate was rejected because no author phase owns its patch."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "delegate_rejected",
                "reason": "integrate_patch_owner_missing",
                "from_agent": source,
                "action_name": "integrate_patch",
                "specialist_task_id": str(params.get("specialist_task_id") or ""),
            },
        )

    def _seed_grid_params(self, request: DelegateRequest) -> None:
        """Seed an explore delegate with what its executor reads from the session."""
        if request.action_name != "explore":
            return
        # Baseline's materialized YAML, which the delegator may override.
        if self.shared_state.baseline_config_path:
            request.params.setdefault("config_path", self.shared_state.baseline_config_path)
        inject_stack_base_params(request.params, self.shared_state, anchor=True)
        self._coord.proposals.inject_explore_runtime_params(request.params)
        self._coord.proposals.credit_predictor_proposals(request.params)

    async def _prepare_specialist_dispatch(self, request: DelegateRequest) -> bool:
        """Warm a specialist delegate's params through the KnowledgePlane.

        False when it is a patch specialist with no git source tree to write into.
        """
        if request.action_name != "specialist":
            return True
        dispatch = self._coord.specialist_dispatch
        await dispatch.warm_specialist_params(request.params)
        return not dispatch.refuse_patch_without_source(request.params)

    async def _admit_under_free_key(self, request: DelegateRequest, key: str) -> str | None:
        """Admit the delegate under the key its task takes: ``key`` stepped past ended twins that measured nothing.

        None once a live twin, a twin that measured it, or an exhausted retry budget policy-denied the delegate.
        """
        unspent = await self._coord.proposals.unspent_key(request.action_name, key)
        if unspent is not None and unspent.holder is None:
            return unspent.key
        action = f"delegate{{action_name={request.action_name!r}}} duplicate"
        if unspent is None:
            denied = PolicyDenied(
                f"{action} idempotency_key exhausted retries for {key!r}",
                rule="duplicate_idempotency_key",
                hint=f"every retry key of {key!r} names a terminated task; send new content",
            )
        elif unspent.holder.state in TERMINAL_STATES:
            task = unspent.holder
            denied = PolicyDenied(
                f"{action} idempotency_key={task.idempotency_key!r}",
                rule="duplicate_idempotency_key",
                hint=f"task {task.task_id} already measured this; read its delegated_result event.",
            )
        else:
            task = unspent.holder
            denied = PolicyDenied(
                f"{action} idempotency_key={task.idempotency_key!r}",
                rule="duplicate_idempotency_key_running",
                hint=(
                    f"task {task.task_id} is still {task.state!r}; wait for its "
                    f"delegated_result event instead of re-sending the same request."
                ),
            )
        await self.record_policy_denied(request.source, request.intent, denied, action_name=request.action_name)
        return None

    async def _explore_grid_covered(
        self, action_name: str, source: str, intent: Intent, params: dict[str, Any]
    ) -> bool:
        """Refuse an explore grid that adds nothing new, recording the denial; report whether it was refused."""
        if action_name != "explore":
            return False
        from ..actions.executors.explore_prefilter.grid_payload import framework_lever_grid
        from ..actions.executors.explore_prefilter.measured_variants import explore_grid_already_covered

        if framework_lever_grid(self.shared_state):
            # The executor adds its pending rewrite levers to this grid, so the grid is not all there is to measure.
            return False

        active = (*await self.tasks.queued(), *await self.tasks.running())
        covered = explore_grid_already_covered(params, active, self.shared_state.explore_search)
        if covered is None:
            return False
        await self.record_policy_denied(source, intent, covered, action_name="explore")
        return True

    async def handle_delegate(self, source: str, intent: Intent) -> None:
        """Validate and enqueue a delegated action as a TaskRegistry task."""
        action_name = intent.payload["action_name"]
        if self._is_pruned_action(action_name):
            await self._note_pruned_advisory(source, action_name)
        denied = await bringup_round_denial(action_name, self.rounds)
        if denied is None:
            denied = self._coord.dispatcher.admission_denial_for_action(
                action_name, params=intent.payload.get("params")
            )
        if denied is not None:
            await self.record_policy_denied(source, intent, denied, action_name=action_name)
            return
        payload = dict(intent.payload)
        _stamp_kb_exposure(self, payload, source=source)
        params = dict(payload.get("params") or {})
        # The work names its task: a key the agent wrote is not part of the work.
        params.pop("idempotency_key", None)
        request = DelegateRequest(source, intent, action_name, params)
        if action_name == "integrate_patch":
            if not await self._admit_integrate_patch(source, intent, params):
                return
            self._coord.proposals.seed_integrate_patch_params(params)
        if action_name == "specialist" and not await self._admit_specialist(source, intent, params):
            return
        self._seed_grid_params(request)
        # Read before a specialist is warmed with this dispatch's context and before a grid's join id goes in,
        # so an identical re-emit lands on the task that already holds the work.
        key = self._coord.proposals.action_key(action_name, params)
        if await self._explore_grid_covered(action_name, source, intent, params):
            return
        free_key = await self._admit_under_free_key(request, key)
        if free_key is None:
            return
        # Past the duplicate check: the warm-up reads the Experience KB and records the injection it carries.
        if not await self._prepare_specialist_dispatch(request):
            return
        # The grid's attempts join its proposal row through this id.
        proposal_id = uuid.uuid4().hex if action_name == "explore" else ""
        if proposal_id:
            params["proposal_msg_id"] = proposal_id
        task = await self._coord.proposals.create_task(action_name, params, free_key)
        # Every task that takes over an open enablement round comes through the hand-off.
        await self._coord.enablement_lane.handoff_enablement_round(task)
        # Only once a task exists, so a dispatch that bailed out above leaves the mandate on offer.
        self._consume_primatune_mandate(action_name, params, task.task_id)
        record_config_proposal(
            self._coord.phase_framework.timeline(),
            proposal_id,
            action_name,
            {**payload, "params": params},
            outcome="delegated",
        )
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "event",
                {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
            )
        )

    def _record_request_failure(self, *, kind: str, request_msg_id: str, result: dict[str, Any]) -> None:
        """Append a failed kernel request to the log the FAILURE RECOVERY prompt block reads."""
        self.shared_state.record_action_failure(action=kind, task_id=request_msg_id, result=result)
        self.shared_state.save(self.session_dir)

    def _record_trace_analyze_outcome(self, payload: dict[str, Any], result: dict[str, Any]) -> None:
        """Cache a successful analysis of a trace, or remember a failure the same trace reproduces."""
        if result.get("status") in ("ok", "succeeded"):
            self.shared_state.record_trace_analyze(payload, result)
        else:
            remember_trace_failure(self.shared_state, payload, result)

    def _record_trace_analyze_on_timeline(
        self,
        *,
        request_msg: Any,
        source: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        cache_hit: bool,
    ) -> None:
        """Mirror a bus-requested analysis onto the visit that was running.

        An analysis dispatched this way opens no roofline event of its own, so
        without this the snapshot counter it advanced has nothing on the
        timeline to account for it.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event_rows import record_trace_analyze_request
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import TraceAnalyzeRun

        record_trace_analyze_request(
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            run=TraceAnalyzeRun(
                run_id=str(getattr(request_msg, "msg_id", "") or ""),
                status=str(result.get("status") or ""),
                result=result,
                requested_by=source,
                request_msg_id=str(getattr(request_msg, "msg_id", "") or ""),
                trace_input=str(payload.get("trace_path") or payload.get("trace_input") or ""),
                top_k=payload.get("top_k"),
                snapshot=self.shared_state.last_trace_analyze,
                cache_hit=cache_hit,
            ),
        )

    async def _reply(self, request_msg: Message, result: dict[str, Any], *, source: str) -> None:
        """Answer a request on the bus with *result*, logging it for FAILURE RECOVERY when it failed.

        The requester awaits ``<kind>_done``, so the reply kind is read back off the request rather than passed in.
        """
        kind = str(request_msg.payload.get("kind") or "")
        status = str(result.get("status", "ok"))
        await self.bus.append_and_seq(
            Message.new(
                request_msg.to_agent,
                request_msg.from_agent,
                "response",
                {
                    "kind": f"{kind}_done",
                    "status": status,
                    "result": result,
                    "source": source,
                },
            )
        )
        if status.lower() in ("failed", "error"):
            self._record_request_failure(kind=kind, request_msg_id=request_msg.msg_id, result=result)

    async def _handle_request(self, source: str, intent: Intent) -> None:
        """Route a REQUEST intent to its programmatic handler."""
        target_agent = str(intent.payload["target_agent"]).strip()
        kind = intent.payload["kind"]
        denied = self._coord.dispatcher.admission_denial_for_request(target_agent, kind)
        if denied is not None:
            await self.record_policy_denied(source, intent, denied)
            return
        # Always record the request on the bus for replay.
        request_msg = Message.new(
            source,
            target_agent,
            "request",
            dict(intent.payload),
        )
        await self.bus.append_and_seq(request_msg)

        if target_agent != "kernel_agent":
            await self._reply(
                request_msg,
                {
                    "status": "failed",
                    "error_class": "unknown_target_agent",
                    "error": f"no handler registered for target_agent={target_agent!r}",
                },
                source=_AUTO_REJECT,
            )
            return
        if not bool(self.shared_state.kernel_enabled):
            await self._reply(
                request_msg,
                {
                    "status": "failed",
                    "error_class": "agent_disabled",
                    "error": "kernel_agent is disabled for this session (--no-kernel)",
                },
                source=_AUTO_REJECT,
            )
            return
        handler = KERNEL_REQUEST_HANDLERS.get(kind)
        if handler is None:
            await self._reply(
                request_msg,
                {
                    "status": "failed",
                    "error_class": "unknown_kernel_kind",
                    "error": f"no programmatic handler for kind={kind!r}",
                    "valid_kinds": sorted(KERNEL_REQUEST_HANDLERS),
                },
                source=_AUTO_REJECT,
            )
            return
        params = intent.payload.get("params") or {}
        request = KernelRequest(request_msg, kind, {**intent.payload, **params})
        outcome = remembered_outcome(self, request)
        if outcome is None:
            outcome = await run_kernel_handler(self, request, handler)
            if outcome is None:
                return
        await self._reply(request_msg, outcome.result, source=outcome.reply_source)
        await self._record_kernel_outcome(request, outcome, requested_by=source)

    async def _record_kernel_outcome(
        self, request: KernelRequest, outcome: KernelRequestOutcome, *, requested_by: str
    ) -> None:
        """Fold what a kernel request produced back into shared state and onto the trajectory timeline."""
        result = outcome.result
        if request.kind == "trace_analyze":
            if not outcome.from_cache:
                self._record_trace_analyze_outcome(request.payload, result)
                self.shared_state.save(self.session_dir)
            self._record_trace_analyze_on_timeline(
                request_msg=request.message,
                source=requested_by,
                payload=request.payload,
                result=result,
                cache_hit=outcome.from_cache,
            )
        if request.kind != "integrate":
            return
        if result.get("status") != "skipped":
            self.shared_state.record_kernel_integrate_result(result)
        if str(result.get("decision", "")).upper() == "KEEP":
            if isinstance(result, dict) and not result.get("gap_canonical_id"):
                payload_gap = str(request.payload.get("gap_canonical_id") or "").strip()
                if payload_gap:
                    result["gap_canonical_id"] = payload_gap
            await self._coord.writeback.record_integrate_keep(result)
        self.shared_state.save(self.session_dir)

    async def _handle_extend_lease(self, source: str, intent: Intent) -> None:
        """Grant a running task more lease time."""
        task_id = str(intent.payload.get("task_id") or "").strip()
        extra_sec = int(intent.payload.get("extra_sec") or 0)
        try:
            new_ttl = await self.tasks.extend_lease(task_id, extra_sec)
        except (TaskNotFound, IllegalTransition) as exc:
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "extend_lease_rejected",
                    "task_id": task_id,
                    "source": source,
                    "error": repr(exc)[:200],
                },
            )
            return
        # Remaining budget = cumulative TTL minus the time already spent running.
        running_sec = 0.0
        try:
            started = to_unix((await self.tasks.get(task_id)).updated_at, 0.0)
        except TaskNotFound:
            started = 0.0
        if started > 0:
            running_sec = max(0.0, time.time() - started)
        # A late extension can arrive after the cumulative task TTL expired but before the worker/reaper acted on it.
        remaining_sec = max(1, int(extra_sec), int(new_ttl - running_sec))
        lanes = await self.locks.heartbeat_by_task(task_id, ttl_sec=remaining_sec)
        # Move the live subprocess's kill instant too, within the room its dispatch left before the phase end.
        granted_sec = grant_wall_budget_extension(task_id, extra_sec)
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "extend_lease",
                "task_id": task_id,
                "source": source,
                "extra_sec": extra_sec,
                "lease_ttl_sec": new_ttl,
                "lease_expires_in_sec": remaining_sec,
                "lanes": lanes,
                "wall_budget_extension_sec": granted_sec,
                "reason": str(intent.payload.get("reason") or "")[:200],
            },
        )

    async def _handle_prune_branch(self, source: str, intent: Intent) -> None:
        """Cancel a family's queued tasks and, for ``scope='family'`` (the default), retire it; running tasks continue."""
        family = intent.payload["family"]
        reason = str(intent.payload.get("reason") or "prune_branch")
        scope = str(intent.payload.get("scope") or _PRUNE_SCOPE_FAMILY).strip()
        drain_only = scope != _PRUNE_SCOPE_FAMILY
        if not drain_only and self.shared_state.add_pruned_family(family):
            self.shared_state.save(self.session_dir)
        # A catalogue family names its member actions; any other value names a single action kind.
        kinds = [name for name, meta in self.action_registry.items() if meta.family == family] or [family]
        cancelled: list[str] = []
        if drain_only and "baseline" in kinds:
            kinds.remove("baseline")
            cancelled = await self._coord.writeback.drain_queued_baselines(reason=reason)
        cancelled += await self.tasks.cancel_family(kinds, reason=reason)
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "event",
                {
                    "kind": "prune_branch",
                    "family": family,
                    "scope": scope,
                    "cancelled_task_ids": cancelled,
                    "reason": intent.payload.get("reason"),
                },
            )
        )

    async def _handle_escalate_strategy_change(self, source: str, intent: Intent) -> None:
        """Broadcast strategy_change, then defer a hint the current phase reads; deny the rest to the sender."""
        payload = dict(intent.payload or {})
        # Always emit the broadcast first.
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "strategy_change",
                payload,
            )
        )
        hint = str(payload.get("next_action_hint") or "").strip()
        if not hint:
            return
        denied = escalate_hint_denial(hint, str(self.shared_state.phase or "").strip().upper())
        if denied is not None:
            await self.record_policy_denied(source, intent, denied)
            return
        # Deferred: the next compute_next_phase acts on the hint.
        self.shared_state.pending_escalate_hint = hint
        self.shared_state.save(self.session_dir)

    async def _handle_send_message(self, source: str, intent: Intent) -> None:
        """Publish a free-form message onto the bus, once per recipient (see :mod:`.message_delivery`)."""
        await publish_send_message(self.bus, self.session_dir, source, intent)

    async def _handle_alert(self, source: str, intent: Intent) -> None:
        """Broadcast an alert message."""
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "alert",
                dict(intent.payload),
            )
        )

    async def record_policy_denied(
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
    ) -> None:
        """Record a PolicyGate denial.

        Logs it, publishes a ``policy_denied`` observation to the denied agent
        and files a denial row on the open phase event.

        Args:
            source (str): The agent whose intent was denied.
            intent (Intent): The denied intent.
            denied (PolicyDenied): The denial carrying rule / hint / reason.
            action_name (str | None): Explicit action name override; falls back
                to ``intent.payload['action_name']``.
        """
        # Surface every PolicyGate denial in the process log (not just the bus)
        # so security rejections are observable in ops logs.
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
            )
        )
        resolved_action = action_name or str((intent.payload or {}).get("action_name") or "")
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseDenial

        phase_event.record_denial(
            PhaseDenial(
                at=PhaseCycle(str(self.shared_state.phase or ""), int(self.shared_state.macro_cycle or 0)),
                actor=source,
                action=resolved_action,
                rule=str(denied.rule or ""),
                hint=str(denied.hint or ""),
            )
        )
