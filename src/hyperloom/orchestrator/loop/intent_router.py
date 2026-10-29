# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Intent routing collaborator for :class:`Coordinator`."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any, NamedTuple

from hyperloom.common.framework_arm import is_upstream_pr_prescreen, review_row_id, verdict_subject
from hyperloom.orchestrator.lever import (
    patch_lever_kind,
    patch_owner_phase,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.common.timeutil import format_exc_brief
from .verdicts import (
    CriticRuling,
    collapse_verdict_map,
    collapse_verdicts,
    serialize_verdict_advisory,
    verdict_held_to_its_rule,
    verdict_map_entry_held_to_its_rule,
)
from hyperloom.common.coerce import to_int, to_unix
from ..bus.message_bus import Message
from ..policy.gate import (
    PolicyDenied,
    PRUNE_BRANCH_SCOPE_FAMILY,
    PRUNE_BRANCH_SCOPE_QUEUED,
)
from ..specialists.wall_budget import grant_wall_budget_extension
from ..state.shared_state import (
    ESCALATE_HINT_SKIP_TO_KERNEL,
    inject_stack_base_params,
    is_valid_escalate_hint,
)
from ..state.analysis_failures import remember_trace_failure
from ..state.experience_citations import merge_citations, normalize_citations, shown_ids
from ..state.task_registry import IllegalTransition, TaskNotFound, TaskSpec
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_INTENT,
    EVENT_PROPOSAL,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    record_event,
    trajectory_scope,
    trajectory_span,
)
from ..kernel.request_handlers import KERNEL_REQUEST_HANDLERS, get_handler
from ..collaborator import CoordinatorCollaborator
from .kernel_request import (
    KernelRequest,
    KernelRequestOutcome,
    remembered_outcome,
    run_kernel_handler,
)
from .message_delivery import publish_send_message

# ``Coordinator`` is intentionally NOT imported (avoids a module-level import cycle with coordinator.py); it is held
# as a back-reference and the annotation below is a deferred string.

log = __import__("logging").getLogger(__name__)

#: Reply source marking an answer the Coordinator wrote itself, refusing a request before any handler ran.
_AUTO_REJECT = "coordinator_auto_reject"


class DelegateRequest(NamedTuple):
    """A delegated action, with the task params being assembled for it."""

    source: str
    intent: Intent
    action_name: str
    params: dict[str, Any]


def _is_specialist_wave(action_name: str, params: Mapping[str, Any]) -> bool:
    """A specialist delegate carrying ``params.tasks`` fans out into one freeform specialist task per entry.

    Each goes out through the normal SpecialistRunner + TaskRegistry + lease + reap path.
    """
    return action_name == "specialist" and isinstance(params.get("tasks"), list) and bool(params["tasks"])


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


def _variant_review_rows(
    payload: Mapping[str, Any] | None,
    held_by_name: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Return one row per named variant, in the order the Critic wrote them.

    A variant the Critic rejected never reaches a bench, so the map is the only
    record that it was judged at all; a collapsed verdict says nothing about
    which variants a grid proceeded on.
    """
    entries = (payload or {}).get("verdict_map")
    if not isinstance(entries, Mapping):
        return []
    held = held_by_name or {}
    rows: list[dict[str, Any]] = []
    for name, entry in entries.items():
        variant = str(name or "")
        if not variant:
            continue
        fields = entry if isinstance(entry, Mapping) else {}
        authored = str(fields.get("verdict") or "")
        effective = str(held.get(variant) or "") or authored
        rows.append(
            {
                "variant_name": variant,
                "verdict": authored,
                "effective_verdict": effective,
                # A hold is visible in the pair, and naming the rule twice
                # would be a second thing to keep in step with the hold itself.
                "held_to_rule": str(fields.get("failure_reason_code") or "") if effective != authored else "",
                "reason": str(fields.get("rationale") or fields.get("reasoning") or ""),
                "failure_reason_code": str(fields.get("failure_reason_code") or ""),
            }
        )
    return rows


def _record_phase_proposal_review(pending: Any, ruling: CriticRuling) -> None:
    """File the Critic's ruling on the proposal row the phase event holds.

    Keyed on the bus message id the verdict named, rather than on the subject
    the framework row is resolved under: the two differ for a proposal
    carrying a candidate id, and this row is about the proposal itself.
    """
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import ProposalReview

        fields = ruling.payload or {}
        phase_event.record_proposal_review(
            ProposalReview(
                proposal_msg_id=proposal_id,
                verdict=ruling.authored or ruling.effective,
                effective_verdict=ruling.effective,
                source=str(fields.get("source") or ""),
                reasoning=str(ruling.reasoning or ""),
                confidence=fields.get("confidence"),
                failure_reason_code=fields.get("failure_reason_code"),
                advisory=ruling.advisory,
                variants=ruling.variant_verdicts,
            )
        )
    except Exception:
        log.debug("phase timeline: proposal review row failed for %s", proposal_id, exc_info=True)


def _record_critic_review(router: Any, pending: Any, ruling: CriticRuling) -> None:
    """Record the Critic's ruling on one proposal, onto the proposal.

    Every action the Critic reviews is recorded, not only the config-arm grid.
    ``authored`` (what the Critic wrote) and ``effective`` (what the loop acted
    on) are both kept: a reject held to a formatting rule is mirrored onto state
    as the reject the Critic wrote. A rejected proposal is settled here, since
    it never reaches a bench and nothing downstream would otherwise resolve it.
    """
    # The framework row below exists only while a framework event is open for
    # this cycle, so the proposal's own row is what carries a ruling reached in
    # any other phase.
    _record_phase_proposal_review(pending, ruling)

    proposal_id = review_row_id(pending.payload or {}, fallback_msg_id=pending.proposal_msg_id)
    if not proposal_id:
        return
    recorder = router._coord.phase_framework.timeline()
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        REVIEWER_CRITIC,
        REVIEWER_CRITIC_UNAVAILABLE,
        STEP_REVIEWED,
        CriticReview,
    )

    authored = ruling.authored
    effective = ruling.effective
    reason = str(ruling.reasoning or "")
    fields = ruling.payload or {}
    grounded = str(fields.get("source") or REVIEWER_CRITIC) == REVIEWER_CRITIC
    recorder.record_proposal_review(
        proposal_id,
        CriticReview(
            verdict=authored or effective,
            effective_verdict=effective,
            held_to_rule=str(fields.get("failure_reason_code") or "") if effective != authored else "",
            reviewer=REVIEWER_CRITIC if grounded else REVIEWER_CRITIC_UNAVAILABLE,
            reason=reason,
            confidence=fields.get("confidence"),
            failure_reason_code=str(fields.get("failure_reason_code") or ""),
            advisory=ruling.advisory,
            variants=ruling.variant_verdicts,
        ),
    )
    recorder.record_proposal_step(
        proposal_id,
        step=STEP_REVIEWED,
        outcome=effective,
        reason=reason,
    )
    if effective == "reject":
        router.settle_dropped_proposal(pending, reason or "critic_rejected")


def _record_phase_proposal_outcome(pending: Any, **outcome: Any) -> None:
    """Record on the proposal row what the loop did with its ruling.

    ``outcome`` carries ``materialized`` / ``denied`` / ``reauthored`` /
    ``patch_verdict_key``, as the framework row names them.
    """
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import ProposalOutcome

        phase_event.record_proposal_outcome(
            ProposalOutcome(
                proposal_msg_id=proposal_id,
                materialized=bool(outcome.get("materialized")),
                denied=bool(outcome.get("denied")),
                reauthored=bool(outcome.get("reauthored")),
                patch_verdict_key=outcome.get("patch_verdict_key"),
            )
        )
    except Exception:
        log.debug("phase timeline: proposal outcome row failed for %s", proposal_id, exc_info=True)


def _record_review_outcome(router: Any, pending: Any, **outcome: Any) -> None:
    """Record what the loop did with a ruling, onto the ruling."""
    _record_phase_proposal_outcome(pending, **outcome)

    proposal_id = review_row_id(pending.payload or {}, fallback_msg_id=pending.proposal_msg_id)
    if not proposal_id:
        return
    recorder = router._coord.phase_framework.timeline()
    if recorder is None:
        return
    recorder.record_proposal_review_outcome(proposal_id, **outcome)


class IntentRouter(CoordinatorCollaborator):
    """Validates and dispatches agent-emitted intents on behalf of a Coordinator."""

    def settle_dropped_proposal(self, pending: Any, reason: str) -> None:
        """Settle a denied proposal as dropped on the framework timeline: it never reaches a bench."""
        proposal_id = review_row_id(pending.payload or {}, fallback_msg_id=pending.proposal_msg_id)
        recorder = self._coord.phase_framework.timeline()
        if not proposal_id or recorder is None:
            return
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import DISPOSITION_DROPPED

        recorder.settle_proposal(proposal_id, disposition=DISPOSITION_DROPPED, reason=reason)

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
        self._stamp_specialist_owner(params)
        return True

    def _consume_primatune_mandate(self, action_name: str, params: dict[str, Any], task_id: str) -> None:
        """Mark the predictor mandate a specialist dispatch cites as consumed by ``task_id``."""
        if action_name != "specialist":
            return
        from ..predictor.mandate import mark_mandate_consumed

        mark_mandate_consumed(self.shared_state, params.get("primatune_mandate_id"), task_id)

    def _stamp_specialist_owner(self, params: dict[str, Any]) -> str:
        """Freeze patch ownership when a specialist task is created."""
        lever = patch_lever_kind(params)
        if lever:
            params["lever_kind"] = lever
        owner = patch_owner_phase(params)
        if not owner:
            from ..specialists.profile import MODE_PATCH, resolve_specialist_profile

            gap_layer = str(params.get("gap_layer") or "").strip().lower()
            active_phase = str(self.shared_state.phase or "").strip().upper()
            # Layer first, phase last: both lanes share one phase, so the live phase does not say which lever a
            # specialist moves.
            if gap_layer == "framework":
                owner = "FRAMEWORK_AGENT"
            elif gap_layer in {"explore", "perf_explore"} or params.get("domain"):
                owner = "EXPLORE"
            # A patch-mode dispatch with no layer names no phase of its own, and integrate_patch is
            # booked to the framework agent, so that is the owner its patch would be attributed to.
            elif resolve_specialist_profile(params).mode == MODE_PATCH or active_phase in {
                "FRAMEWORK",
                "FRAMEWORK_AGENT",
            }:
                owner = "FRAMEWORK_AGENT"
        if owner:
            params["source_phase"] = owner
        return owner

    async def _stamp_integrate_patch_owner(
        self,
        params: dict[str, Any],
    ) -> str:
        """Copy immutable author ownership from the originating specialist."""
        owner = patch_owner_phase(params)
        if owner:
            params["source_phase"] = owner
            return owner
        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return ""
        try:
            specialist = await self.tasks.get(specialist_task_id)
        except TaskNotFound:
            return ""
        specialist_params = dict(getattr(specialist, "params", None) or {})
        owner = patch_owner_phase(specialist_params)
        if not owner:
            return ""
        for key in (
            "domain",
            "provenance",
            "gap_canonical_id",
            "gap_layer",
            "framework_agent_authoring",
            "framework_agent_candidate_id",
            # The lever the specialist was dispatched against; the patch that lands is the same lever, so it is copied
            # rather than re-derived.
            "lever_kind",
        ):
            value = specialist_params.get(key)
            if value not in (None, "", [], {}):
                params.setdefault(key, value)
        params["source_phase"] = owner
        return owner

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
            it = intent.type
            handlers: dict[IntentType, Any] = {
                IntentType.PROPOSE_ACTION: self._handle_propose_action,
                IntentType.REVIEW_VERDICT: self._handle_review_verdict,
                IntentType.DELEGATE: self.handle_delegate,
                IntentType.REQUEST: self._handle_request,
                IntentType.EXTEND_LEASE: self._handle_extend_lease,
                IntentType.PRUNE_BRANCH: self._handle_prune_branch,
                IntentType.ESCALATE_STRATEGY_CHANGE: self._handle_escalate_strategy_change,
                IntentType.SEND_MESSAGE: self._handle_send_message,
                IntentType.ALERT: self._handle_alert,
            }
            handler = handlers.get(it)
            if handler is not None:
                await handler(source, intent)
            else:
                # Unknown / unhandled intent — record for replay.
                await self.bus.record_observation(
                    source,
                    "observation",
                    {"intent": it.value, "payload": intent.payload},
                )
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
            except Exception:
                log.exception("failed to record handle_intent_exception observation")
            return

    def _is_pruned_action(self, action_name: str) -> bool:
        """Whether ``prune_branch`` retired ``action_name`` itself or its catalogue family."""
        meta = self.action_registry.get(action_name)
        return self.shared_state.is_pruned(action_name) or (
            meta is not None and self.shared_state.is_pruned(meta.family)
        )

    async def _note_pruned_advisory(self, source: str, action_name: str, kind: str) -> None:
        """Tell the proposer the action sits in a pruned family; the action is still admitted."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": kind,
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

    async def _reject_missing_patch_owner(
        self, source: str, action_name: str, params: dict[str, Any], kind: str
    ) -> None:
        """Record that an integrate_patch intent was rejected because no author phase owns its patch."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": kind,
                "reason": "integrate_patch_owner_missing",
                "from_agent": source,
                "action_name": action_name,
                "specialist_task_id": str(params.get("specialist_task_id") or ""),
            },
        )

    async def _handle_propose_action(self, source: str, intent: Intent) -> None:
        """Gate a proposed action and enqueue it for Critic Review."""
        action_name = intent.payload["action_name"]
        # Pruned families are advisory: proposal still queues with an advisory note.
        if self._is_pruned_action(action_name):
            await self._note_pruned_advisory(source, action_name, "proposal_pruned_advisory")
        denied = self._coord.dispatcher.admission_denial_for_action(action_name, params=intent.payload.get("params"))
        if denied is not None:
            await self.record_policy_denied(source, intent, denied)
            return
        payload = dict(intent.payload)
        _stamp_kb_exposure(self, payload, source=source)
        if action_name == "integrate_patch":
            params = dict(payload.get("params") or {})
            if not await self._stamp_integrate_patch_owner(params):
                await self._reject_missing_patch_owner(source, action_name, params, "proposal_rejected")
                return
            payload["params"] = params
        from .proposals import record_proposal

        await record_proposal(
            self._coord,
            from_agent=source,
            action_name=action_name,
            predicted_gain_pct=float(intent.payload.get("predicted_gain_pct", 0.0)),
            payload=payload,
        )

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        """Apply a Critic ``review_verdict`` to its target proposal."""
        target = intent.payload["target_proposal_msg_id"]
        pending = self.state.pending_proposals.get(target)
        verdict_map = intent.payload.get("verdict_map")
        single_verdict = intent.payload.get("verdict")
        if pending is None:
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "verdict_for_unknown_proposal",
                    "target": target,
                    "verdict": single_verdict or "",
                    "verdict_map": bool(verdict_map),
                },
            )
            return
        verdict = await self._record_verdict_hold(
            verdict_held_to_its_rule(intent.payload, action_name=pending.action_name),
            target=target,
        )
        authored = str(single_verdict or "").strip()
        approved_variant_names: set[str] | None = None
        held_by_name: dict[str, str] = {}
        if not verdict and isinstance(verdict_map, dict) and verdict_map:
            held_by_name = await self._held_verdict_map(
                verdict_map,
                target=target,
                action_name=pending.action_name,
                payload=intent.payload,
            )
            verdict, approved_variant_names = collapse_verdict_map(held_by_name)
            authored = collapse_verdicts(
                str((entry or {}).get("verdict") or "").strip() for entry in verdict_map.values()
            )
            self._log_mixed_verdict_map_collapse(target, verdict, held_by_name)
        await self._handle_single_verdict(
            source=source,
            pending=pending,
            ruling=CriticRuling(
                verdict=verdict,
                authored_verdict=authored,
                reasoning=str(intent.payload.get("reasoning") or ""),
                advisory=serialize_verdict_advisory(intent.payload),
                approved_variant_names=approved_variant_names,
                payload=intent.payload,
                variant_verdicts=_variant_review_rows(intent.payload, held_by_name),
            ),
        )

    async def _held_verdict_map(
        self,
        verdict_map: dict[str, Any],
        *,
        target: str,
        action_name: str,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        """Hold each ``verdict_map`` entry to its cited rule."""
        held_by_name: dict[str, str] = {}
        for name, entry in verdict_map.items():
            held_by_name[str(name)] = await self._record_verdict_hold(
                verdict_map_entry_held_to_its_rule(entry, payload, action_name=action_name),
                target=target,
                variant=str(name),
            )
        return held_by_name

    def _log_mixed_verdict_map_collapse(
        self,
        target: str,
        verdict: str,
        held_by_name: dict[str, str],
    ) -> None:
        """Log when a mixed map still proceeds, for traceability."""
        sub_verdicts = list(held_by_name.values())
        if verdict in ("approve", "advise") and any(sv in ("reject", "needs_review") for sv in sub_verdicts):
            log.warning(
                "review_verdict collapse: target=%s collapsed to %r (sub_verdicts=%r)",
                target,
                verdict,
                sub_verdicts,
            )

    async def _record_verdict_hold(
        self,
        held: tuple[str, str],
        *,
        target: str,
        variant: str = "",
    ) -> str:
        """Return the verdict to act on, recording any hold to a cited rule."""
        verdict, downgraded_from_code = held
        if not downgraded_from_code:
            return verdict
        log.warning(
            "review_verdict held to its rule: target=%s variant=%s reject -> %s (reason_code=%s)",
            target,
            variant or "-",
            verdict,
            downgraded_from_code,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "verdict_downgraded_to_rule_verdict",
                "target": target,
                "variant": variant,
                "from_verdict": "reject",
                "to_verdict": verdict,
                "failure_reason_code": downgraded_from_code,
            },
        )
        return verdict

    async def _handle_single_verdict(
        self,
        *,
        source: str,
        pending: Any,
        ruling: CriticRuling,
    ) -> None:
        """Apply one collapsed verdict: approve/advise materialise, reject may rearm.

        ``ruling.verdict`` is one of approve / advise / reject / needs_review.
        ``ruling.authored`` is what the Critic itself wrote before any hold to
        a cited rule; it is mirrored onto ``specialist_patch_verdicts`` in place
        of ``verdict``. ``ruling.approved_variant_names`` restricts an explore
        grid to the named variants; ``None`` keeps the full proposal.
        """
        verdict = ruling.verdict
        self.state.pending_proposals.pop(pending.proposal_msg_id, None)
        if is_upstream_pr_prescreen(pending.action_name, pending.payload):
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "framework_agent_verdict_received",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "candidate_id": str((pending.payload or {}).get("framework_agent_candidate_id") or ""),
                    "verdict": verdict,
                },
            )
        rebroadcast_payload: dict[str, Any] = {
            "target_proposal_msg_id": pending.proposal_msg_id,
            "verdict": verdict,
            "reasoning": ruling.reasoning,
        }
        if ruling.advisory:
            rebroadcast_payload.update(ruling.advisory)
        await self.bus.append_and_seq(
            Message.new(
                source,
                pending.from_agent,
                "review_verdict",
                rebroadcast_payload,
                in_reply_to=pending.proposal_msg_id,
            )
        )
        # Mirror specialist / integrate_patch verdicts onto SharedState so PolicyGate's integrate_patch gate can
        # consult them on the next tick.
        patch_verdict = ruling.authored
        try:
            pa_params = pending.payload.get("params") or {}
        except AttributeError:
            pa_params = {}
        sid_candidate = ""
        if pending.action_name == "integrate_patch":
            # A pre-screen carries its candidate id at the top level, not in params.
            sid_candidate = verdict_subject({**pa_params, **(pending.payload or {})})
        elif pending.action_name == "specialist":
            # Critic verdict on the specialist proposal counts as the verdict on its patches; task_id is the key.
            sid_candidate = str(pa_params.get("task_id") or "").strip()
        if sid_candidate and patch_verdict:
            try:
                self.shared_state.record_specialist_patch_verdict(
                    sid_candidate,
                    patch_verdict,
                )
                self.shared_state.save(self.session_dir)
            except OSError:
                log.exception(
                    "failed to mirror critic verdict for specialist task=%s",
                    sid_candidate,
                )
        _record_critic_review(self, pending, ruling)
        _record_review_outcome(
            self,
            pending,
            materialized=verdict in ("approve", "advise"),
            denied=verdict == "reject",
            reauthored=verdict == "needs_review",
            patch_verdict_key=sid_candidate if patch_verdict else "",
        )
        with trajectory_span(
            EVENT_PROPOSAL,
            span_id=pending.proposal_msg_id,
            attributes={"name": pending.action_name, "verdict": verdict},
        ) as proposal_span:
            await self._apply_verdict_outcome(
                pending,
                ruling=ruling,
                pa_params=pa_params,
                sid_candidate=sid_candidate,
            )
            proposal_span.finish(
                STATUS_COMPLETED if verdict in ("approve", "advise") else STATUS_CANCELLED,
                task_id=pending.task_id,
            )

    async def _apply_verdict_outcome(
        self,
        pending: Any,
        *,
        ruling: CriticRuling,
        pa_params: Mapping[str, Any],
        sid_candidate: str,
    ) -> None:
        """Materialise, deny, or send back a proposal according to its collapsed verdict."""
        from ..phases.geak_native_revalidation import record_review

        verdict = ruling.verdict
        if record_review(self, pending, verdict=verdict, reasoning=ruling.reasoning):
            return
        if verdict in ("approve", "advise"):
            await self._coord.proposals.materialize_approved_proposal(
                pending,
                approved_variant_names=ruling.approved_variant_names,
            )
        elif verdict == "reject" and is_upstream_pr_prescreen(pending.action_name, pending.payload):
            await self._coord.phase_framework.record_critic_denial(pending, ruling.reasoning)
        elif verdict == "reject" and pending.action_name == "integrate_patch" and bool(pa_params.get("enablement")):
            await self._coord.enablement_lane.maybe_rearm_enablement(
                {"enablement": True, "status": "reverted", "reason": "critic_rejected"}
            )
        elif verdict == "needs_review":
            await self._coord.phase_framework.maybe_reauthor_from_critic_feedback(pending, ruling.advisory)

    def _seed_grid_params(self, request: DelegateRequest) -> None:
        """Seed an explore delegate with the params an approved proposal would have carried."""
        if request.action_name != "explore":
            return
        # Baseline's materialized YAML, which the delegator may override.
        if self.shared_state.baseline_config_path:
            request.params.setdefault("config_path", self.shared_state.baseline_config_path)
        inject_stack_base_params(request.params, self.shared_state, anchor=True)
        self._coord.proposals.inject_explore_runtime_params(request.params)
        self._coord.proposals.credit_predictor_proposals(request.params)

    async def _prepare_specialist_dispatch(self, request: DelegateRequest, nested_idempotency_key: Any) -> bool:
        """Warm a specialist delegate's params through the KnowledgePlane.

        False when it is a patch specialist with no git source tree to write into.
        """
        if request.action_name != "specialist":
            return True
        dispatch = self._coord.specialist_dispatch
        await dispatch.warm_specialist_params(request.params)
        task_id = str(request.intent.payload.get("idempotency_key") or nested_idempotency_key or "")
        return not dispatch.refuse_patch_without_source(request.params, task_id=task_id)

    async def _create_delegated_task(self, request: DelegateRequest, raw_key: Any) -> Any | None:
        """Register the delegate as a task, stepping its idempotency key past terminated twins.

        Returns None once a live twin, or an exhausted retry budget, policy-denied the delegate.
        """
        terminal_states = {
            "succeeded",
            "failed",
            "cancelled",
        }
        task = None
        for attempt in range(6):
            idempotency_key = str(raw_key) if attempt == 0 else f"{raw_key}-retry{attempt}"
            lanes, ttl = self._coord.dispatcher.registry_lanes_ttl(request.action_name)
            if request.action_name == "specialist":
                from ..specialists.profile import requires_gpu, specialist_lanes

                lanes = tuple(specialist_lanes(request.params, list(lanes)))
                if requires_gpu(request.params):
                    ttl = self._coord.dispatcher.gpu_lease_ttl_sec(int(ttl or 0), params=request.params)
            task, was_existing = await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind=request.action_name,
                    params=request.params,
                    idempotency_key=idempotency_key,
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="llm",
                )
            )
            if not was_existing:
                return task
            if task.state not in terminal_states:
                hint = (
                    f"task {task.task_id} is still {task.state!r}; wait for the "
                    f"delegated_result event instead of re-emitting the same key."
                )
                await self.record_policy_denied(
                    request.source,
                    request.intent,
                    PolicyDenied(
                        f"delegate{{action_name={request.action_name!r}}} duplicate "
                        f"idempotency_key={idempotency_key!r}",
                        rule="duplicate_idempotency_key_running",
                        hint=hint,
                    ),
                    action_name=request.action_name,
                )
                return None
        hint = (
            f"task {task.task_id if task else '?'} terminated and could not "
            f"allocate a fresh idempotency_key after 5 retries"
        )
        await self.record_policy_denied(
            request.source,
            request.intent,
            PolicyDenied(
                f"delegate{{action_name={request.action_name!r}}} duplicate "
                f"idempotency_key exhausted retries for {raw_key!r}",
                rule="duplicate_idempotency_key",
                hint=hint,
            ),
            action_name=request.action_name,
        )
        return None

    async def _explore_grid_covered(
        self, action_name: str, source: str, intent: Intent, params: dict[str, Any], idempotency_key: str
    ) -> bool:
        """Refuse an explore grid that adds nothing new, recording the denial; report whether it was refused.

        A task already queued or running under ``idempotency_key`` is not counted against the grid: re-sending the key
        is answered as a duplicate by the enqueue below, not as coverage.
        """
        if action_name != "explore":
            return False
        from ..actions.executors.explore import framework_lever_grid
        from ..actions.executors.measured_variants import explore_grid_already_covered

        if framework_lever_grid(self.shared_state):
            # The executor adds its pending rewrite levers to this grid, so the grid is not all there is to measure.
            return False

        retry_prefix = f"{idempotency_key}-retry"
        active = [
            task
            for task in (*await self.tasks.queued(), *await self.tasks.running())
            if task.idempotency_key != idempotency_key and not task.idempotency_key.startswith(retry_prefix)
        ]
        covered = explore_grid_already_covered(params, active, self.shared_state.explore_search)
        if covered is None:
            return False
        await self.record_policy_denied(source, intent, covered, action_name="explore")
        return True

    async def handle_delegate(self, source: str, intent: Intent) -> None:
        """Validate and enqueue a delegated action as a TaskRegistry task."""
        action_name = intent.payload["action_name"]
        if self._is_pruned_action(action_name):
            await self._note_pruned_advisory(source, action_name, "delegate_pruned_advisory")
        denied = self._coord.dispatcher.admission_denial_for_action(action_name, params=intent.payload.get("params"))
        if denied is not None:
            await self.record_policy_denied(source, intent, denied, action_name=action_name)
            return
        # delegate explore runs variants directly (no Critic pre-review).
        payload = dict(intent.payload)
        _stamp_kb_exposure(self, payload, source=source)
        params = dict(payload.get("params") or {})
        request = DelegateRequest(source, intent, action_name, params)
        if action_name == "integrate_patch":
            if not await self._stamp_integrate_patch_owner(params):
                await self._reject_missing_patch_owner(source, action_name, params, "delegate_rejected")
                return
            self._coord.proposals.floor_keep_threshold(params)
        if action_name == "specialist" and not await self._admit_specialist(source, intent, params):
            return
        # idempotency_key is top-level per schema; strip a nested compat alias.
        nested_idempotency_key = params.pop("idempotency_key", None)
        self._seed_grid_params(request)
        if _is_specialist_wave(action_name, params):
            await self._coord.specialist_dispatch.fan_out_specialist_wave(source, intent, params)
            return
        if not await self._prepare_specialist_dispatch(request, nested_idempotency_key):
            return
        # Idempotency-key chain: top-level -> nested compat alias -> content-fingerprint auto-key.
        raw_key = intent.payload.get("idempotency_key") or nested_idempotency_key
        if not raw_key:
            content_fp = hashlib.sha1(
                json.dumps(params, sort_keys=True, default=str).encode(),
                usedforsecurity=False,
            ).hexdigest()[:10]
            raw_key = f"{source}:{action_name}:t{int(self.shared_state.tick or 0)}:{content_fp}"
        idempotency_key = str(raw_key)
        if await self._explore_grid_covered(action_name, source, intent, params, idempotency_key):
            return
        # The grid's attempts join its proposal row through this id, as an approved proposal's do. Added after the
        # key is derived, so an identical re-emit still lands on the task already running.
        proposal_id = uuid.uuid4().hex if action_name == "explore" else ""
        if proposal_id:
            params["proposal_msg_id"] = proposal_id
        task = await self._create_delegated_task(request, raw_key)
        if task is None:
            return
        self.shared_state.reset_policy_denial_streak(action_name)
        # Only once a task exists, so a dispatch that bailed out above leaves the mandate on offer.
        self._consume_primatune_mandate(action_name, params, task.task_id)
        from .proposals import record_config_proposal

        record_config_proposal(
            self._coord, proposal_id, action_name, {**payload, "params": params}, outcome="delegated"
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
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_trace_analyze_request
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
                    "in_reply_to": request_msg.msg_id,
                    "kind": f"{kind}_done",
                    "status": status,
                    "result": result,
                    "source": source,
                },
                in_reply_to=request_msg.msg_id,
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
        handler = get_handler(kind)
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
        """Cancel a family's queued tasks and, unless ``scope`` is ``queued``, retire the family; running tasks continue."""
        family = intent.payload["family"]
        reason = str(intent.payload.get("reason") or "prune_branch")
        scope = str(intent.payload.get("scope") or PRUNE_BRANCH_SCOPE_FAMILY).strip()
        drain_only = scope == PRUNE_BRANCH_SCOPE_QUEUED
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
        """Broadcast strategy_change, then defer a closed-vocab hint; drop or deny the rest."""
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
        if not is_valid_escalate_hint(hint):
            denied = PolicyDenied(
                f"escalate_strategy_change hint {hint!r} is not accepted",
                rule="escalate_hint_not_in_vocab",
                hint="Request a phase advance such as skip_to_kernel; the run closes itself.",
            )
            await self.record_policy_denied(source, intent, denied)
            return
        from ..phases.machine_state import PHASE_FRAMEWORK_AGENT

        phase = str(self.shared_state.phase or "").strip().upper()
        if hint == ESCALATE_HINT_SKIP_TO_KERNEL and phase != PHASE_FRAMEWORK_AGENT:
            # Only the FRAMEWORK_AGENT exit reads it; left pending, it would end that phase on arrival.
            log.info("escalate_strategy_change: dropped %r emitted in phase %s", hint, phase or "<none>")
            return
        # Deferred: the next compute_next_phase acts on the hint.
        self.shared_state.set_pending_escalate_hint(hint)
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

        Publishes a ``policy_denied`` observation and records the denial streak.
        The streak is a fact for LLM self-correction only: there is no
        auto-prune and no ``policy_loop`` stop triggered from it.

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
        # Streak counter is a fact for LLM self-correction only; the system does not auto-prune or stop on it.
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            intent_payload=intent.payload,
        )
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseDenial

        payload = intent.payload or {}
        proposal_msg_id = (
            payload.get("proposal_msg_id") or payload.get("target_proposal_msg_id") or payload.get("proposal_id")
        )
        phase_event.record_denial(
            PhaseDenial(
                at=PhaseCycle(str(self.shared_state.phase or ""), int(self.shared_state.macro_cycle or 0)),
                actor=source,
                proposal_msg_id=str(proposal_msg_id) if proposal_msg_id else None,
                action=resolved_action,
                rule=str(denied.rule or ""),
                hint=str(denied.hint or ""),
            )
        )
