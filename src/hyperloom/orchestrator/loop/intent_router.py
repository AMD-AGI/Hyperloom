# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Intent routing collaborator for :class:`Coordinator`."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any

from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    patch_lever_kind,
    patch_owner_phase,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from .coordinator_helpers import (
    _parse_iso_unix,
    coerce_needs_gpu,
    collapse_verdict_map,
    collapse_verdicts,
    format_exc_brief,
    serialize_verdict_advisory,
    verdict_held_to_its_rule,
    verdict_map_entry_held_to_its_rule,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ..bus.message_bus import Message, TOPIC_ALLOWLIST
from ..policy.gate import (
    patch_verdict_subject,
    PolicyDenied,
    PRUNE_BRANCH_SCOPE_FAMILY,
    PRUNE_BRANCH_SCOPE_QUEUED,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import IllegalTransition, TaskNotFound
from ..kernel.request_handlers import KERNEL_REQUEST_HANDLERS, get_handler
from ..phases.machine_state import KERNEL_HEARTBEAT_SEC as _KERNEL_HEARTBEAT_SEC

# ``Coordinator`` is intentionally NOT imported (avoids a module-level import cycle with coordinator.py); it is held
# as a back-reference and the annotation below is a deferred string.

log = __import__("logging").getLogger(__name__)


# IntentType -> the ``Coordinator`` handler method it dispatches to.
_INTENT_DISPATCH: dict[IntentType, str] = {
    IntentType.PROPOSE_ACTION: "_handle_propose_action",
    IntentType.REVIEW_VERDICT: "_handle_review_verdict",
    IntentType.DELEGATE: "_handle_delegate",
    IntentType.REQUEST: "_handle_request",
    IntentType.RESPONSE: "_handle_response",
    IntentType.EXTEND_LEASE: "_handle_extend_lease",
    IntentType.PRUNE_BRANCH: "_handle_prune_branch",
    IntentType.ESCALATE_STRATEGY_CHANGE: "_handle_escalate_strategy_change",
    IntentType.SEND_MESSAGE: "_handle_send_message",
    IntentType.ALERT: "_handle_alert",
    IntentType.UPDATE_STATE: "_handle_update_state",
}


def _is_upstream_pr_candidate(pending: Any) -> bool:
    """True for an ``integrate_patch`` proposal that pre-screens a PR candidate."""
    if getattr(pending, "action_name", "") != "integrate_patch":
        return False
    return bool((getattr(pending, "payload", None) or {}).get("framework_agent_candidate_id"))


def _record_config_proposal(router: Any, pending: Any) -> None:
    """Record one config-arm grid on the framework event, as it is proposed.

    The arm's measured attempts already name this row through their
    ``proposal_ref``, so without it the whole upstream of the config arm is a
    dangling reference: the event says six variants were benched and cannot
    say what was proposed or who proposed it.

    Recorded when the grid is proposed rather than when it is approved, so one
    the Critic denies is still on record as a thing the phase pursued and
    dropped -- which is the reading the arm's plateau acts on.

    One row per grid, not per variant, because that is what the attempts point
    at: the config arm proposes in batches and the Critic rules on them by
    name, so a grid of six variants is one proposal with six attempts.

    Module-level and self-defending like the phase's other recording helpers:
    this seam is driven by lightweight stand-ins in tests, and a proposal must
    never fail to route because its observability did.

    Args:
        router: The intent router, or a stand-in; the recorder is read off it.
        pending: The pending proposal just minted.
    """
    if str(getattr(pending, "action_name", "") or "") != "explore":
        return
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    getter = getattr(router, "_framework_timeline", None)
    recorder = getter() if callable(getter) else None
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        ARM_CONFIG,
        PRODUCER_ORCHESTRATION,
        STEP_PROPOSED,
        producer_for_provenance,
    )

    params = (getattr(pending, "payload", None) or {}).get("params") or {}
    grid = [row for row in (params.get("grid") or []) if isinstance(row, dict)]
    labels = {str(row.get("provenance") or "").strip() for row in grid}
    if len(labels) == 1:
        producer, producer_ref = producer_for_provenance(next(iter(labels)))
    else:
        # A grid mixing provenances was assembled by the orchestration agent:
        # a specialist's own config arrives as a single-provenance proposal of
        # its own. Either way each variant keeps its label on its attempt, so
        # the mix is not lost by naming the assembler here.
        producer, producer_ref = PRODUCER_ORCHESTRATION, ""
    scopes = {str(row.get("scope") or "").strip() for row in grid if str(row.get("scope") or "").strip()}
    try:
        recorder.record_proposal(
            proposal_id,
            arm=ARM_CONFIG,
            producer=producer,
            producer_ref=producer_ref,
            lever_kind=LEVER_CONFIG,
            scope=scopes.pop() if len(scopes) == 1 else "",
        )
        recorder.record_proposal_step(proposal_id, step=STEP_PROPOSED, outcome="submitted")
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug(
            "framework timeline: config proposal row failed for %s",
            proposal_id,
            exc_info=True,
        )


def _variant_review_rows(
    payload: Mapping[str, Any] | None,
    held_by_name: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Return the per-variant rulings a ``verdict_map`` carried.

    A variant the Critic rejected never reaches a bench, so its ruling has no
    attempt row to live on and the map is the only record that it was judged
    at all. Kept per variant rather than collapsed, because the collapse is
    deliberately lossy: a grid proceeds on its approved subset, and the summary
    verdict says nothing about which variants that was.

    Args:
        payload: The ``review_verdict`` payload, read for ``verdict_map``.
        held_by_name: What each entry was held to, keyed by variant name.

    Returns:
        One row per named variant, in the order the Critic wrote them.
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


def _review_subject(pending: Any) -> str:
    """Return the proposal row a ruling belongs on.

    The two arms identify a proposal differently: a configuration grid is known
    by the bus message that raised it, an upstream candidate by its candidate
    id, and the row already recorded under that id is the one the ruling is
    about. Resolving it here is what keeps a review on the proposal it judged
    instead of opening a second, near-empty row beside it.

    Args:
        pending: The proposal reviewed.

    Returns:
        The candidate id when the proposal carries one, else the bus message id.
    """
    payload = getattr(pending, "payload", None) or {}
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    candidate = str(
        payload.get("framework_agent_candidate_id") or params.get("framework_agent_candidate_id") or ""
    ).strip()
    return candidate or str(getattr(pending, "proposal_msg_id", "") or "")


def _phase_scope(router: Any) -> tuple[str, int]:
    """The phase and macro cycle a proposal is being raised in.

    Args:
        router: The intent router, or a stand-in.

    Returns:
        tuple[str, int]: The phase name and macro cycle, ``("", 0)`` when the
            stand-in carries no state.
    """
    state = getattr(router, "shared_state", None) or getattr(router, "state", None)
    phase = str(getattr(state, "phase", "") or "")
    try:
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
    except (TypeError, ValueError):
        cycle = 0
    return phase, cycle


def _record_phase_proposal(router: Any, pending: Any) -> None:
    """Record one proposal against the phase that raised it.

    Every proposal, not only the ones a framework arm claims: this is the row
    the Critic's ruling is filed on, and a ruling on a KERNEL ``kernel_opt`` or
    on an action no arm maps to had no subject row anywhere before this.

    Module-level and self-defending like the phase's other recording helpers:
    this seam is driven by lightweight stand-ins in tests, and a proposal must
    never fail to route because its observability did.

    Args:
        router: The intent router, or a stand-in.
        pending: The pending proposal just minted.
    """
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    phase, macro_cycle = _phase_scope(router)
    if not phase:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        payload = getattr(pending, "payload", None) or {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        phase_event.record_proposal(
            proposal_msg_id=proposal_id,
            action=str(getattr(pending, "action_name", "") or ""),
            phase=phase,
            macro_cycle=macro_cycle,
            from_agent=str(getattr(pending, "from_agent", "") or ""),
            tick=int(getattr(getattr(router, "shared_state", None), "tick", 0) or 0),
            predicted_gain_pct=getattr(pending, "predicted_gain_pct", None),
            candidate_id=payload.get("framework_agent_candidate_id") or params.get("framework_agent_candidate_id"),
            variant_name=payload.get("variant_name") or params.get("variant_name"),
        )
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug("phase timeline: proposal row failed for %s", proposal_id, exc_info=True)


def _record_phase_proposal_review(
    pending: Any,
    *,
    authored: str,
    effective: str,
    reason: str,
    payload: Mapping[str, Any] | None = None,
    advisory: Mapping[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> None:
    """File the Critic's ruling on the proposal row the phase event holds.

    Keyed on the bus message id the verdict named, rather than on the subject
    the framework row is resolved under: the two differ for a proposal
    carrying a candidate id, and this row is about the proposal itself.

    Args:
        pending: The proposal reviewed.
        authored: The verdict the Critic wrote.
        effective: The verdict the loop acted on.
        reason: The Critic's reasoning.
        payload: The ``review_verdict`` payload.
        advisory: The advisory block serialised for the rebroadcast.
        variants: Per-variant rulings, when a ``verdict_map`` named any.
    """
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        fields = payload or {}
        advice = advisory or {}
        phase_event.record_proposal_review(
            proposal_msg_id=proposal_id,
            verdict=authored or effective,
            effective_verdict=effective,
            source=str(fields.get("source") or ""),
            reasoning=reason,
            confidence=fields.get("confidence"),
            failure_reason_code=fields.get("failure_reason_code"),
            required_evidence=advice.get("required_evidence"),
            risks=advice.get("risks"),
            advice_text=advice.get("advice_text"),
            alternative_action=advice.get("alternative_action"),
            variants=variants,
        )
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug("phase timeline: proposal review row failed for %s", proposal_id, exc_info=True)


def _record_critic_review(
    router: Any,
    pending: Any,
    *,
    authored: str,
    effective: str,
    reason: str,
    payload: Mapping[str, Any] | None = None,
    advisory: Mapping[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> None:
    """Record the Critic's ruling on one proposal, onto the proposal.

    Every action the Critic reviews is recorded, not only the config-arm grid:
    a rejected ``integrate_patch`` is the reason a patch never landed, and
    keeping that ruling only in the state mirror the patch gate consults left
    the round reading as though the patch had never been authored.

    Both verdicts are kept because they are different facts and the loop
    already treats them so: a reject held to a formatting rule is mirrored
    onto state as the reject the Critic wrote, not as what the hold made of
    it. The review carries what the Critic ruled; the lifecycle step carries
    what the loop then did with it.

    A rejected proposal is settled here. It never reaches a bench, so nothing
    downstream would otherwise resolve it, and a proposal left ``pending``
    reads as one the phase never finished deciding.

    Args:
        router: The intent router, or a stand-in.
        pending: The proposal reviewed.
        authored: The verdict the Critic wrote.
        effective: The verdict the loop acted on.
        reason: The Critic's reasoning.
        payload: The ``review_verdict`` payload, read for the grounds the
            Critic stated -- source, confidence and the rule it cited.
        advisory: The advisory block already serialised for the rebroadcast.
        variants: Per-variant rulings, when a ``verdict_map`` named any.
    """
    # Onto the proposal's own row first. The framework row below exists only
    # while a framework event is open for this cycle, so a ruling reached in
    # any other phase -- or after this one exited -- would otherwise be acted
    # on with nothing recording what the Critic said.
    _record_phase_proposal_review(
        pending,
        authored=authored,
        effective=effective,
        reason=reason,
        payload=payload,
        advisory=advisory,
        variants=variants,
    )

    proposal_id = _review_subject(pending)
    if not proposal_id:
        return
    getter = getattr(router, "_framework_timeline", None)
    recorder = getter() if callable(getter) else None
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        DISPOSITION_DROPPED,
        REVIEWER_CRITIC,
        REVIEWER_CRITIC_UNAVAILABLE,
        STEP_REVIEWED,
    )

    fields = payload or {}
    grounded = str(fields.get("source") or REVIEWER_CRITIC) == REVIEWER_CRITIC
    try:
        recorder.record_proposal_review(
            proposal_id,
            verdict=authored or effective,
            effective_verdict=effective,
            held_to_rule=str(fields.get("failure_reason_code") or "") if effective != authored else "",
            reviewer=REVIEWER_CRITIC if grounded else REVIEWER_CRITIC_UNAVAILABLE,
            reason=reason,
            confidence=fields.get("confidence"),
            failure_reason_code=str(fields.get("failure_reason_code") or ""),
            advisory=advisory,
            variants=variants,
        )
        recorder.record_proposal_step(
            proposal_id,
            step=STEP_REVIEWED,
            outcome=effective,
            reason=reason,
        )
        if effective == "reject":
            recorder.settle_proposal(
                proposal_id,
                disposition=DISPOSITION_DROPPED,
                reason=reason or "critic_rejected",
            )
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug(
            "framework timeline: critic review row failed for %s",
            proposal_id,
            exc_info=True,
        )


def _record_phase_proposal_outcome(pending: Any, **outcome: Any) -> None:
    """Record on the proposal row what the loop did with its ruling.

    Args:
        pending: The proposal ruled on.
        **outcome: ``materialized`` / ``denied`` / ``reauthored`` /
            ``patch_verdict_key``, as the framework row names them.
    """
    proposal_id = str(getattr(pending, "proposal_msg_id", "") or "")
    if not proposal_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        phase_event.record_proposal_outcome(
            proposal_msg_id=proposal_id,
            materialized=bool(outcome.get("materialized")),
            denied=bool(outcome.get("denied")),
            reauthored=bool(outcome.get("reauthored")),
            patch_verdict_key=outcome.get("patch_verdict_key"),
        )
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug("phase timeline: proposal outcome row failed for %s", proposal_id, exc_info=True)


def _record_review_outcome(router: Any, pending: Any, **outcome: Any) -> None:
    """Record what the loop did with a ruling, onto the ruling.

    Args:
        router: The intent router, or a stand-in.
        pending: The proposal ruled on.
        **outcome: The fields ``record_proposal_review_outcome`` accepts.
    """
    _record_phase_proposal_outcome(pending, **outcome)

    proposal_id = _review_subject(pending)
    if not proposal_id:
        return
    getter = getattr(router, "_framework_timeline", None)
    recorder = getter() if callable(getter) else None
    if recorder is None:
        return
    try:
        recorder.record_proposal_review_outcome(proposal_id, **outcome)
    except Exception:  # noqa: BLE001 — observability cannot change routing
        log.debug(
            "framework timeline: review outcome row failed for %s",
            proposal_id,
            exc_info=True,
        )


class IntentRouter:
    """Validates and dispatches agent-emitted intents on behalf of a Coordinator."""

    def __init__(self, coordinator: Any) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        # Attributes not defined on the router resolve onto the coordinator.
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _stamp_specialist_owner(self, params: dict[str, Any]) -> str:
        """Freeze patch ownership when a specialist task is created."""
        lever = patch_lever_kind(params)
        if lever:
            params["lever_kind"] = lever
        owner = patch_owner_phase(params)
        if not owner:
            gap_layer = str(params.get("gap_layer") or "").strip().lower()
            active_phase = str(getattr(self.shared_state, "phase", "") or "").strip().upper()
            # Layer first, phase last: both lanes share one phase, so the live phase no longer says which lever a
            # specialist moves.
            if gap_layer == "framework":
                owner = "FRAMEWORK_AGENT"
            elif gap_layer in {"explore", "perf_explore"} or params.get("domain"):
                owner = "EXPLORE"
            elif active_phase in {"FRAMEWORK", "FRAMEWORK_AGENT"}:
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
            "source_domain",
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

    async def _handle_intent(self, source: str, intent: Intent) -> None:
        """Validate an emitted intent through PolicyGate, then route it."""
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
                    source,
                    "observation",
                    {"intent": it.value, "payload": intent.payload},
                )
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
        """Gate a proposed action and enqueue it for Critic Review."""
        action_name = intent.payload["action_name"]
        # Pruned families are advisory: proposal still queues with an advisory note.
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator",
                "observation",
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
        denied = self._admission_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        payload = dict(intent.payload)
        if action_name == "integrate_patch":
            params = dict(payload.get("params") or {})
            if not await self._stamp_integrate_patch_owner(params):
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_rejected",
                        "reason": "integrate_patch_owner_missing",
                        "from_agent": source,
                        "action_name": action_name,
                        "specialist_task_id": str(params.get("specialist_task_id") or ""),
                    },
                )
                return
            payload["params"] = params
        msg = Message.new(
            source,
            "*",
            "proposal",
            {**payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        from .coordinator import PendingProposal

        pending = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent=source,
            action_name=action_name,
            predicted_gain_pct=float(intent.payload.get("predicted_gain_pct", 0.0)),
            payload=payload,
        )
        self.state.pending_proposals[msg.msg_id] = pending
        _record_phase_proposal(self, pending)
        _record_config_proposal(self, pending)

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        """Apply a Critic ``review_verdict`` to its target proposal."""
        target = intent.payload["target_proposal_msg_id"]
        pending = self.state.pending_proposals.get(target)
        verdict_map = intent.payload.get("verdict_map")
        single_verdict = intent.payload.get("verdict")
        if pending is None:
            await self._record_observation(
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
        await self._coord._handle_single_verdict(
            source=source,
            pending=pending,
            verdict=verdict,
            authored_verdict=authored,
            reasoning=str(intent.payload.get("reasoning") or ""),
            advisory=serialize_verdict_advisory(intent.payload),
            approved_variant_names=approved_variant_names,
            payload=intent.payload,
            variant_verdicts=_variant_review_rows(intent.payload, held_by_name),
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
        try:
            sub_verdicts = list(held_by_name.values())
            if verdict in ("approve", "advise") and any(sv in ("reject", "needs_review") for sv in sub_verdicts):
                log.warning(
                    "review_verdict collapse: target=%s collapsed to %r (sub_verdicts=%r)",
                    target,
                    verdict,
                    sub_verdicts,
                )
        except Exception:  # noqa: BLE001 - audit log must never affect flow
            pass

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
        await self._record_observation(
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
        verdict: str,
        reasoning: str,
        authored_verdict: str = "",
        advisory: dict[str, Any] | None = None,
        approved_variant_names: set[str] | None = None,
        payload: Mapping[str, Any] | None = None,
        variant_verdicts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Apply one collapsed verdict: approve/advise materialise, reject may rearm.

        Args:
            source: The agent emitting the verdict.
            pending: The pending proposal the verdict targets.
            verdict: The collapsed verdict (approve / advise / reject / needs_review).
            reasoning: Free-text reasoning recorded with the verdict.
            authored_verdict: The verdict the Critic itself wrote, before any
                hold to a cited rule. Mirrored onto ``specialist_patch_verdicts``
                in place of ``verdict``; defaults to ``verdict``.
            advisory: Pre-serialised advisory fields (``required_evidence`` /
                ``risks`` / ``advice_text`` / ``alternative_action`` /
                ``notes`` / ``kb_evidence`` / ``packet_evidence``) to carry on
                the rebroadcast payload so the full Critic context reaches the
                orchestration inbox and downstream consumers.
            approved_variant_names: When a ``verdict_map`` named proceedable
                variants, restrict an explore grid to those names; ``None``
                keeps the full proposal.
            payload: The ``review_verdict`` payload, recorded onto the
                proposal for the grounds the Critic stated.
            variant_verdicts: The per-variant rulings a ``verdict_map``
                carried, recorded onto the proposal.
        """
        pending.decided = True
        pending.verdict = verdict
        if _is_upstream_pr_candidate(pending):
            await self._record_observation(
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
            "reasoning": reasoning,
        }
        if advisory:
            rebroadcast_payload.update(advisory)
        await self.bus.append_and_seq(
            Message.new(
                source,
                pending.from_agent,
                "review_verdict",
                rebroadcast_payload,
                priority=0 if verdict == "reject" else 1,
                in_reply_to=pending.proposal_msg_id,
            )
        )
        # Mirror specialist / integrate_patch verdicts onto SharedState so PolicyGate's integrate_patch gate can
        # consult them on the next tick.
        patch_verdict = str(authored_verdict or verdict).strip()
        try:
            pa_params = pending.payload.get("params") or {}
        except AttributeError:
            pa_params = {}
        sid_candidate = ""
        if pending.action_name == "integrate_patch":
            # A pre-screen carries its candidate id at the top level, not in params.
            sid_candidate = patch_verdict_subject({**pa_params, **(pending.payload or {})})
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
            except Exception:  # noqa: BLE001 — best-effort mirror
                log.exception(
                    "failed to mirror critic verdict for specialist task=%s",
                    sid_candidate,
                )
        _record_critic_review(
            self,
            pending,
            authored=patch_verdict,
            effective=str(verdict or "").strip(),
            reason=str(reasoning or ""),
            payload=payload,
            advisory=advisory,
            variants=variant_verdicts,
        )
        _record_review_outcome(
            self,
            pending,
            materialized=verdict in ("approve", "advise"),
            denied=verdict == "reject",
            reauthored=verdict == "needs_review",
            patch_verdict_key=sid_candidate if patch_verdict else "",
        )
        # Both `approve` and `advise` mean "dispatch may proceed"; treat them
        # identically for materialization.
        if verdict in ("approve", "advise"):
            await self._materialize_approved_proposal(
                pending,
                approved_variant_names=approved_variant_names,
            )
        elif verdict == "reject" and _is_upstream_pr_candidate(pending):
            # Record the critic_denied row so the candidate pump advances.
            await self._coord._record_framework_agent_critic_denied(
                pending,
                reasoning,
            )
        elif verdict == "reject" and pending.action_name == "integrate_patch" and bool(pa_params.get("enablement")):
            # A Critic-rejected ENABLEMENT integrate_patch never reaches the executor, so the normal integrate-result
            # rearm never fires.
            try:
                self._coord._maybe_rearm_enablement(
                    {"enablement": True, "status": "reverted", "reason": "critic_rejected"}
                )
            except Exception:  # noqa: BLE001 — accounting must never wedge the loop
                log.exception(
                    "enablement rearm on critic-reject failed for task=%s",
                    sid_candidate,
                )
        elif verdict == "needs_review":
            await self._coord._maybe_reauthor_from_critic_feedback(
                pending,
                advisory,
            )

    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        """Validate and enqueue a delegated action as a TaskRegistry task."""
        action_name = intent.payload["action_name"]
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
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
        denied = self._admission_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(
                source,
                intent,
                denied,
                action_name=action_name,
            )
            return
        # delegate explore runs variants directly (no Critic pre-review).
        params = dict(intent.payload.get("params") or {})
        if action_name == "integrate_patch":
            if not await self._stamp_integrate_patch_owner(params):
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "delegate_rejected",
                        "reason": "integrate_patch_owner_missing",
                        "from_agent": source,
                        "action_name": action_name,
                        "specialist_task_id": str(params.get("specialist_task_id") or ""),
                    },
                )
                return
        if action_name == "specialist":
            # Capture proposal ownership at dispatch.
            self._stamp_specialist_owner(params)
        # idempotency_key is top-level per schema; strip a nested compat alias.
        nested_idempotency_key = params.pop("idempotency_key", None)
        # Plumb baseline's materialized YAML into grid-style tasks (delegator may override).
        if action_name in ("sweep", "explore") and self.shared_state.baseline_config_path:
            params.setdefault("config_path", self.shared_state.baseline_config_path)
        # Delegates skip _materialize_approved_proposal, so seed the same params here.
        if action_name in ("sweep", "explore"):
            inject_stack_base_params(params, self.shared_state, anchor=True)
        if action_name == "explore":
            self._inject_explore_runtime_params(params)
        # Wave sugar: a specialist delegate carrying params.tasks=[...] fans out into N standard freeform specialist
        # tasks, each dispatched through the normal SpecialistRunner + TaskRegistry + lease + reap path.
        if (
            action_name == "specialist"
            and isinstance(
                params.get("tasks"),
                list,
            )
            and params["tasks"]
        ):
            await self._fan_out_specialist_wave(source, intent, params)
            return
        # Specialist pre-dispatch warmup via KnowledgePlane.
        if action_name == "specialist":
            await self._warm_specialist_params(params)
        # Idempotency-key chain: top-level -> nested compat alias -> content-fingerprint auto-key.
        raw_key = intent.payload.get("idempotency_key") or nested_idempotency_key
        if not raw_key:
            content_fp = hashlib.sha1(
                json.dumps(params, sort_keys=True, default=str).encode(),
                usedforsecurity=False,
            ).hexdigest()[:10]
            raw_key = f"{source}:{action_name}:t{int(self.shared_state.tick or 0)}:{content_fp}"
        idempotency_key = str(raw_key)
        terminal_states = {
            "succeeded",
            "failed",
            "cancelled",
        }
        task = None
        was_existing = False
        for attempt in range(6):
            idempotency_key = str(raw_key) if attempt == 0 else f"{raw_key}-retry{attempt}"
            lanes, ttl = self._registry_lanes_ttl(action_name)
            # Bench-enabled specialists serialize against the other GPU benchmark/profile/server work via
            # benchmark_lane (research_lane alone conflicts with nothing).
            if action_name == "specialist":
                from ..specialists.profile import resolve_specialist_profile

                if resolve_specialist_profile(params).reserves_benchmark_lane:
                    lanes = tuple(dict.fromkeys((*lanes, "benchmark_lane")))
                # Any GPU-holding specialist serializes against serving via gpu_research_lane.
                needs_gpu = coerce_needs_gpu(params.get("needs_gpu", False))
                if needs_gpu:
                    lanes = tuple(dict.fromkeys((*lanes, "gpu_research_lane")))
                    try:
                        # Shared with the GPU-pool lease so the two TTLs never drift.
                        ttl = self._coord._gpu_lease_ttl_sec(
                            int(ttl or 0),
                            params=params,
                        )
                    except Exception:  # noqa: BLE001 — fall back to registry ttl
                        log.exception(
                            "WS2: failed to re-source gpu_research_lane TTL; using registry default",
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
                    source,
                    intent,
                    PolicyDenied(
                        f"delegate{{action_name={action_name!r}}} duplicate idempotency_key={idempotency_key!r}",
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
                source,
                intent,
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
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "event",
                {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
            )
        )

    @asynccontextmanager
    async def _kernel_step_heartbeat(self, kind: str, started: float):
        """Keep orchestration's bus timestamp moving through an inline step."""

        # Re-stamped per beat rather than once at the start, so a stamp that outlives its process expires instead of
        # muting the KERNEL idle guard.
        def _mark_running() -> None:
            self.shared_state.kernel_inline_step_seen_unix = time.time()

        async def _beat() -> None:
            while True:
                await asyncio.sleep(_KERNEL_HEARTBEAT_SEC)
                _mark_running()
                await self.bus.append_and_seq(
                    Message.new(
                        "orchestration",
                        "*",
                        "observation",
                        {
                            "kind": "kernel_step_running",
                            "step": kind,
                            "elapsed_sec": round(time.monotonic() - started, 1),
                        },
                    )
                )

        _mark_running()
        task = asyncio.create_task(_beat())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self.shared_state.kernel_inline_step_seen_unix = 0.0

    def _record_request_failure(self, *, kind: str, request_msg_id: str, result: dict[str, Any]) -> None:
        """Append a failed kernel request to the log the FAILURE RECOVERY prompt block reads."""
        self.shared_state.record_action_failure(action=kind, task_id=request_msg_id, result=result)
        self.shared_state.save(self.session_dir)

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
        timeline to account for it, and the kernel table it produced reaches a
        reader only through the state projection.

        Best-effort: a failed record never breaks the request.
        """
        try:
            from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_trace_analyze_request

            record_trace_analyze_request(
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                run_id=str(getattr(request_msg, "msg_id", "") or ""),
                status=str(result.get("status") or ""),
                result=result,
                requested_by=source,
                request_msg_id=str(getattr(request_msg, "msg_id", "") or ""),
                trace_input=str(payload.get("trace_path") or payload.get("trace_input") or ""),
                top_k=payload.get("top_k"),
                snapshot=getattr(self.shared_state, "last_trace_analyze", None),
                cache_hit=cache_hit,
            )
        except Exception:  # noqa: BLE001 — observability cannot change routing
            log.debug("kernel timeline: trace_analyze request capture failed", exc_info=True)

    async def _handle_request(self, source: str, intent: Intent) -> None:
        """Route a REQUEST intent to its programmatic handler."""
        from .coordinator import _lifecycle_paths

        target_agent = intent.payload["target_agent"]
        kind = intent.payload["kind"]
        denied = self._sequence_denial_for_request(target_agent, kind)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        # Always record the request on the bus for the kernel reactor / replay.
        request_msg = Message.new(
            source,
            target_agent,
            "request",
            dict(intent.payload),
            priority=1,
        )
        await self.bus.append_and_seq(request_msg)

        if target_agent == "kernel_agent":
            if not bool(getattr(self.shared_state, "kernel_enabled", True)):
                _fail_result = {
                    "status": "failed",
                    "error_class": "agent_disabled",
                    "error": "kernel_agent is disabled for this session (--no-kernel)",
                }
                await self.bus.append_and_seq(
                    Message.new(
                        target_agent,
                        source,
                        "response",
                        {
                            "in_reply_to": request_msg.msg_id,
                            "kind": f"{kind}_done",
                            "status": "failed",
                            "result": _fail_result,
                            "source": "coordinator_auto_reject",
                        },
                        in_reply_to=request_msg.msg_id,
                        priority=1,
                    )
                )
                self._record_request_failure(kind=kind, request_msg_id=request_msg.msg_id, result=_fail_result)
                return
            handler = get_handler(kind)
            if handler is None:
                _fail_result = {
                    "status": "failed",
                    "error_class": "unknown_kernel_kind",
                    "error": f"no programmatic handler for kind={kind!r}",
                    "valid_kinds": sorted(KERNEL_REQUEST_HANDLERS),
                }
                await self.bus.append_and_seq(
                    Message.new(
                        target_agent,
                        source,
                        "response",
                        {
                            "in_reply_to": request_msg.msg_id,
                            "kind": f"{kind}_done",
                            "status": "failed",
                            "result": _fail_result,
                            "source": "coordinator_auto_reject",
                        },
                        in_reply_to=request_msg.msg_id,
                        priority=1,
                    )
                )
                self._record_request_failure(kind=kind, request_msg_id=request_msg.msg_id, result=_fail_result)
                return
            params = intent.payload.get("params") or {}
            merged_payload = {**intent.payload, **params}
            # Roofline data is read from the last_trace_analyze cache rather than auto-injected here.
            cache_hit_source = None
            cached_result = self._cached_kernel_request(kind, merged_payload)
            if cached_result is not None:
                result = cached_result
                cache_hit_source = "shared_state_cache"
                # A cache hit never runs the handler; emit a single END (detail=cache_hit) so the lifecycle log
                # records the step.
                self._emit_lifecycle(
                    step=kind,
                    status="END",
                    artifacts=_lifecycle_paths(result),
                    detail="cache_hit",
                )
            else:
                rejected = self.shared_state.find_rejected_kernel_patch(merged_payload) if kind == "integrate" else None
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
                    # A short-circuited integrate never runs the handler; emit a lone END recording the rejection.
                    self._emit_lifecycle(
                        step=kind,
                        status="END",
                        artifacts=_lifecycle_paths(result),
                        detail="rejected",
                    )
                else:
                    # Inject base_tput from current_best.tput when an integrate request omits it; operator value wins.
                    if kind == "integrate" and not merged_payload.get("base_tput"):
                        cb_tput = (self.shared_state.current_best or {}).get("tput")
                        if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                            merged_payload["base_tput"] = float(cb_tput)

                    handler_kwargs: dict[str, Any] = {
                        "session_dir": self.session_dir,
                    }
                    # Bracket the programmatic kernel step with START / END lifecycle events.
                    _lc_t0 = time.monotonic()
                    self._emit_lifecycle(
                        step=kind,
                        status="START",
                        artifacts=_lifecycle_paths(merged_payload),
                    )
                    try:
                        async with self._kernel_step_heartbeat(kind, _lc_t0):
                            result = await handler(
                                merged_payload,
                                **handler_kwargs,
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.exception(
                            "kernel_request_handler[%s] crashed for source=%s",
                            kind,
                            source,
                        )
                        result = {
                            "status": "failed",
                            "error_class": "handler_exception",
                            "error": repr(exc),
                        }
                    # A block-FP8 GEMM run may have executed an inline Roofline whose refreshed profile fields only
                    # live in state.json.
                    if kind == "run_gemm_tuning":
                        self._sync_profile_state_after_gemm_roofline(result)
                    _lc_status = "ERROR" if str(result.get("status", "")).lower() in ("failed", "error") else "END"
                    _lc_detail = " ".join(
                        str(p)
                        for p in (
                            result.get("decision"),
                            result.get("status"),
                            f"kernel={result.get('kernel_id')}" if result.get("kernel_id") else "",
                        )
                        if p
                    )
                    self._emit_lifecycle(
                        step=kind,
                        status=_lc_status,
                        artifacts=_lifecycle_paths(result),
                        detail=_lc_detail,
                        duration_s=time.monotonic() - _lc_t0,
                    )
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent",
                    source,
                    "response",
                    {
                        "in_reply_to": request_msg.msg_id,
                        "kind": f"{kind}_done",
                        "status": result.get("status", "ok"),
                        "result": result,
                        "source": cache_hit_source or "programmatic_handler",
                    },
                    in_reply_to=request_msg.msg_id,
                    priority=1,
                )
            )
            if str(result.get("status", "")).lower() in ("failed", "error"):
                self._record_request_failure(kind=kind, request_msg_id=request_msg.msg_id, result=result)
            # Cache trace_analyze output (successful runs only).
            if kind == "trace_analyze" and cache_hit_source is None and result.get("status") in ("ok", "succeeded"):
                self.shared_state.record_trace_analyze(merged_payload, result)
                self.shared_state.save(self.session_dir)
            if kind == "trace_analyze":
                self._record_trace_analyze_on_timeline(
                    request_msg=request_msg,
                    source=source,
                    payload=merged_payload,
                    result=result,
                    cache_hit=cache_hit_source is not None,
                )
            if kind == "run_gemm_tuning":
                await self._handle_gemm_tuning_result(result)
            if kind == "integrate":
                if result.get("status") != "skipped":
                    self.shared_state.record_kernel_integrate_result(result)
                decision = str(result.get("decision", "")).upper()
                if decision == "KEEP":
                    if isinstance(result, dict) and not result.get("gap_canonical_id"):
                        payload_gap = str(merged_payload.get("gap_canonical_id") or "").strip()
                        if payload_gap:
                            result["gap_canonical_id"] = payload_gap
                    await self._record_integrate_keep(result)
                self.shared_state.save(self.session_dir)
        else:
            _fail_result = {
                "status": "failed",
                "error_class": "unknown_target_agent",
                "error": f"no handler registered for target_agent={target_agent!r}",
            }
            await self.bus.append_and_seq(
                Message.new(
                    target_agent,
                    source,
                    "response",
                    {
                        "in_reply_to": request_msg.msg_id,
                        "kind": f"{kind}_done",
                        "status": "failed",
                        "result": _fail_result,
                        "source": "coordinator_auto_reject",
                    },
                    in_reply_to=request_msg.msg_id,
                    priority=1,
                )
            )
            self._record_request_failure(kind=kind, request_msg_id=request_msg.msg_id, result=_fail_result)

    async def _handle_response(self, source: str, intent: Intent) -> None:
        """Route a RESPONSE intent back to the original requester."""
        in_reply_to = intent.payload["in_reply_to"]
        # Locate the original requester so we can address the response.
        original = await self.bus.lookup_by_id(in_reply_to)
        target = original.from_agent if original else "*"
        await self.bus.append_and_seq(
            Message.new(
                source,
                target,
                "response",
                dict(intent.payload),
                in_reply_to=in_reply_to,
                priority=1,
            )
        )

    async def _handle_extend_lease(self, source: str, intent: Intent) -> None:
        """Grant a running task more lease time."""
        task_id = str(intent.payload.get("task_id") or "").strip()
        extra_sec = int(intent.payload.get("extra_sec") or 0)
        try:
            new_ttl = await self.tasks.extend_lease(task_id, extra_sec)
        except (TaskNotFound, IllegalTransition) as exc:
            await self._record_observation(
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
            task = await self.tasks.get(task_id)
            started = _parse_iso_unix(task.updated_at)
            if started > 0:
                running_sec = max(0.0, time.time() - started)
        except Exception:  # noqa: BLE001 — fall back to the full TTL
            log.exception("extend_lease: could not read running age for task=%s", task_id)
        # A late extension can arrive after the cumulative task TTL expired but before the worker/reaper acted on it.
        remaining_sec = max(1, int(extra_sec), int(new_ttl - running_sec))
        lanes = await self.locks.heartbeat_by_task(task_id, ttl_sec=remaining_sec)
        gpu_error = ""
        try:
            gpus = await self.gpu_specialist_pool.extend(task_id, remaining_sec)
        except Exception as exc:  # noqa: BLE001 — lane extension already landed
            log.exception("extend_lease: GPU lease refresh failed for task=%s", task_id)
            gpus = 0
            gpu_error = repr(exc)[:200]
        # Push the live subprocess's hard wall-clock kill deadline out too, so the extension actually buys the
        # specialist more time to run.
        wall_budget_error = ""
        try:
            from ..specialists.subprocess_ import grant_wall_budget_extension

            grant_wall_budget_extension(task_id, extra_sec)
        except Exception as exc:  # noqa: BLE001 — lease rows already moved
            log.exception("extend_lease: wall-budget extension failed for task=%s", task_id)
            wall_budget_error = repr(exc)[:200]
        # A swallowed GPU or wall-budget failure would leave the lane extended while the GPU reaper or subprocess
        # wall-clock cap can still interrupt the work — report the partial extension as degraded.
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "extend_lease_degraded" if gpu_error or wall_budget_error else "extend_lease",
                "task_id": task_id,
                "source": source,
                "extra_sec": extra_sec,
                "lease_ttl_sec": new_ttl,
                "lease_expires_in_sec": remaining_sec,
                "lanes": lanes,
                "gpu_rows": gpus,
                **({"gpu_refresh_error": gpu_error} if gpu_error else {}),
                **({"wall_budget_extension_error": wall_budget_error} if wall_budget_error else {}),
                "reason": str(intent.payload.get("reason") or "")[:200],
            },
        )

    async def _handle_prune_branch(self, source: str, intent: Intent) -> None:
        """Prune an action family and cancel its in-flight tasks."""
        family = intent.payload["family"]
        reason = str(intent.payload.get("reason") or "prune_branch")
        scope = str(intent.payload.get("scope") or PRUNE_BRANCH_SCOPE_FAMILY).strip()
        drain_only = scope == PRUNE_BRANCH_SCOPE_QUEUED
        if not drain_only and self.shared_state.add_pruned_family(family):
            self.shared_state.save(self.session_dir)
        if drain_only and family == "baseline":
            cancelled = await self._drain_queued_baselines(reason=reason)
        else:
            cancelled = await self.tasks.cancel_family([family], reason=reason)
        # A pruned explore family can take the GEAK 2b rebench with it; settle the slot so KERNEL is not held open
        # waiting on a task that will never run.
        if cancelled:
            from ..phases.geak_rebench import settle_dangling_geak_pending

            try:
                if await settle_dangling_geak_pending(
                    self.tasks,
                    self.shared_state,
                    reason=f"prune_branch:{family}",
                ):
                    self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — prune must not fail on bookkeeping
                log.exception("prune_branch: GEAK pending settle failed")
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
        """Process ``escalate_strategy_change``: broadcast strategy_change, act on closed-vocab hints, drop unknown hints."""
        payload = dict(intent.payload or {})
        # Always emit the broadcast first.
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "strategy_change",
                payload,
                priority=0,
            )
        )
        from ..phases.machine_state import (
            ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
            ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
            ESCALATE_HINT_SKIP_TO_CLOSE,
            PHASE_FRAMEWORK_AGENT,
            PHASE_KERNEL_AGENT,
            apply_escalate_budget_bump,
            is_valid_escalate_hint,
        )

        hint = str(payload.get("next_action_hint") or "").strip()
        if not hint or not is_valid_escalate_hint(hint):
            return
        # Pre-enablement close guard: drop a premature ``skip_to_close`` while the model is not yet runnable and let
        # the enablement loop continue.
        if hint == ESCALATE_HINT_SKIP_TO_CLOSE and self.shared_state.enablement_close_guard_active():
            log.info(
                "escalate_strategy_change: dropping premature skip_to_close from %s "
                "(pre-enablement: baseline not established; enablement loop still active)",
                source,
            )
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "observation",
                    {
                        "kind": "enablement_skip_to_close_suppressed",
                        "source": source,
                        "phase": (self.shared_state.phase or ""),
                    },
                )
            )
            return
        # extend_*_budget mutates phase_budget_pct directly.
        now_ts = datetime.now(timezone.utc).isoformat()
        if hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct,
                phase=PHASE_FRAMEWORK_AGENT,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        if hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct,
                phase=PHASE_KERNEL_AGENT,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # skip_to_kernel / skip_to_close are deferred; next compute_next_phase picks them up.
        self.shared_state.set_pending_escalate_hint(hint)
        self.shared_state.save(self.session_dir)

    async def _handle_send_message(self, source: str, intent: Intent) -> None:
        """Publish a free-form message onto the bus."""
        topic = intent.payload.get("topic", "observation")
        if topic not in TOPIC_ALLOWLIST:
            # Soft-degrade unknown topic.
            topic = "observation"
        to_agent = intent.payload.get("to") or "*"
        await self.bus.append_and_seq(
            Message.new(
                source,
                to_agent,
                topic,
                {k: v for k, v in intent.payload.items() if k != "to"},
            )
        )
        if str(to_agent).startswith(SPECIALIST_FROM_AGENT_PREFIX):
            self._deliver_specialist_inbox(source, str(to_agent), intent.payload)

    def _deliver_specialist_inbox(self, source: str, to_agent: str, payload: dict[str, Any]) -> None:
        """Append a message to a running specialist's workspace inbox."""
        task_id = to_agent[len(SPECIALIST_FROM_AGENT_PREFIX) :].strip()
        if not task_id:
            return
        try:
            workspace = runs_dir(self.session_dir, "specialist", task_id)
            workspace.mkdir(parents=True, exist_ok=True)
            # The prompt advertises the worktree when one exists; match it.
            worktree = workspace / "worktree"
            inbox = (worktree if worktree.is_dir() else workspace) / "inbox.json"
            existing: list[Any] = []
            if inbox.exists():
                loaded = json.loads(inbox.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            existing.append(
                {
                    "from": source,
                    "ts": now_iso(),
                    "body": {k: v for k, v in payload.items() if k not in ("to", "topic")},
                }
            )
            # Keep the last 32 so the file stays prompt-sized.
            tmp = inbox.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing[-32:], indent=2), encoding="utf-8")
            tmp.replace(inbox)
        except Exception:  # noqa: BLE001 — steering is best-effort
            log.exception("failed to deliver inbox message to %s", to_agent)

    async def _handle_alert(self, source: str, intent: Intent) -> None:
        """Broadcast an alert message, prioritized by severity."""
        prio = 0 if intent.payload.get("severity") == "high" else 1
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "alert",
                dict(intent.payload),
                priority=prio,
            )
        )

    async def _handle_update_state(self, source: str, intent: Intent) -> None:
        """Apply agent-requested SharedState changes and report the result."""
        # Apply to persistent SharedState (PolicyGate enforces core-field writes).
        applied = self.shared_state.apply_changes(
            intent.payload["changes"],
            allow_core=False,
        )
        if applied:
            self.shared_state.save(self.session_dir)
        await self.bus.append_and_seq(
            Message.new(
                source,
                "*",
                "observation",
                {
                    "kind": "update_state",
                    "changes": applied,
                    "rejected": sorted(set(intent.payload["changes"]) - set(applied)),
                },
            )
        )
