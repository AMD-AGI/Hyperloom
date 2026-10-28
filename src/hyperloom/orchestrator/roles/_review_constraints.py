# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a Critic turn's proposals have to be judged against, stamped onto the judge bundle before it is reasoned over."""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_ENABLEMENT,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    patch_lever_kind,
)


def apply_review_constraints(
    judge_bundle: dict[str, Any],
    *,
    phase: str,
    action_verdict_policy: dict[str, str],
) -> None:
    """Stamp this turn's review constraints onto the judge bundle: policy, phase, lever and the advisory rules."""
    if action_verdict_policy:
        rc = judge_bundle.setdefault("review_constraints", {})
        if not isinstance(rc, dict):
            rc = {}
            judge_bundle["review_constraints"] = rc
        rc["action_verdict_policy"] = dict(action_verdict_policy)
    _inject_phase_constraints(judge_bundle, phase)
    # The lever says what a KEEP has to clear; the phase cannot, since one phase carries several levers.
    _inject_lever_orientation(judge_bundle, _first_proposal(judge_bundle))
    _maybe_inject_cross_domain_constraints(judge_bundle)
    _maybe_inject_quantitative_claim_constraint(judge_bundle)


def _first_proposal(judge_bundle: dict[str, Any]) -> dict[str, Any] | None:
    """The bundle's first proposal, when it holds one shaped like a proposal."""
    proposals = judge_bundle.get("proposals") or []
    if not isinstance(proposals, list) or not proposals:
        return None
    first = proposals[0]
    return first if isinstance(first, dict) else None


def _proposal_scope_literal(proposal: dict[str, Any]) -> str:
    """Read the ``scope`` dial off a proposal (top-level or nested ``params``)."""
    if not isinstance(proposal, dict):
        return ""
    top = proposal.get("scope")
    if isinstance(top, str) and top.strip():
        return top.strip()
    params = proposal.get("params") or {}
    if isinstance(params, dict):
        nested = params.get("scope")
        if isinstance(nested, str):
            return nested.strip()
    return ""


# Per-phase review orientation; only the live phase's entry is injected.
_PHASE_ORIENTATION: dict[str, str] = {
    "PRELUDE": (
        "Typical proposals are `target_analysis` and `baseline`. If something "
        "else slips through (PolicyGate R1 should already have blocked it), "
        "`advise` with a phase hint rather than reject."
    ),
    "ENABLEMENT": (
        "Typical proposals are the enablement lane's `integrate_patch` rounds. "
        "The gate is runnability plus the accuracy floor, not throughput: a patch "
        "that boots the model and holds accuracy is a legitimate KEEP even with "
        "no throughput gain. The before/after benchmark gate does not apply — a "
        "booting baseline does not yet exist. `approve` when the patch clearly "
        "makes the combo runnable; `reject` if it worsens the crash or lowers "
        "accuracy below the floor; `advise` otherwise."
    ),
    "FRAMEWORK_AGENT": (
        "Typical proposals are `explore`, `specialist` and `integrate_patch`. "
        "Specialist-style proposal_set packets arrive as "
        "`propose_action='explore'` with a `variants` array — return one "
        "verdict dict per variant msg_id; missing entries are treated as "
        "`needs_review`. What a KEEP has to clear depends on the lever the "
        "proposal moves, not on the phase — see `review_constraints."
        "lever_orientation` when it is present."
    ),
    "KERNEL_AGENT": (
        "Typical proposals are the KERNEL_AGENT_OWNED_ACTIONS (proxied via "
        "REQUEST) plus auto-managed `profile` / `roofline`. Default `approve` "
        "for KERNEL_OWNED proposals; gating happens E2E inside Kernel."
    ),
    "SWEEP": ("Typical proposal is `sweep`. Mismatches → `advise` with the phase hint."),
    "CLOSE": (
        "Typical proposals are `report` and `session_breakdown`. Both are "
        "archival: they transcribe existing state and introduce no new "
        "measurement, so the before/after gate does not apply."
    ),
}


#: Orientation by the lever a proposal moves, not by phase: one phase carries
#: several levers, so a phase-keyed entry telling the Critic that flat gain is a
#: legitimate KEEP would extend that to configuration search. The deterministic
#: layer routes on payload markers rather than phase; this matches it.
_LEVER_ORIENTATION: dict[str, str] = {
    LEVER_UPSTREAM_PR: (
        "This lands an upstream diff nobody here wrote. Judge whether it is "
        "worth measuring and whether it can be rolled back — the measurement "
        "itself is the executor's gate, not yours."
    ),
    LEVER_ENABLEMENT: (
        "The gate here is runnability plus the accuracy floor, not throughput: "
        "a candidate that boots and holds accuracy is a legitimate KEEP even at "
        "flat gain. Pre-boot, the production evidence cannot exist yet."
    ),
    LEVER_SOURCE_PATCH: (
        "A patch written for this session. It changes the source tree, so "
        "rollback and blast radius carry the weight; throughput is measured "
        "afterwards and is not yours to predict."
    ),
    LEVER_CONFIG: (
        "Server arguments and environment only — nothing on disk changes and a "
        "revert is a non-composition. Judge the reasoning and the accuracy "
        "risk; the cost of being wrong is one bench."
    ),
}


def _inject_lever_orientation(judge_bundle: dict[str, Any], payload: dict[str, Any] | None) -> None:
    """Stamp the orientation for the lever this proposal moves, when known."""
    lever = patch_lever_kind(payload if isinstance(payload, dict) else None)
    if not lever:
        params = (payload or {}).get("params") if isinstance(payload, dict) else None
        lever = patch_lever_kind(params if isinstance(params, dict) else None)
    orientation = _LEVER_ORIENTATION.get(lever)
    if not orientation:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    rc["lever_kind"] = lever
    rc["lever_orientation"] = orientation


def _inject_phase_constraints(judge_bundle: dict[str, Any], phase: str) -> None:
    """Stamp the live phase and its review orientation onto the judge bundle."""
    normalized = (phase or "").strip().upper()
    if normalized not in _PHASE_ORIENTATION:
        return
    judge_bundle["phase"] = normalized
    rc = judge_bundle.setdefault("review_constraints", {})
    rc["phase"] = normalized
    rc["phase_orientation"] = _PHASE_ORIENTATION[normalized]


def _maybe_inject_cross_domain_constraints(judge_bundle: dict[str, Any]) -> None:
    """Set ``review_constraints.cross_domain`` + rule descriptors when any proposal is cross-domain (unified ``scope == 'domains'`` dial)."""
    proposals = judge_bundle.get("proposals") or []
    if not isinstance(proposals, list):
        return
    from ..specialists.patch_safety import (
        SCOPE_DOMAINS_LITERAL,
        cross_domain_rule_descriptors,
    )

    has_cross_domain = any(_proposal_scope_literal(p) == SCOPE_DOMAINS_LITERAL for p in proposals)
    if not has_cross_domain:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    if not isinstance(rc, dict):
        rc = {}
        judge_bundle["review_constraints"] = rc
    rc["cross_domain"] = True
    rc["cross_domain_rules"] = cross_domain_rule_descriptors()


def _maybe_inject_quantitative_claim_constraint(judge_bundle: dict[str, Any]) -> None:
    """Set ``review_constraints.quantitative_claim_rule`` from the enforced list."""
    from ..specialists.patch_safety import (
        advisory_rules_govern,
        quantitative_claim_rule_descriptor,
    )

    proposals = judge_bundle.get("proposals") or []
    if not isinstance(proposals, list):
        return
    governed = any(advisory_rules_govern(str(p.get("action_name") or "")) for p in proposals if isinstance(p, dict))
    if not governed:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    if not isinstance(rc, dict):
        rc = {}
        judge_bundle["review_constraints"] = rc
    rc["quantitative_claim_rule"] = quantitative_claim_rule_descriptor()
