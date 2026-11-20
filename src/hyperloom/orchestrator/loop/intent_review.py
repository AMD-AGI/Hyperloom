# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Records of the Critic's rulings on proposals and of what the loop did with them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.common.framework_arm import review_row_id, verdict_subject
from .verdicts import CriticRuling


def variant_review_rows(
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


def patch_verdict_subject(pending: Any, pa_params: Mapping[str, Any]) -> str:
    """The key a proposal's Critic verdict is mirrored under as a patch verdict, or ``""`` when none."""
    sid_candidate = ""
    if pending.action_name == "integrate_patch":
        # A pre-screen carries its candidate id at the top level, not in params.
        sid_candidate = verdict_subject({**pa_params, **(pending.payload or {})})
    elif pending.action_name == "specialist":
        # Critic verdict on the specialist proposal counts as the verdict on its patches; task_id is the key.
        sid_candidate = str(pa_params.get("task_id") or "").strip()
    return sid_candidate


def record_critic_review(router: Any, pending: Any, ruling: CriticRuling) -> None:
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


def record_review_outcome(router: Any, pending: Any, **outcome: Any) -> None:
    """Record what the loop did with a ruling, onto the ruling."""
    _record_phase_proposal_outcome(pending, **outcome)

    proposal_id = review_row_id(pending.payload or {}, fallback_msg_id=pending.proposal_msg_id)
    if not proposal_id:
        return
    recorder = router._coord.phase_framework.timeline()
    if recorder is None:
        return
    recorder.record_proposal_review_outcome(proposal_id, **outcome)
