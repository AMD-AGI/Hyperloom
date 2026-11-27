# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Records of the Critic's rulings on proposals and of what the loop did with them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.common.framework_arm import review_row_id, verdict_subject
from .verdicts import CriticRuling


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
            verdict=ruling.verdict,
            source=str(fields.get("source") or ""),
            reasoning=str(ruling.reasoning or ""),
            confidence=fields.get("confidence"),
            advisory=ruling.advisory,
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
    A rejected proposal is settled here, since it never reaches a bench and
    nothing downstream would otherwise resolve it.
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

    verdict = ruling.verdict
    reason = str(ruling.reasoning or "")
    fields = ruling.payload or {}
    grounded = str(fields.get("source") or REVIEWER_CRITIC) == REVIEWER_CRITIC
    recorder.record_proposal_review(
        proposal_id,
        CriticReview(
            verdict=verdict,
            reviewer=REVIEWER_CRITIC if grounded else REVIEWER_CRITIC_UNAVAILABLE,
            reason=reason,
            confidence=fields.get("confidence"),
            advisory=ruling.advisory,
        ),
    )
    recorder.record_proposal_step(
        proposal_id,
        step=STEP_REVIEWED,
        outcome=verdict,
        reason=reason,
    )
    if verdict == "reject":
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
