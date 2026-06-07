# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic-side review primitives for ``dynamic_action``.

Safety-invariant second defence layer (provenance literal, forbidden
quantitative fields, numeric-claim regex). Strategy-level cross-domain
quality is the LLM Critic's call, offered via ``CROSS_DOMAIN_RULES``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..session_paths import dynamic_action_critic_verdict_path
from .dynamic_action_proposal import (
    EXPECTED_PROVENANCE,
    FORBIDDEN_PROPOSAL_FIELDS,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossDomainRule:
    """One review rule injected into the Critic prompt when
    ``review_constraints.cross_domain == True``."""

    rule_id: str
    description: str
    failure_verdict: str
    failure_reason_code: str


CROSS_DOMAIN_RULES: tuple[CrossDomainRule, ...] = (
    CrossDomainRule(
        rule_id="rationale_per_domain",
        description=(
            "Proposal SHOULD give an independent rationale for each "
            "domain listed in scope_domains — why this change is "
            "necessary within that domain's boundary."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_rationale_incomplete",
    ),
    CrossDomainRule(
        rule_id="coupling_and_side_effects",
        description=(
            "Proposal SHOULD name the cross-domain coupling points "
            "(why these changes must happen together) AND at least "
            "one potential side effect of the combination."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_coupling_unspecified",
    ),
    CrossDomainRule(
        rule_id="motivation_gap_valid",
        description=(
            "Proposal SHOULD show that no single specialist could "
            "surface this combination within its own-domain prompt. "
            "A simple specialist-A + specialist-B concatenation is "
            "a grid combo (explore.params.grid), not a dynamic "
            "action; advise when the motivation degenerates so the "
            "stack rebench + KEEP threshold can adjudicate."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_motivation_invalid",
    ),
)


CRITIC_VERDICT_FIELDS: frozenset[str] = frozenset({
    "dyn_id",
    "verdict",
    "reason_codes",
    "reviewer_notes",
    "applied_rules",
    "cross_domain_flag",
})

ALLOWED_VERDICTS: frozenset[str] = frozenset({
    "approve", "advise", "reject", "revise",
})


# Mirrors the runner's numeric-claim regex so the Critic still rejects smuggled numbers.
_NUMERIC_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|us|tok/s|qps|tps)\b", re.I),
    re.compile(r"\bspeedup\s*(?:of|=)?\s*\d", re.I),
)


def _numeric_hits(text: str) -> list[str]:
    out: list[str] = []
    for pat in _NUMERIC_CLAIM_PATTERNS:
        for m in pat.finditer(text or ""):
            out.append(m.group(0))
    return out


@dataclass
class CrossDomainPreverdict:
    """Pre-LLM verdict produced by the safety-invariant checks.

    On ``approve`` the LLM-critic is sole authority; the mechanical layer only
    short-circuits to ``reject`` on a safety-invariant violation.
    """

    verdict: str
    reason_codes: list[str] = field(default_factory=list)
    reviewer_notes: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)
    cross_domain_flag: bool = True

    def is_blocking(self) -> bool:
        return self.verdict in {"reject", "revise"}


def classify_proposal_for_critic(
    proposal_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Decide how the Critic should treat one proposal.

    Returns ``(bundle_action_class, review_constraints)``; the latter carries
    ``cross_domain=True`` + rules when ``provenance == "dynamic"`` (strict,
    case-sensitive so a forged ``DYNAMIC`` cannot slip past), else empty.
    """
    provenance = str(
        (proposal_payload or {}).get("provenance") or "",
    ).strip()
    bundle_action_class = "patch_landing"
    if provenance != EXPECTED_PROVENANCE:
        return bundle_action_class, {}
    review_constraints: dict[str, Any] = {
        "cross_domain": True,
        "cross_domain_rules": [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "failure_verdict": r.failure_verdict,
                "failure_reason_code": r.failure_reason_code,
            }
            for r in CROSS_DOMAIN_RULES
        ],
    }
    return bundle_action_class, review_constraints


def is_cross_domain_proposal(proposal_payload: dict[str, Any]) -> bool:
    """Convenience predicate; mirrors :func:`classify_proposal_for_critic`."""
    _, rc = classify_proposal_for_critic(proposal_payload)
    return bool(rc.get("cross_domain"))


def run_mechanical_cross_domain_checks(
    proposal_payload: dict[str, Any],
    *,
    spec_scope_domains: list[str],
) -> CrossDomainPreverdict:
    """Apply the fail-fast safety guards (provenance + schema).

    Only safety invariants block: ``provenance == 'dynamic'``, no
    :data:`FORBIDDEN_PROPOSAL_FIELDS`, no numeric claim in the qualitative
    argument. ``spec_scope_domains`` is accepted for compat but not consulted.
    """
    del spec_scope_domains  # safety guards do not consult scope
    pre = CrossDomainPreverdict(verdict="approve", cross_domain_flag=True)

    provenance = str(
        (proposal_payload or {}).get("provenance") or "",
    ).strip()
    if provenance != EXPECTED_PROVENANCE:
        pre.verdict = "reject"
        pre.reason_codes.append("dynamic_provenance_violation")
        pre.applied_rules.append("provenance_literal")
        pre.reviewer_notes.append(
            f"provenance must be {EXPECTED_PROVENANCE!r}, got {provenance!r}",
        )
        return pre
    pre.applied_rules.append("provenance_literal")

    forbidden_present = sorted(
        set((proposal_payload or {}).keys()) & FORBIDDEN_PROPOSAL_FIELDS,
    )
    if forbidden_present:
        pre.verdict = "reject"
        pre.reason_codes.append("dynamic_quantitative_claim_violation")
        pre.applied_rules.append("forbidden_fields")
        pre.reviewer_notes.append(
            f"forbidden field(s) present: {forbidden_present!r}",
        )
        return pre
    pre.applied_rules.append("forbidden_fields")

    qualitative = str(
        proposal_payload.get("expected_qualitative_argument") or "",
    )
    nh = _numeric_hits(qualitative)
    if nh:
        pre.verdict = "reject"
        pre.reason_codes.append("dynamic_quantitative_claim_violation")
        pre.applied_rules.append("qualitative_no_numeric_claims")
        pre.reviewer_notes.append(
            f"numeric-claim regex hits in qualitative argument: {nh!r}",
        )
        return pre
    pre.applied_rules.append("qualitative_no_numeric_claims")

    return pre


def build_critic_verdict_envelope(
    *,
    dyn_id: str,
    verdict: str,
    reason_codes: list[str] | None = None,
    reviewer_notes: list[str] | str | None = None,
    applied_rules: list[str] | None = None,
    cross_domain_flag: bool = True,
) -> dict[str, Any]:
    """Build the on-disk shape with field-set closure enforced."""
    v = (verdict or "").strip().lower()
    if v not in ALLOWED_VERDICTS:
        raise ValueError(
            f"build_critic_verdict_envelope: verdict={verdict!r} not in "
            f"{sorted(ALLOWED_VERDICTS)!r}"
        )
    notes_value: list[str]
    if reviewer_notes is None:
        notes_value = []
    elif isinstance(reviewer_notes, str):
        notes_value = [reviewer_notes] if reviewer_notes.strip() else []
    else:
        notes_value = [str(n) for n in reviewer_notes if str(n).strip()]
    envelope = {
        "dyn_id": str(dyn_id),
        "verdict": v,
        "reason_codes": list(reason_codes or []),
        "reviewer_notes": notes_value,
        "applied_rules": list(applied_rules or []),
        "cross_domain_flag": bool(cross_domain_flag),
    }
    extra = set(envelope.keys()) - CRITIC_VERDICT_FIELDS
    if extra:
        raise ValueError(
            f"critic verdict envelope has unknown fields: {sorted(extra)!r}"
        )
    return envelope


def write_critic_verdict(
    session_dir: Path,
    dyn_id: str,
    envelope: dict[str, Any],
) -> Path:
    """Persist ``critic_verdict.json`` under the dyn_id artefact dir; returns the path."""
    target = dynamic_action_critic_verdict_path(session_dir, dyn_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(envelope, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return target


__all__ = [
    "ALLOWED_VERDICTS",
    "CRITIC_VERDICT_FIELDS",
    "CROSS_DOMAIN_RULES",
    "CrossDomainPreverdict",
    "CrossDomainRule",
    "build_critic_verdict_envelope",
    "classify_proposal_for_critic",
    "is_cross_domain_proposal",
    "run_mechanical_cross_domain_checks",
    "write_critic_verdict",
]
