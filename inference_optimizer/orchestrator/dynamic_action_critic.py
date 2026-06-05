"""Critic-side review primitives for ``dynamic_action``.

The runner already validates the proposal payload before the patch
reaches the Critic. This module is the safety-invariant second defence
layer (provenance literal, forbidden quantitative fields, numeric-claim
regex inside the qualitative argument) so a regression upstream cannot
smuggle a forged proposal past the Critic boundary. Strategy-level
cross-domain quality (rationale completeness, coupling/side-effect
articulation, motivation gap) is the LLM Critic's call and is offered
to it as advisory descriptors via ``CROSS_DOMAIN_RULES``; the
mechanical layer no longer enforces those rules.

Public surface:

* :data:`CROSS_DOMAIN_RULES` — advisory rule descriptors injected into
  the Critic prompt (LLM judgement guide, not mechanically enforced);
* :data:`CRITIC_VERDICT_FIELDS` — closed envelope for
  ``critic_verdict.json``;
* :class:`CrossDomainPreverdict` — safety-invariant check result;
* :func:`classify_proposal_for_critic` — maps a proposal to
  ``(bundle_action_class, review_constraints)``;
* :func:`run_mechanical_cross_domain_checks` — fail-fast provenance +
  forbidden-field + numeric-claim guards (safety invariants only);
* :func:`build_critic_verdict_envelope` — assembles the on-disk shape;
* :func:`write_critic_verdict` — persists the envelope under the
  dispatch artefact dir.
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


# ---------------------------------------------------------------------------
# Rule catalogue
# ---------------------------------------------------------------------------
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
            "Proposal MUST give an independent rationale for each "
            "domain listed in scope_domains — why this change is "
            "necessary within that domain's boundary."
        ),
        failure_verdict="revise",
        failure_reason_code="cross_domain_rationale_incomplete",
    ),
    CrossDomainRule(
        rule_id="coupling_and_side_effects",
        description=(
            "Proposal MUST name the cross-domain coupling points "
            "(why these changes must happen together) AND at least "
            "one potential side effect of the combination."
        ),
        failure_verdict="revise",
        failure_reason_code="cross_domain_coupling_unspecified",
    ),
    CrossDomainRule(
        rule_id="motivation_gap_valid",
        description=(
            "Proposal MUST show that no single specialist could "
            "surface this combination within its own-domain prompt. "
            "A simple specialist-A + specialist-B concatenation is "
            "a grid combo (explore.params.grid), not a dynamic "
            "action; reject when the motivation degenerates."
        ),
        failure_verdict="reject",
        failure_reason_code="cross_domain_motivation_invalid",
    ),
)


# ---------------------------------------------------------------------------
# Closed verdict envelope
# ---------------------------------------------------------------------------
CRITIC_VERDICT_FIELDS: frozenset[str] = frozenset({
    "dyn_id",
    "verdict",
    "reason_codes",
    "reviewer_notes",
    "applied_rules",
    "cross_domain_flag",
})

ALLOWED_VERDICTS: frozenset[str] = frozenset({"approve", "reject", "revise"})


# ---------------------------------------------------------------------------
# Mechanical-check primitives (safety invariants only)
# ---------------------------------------------------------------------------
# Mirrors the runner's numeric-claim regex so the Critic boundary
# still rejects smuggled numbers if the runner is bypassed.
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


# ---------------------------------------------------------------------------
# Pre-verdict envelope (mechanical checks)
# ---------------------------------------------------------------------------
@dataclass
class CrossDomainPreverdict:
    """Pre-LLM verdict produced by the safety-invariant checks.

    When ``verdict == "approve"`` the LLM-critic runs and is the sole
    authority on the strategy verdict (approve / revise / reject). The
    mechanical layer only short-circuits on a clear safety-invariant
    violation (forged provenance, forbidden field, smuggled numeric
    claim) — in that case it sets ``verdict == "reject"`` and the
    Critic call is skipped. ``revise`` is no longer emitted from this
    layer; it was previously raised by strategy keyword checks that
    have been delegated to the LLM Critic, but ``is_blocking`` still
    treats it as blocking for any caller that hand-builds a preverdict.
    """

    verdict: str
    reason_codes: list[str] = field(default_factory=list)
    reviewer_notes: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)
    cross_domain_flag: bool = True

    def is_blocking(self) -> bool:
        return self.verdict in {"reject", "revise"}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def classify_proposal_for_critic(
    proposal_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Decide how the Critic should treat one proposal.

    Returns ``(bundle_action_class, review_constraints)``:

    * ``bundle_action_class`` is always ``"patch_landing"`` (never
      branches by source).
    * ``review_constraints`` carries ``cross_domain=True`` plus the
      rule descriptors when ``provenance == "dynamic"``; otherwise
      the dict is empty (specialist patches are unaffected).

    The ``provenance`` match is strict + case-sensitive so a forged
    ``DYNAMIC`` / ``Dynamic`` cannot slip past this layer.
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


# ---------------------------------------------------------------------------
# Mechanical checks
# ---------------------------------------------------------------------------
def run_mechanical_cross_domain_checks(
    proposal_payload: dict[str, Any],
    *,
    spec_scope_domains: list[str],
) -> CrossDomainPreverdict:
    """Apply the fail-fast safety guards (provenance + schema).

    Only safety invariants block here: the proposal MUST declare
    ``provenance == 'dynamic'``, MUST NOT carry any field in
    :data:`FORBIDDEN_PROPOSAL_FIELDS`, and the qualitative argument
    MUST NOT smuggle quantitative claims (numeric-claim regex). Every
    other ("strategy") quality dimension — per-domain rationale
    coverage, coupling articulation, side-effect mention, grid-combo
    motivation — is the LLM Critic's call now; the mechanical layer no
    longer rewrites or downgrades those.

    ``applied_rules`` records the safety-guard id list for audit so
    downstream readers can still tell which checks ran.
    ``spec_scope_domains`` is accepted for backward compatibility with
    callers that pass it through; the safety-invariant checks do not
    consult it.
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


# ---------------------------------------------------------------------------
# Verdict envelope writer
# ---------------------------------------------------------------------------
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
    """Persist ``critic_verdict.json`` under the dyn_id artefact dir.

    Caller is expected to have validated the envelope via
    :func:`build_critic_verdict_envelope`. Returns the path written.
    """
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
