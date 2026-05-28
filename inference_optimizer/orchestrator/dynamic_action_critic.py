"""Critic-side cross-domain review primitives for ``dynamic_action``.

The runner already validates the proposal payload before the patch
reaches the Critic. This module is the second defence layer so a
regression upstream cannot silently bypass the cross-domain red lines.

Public surface:

* :data:`CROSS_DOMAIN_RULES` — the three rule descriptors the Critic
  prompt cites verbatim;
* :data:`CRITIC_VERDICT_FIELDS` — closed envelope for
  ``critic_verdict.json``;
* :class:`CrossDomainPreverdict` — mechanical-check result;
* :func:`classify_proposal_for_critic` — maps a proposal to
  ``(bundle_action_class, review_constraints)``;
* :func:`run_mechanical_cross_domain_checks` — applies the three rules
  plus the fail-fast provenance + forbidden-field guards;
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
from typing import Any, Iterable

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
# Mechanical-check primitives
# ---------------------------------------------------------------------------
# Coupling keywords — "why must these changes happen together?".
_COUPLING_KEYWORDS: tuple[str, ...] = (
    "coupl", "联动", "interaction", "depend", "依赖", "interface",
    "interplay", "joint", "together", "synerg", "interact",
    "trigger", "follow",
)

# Side-effect keywords — "what could go wrong?".
_SIDE_EFFECT_KEYWORDS: tuple[str, ...] = (
    "side effect", "side-effect", "regression", "副作用", "trade-off",
    "tradeoff", "risk", "downside", "may decrease", "may degrade",
    "could degrade", "could regress", "可能下降", "可能退化",
    "may break", "could break",
)

# Phrases that justify a cross-domain dispatch (no single specialist
# could surface it) — informational; the negative set drives the hard
# reject.
_MOTIVATION_VALID_KEYWORDS: tuple[str, ...] = (
    "no single specialist", "no single domain",
    "single specialist cannot", "single domain cannot",
    "specialist boundary", "single-domain boundary",
    "cross-domain only", "outside specialist scope",
    "无法由单个 specialist", "无法由单 domain",
    "specialist 边界", "跨域专属",
)
# Phrases that signal a grid-combo masquerading as dynamic_action.
_MOTIVATION_INVALID_KEYWORDS: tuple[str, ...] = (
    "just stack", "simple combination of",
    "concatenate two specialist", "concatenat", "merge of specialist",
    "拼接", "拼合", "组合 specialist a 和 specialist b",
)

# Mirrors the runner's numeric-claim regex so the Critic boundary
# still rejects smuggled numbers if the runner is bypassed.
_NUMERIC_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|us|tok/s|qps|tps)\b", re.I),
    re.compile(r"\bspeedup\s*(?:of|=)?\s*\d", re.I),
)


def _has_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    lower = (text or "").lower()
    return any(k in lower for k in keywords)


def _missing_domain_mentions(text: str, scope_domains: list[str]) -> list[str]:
    lower = (text or "").lower()
    return [d for d in scope_domains if d.lower() not in lower]


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
    """Pre-LLM verdict produced by the mechanical checks.

    When ``verdict == "approve"`` the LLM-critic still runs and may
    down-rank to ``revise`` / ``reject``. The mechanical layer is a
    *floor* — it can only tighten the LLM's verdict, never loosen it.
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
    """Apply the three review rules + the fail-fast guards.

    The mechanical layer is conservative — it blocks on clear-cut
    violations and otherwise falls through to ``approve`` so the
    LLM-critic still gets the final say. ``applied_rules`` records
    the full id list (passed + failed) for audit.
    """
    pre = CrossDomainPreverdict(verdict="approve", cross_domain_flag=True)

    # Provenance literal — the last of the three defence layers.
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

    # Forbidden quantitative fields mirror the runner-side validator.
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

    scope_domains = [
        str(d or "").strip()
        for d in proposal_payload.get("scope_domains") or ()
        if str(d or "").strip()
    ]
    # Spec is the truth set; proposal's own scope is used only to keep
    # a tighter declared scope fairly evaluated.
    truth_set = list(spec_scope_domains or scope_domains)
    rationale = str(proposal_payload.get("cross_domain_rationale") or "")
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

    # rationale_per_domain — every truth-set domain must appear in
    # the rationale text.
    missing = _missing_domain_mentions(rationale, truth_set)
    pre.applied_rules.append("rationale_per_domain")
    if missing:
        pre.verdict = "revise"
        pre.reason_codes.append("cross_domain_rationale_incomplete")
        pre.reviewer_notes.append(
            f"rationale missing per-domain coverage for: {missing!r}",
        )
        return pre

    # coupling_and_side_effects — rationale must mention both why
    # changes happen together and one potential side effect.
    pre.applied_rules.append("coupling_and_side_effects")
    has_coupling = _has_any_keyword(rationale, _COUPLING_KEYWORDS)
    has_side_effect = _has_any_keyword(rationale, _SIDE_EFFECT_KEYWORDS)
    if not (has_coupling and has_side_effect):
        pre.verdict = "revise"
        pre.reason_codes.append("cross_domain_coupling_unspecified")
        missing_parts: list[str] = []
        if not has_coupling:
            missing_parts.append("coupling")
        if not has_side_effect:
            missing_parts.append("side_effect")
        pre.reviewer_notes.append(
            f"rationale missing: {missing_parts!r}",
        )
        return pre

    # motivation_gap_valid — hard reject when the rationale describes
    # a grid-combo masquerading as a dynamic_action.
    pre.applied_rules.append("motivation_gap_valid")
    if _has_any_keyword(rationale, _MOTIVATION_INVALID_KEYWORDS):
        pre.verdict = "reject"
        pre.reason_codes.append("cross_domain_motivation_invalid")
        pre.reviewer_notes.append(
            "rationale describes a grid-combo (specialist A + B "
            "stacking), which belongs in explore.params.grid, not "
            "in a dynamic_action dispatch.",
        )
        return pre

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
