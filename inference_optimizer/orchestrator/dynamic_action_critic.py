"""dynamic_action.MD P4 — Critic-side cross-domain review primitives.

The runner (P3) already validates proposal payload schema and the
numeric-claim guard before the patch reaches the Critic. P4 layers the
**second** defense on the Critic boundary so a regression in P3 (or a
manual proposal_set surgery) cannot silently bypass the §1.2 red lines.

Public surface:

* :data:`CROSS_DOMAIN_RULES` — the three rule descriptors (P4 §4) the
  Critic prompt cites verbatim;
* :data:`CRITIC_VERDICT_FIELDS` — closed envelope for
  ``critic_verdict.json`` (P4 §5.3);
* :class:`CrossDomainPreverdict` — mechanical-check result wrapping a
  ``verdict`` + ``reason_codes`` + ``applied_rules`` triple;
* :func:`classify_proposal_for_critic` — entry point: maps one proposal
  payload to ``(bundle_action_class, review_constraints)`` (P4 §3.1);
* :func:`run_mechanical_cross_domain_checks` — applies the three rules
  plus the §9 #6 / #7 fail-fast guards;
* :func:`build_critic_verdict_envelope` — assembles the on-disk shape;
* :func:`write_critic_verdict` — persists the envelope alongside
  spec.json / seed_kit.json / proposal_set.json.
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
# Rule catalogue (P4 §4)
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
# Closed verdict envelope (P4 §5.3)
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
# P4 §4 rule 2 — sub-agent must talk about *why these have to happen
# together*. Substring containment over a small list of coupling
# keywords is the lightest deterministic check that catches the obvious
# "two unrelated patches glued together" failure mode.
_COUPLING_KEYWORDS: tuple[str, ...] = (
    "coupl", "联动", "interaction", "depend", "依赖", "interface",
    "interplay", "joint", "together", "synerg", "interact",
    "trigger", "follow",
)

# Words that signal "this proposal acknowledges a potential downside"
# (P4 §4 rule 2 second half). A single hit is enough; the rule cap
# stays soft so legitimate proposals with diverse phrasing pass.
_SIDE_EFFECT_KEYWORDS: tuple[str, ...] = (
    "side effect", "side-effect", "regression", "副作用", "trade-off",
    "tradeoff", "risk", "downside", "may decrease", "may degrade",
    "could degrade", "could regress", "可能下降", "可能退化",
    "may break", "could break",
)

# P4 §4 rule 3 — motivation must explicitly say "no specialist alone
# can do this". A grid-combo justification ("just stack proposals
# from spec A and spec B") is a hard reject.
_MOTIVATION_VALID_KEYWORDS: tuple[str, ...] = (
    "no single specialist", "no single domain",
    "single specialist cannot", "single domain cannot",
    "specialist boundary", "single-domain boundary",
    "cross-domain only", "outside specialist scope",
    "无法由单个 specialist", "无法由单 domain",
    "specialist 边界", "跨域专属",
)
_MOTIVATION_INVALID_KEYWORDS: tuple[str, ...] = (
    "just stack", "simple combination of",
    "concatenate two specialist", "concatenat", "merge of specialist",
    "拼接", "拼合", "组合 specialist a 和 specialist b",
)

# Re-export the runner's numeric-claim patterns so the Critic boundary
# applies the same regex even if the runner is bypassed.
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
    """Pre-LLM verdict produced by mechanical checks.

    When ``verdict == "approve"`` the LLM-critic still runs (its
    job is to apply the §6 patch_landing four-checklist + any
    higher-level judgement). The mechanical layer is a *floor* — it
    can only down-rank the LLM's verdict, never up-rank it.
    """

    verdict: str
    reason_codes: list[str] = field(default_factory=list)
    reviewer_notes: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)
    cross_domain_flag: bool = True

    def is_blocking(self) -> bool:
        return self.verdict in {"reject", "revise"}


# ---------------------------------------------------------------------------
# Classifier (P4 §3.1)
# ---------------------------------------------------------------------------
def classify_proposal_for_critic(
    proposal_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Decide how the Critic should treat one proposal.

    Returns ``(bundle_action_class, review_constraints)``:

    * ``bundle_action_class`` is always ``"patch_landing"`` (D-B
      decision — never branches by source).
    * ``review_constraints`` carries ``cross_domain=True`` plus the
      rule descriptors when the proposal's ``provenance`` literal is
      ``"dynamic"``; otherwise the dict is empty (specialist patches
      are unaffected).
    """
    provenance = str(
        (proposal_payload or {}).get("provenance") or "",
    ).strip().lower()
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
# Mechanical checks (P4 §9 #2 / #3 / #4 / #6 / #7)
# ---------------------------------------------------------------------------
def run_mechanical_cross_domain_checks(
    proposal_payload: dict[str, Any],
    *,
    spec_scope_domains: list[str],
) -> CrossDomainPreverdict:
    """Apply the three §4 rules + the §9 #6/#7 fail-fast guards.

    Notes
    -----
    * The mechanical layer is intentionally conservative — it only
      *blocks* on clear-cut violations and falls through to APPROVE
      otherwise so the LLM-critic's patch_landing checklist still
      gets the final say.
    * ``applied_rules`` records the full ID list (passed + failed)
      so the audit can reconstruct which rules ran.
    """
    pre = CrossDomainPreverdict(verdict="approve", cross_domain_flag=True)

    # §9 #6 — fail-fast provenance literal check (last line of
    # defence; P1 IR-4 + P3 runner schema were the first two).
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

    # §9 #7 — forbidden quantitative fields (mirrors the P3 runner
    # validator; second defence at the critic boundary).
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
    # The spec is the truth set; we still record the proposal's own
    # scope for the rule-1 / rule-2 / rule-3 mentions check so a
    # tighter scope inside the proposal still gets fairly evaluated.
    truth_set = list(spec_scope_domains or scope_domains)
    rationale = str(proposal_payload.get("cross_domain_rationale") or "")
    qualitative = str(
        proposal_payload.get("expected_qualitative_argument") or "",
    )

    # P3 runner validator already enforces numeric_claims at sub-agent
    # boundary; defence in depth at critic boundary too (§9 #7 cousin
    # for the qualitative argument).
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

    # P4 §4 rule 1 — every truth-set domain must show up in the
    # rationale text. Missing → REVISE (sub-agent can patch the
    # rationale without re-emitting the whole proposal in v2; in v1
    # REVISE is handled identically to REJECT per §5.2).
    missing = _missing_domain_mentions(rationale, truth_set)
    pre.applied_rules.append("rationale_per_domain")
    if missing:
        pre.verdict = "revise"
        pre.reason_codes.append("cross_domain_rationale_incomplete")
        pre.reviewer_notes.append(
            f"rationale missing per-domain coverage for: {missing!r}",
        )
        return pre

    # P4 §4 rule 2 — coupling + side-effect keywords.
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

    # P4 §4 rule 3 — motivation gap valid. Hard reject when the
    # rationale explicitly says "stack/concatenate specialist
    # output"; soft pass otherwise (LLM-critic has the final say on
    # the harder cases).
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
# Verdict envelope writer (P4 §5.3)
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
