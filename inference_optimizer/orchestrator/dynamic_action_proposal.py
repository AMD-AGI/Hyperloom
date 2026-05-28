"""dynamic_action.MD P3 §5 + §8 — proposal validator + terminal states.

The runner calls :func:`validate_proposal` on every ``emit_proposal``
tool call; only validated payloads land in ``proposal_set.json``.

Terminal-state enum (P3 §8) tags every runner exit so the P6 state
machine reads a single canonical label rather than parsing logs.

Reject reasons are stable strings — the runner echoes them into the
sub_agent_journal so the sub-agent can iterate within the
``MAX_PROPOSAL_REJECTS`` cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Terminal states (P3 §8)
# ---------------------------------------------------------------------------
class DynamicRunnerTerminalState(str, Enum):
    """Final outcome label written to the per-dispatch summary."""

    COMPLETED = "COMPLETED"
    COMPLETED_EMPTY = "COMPLETED_EMPTY"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


# Reason vocabulary attached to each terminal state. Closed enums so the
# P6 state machine cannot grow new reasons by accident.
TERMINAL_REASONS: dict[DynamicRunnerTerminalState, frozenset[str]] = {
    DynamicRunnerTerminalState.COMPLETED: frozenset({"emit_proposal"}),
    DynamicRunnerTerminalState.COMPLETED_EMPTY: frozenset({"emit_empty"}),
    DynamicRunnerTerminalState.TIMED_OUT: frozenset({
        "wall_clock_exhausted",
        "turn_cap_exhausted",
    }),
    DynamicRunnerTerminalState.FAILED: frozenset({
        "proposal_validation_failed",
        "subprocess_crashed",
        "unparsable_output",
        "runner_internal_error",
    }),
    DynamicRunnerTerminalState.ABANDONED: frozenset({"external_kill"}),
}


# ---------------------------------------------------------------------------
# Proposal schema (P3 §5)
# ---------------------------------------------------------------------------
ALLOWED_PROPOSAL_FIELDS: frozenset[str] = frozenset({
    "name",
    "provenance",
    "patch_text",
    "scope_domains",
    "cross_domain_rationale",
    "expected_qualitative_argument",
})

REQUIRED_PROPOSAL_FIELDS: tuple[str, ...] = (
    "name",
    "provenance",
    "patch_text",
    "scope_domains",
    "cross_domain_rationale",
    "expected_qualitative_argument",
)

# P3 §5.2 — explicit denial of quantitative / priority fields. The
# validator rejects any of these even when present with falsy values.
FORBIDDEN_PROPOSAL_FIELDS: frozenset[str] = frozenset({
    "expected_gain",
    "expected_gain_pct",
    "bench_evidence",
    "confidence",
    "score",
    "rank",
    "force_provenance",
})

# P3 §5.1 — provenance is a literal, no composite form.
EXPECTED_PROVENANCE: str = "dynamic"

# Q3 — proposal_set length is hard-capped at 1.
MAX_PROPOSAL_SET_LEN: int = 1

# P3 §5.3 — consecutive validation rejects before the runner FAILs.
MAX_PROPOSAL_REJECTS: int = 2

# P3 §5.3 — numeric-claim regex catches obvious "X%", "X.Yx",
# "X.Y tok/s" patterns the sub-agent might use to smuggle bench
# numbers into the qualitative argument.
_NUMERIC_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|us|tok/s|qps|tps)\b", re.I),
    re.compile(r"\bspeedup\s*(?:of|=)?\s*\d", re.I),
)

# Unified diff sanity: must contain at least one ``@@`` hunk header.
_UNIFIED_DIFF_HUNK_RE: re.Pattern[str] = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M,
)


# ---------------------------------------------------------------------------
# Result envelopes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProposalValidationResult:
    """Outcome of validating one ``emit_proposal`` payload."""

    ok: bool
    normalised: dict[str, Any] | None = None
    reason: str = ""
    detail: str = ""

    def to_journal_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
        }


def _numeric_claims(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in _NUMERIC_CLAIM_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return []


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------
def validate_proposal(
    proposal: dict[str, Any],
    *,
    spec_scope_domains: list[str],
) -> ProposalValidationResult:
    """Apply the P3 §5.3 checks to one proposal payload.

    ``spec_scope_domains`` is the scope_domains list captured in the
    dispatch ``spec.json`` (the canonical truth set; the proposal's
    own list must be a subset).

    Empty payload + empty patch is **not** treated here — the runner
    interprets ``emit_proposal`` with ``patch_text == ""`` as the
    COMPLETED_EMPTY signal before calling the validator.
    """
    if not isinstance(proposal, dict):
        return ProposalValidationResult(
            ok=False, reason="payload_must_be_dict",
        )

    keys = set(proposal.keys())
    forbidden = sorted(keys & FORBIDDEN_PROPOSAL_FIELDS)
    if forbidden:
        return ProposalValidationResult(
            ok=False, reason="forbidden_field_present",
            detail=f"fields={forbidden!r}",
        )
    extra = sorted(keys - ALLOWED_PROPOSAL_FIELDS)
    if extra:
        return ProposalValidationResult(
            ok=False, reason="unknown_field_present",
            detail=f"fields={extra!r}",
        )
    missing = [f for f in REQUIRED_PROPOSAL_FIELDS if f not in proposal]
    if missing:
        return ProposalValidationResult(
            ok=False, reason="missing_required_field",
            detail=f"fields={missing!r}",
        )

    provenance = str(proposal.get("provenance") or "").strip()
    if provenance != EXPECTED_PROVENANCE:
        return ProposalValidationResult(
            ok=False, reason="provenance_must_be_dynamic",
            detail=f"got={provenance!r}",
        )

    scope = [s.strip() for s in _coerce_string_list(
        proposal.get("scope_domains"),
    ) if s and str(s).strip()]
    if not scope:
        return ProposalValidationResult(
            ok=False, reason="scope_domains_empty",
        )
    spec_set = {str(s).strip() for s in spec_scope_domains or () if s}
    extra_domains = sorted(set(scope) - spec_set)
    if extra_domains:
        return ProposalValidationResult(
            ok=False, reason="scope_domains_not_subset",
            detail=(
                f"extra={extra_domains!r}; spec={sorted(spec_set)!r}"
            ),
        )

    patch = str(proposal.get("patch_text") or "")
    if not patch.strip():
        return ProposalValidationResult(
            ok=False, reason="patch_text_empty",
        )
    if not _UNIFIED_DIFF_HUNK_RE.search(patch):
        return ProposalValidationResult(
            ok=False, reason="patch_text_not_unified_diff",
        )

    rationale = str(proposal.get("cross_domain_rationale") or "")
    if not rationale.strip():
        return ProposalValidationResult(
            ok=False, reason="cross_domain_rationale_empty",
        )
    missing_mentions = [
        d for d in scope
        if d.lower() not in rationale.lower()
    ]
    if missing_mentions:
        return ProposalValidationResult(
            ok=False, reason="cross_domain_rationale_missing_domain_mention",
            detail=f"missing={missing_mentions!r}",
        )

    qualitative = str(proposal.get("expected_qualitative_argument") or "")
    if not qualitative.strip():
        return ProposalValidationResult(
            ok=False, reason="expected_qualitative_argument_empty",
        )
    numeric_hits = _numeric_claims(qualitative)
    if numeric_hits:
        return ProposalValidationResult(
            ok=False, reason="numeric_claim_in_qualitative_argument",
            detail=f"hits={numeric_hits!r}",
        )

    normalised = {
        "name": str(proposal["name"]).strip(),
        "provenance": EXPECTED_PROVENANCE,
        "patch_text": patch,
        "scope_domains": scope,
        "cross_domain_rationale": rationale.strip(),
        "expected_qualitative_argument": qualitative.strip(),
    }
    return ProposalValidationResult(ok=True, normalised=normalised)


def build_proposal_set_payload(
    *, dyn_id: str, normalised_proposal: dict[str, Any] | None,
    journal_path: str,
) -> dict[str, Any]:
    """Build the ``proposal_set.json`` body in the schema P5 expects.

    ``normalised_proposal=None`` represents the COMPLETED_EMPTY signal
    (sub-agent declared no feasible cross-domain combo). The
    specialist-equivalent empty path consumes this verbatim."""
    if normalised_proposal is None:
        return {
            "dyn_id": str(dyn_id),
            "proposal_set": [],
            "empty": True,
            "journal_path": journal_path,
        }
    return {
        "dyn_id": str(dyn_id),
        "proposal_set": [normalised_proposal],
        "empty": False,
        "journal_path": journal_path,
    }


__all__ = [
    "ALLOWED_PROPOSAL_FIELDS",
    "DynamicRunnerTerminalState",
    "EXPECTED_PROVENANCE",
    "FORBIDDEN_PROPOSAL_FIELDS",
    "MAX_PROPOSAL_REJECTS",
    "MAX_PROPOSAL_SET_LEN",
    "ProposalValidationResult",
    "REQUIRED_PROPOSAL_FIELDS",
    "TERMINAL_REASONS",
    "build_proposal_set_payload",
    "validate_proposal",
]
