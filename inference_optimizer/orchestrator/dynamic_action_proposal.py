"""Proposal validator + lifecycle / terminal-state enums.

The runner calls :func:`validate_proposal` on every ``emit_proposal``
tool call; only validated payloads land in ``proposal_set.json``.

Terminal-state and lifecycle enums anchor the canonical labels read
by the state machine. Reject reasons are stable strings — the runner
echoes them into the journal so the sub-agent can iterate within
:data:`MAX_PROPOSAL_REJECTS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Runner terminal states
# ---------------------------------------------------------------------------
class DynamicRunnerTerminalState(str, Enum):
    """Final outcome label written to the per-dispatch summary."""

    COMPLETED = "COMPLETED"
    COMPLETED_EMPTY = "COMPLETED_EMPTY"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


# Closed reason vocabulary per terminal state.
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
# Lifecycle states
# ---------------------------------------------------------------------------
class DynamicActionStatus(str, Enum):
    """``SharedState.dynamic_actions[dyn_id].status`` vocabulary.

    * Non-terminal: ``DISPATCHED`` → ``SUB_AGENT_RUNNING`` →
      ``SUB_AGENT_DONE`` → ``AWAITING_CRITIC`` → ``INTEGRATING``;
    * Terminal: ``COMPLETED_EMPTY`` / ``TIMED_OUT`` / ``FAILED`` /
      ``CRITIC_REJECTED`` / ``INTEGRATE_FAILED`` / ``KEPT`` /
      ``REVERTED``;
    * Special terminal: ``ABANDONED`` set by the resume sweep.
    """

    DISPATCHED = "DISPATCHED"
    SUB_AGENT_RUNNING = "SUB_AGENT_RUNNING"
    SUB_AGENT_DONE = "SUB_AGENT_DONE"
    AWAITING_CRITIC = "AWAITING_CRITIC"
    INTEGRATING = "INTEGRATING"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"
    COMPLETED_EMPTY = "COMPLETED_EMPTY"
    CRITIC_REJECTED = "CRITIC_REJECTED"
    INTEGRATE_FAILED = "INTEGRATE_FAILED"
    REVERTED = "REVERTED"
    KEPT = "KEPT"
    ABANDONED = "ABANDONED"


# Map runner terminal state → initial lifecycle status. COMPLETED
# goes straight to ``AWAITING_CRITIC`` because the runner-done →
# critic-bound step is synchronous.
RUNNER_STATE_TO_STATUS: dict[DynamicRunnerTerminalState, DynamicActionStatus] = {
    DynamicRunnerTerminalState.COMPLETED: DynamicActionStatus.AWAITING_CRITIC,
    DynamicRunnerTerminalState.COMPLETED_EMPTY: DynamicActionStatus.COMPLETED_EMPTY,
    DynamicRunnerTerminalState.TIMED_OUT: DynamicActionStatus.TIMED_OUT,
    DynamicRunnerTerminalState.FAILED: DynamicActionStatus.FAILED,
    DynamicRunnerTerminalState.ABANDONED: DynamicActionStatus.ABANDONED,
}


TERMINAL_LIFECYCLE_STATUSES: frozenset[DynamicActionStatus] = frozenset({
    DynamicActionStatus.TIMED_OUT,
    DynamicActionStatus.FAILED,
    DynamicActionStatus.COMPLETED_EMPTY,
    DynamicActionStatus.CRITIC_REJECTED,
    DynamicActionStatus.INTEGRATE_FAILED,
    DynamicActionStatus.REVERTED,
    DynamicActionStatus.KEPT,
    DynamicActionStatus.ABANDONED,
})


# ---------------------------------------------------------------------------
# Transition table — Coordinator-only writes; terminal states are sinks.
# ---------------------------------------------------------------------------
ALLOWED_TRANSITIONS: dict[DynamicActionStatus, frozenset[DynamicActionStatus]] = {
    DynamicActionStatus.DISPATCHED: frozenset({
        DynamicActionStatus.SUB_AGENT_RUNNING,
        DynamicActionStatus.TIMED_OUT,
        DynamicActionStatus.FAILED,
        DynamicActionStatus.ABANDONED,
    }),
    DynamicActionStatus.SUB_AGENT_RUNNING: frozenset({
        DynamicActionStatus.SUB_AGENT_DONE,
        DynamicActionStatus.COMPLETED_EMPTY,
        DynamicActionStatus.TIMED_OUT,
        DynamicActionStatus.FAILED,
        DynamicActionStatus.ABANDONED,
    }),
    DynamicActionStatus.SUB_AGENT_DONE: frozenset({
        DynamicActionStatus.AWAITING_CRITIC,
        DynamicActionStatus.CRITIC_REJECTED,
        DynamicActionStatus.ABANDONED,
    }),
    DynamicActionStatus.AWAITING_CRITIC: frozenset({
        DynamicActionStatus.INTEGRATING,
        DynamicActionStatus.CRITIC_REJECTED,
        DynamicActionStatus.ABANDONED,
    }),
    DynamicActionStatus.INTEGRATING: frozenset({
        DynamicActionStatus.KEPT,
        DynamicActionStatus.REVERTED,
        DynamicActionStatus.INTEGRATE_FAILED,
        DynamicActionStatus.ABANDONED,
    }),
}
# Terminal states implicitly map to an empty allowed-set (locked).
for _terminal in TERMINAL_LIFECYCLE_STATUSES:
    ALLOWED_TRANSITIONS.setdefault(_terminal, frozenset())


def can_transition(
    from_state: DynamicActionStatus | str | None,
    to_state: DynamicActionStatus | str,
) -> bool:
    """Return True iff ``from_state → to_state`` is permitted.

    Missing source (no prior summary) is treated as creation; only
    ``DISPATCHED`` is a legal first state. Same-status re-writes are
    idempotent.

    Args:
        from_state (DynamicActionStatus | str | None): Current status,
            or ``None``/empty for a first write.
        to_state (DynamicActionStatus | str): Target status.

    Returns:
        bool: ``True`` when the transition is permitted.
    """
    target = (
        to_state if isinstance(to_state, DynamicActionStatus)
        else DynamicActionStatus(str(to_state))
    )
    if from_state is None or from_state == "":
        return target == DynamicActionStatus.DISPATCHED
    src = (
        from_state if isinstance(from_state, DynamicActionStatus)
        else DynamicActionStatus(str(from_state))
    )
    if src == target:
        return True
    allowed = ALLOWED_TRANSITIONS.get(src, frozenset())
    return target in allowed


# ---------------------------------------------------------------------------
# Prompt-friendly flattened ``last_outcome`` label per status.
# ---------------------------------------------------------------------------
LAST_OUTCOME_BY_STATUS: dict[DynamicActionStatus, str] = {
    DynamicActionStatus.DISPATCHED: "running",
    DynamicActionStatus.SUB_AGENT_RUNNING: "running",
    DynamicActionStatus.AWAITING_CRITIC: "awaiting_review",
    DynamicActionStatus.SUB_AGENT_DONE: "awaiting_review",
    DynamicActionStatus.INTEGRATING: "evaluating",
    DynamicActionStatus.COMPLETED_EMPTY: "empty",
    DynamicActionStatus.TIMED_OUT: "timeout",
    DynamicActionStatus.FAILED: "failed",
    DynamicActionStatus.CRITIC_REJECTED: "rejected",
    DynamicActionStatus.INTEGRATE_FAILED: "apply_failed",
    DynamicActionStatus.KEPT: "success",
    DynamicActionStatus.REVERTED: "no_gain",
    DynamicActionStatus.ABANDONED: "abandoned",
}


# ---------------------------------------------------------------------------
# Closed field set the orchestration prompt's dynamic_action section
# renders. On-disk summaries may carry extra audit fields; only the
# names below leak into the prompt projection.
#
# ``cumulative_gain`` currently holds the single integrate
# ``delta_pct`` (one-shot dispatch); the name reserves space for a
# multi-run total if sub-agent re-dispatch lands later.
# ---------------------------------------------------------------------------
SUMMARY_PROMPT_FIELDS: frozenset[str] = frozenset({
    "dyn_id",
    "status",
    "dispatched_at",
    "round_index",
    "scope_domains",
    "motivation_gap_short",
    "verdict",
    "cumulative_gain",
    "last_outcome",
    "artifact_path",
    "updated_at",
})

# Hard cap on the prompt-facing motivation excerpt.
MOTIVATION_GAP_SHORT_MAX_CHARS: int = 200


# ---------------------------------------------------------------------------
# Proposal schema
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

# Quantitative / priority fields rejected outright on any proposal.
FORBIDDEN_PROPOSAL_FIELDS: frozenset[str] = frozenset({
    "expected_gain",
    "expected_gain_pct",
    "bench_evidence",
    "confidence",
    "score",
    "rank",
    "force_provenance",
})

# Provenance is a strict literal, no composite form.
EXPECTED_PROVENANCE: str = "dynamic"

# Upper bound on ``name`` length; keeps the synthesised patch filename
# (``001_<name>.patch``) well under the filesystem limit.
MAX_PROPOSAL_NAME_CHARS: int = 80

# Cap on the number of entries the runner accepts per dispatch.
MAX_PROPOSAL_SET_LEN: int = 1

# Consecutive validation rejects before the runner gives up.
MAX_PROPOSAL_REJECTS: int = 2

# Catches obvious numeric speedup claims smuggled into the qualitative
# argument (``X%`` / ``X.Yx`` / ``X.Y ms`` / ``speedup of X``).
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

# Git emits these metadata lines on every diff; hand-crafted patches
# typically do not. Stripping them keeps the cumulative-diff check on
# hunk semantics, not metadata.
_DIFF_NORMALISE_DROP_LINES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+(?: \d+)?$", re.M),
    re.compile(r"^diff --git a/.* b/.*$", re.M),
    re.compile(r"^new file mode \d+$", re.M),
    re.compile(r"^deleted file mode \d+$", re.M),
    re.compile(r"^similarity index \d+%$", re.M),
)


def _normalise_diff_for_compare(text: str) -> str:
    """Return a diff's canonical form for byte comparison.

    Strips git-only metadata lines, trailing whitespace, and blank
    lines so two diffs can be compared on hunk semantics.

    Args:
        text (str): Raw diff text to normalise.

    Returns:
        str: The normalised diff body.
    """
    body = text or ""
    for pat in _DIFF_NORMALISE_DROP_LINES:
        body = pat.sub("", body)
    return "\n".join(
        line.rstrip() for line in body.splitlines() if line.strip()
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
        """Render the validation result for the runner journal.

        Returns:
            dict[str, Any]: The ``ok`` flag plus ``reason`` and
            ``detail`` strings.
        """
        return {
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
        }


def _numeric_claims(text: str) -> list[str]:
    """Return numeric speedup-claim substrings found in ``text``.

    Args:
        text (str): Text to scan with the numeric-claim regexes.

    Returns:
        list[str]: Every matched numeric-claim substring; empty when
        none match.
    """
    hits: list[str] = []
    for pattern in _NUMERIC_CLAIM_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits


def _coerce_string_list(value: Any) -> list[str]:
    """Coerce a scalar or iterable into a list of strings.

    Args:
        value (Any): ``None``, a string, or an iterable of values.

    Returns:
        list[str]: ``[]`` for ``None``/non-iterables, a single-element
        list for a string, otherwise each item stringified.
    """
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
    worktree_cumulative_diff: str | None = None,
) -> ProposalValidationResult:
    """Apply the closed-schema checks to one proposal payload.

    ``spec_scope_domains`` is the truth set from the dispatch
    ``spec.json``; the proposal's own list must be a subset.

    ``worktree_cumulative_diff`` is ``git diff HEAD`` from the
    sub-agent's worktree captured at ``emit_proposal`` time. When
    non-empty the proposal's ``patch_text`` MUST match it (after
    git-metadata + whitespace normalisation). ``None`` disables the
    check (no worktree / git failure / test fixture).

    Empty ``patch_text`` is handled upstream as the ``COMPLETED_EMPTY``
    signal and is not reached here.

    Args:
        proposal (dict[str, Any]): The ``emit_proposal`` payload.
        spec_scope_domains (list[str]): Authoritative scope domains; the
            proposal's domains must be a subset.
        worktree_cumulative_diff (str | None): ``git diff HEAD`` from the
            worktree; when non-empty the patch must match it. ``None``
            disables the check.

    Returns:
        ProposalValidationResult: ``ok=True`` with the normalised
        payload, or ``ok=False`` with a stable reason code and detail.
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

    name = str(proposal.get("name") or "").strip()
    if len(name) > MAX_PROPOSAL_NAME_CHARS:
        return ProposalValidationResult(
            ok=False, reason="name_too_long",
            detail=f"len={len(name)} max={MAX_PROPOSAL_NAME_CHARS}",
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
    if worktree_cumulative_diff:
        if _normalise_diff_for_compare(patch) != _normalise_diff_for_compare(
            worktree_cumulative_diff,
        ):
            return ProposalValidationResult(
                ok=False,
                reason="patch_text_not_cumulative_diff",
                detail=(
                    "proposal.patch_text does not match the worktree's "
                    "git diff HEAD; emit_proposal MUST carry the "
                    "cumulative diff when the sub-agent has applied "
                    "patches during iteration."
                ),
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
    """Build the ``proposal_set.json`` body.

    ``normalised_proposal=None`` represents the ``COMPLETED_EMPTY``
    signal (sub-agent declared no feasible cross-domain combo).

    Args:
        dyn_id (str): Dynamic-action identifier.
        normalised_proposal (dict[str, Any] | None): Validated proposal,
            or ``None`` for the empty signal.
        journal_path (str): Path to the runner journal recorded in the
            payload.

    Returns:
        dict[str, Any]: The ``proposal_set.json`` body with the proposal
        list and ``empty`` flag.
    """
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
    "ALLOWED_TRANSITIONS",
    "DynamicActionStatus",
    "DynamicRunnerTerminalState",
    "EXPECTED_PROVENANCE",
    "FORBIDDEN_PROPOSAL_FIELDS",
    "LAST_OUTCOME_BY_STATUS",
    "MAX_PROPOSAL_NAME_CHARS",
    "MAX_PROPOSAL_REJECTS",
    "MAX_PROPOSAL_SET_LEN",
    "MOTIVATION_GAP_SHORT_MAX_CHARS",
    "ProposalValidationResult",
    "REQUIRED_PROPOSAL_FIELDS",
    "RUNNER_STATE_TO_STATUS",
    "SUMMARY_PROMPT_FIELDS",
    "TERMINAL_LIFECYCLE_STATUSES",
    "TERMINAL_REASONS",
    "build_proposal_set_payload",
    "can_transition",
    "validate_proposal",
]
