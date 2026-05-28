"""dynamic_action.MD P2 §5 — seed kit assembler.

Closed field set, deterministic selection rules, hard token cap. The
sub-agent's whole input surface is what this module returns.

Public surface:

* :data:`SEED_KIT_FIELDS` — frozenset of allowed top-level keys; any
  caller-introduced field name not in this set is a P2 §5.2 violation.
* :data:`MAX_SEED_KIT_TOKENS` / per-section caps — DEFAULT values from
  ``action_dynamic_plan/00_README.md §3``.
* :class:`SeedKitAssemblyError` — raised on any invariant violation
  (token overflow, schema mismatch). The Coordinator treats this as
  a non-retryable dispatch failure.
* :func:`assemble_seed_kit(state, payload)` — returns the dict that
  will be JSON-dumped into ``seed_kit.json``. Side-effect free.
* :func:`estimate_tokens(text)` — char-based estimator (≈ 4 chars
  per token) used for the cap check.

Selection rules are deterministic (P2 §5.2 b): no LLM scoring, no
randomness. Adding a new selection signal requires bumping the
schema version + extending :data:`SEED_KIT_FIELDS`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .framework_paths import resolve_source_file_allowlist


SEED_KIT_FIELDS: frozenset[str] = frozenset({
    "motivation_gap_text",
    "roofline_summary",
    "profile_keyslices",
    "kept_patches",
    "reverted_patches",
    "kb_pitfalls",
    "source_root_hints",
})

# P2 §5.1 — total seed kit budget.
MAX_SEED_KIT_TOKENS: int = 8_000

# Per-section item caps. Combined with the per-item token caps below,
# these enforce P2 §5.1 column "量级上限".
MAX_PROFILE_KEYSLICES: int = 6
MAX_KEPT_PATCHES: int = 20
MAX_REVERTED_PATCHES: int = 10
MAX_KB_PITFALLS: int = 10

# Soft per-section caps (chars; converted to tokens at the end). Used
# to keep the overall budget well below MAX_SEED_KIT_TOKENS.
_MAX_MOTIVATION_CHARS: int = 4_000
_MAX_ROOFLINE_CHARS: int = 4_000
_MAX_PITFALL_CHARS_EACH: int = 600
_MAX_RATIONALE_CHARS_EACH: int = 240

# Char-to-token estimator: GPT-family tokenisers average ~4 chars per
# English token. A coarse estimator is sufficient for an enforcement
# floor — actual tokenisation happens at the sub-agent boundary.
_CHARS_PER_TOKEN: float = 4.0


class SeedKitAssemblyError(RuntimeError):
    """Raised when the seed kit cannot be assembled within the
    declared invariants (token cap, schema closure, etc.). The
    Coordinator catches this and rolls back the dispatch."""


@dataclass(frozen=True)
class SeedKitResult:
    """Assembler output."""

    payload: dict[str, Any]
    degraded: bool
    total_tokens: int


def estimate_tokens(text: str) -> int:
    """Coarse char-based token estimator (≈ 4 chars per token)."""
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_kernel_only_entry(entry: dict[str, Any]) -> bool:
    """``scope_domains == ['kernel']`` is denied at PolicyGate (P1 §4.1
    group C), but the assembler still defensively filters any patch /
    pitfall that names a kernel-only domain so the closed contract
    survives schema drift in upstream data sources."""
    action = str(entry.get("action") or "").strip().lower()
    return action.startswith("kernel_") or action == "integrate"


# ---------------------------------------------------------------------------
# Section assemblers — every helper returns a serialisable list/string
# obeying the per-section caps; the orchestrator-side ``assemble_seed_kit``
# is the only public composer.
# ---------------------------------------------------------------------------
def _roofline_summary(state: Any) -> str:
    """One-paragraph roofline digest sourced from
    ``last_trace_analyze.analysis_md_text`` (the verbatim
    Coordinator-cached roofline text). Empty when the cache is cold."""
    snap = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(snap, dict):
        return ""
    text = str(snap.get("analysis_md_text") or "").strip()
    return _truncate(text, _MAX_ROOFLINE_CHARS)


def _profile_keyslices(state: Any, scope_domains: list[str]) -> list[dict[str, Any]]:
    """Top-N hot kernels from ``last_trace_analyze.hot_kernels_top15``.

    Selection rule: take entries with the highest ``gpu_pct``; emit
    name / gpu_pct / bottleneck / arithmetic_intensity / source_file.
    ``scope_domains`` is passed for future filtering (P3 may narrow
    by domain affinity); P2 keeps the deterministic top-N rule.
    """
    snap = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(snap, dict):
        return []
    hot = snap.get("hot_kernels_top15") or []
    if not isinstance(hot, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for entry in hot:
        if not isinstance(entry, dict):
            continue
        cleaned.append({
            "name": entry.get("name"),
            "gpu_pct": entry.get("gpu_pct"),
            "bottleneck": entry.get("bottleneck"),
            "arithmetic_intensity": entry.get("arithmetic_intensity"),
            "source_file": entry.get("source_file"),
        })
    cleaned.sort(key=lambda e: float(e.get("gpu_pct") or 0.0), reverse=True)
    return cleaned[:MAX_PROFILE_KEYSLICES]


def _kept_patches(state: Any) -> list[dict[str, Any]]:
    """The most recent KEEP'd variants. Sources, in order:

    1. ``explore_search.accepted`` (canonical promote ledger).
    2. ``optimization_stack`` (fallback for legacy sessions without
       ``explore_search``).
    """
    accepted = []
    search = getattr(state, "explore_search", None) or {}
    if isinstance(search, dict):
        accepted = list(search.get("accepted") or [])
    if not accepted:
        accepted = list(getattr(state, "optimization_stack", []) or [])
    rows: list[dict[str, Any]] = []
    for entry in accepted[-MAX_KEPT_PATCHES:]:
        if not isinstance(entry, dict):
            continue
        if _is_kernel_only_entry(entry):
            continue
        rows.append({
            "name": entry.get("name") or entry.get("variant_name"),
            "action": entry.get("action") or "explore",
            "gain_pct": entry.get("gain_pct"),
            "rationale": _truncate(
                str(entry.get("rationale") or ""), _MAX_RATIONALE_CHARS_EACH,
            ),
        })
    return rows


def _reverted_patches(state: Any) -> list[dict[str, Any]]:
    """Most recent REVERT entries from ``explore_search.rejected``."""
    search = getattr(state, "explore_search", None) or {}
    if not isinstance(search, dict):
        return []
    rejected = list(search.get("rejected") or [])
    rows: list[dict[str, Any]] = []
    for entry in rejected[-MAX_REVERTED_PATCHES:]:
        if not isinstance(entry, dict):
            continue
        if _is_kernel_only_entry(entry):
            continue
        rows.append({
            "name": entry.get("name") or entry.get("variant_name"),
            "reason": str(entry.get("reason") or "").strip(),
            "gain_pct": entry.get("gain_pct"),
        })
    return rows


def _kb_pitfalls(
    state: Any, scope_domains: list[str], motivation: str,
) -> list[dict[str, Any]]:
    """Filter ``warm_start_pitfalls`` by keyword overlap with
    ``scope_domains`` ∪ keywords extracted from ``motivation``.

    The matcher is intentionally simple substring containment — the
    KB pitfall objects are free-form text blobs and P2 §5.2 b forbids
    LLM scoring. Top-K returned in source order; the warm cache is
    already ranked by Cortex relevance."""
    raw = getattr(state, "warm_start_pitfalls", None) or []
    if not isinstance(raw, list) or not raw:
        return []
    keywords = {d.lower() for d in scope_domains if d}
    for tok in motivation.lower().split():
        cleaned = tok.strip(".,;:!?()[]{}\"'`").strip()
        if len(cleaned) >= 4:
            keywords.add(cleaned)
    rows: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text_blob = (
            str(entry.get("raw") or entry.get("text") or "")
        ).lower()
        if not text_blob:
            continue
        if keywords and not any(k in text_blob for k in keywords):
            continue
        rows.append({
            "text": _truncate(
                str(entry.get("raw") or entry.get("text") or ""),
                _MAX_PITFALL_CHARS_EACH,
            ),
            "domain": entry.get("domain") or "",
        })
        if len(rows) >= MAX_KB_PITFALLS:
            break
    return rows


def _source_root_hints() -> list[str]:
    """Framework source roots (env-resolved). Empty list when
    ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`` is unset."""
    return list(resolve_source_file_allowlist())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def assemble_seed_kit(
    state: Any, payload: dict[str, Any],
) -> SeedKitResult:
    """Compose the closed seed kit dict for one dynamic_action dispatch.

    Inputs:

    * ``state``    — a SharedState snapshot; only ``last_trace_analyze``
      / ``explore_search`` / ``optimization_stack`` / ``warm_start_pitfalls``
      are consulted. Defensively tolerates a thin double for tests.
    * ``payload``  — the validated dispatch payload (P1 §3); must carry
      ``motivation_gap_text`` (non-empty) and ``scope_domains``
      (list ≥ 2). PolicyGate has already enforced both invariants by
      the time this function is reached.

    Output:

    * :class:`SeedKitResult` with ``payload`` ready for JSON dump,
      ``degraded`` flag (True iff one or more best-effort sources
      returned empty), and ``total_tokens`` for audit.

    Raises:

    * :class:`SeedKitAssemblyError` — total tokens > MAX_SEED_KIT_TOKENS
      or any produced field is not in :data:`SEED_KIT_FIELDS`. The
      Coordinator rolls back the dispatch.
    """
    motivation_raw = str(payload.get("motivation_gap_text") or "").strip()
    if not motivation_raw:
        raise SeedKitAssemblyError(
            "seed kit assembler: payload.motivation_gap_text is empty "
            "(PolicyGate should have rejected before reaching here)"
        )
    scope_domains_raw = payload.get("scope_domains") or ()
    scope_domains = [
        str(d or "").strip() for d in scope_domains_raw if str(d or "").strip()
    ]
    motivation = _truncate(motivation_raw, _MAX_MOTIVATION_CHARS)
    roofline = _roofline_summary(state)
    profile = _profile_keyslices(state, scope_domains)
    kept = _kept_patches(state)
    reverted = _reverted_patches(state)
    pitfalls = _kb_pitfalls(state, scope_domains, motivation)
    sources = _source_root_hints()
    out: dict[str, Any] = {
        "motivation_gap_text": motivation,
        "roofline_summary": roofline,
        "profile_keyslices": profile,
        "kept_patches": kept,
        "reverted_patches": reverted,
        "kb_pitfalls": pitfalls,
        "source_root_hints": sources,
    }
    extra_keys = set(out.keys()) - SEED_KIT_FIELDS
    if extra_keys:
        raise SeedKitAssemblyError(
            f"seed kit assembler emitted disallowed fields: "
            f"{sorted(extra_keys)!r}; SEED_KIT_FIELDS is the canonical "
            f"closed set."
        )
    serialised = json.dumps(out, sort_keys=True)
    total_tokens = estimate_tokens(serialised)
    if total_tokens > MAX_SEED_KIT_TOKENS:
        raise SeedKitAssemblyError(
            f"seed kit estimated tokens={total_tokens} exceeds "
            f"MAX_SEED_KIT_TOKENS={MAX_SEED_KIT_TOKENS}; tighten the "
            f"per-section caps before re-dispatching."
        )
    degraded = (
        not roofline
        or not profile
        or not kept
        or not pitfalls
    )
    return SeedKitResult(
        payload=out, degraded=degraded, total_tokens=total_tokens,
    )


__all__ = [
    "MAX_KB_PITFALLS",
    "MAX_KEPT_PATCHES",
    "MAX_PROFILE_KEYSLICES",
    "MAX_REVERTED_PATCHES",
    "MAX_SEED_KIT_TOKENS",
    "SEED_KIT_FIELDS",
    "SeedKitAssemblyError",
    "SeedKitResult",
    "assemble_seed_kit",
    "estimate_tokens",
]
