"""Seed-kit assembler for the ``dynamic_action`` sub-agent.

The sub-agent's whole input surface is what this module returns.
Closed field set, deterministic selection rules, hard token cap.

Public surface:

* :data:`SEED_KIT_FIELDS` — frozenset of allowed top-level keys; any
  other key in the output is rejected.
* :data:`MAX_SEED_KIT_TOKENS` / per-section caps — token budget.
* :class:`SeedKitAssemblyError` — raised on token overflow or schema
  mismatch; the Coordinator treats this as a non-retryable dispatch
  failure.
* :func:`assemble_seed_kit(state, payload)` — side-effect-free; returns
  the dict that will be JSON-dumped into ``seed_kit.json``.
* :func:`estimate_tokens(text)` — char-based estimator (~4 chars per
  token).

Selection is deterministic (no LLM scoring, no randomness). A new
selection signal requires extending :data:`SEED_KIT_FIELDS`.
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

# Total seed-kit budget.
MAX_SEED_KIT_TOKENS: int = 8_000

# Per-section item count caps.
MAX_PROFILE_KEYSLICES: int = 6
MAX_KEPT_PATCHES: int = 20
MAX_REVERTED_PATCHES: int = 10
MAX_KB_PITFALLS: int = 10

# Per-section char budgets (≈ 4 chars per token), kept well below
# ``MAX_SEED_KIT_TOKENS`` so all sections combined stay within budget.
_MAX_MOTIVATION_CHARS: int = 4_000
_MAX_ROOFLINE_CHARS: int = 4_000
_MAX_PITFALL_CHARS_EACH: int = 600
_MAX_RATIONALE_CHARS_EACH: int = 240

_CHARS_PER_TOKEN: float = 4.0


class SeedKitAssemblyError(RuntimeError):
    """Token overflow or schema violation during seed-kit assembly.

    The Coordinator catches this and rolls back the dispatch."""


@dataclass(frozen=True)
class SeedKitResult:
    """Assembler output."""

    payload: dict[str, Any]
    degraded: bool
    total_tokens: int


def estimate_tokens(text: str) -> int:
    """Char-based token estimator (~4 chars per token)."""
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_kernel_only_entry(entry: dict[str, Any]) -> bool:
    """True for patches whose action is kernel-only; defensively
    filtered out of the seed kit even though PolicyGate already
    rejects kernel-only scope at dispatch."""
    action = str(entry.get("action") or "").strip().lower()
    return action.startswith("kernel_") or action == "integrate"


# ---------------------------------------------------------------------------
# Section assemblers
# ---------------------------------------------------------------------------
def _roofline_summary(state: Any) -> str:
    """One-paragraph roofline digest from
    ``last_trace_analyze.analysis_md_text``; empty when the cache is
    cold."""
    snap = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(snap, dict):
        return ""
    text = str(snap.get("analysis_md_text") or "").strip()
    return _truncate(text, _MAX_ROOFLINE_CHARS)


def _profile_keyslices(state: Any, scope_domains: list[str]) -> list[dict[str, Any]]:
    """Top-N hot kernels from ``last_trace_analyze.hot_kernels_top15``,
    sorted by ``gpu_pct`` descending. ``scope_domains`` is reserved
    for future domain-affinity filtering."""
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
    """Most recent KEEP'd variants from ``explore_search.accepted``;
    falls back to ``optimization_stack`` when ``explore_search`` is
    absent."""
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
    """Filter ``warm_start_pitfalls`` by substring overlap with
    ``scope_domains`` ∪ keywords extracted from ``motivation``.

    Substring containment only (no LLM scoring). Top-K returned in
    source order; the warm cache is already ranked by relevance."""
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
    """Resolved framework source roots; empty when the env var is
    unset."""
    return list(resolve_source_file_allowlist())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def assemble_seed_kit(
    state: Any, payload: dict[str, Any],
) -> SeedKitResult:
    """Compose the closed seed-kit dict for one dispatch.

    ``state`` is a SharedState snapshot; only ``last_trace_analyze``
    / ``explore_search`` / ``optimization_stack`` /
    ``warm_start_pitfalls`` are consulted (thin doubles tolerated).

    ``payload`` is the validated dispatch payload; PolicyGate has
    already enforced the non-empty ``motivation_gap_text`` + ≥ 2
    ``scope_domains`` invariants.

    Returns a :class:`SeedKitResult` with ``degraded=True`` when one
    or more best-effort sources came back empty. Raises
    :class:`SeedKitAssemblyError` on token overflow or schema
    violation; the Coordinator rolls back the dispatch.
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
