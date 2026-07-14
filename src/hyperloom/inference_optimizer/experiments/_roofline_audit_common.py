# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared state-extraction helpers for the roofline-v2 N7 scripts.

``audit_roofline_decisions`` and ``verify_roofline_v2`` both read a Hyperloom
session's ``state.json`` and derive the same decision-quality signals (discovered
vs. proposed flags, analysis.md grounding, prompt-cache hit rate). These helpers
are the single source of truth for that extraction so the two scripts stay in
lockstep. Read-only, pure stdlib.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from hyperloom.common.jsonio import read_json


# Keywords the LLM is likely to quote from analysis.md when grounding a
# PRUNE_BRANCH reason or propose-action note; used as a proxy for how often the
# report was actually consulted.
ANALYSIS_MD_KEYWORDS = (
    "analysis.md",
    "saturated",
    "comm-bound",
    "memory-bound",
    "compute-bound",
    "efficiency",
    "Top Operations",
    "Executive Summary",
    "Recommendations",
    "snapshot",
    "rcclAllreduce",
    "bottleneck",
)

FLAG_PATTERN = re.compile(r"--[a-z][a-z0-9_-]+")


class _CacheCounts(Protocol):
    """Structural type for objects carrying prompt-cache token counters."""

    cache_creation_input_tokens: int
    cache_read_input_tokens: int


def safe_load_state(session_dir: Path) -> dict[str, Any] | None:
    """Load ``state.json`` from a session directory if present and valid.

    Args:
        session_dir (Path): Session directory expected to contain
            ``state.json``.

    Returns:
        dict[str, Any] | None: Parsed state mapping, or ``None`` when missing
        or unreadable.
    """
    return read_json(session_dir / "state.json", default=None)


def flatten_discovered_flag_names(state: dict[str, Any]) -> set[str]:
    """Collect every discovered flag name across all frameworks.

    Walks ``discovered_flags[framework].{backend_flags,param_flags}`` and
    unions the flag names.

    Args:
        state (dict[str, Any]): Parsed ``state.json`` mapping.

    Returns:
        set[str]: Set of discovered flag names; empty when none are present.
    """
    discovered = state.get("discovered_flags") or {}
    names: set[str] = set()
    if not isinstance(discovered, dict):
        return names
    for entry in discovered.values():
        if not isinstance(entry, dict):
            continue
        for key in ("backend_flags", "param_flags"):
            lst = entry.get(key) or []
            if isinstance(lst, (list, tuple)):
                names.update(str(f) for f in lst if f)
    return names


def extract_proposed_flags(state: dict[str, Any]) -> list[str]:
    """Pull every ``--flag-name`` proposed across explore variants.

    Scans ``explore_attempts`` and ``explore_search.tested`` entries for
    ``extra_server_args`` strings and extracts flag tokens.

    Args:
        state (dict[str, Any]): Parsed ``state.json`` mapping.

    Returns:
        list[str]: Proposed flag names; may contain duplicates.
    """
    found: list[str] = []
    attempts = state.get("explore_attempts") or []
    if isinstance(attempts, list):
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            args = str(entry.get("extra_server_args") or "")
            if args:
                found.extend(FLAG_PATTERN.findall(args))
    # Also walk explore_search.tested for fingerprints that may not have ended
    # up in attempts (idempotency short-circuits).
    sub = state.get("explore_search") or {}
    tested = sub.get("tested") if isinstance(sub, dict) else None
    if isinstance(tested, dict):
        for snap in tested.values():
            if isinstance(snap, dict):
                args = str(snap.get("extra_server_args") or "")
                if args:
                    found.extend(FLAG_PATTERN.findall(args))
    return found


def count_analysis_md_references(state: dict[str, Any]) -> int:
    """Count prune reasons and explore notes grounded on analysis.md.

    Scans ``pruned_families[*].reason`` and ``explore_attempts[*].notes`` for
    any :data:`ANALYSIS_MD_KEYWORDS` token (a proxy for how often the LLM
    quoted the cached report when justifying a decision).

    Args:
        state (dict[str, Any]): Parsed ``state.json`` mapping.

    Returns:
        int: Number of reason/notes strings containing at least one keyword.
    """
    count = 0
    pruned = state.get("pruned_families") or []
    if isinstance(pruned, list):
        for entry in pruned:
            if isinstance(entry, dict):
                reason = str(entry.get("reason") or "").lower()
                if any(kw.lower() in reason for kw in ANALYSIS_MD_KEYWORDS):
                    count += 1
    attempts = state.get("explore_attempts") or []
    if isinstance(attempts, list):
        for entry in attempts:
            if isinstance(entry, dict):
                notes = str(entry.get("notes") or "").lower()
                if any(kw.lower() in notes for kw in ANALYSIS_MD_KEYWORDS):
                    count += 1
    return count


def cache_hit_rate(obj: _CacheCounts) -> float:
    """Compute the prompt-cache read hit rate from token counters.

    Args:
        obj (_CacheCounts): Object carrying ``cache_creation_input_tokens`` and
            ``cache_read_input_tokens``.

    Returns:
        float: ``cache_read / (cache_creation + cache_read)`` in [0, 1], or
        ``0.0`` when no cache tokens were recorded.
    """
    total = obj.cache_creation_input_tokens + obj.cache_read_input_tokens
    if total <= 0:
        return 0.0
    return obj.cache_read_input_tokens / total


def aggregate_cache_tokens(session_dir: Path) -> tuple[int, int]:
    """Return (cache_creation, cache_read) summed over the session LLM ledger.

    Reads ``reports/trace/llm_calls.jsonl`` (+ ext shards) via the breakdown
    collector, so the figures match ``session_breakdown.json``. This replaces the
    never-written ``state["tick_cache_metrics"]`` source. Best-effort ``(0, 0)``.
    """
    from hyperloom.inference_optimizer.breakdown.collectors.decision import (
        aggregate_session_cache_tokens,
    )

    return aggregate_session_cache_tokens(session_dir)
