#!/usr/bin/env python3
"""Roofline-v2 N7: baseline vs exp verification per design §10.5.

Compares two Hyperloom session_dirs (typically a `main` baseline and
a `feature/xiaofei/roofline-v2` experiment) and prints a side-by-side
table covering the §10.2 main success criteria + §10.3 v2-specific
cache-and-decision quality criteria.

Usage::

    python -m inference_optimizer.scripts.verify_roofline_v2 \\
        --baseline /tmp/roofline-v2/qwen3-baseline \\
        --exp      /tmp/roofline-v2/qwen3-exp

Exit codes (machine-readable for CI):
* ``0`` PASS    — delta gain ≥ +5.0% (design §10.2 main hard target met)
* ``2`` PARTIAL — delta gain in (0%, +5.0%); positive but below target
* ``1`` FAIL    — delta gain ≤ 0% or session metadata unreadable

The script is read-only over ``state.json`` and pure-Python (no
pandas / numpy) so it runs anywhere a session_dir is mounted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
@dataclass
class SessionMetrics:
    """Subset of SharedState surfaced in the comparison table."""

    session_dir: Path
    has_state_json: bool = False
    error: str = ""

    cumulative_gain_validated_pct: float = 0.0
    wall_clock_min: float = 0.0
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    pruned_families_raw: list[dict[str, Any]] = field(default_factory=list)

    roofline_action_count: int = 0
    profile_action_count: int = 0
    snapshot_id: int = 0
    analysis_md_text: str = ""

    # Cache metrics (sum across all backend.calls if surfaced in
    # state.json; otherwise 0 — the backend emits these to backend.calls
    # only, so verify can't read them unless an audit hook pushes them
    # to SharedState; tracked here for forward-compatibility).
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Decision quality (derived in derive())
    analysis_md_referenced_count: int = 0
    hallucinated_flag_count: int = 0
    discovered_flag_names: set[str] = field(default_factory=set)
    prune_branch_count_orchestration: int = 0


# Keywords the LLM is likely to quote from analysis.md when grounding
# a PRUNE_BRANCH reason or propose-action note. The LLM is instructed
# to quote the report; we count how often that happened.
_ANALYSIS_MD_KEYWORDS = (
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


def _safe_load_state(session_dir: Path) -> dict[str, Any] | None:
    p = session_dir / "state.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_clock_min(state: dict[str, Any]) -> float:
    """Approximate session duration from baseline ts → last validated
    gain ts (or fallback to latest attempt ts)."""
    starts: list[datetime] = []
    baseline_attempts = state.get("baseline_attempts") or []
    for entry in baseline_attempts:
        if isinstance(entry, dict):
            dt = _parse_iso(entry.get("ts"))
            if dt is not None:
                starts.append(dt)
                break
    end_dt = _parse_iso(state.get("cumulative_gain_validated_ts"))
    if end_dt is None:
        for action in ("validate_stack", "params", "backends",
                       "sweep", "profile", "baseline"):
            attempts = state.get(f"{action}_attempts") or []
            if not attempts:
                continue
            last = attempts[-1]
            if isinstance(last, dict):
                dt = _parse_iso(last.get("ts"))
                if dt is not None and (end_dt is None or dt > end_dt):
                    end_dt = dt
    if not starts or end_dt is None:
        return 0.0
    return max(0.0, (end_dt - min(starts)).total_seconds() / 60.0)


def _count_action_attempts(state: dict[str, Any], action: str) -> int:
    """Length of `<action>_attempts` list; 0 when missing / non-list."""
    attempts = state.get(f"{action}_attempts") or []
    return len(attempts) if isinstance(attempts, list) else 0


def _flatten_discovered_flag_names(state: dict[str, Any]) -> set[str]:
    """Union of every `--flag-name` in `discovered_flags[fw].{backend,param}_flags`."""
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


_FLAG_PATTERN = re.compile(r"--[a-z][a-z0-9_-]+")


def _extract_proposed_flags(state: dict[str, Any]) -> list[str]:
    """Pull every `--flag-name` mentioned in any explore variant the
    LLM proposed (across attempts_history)."""
    found: list[str] = []
    attempts = state.get("explore_attempts") or []
    if isinstance(attempts, list):
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            args = str(entry.get("extra_server_args") or "")
            if args:
                found.extend(_FLAG_PATTERN.findall(args))
    # Also walk explore_search.tested for fingerprints that may not
    # have ended up in attempts (idempotency short-circuits).
    sub = state.get("explore_search") or {}
    tested = sub.get("tested") if isinstance(sub, dict) else None
    if isinstance(tested, dict):
        for snap in tested.values():
            if isinstance(snap, dict):
                args = str(snap.get("extra_server_args") or "")
                if args:
                    found.extend(_FLAG_PATTERN.findall(args))
    return found


def _count_orchestration_prune_branch(state: dict[str, Any]) -> int:
    """Count `pruned_families` entries with source="orchestration".

    `pruned_families` may be a list of dicts (`{family, reason,
    source, ts}`) or a list of strings (legacy / Robustness inserts).
    Strings are NOT counted as orchestration-emitted because the
    source provenance is unknown.
    """
    pruned = state.get("pruned_families") or []
    if not isinstance(pruned, list):
        return 0
    count = 0
    for entry in pruned:
        if isinstance(entry, dict) and entry.get("source") == "orchestration":
            count += 1
    return count


def _count_analysis_md_references(state: dict[str, Any]) -> int:
    """Scan PRUNE_BRANCH reasons + propose-action notes for keywords
    that indicate the LLM grounded its decision on the cached
    analysis.md (per §8.7 orchestration.md guidance).

    Sources:
    * `pruned_families[*].reason` (orchestration-sourced)
    * `attempts_history` entries' `notes` field (when present)
    """
    count = 0
    pruned = state.get("pruned_families") or []
    if isinstance(pruned, list):
        for entry in pruned:
            if not isinstance(entry, dict):
                continue
            reason = str(entry.get("reason") or "")
            for kw in _ANALYSIS_MD_KEYWORDS:
                if kw.lower() in reason.lower():
                    count += 1
                    break
    attempts = state.get("explore_attempts") or []
    if isinstance(attempts, list):
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            notes = str(entry.get("notes") or "")
            for kw in _ANALYSIS_MD_KEYWORDS:
                if kw.lower() in notes.lower():
                    count += 1
                    break
    return count


def extract(session_dir: Path) -> SessionMetrics:
    m = SessionMetrics(session_dir=session_dir)
    state = _safe_load_state(session_dir)
    if state is None:
        m.error = f"state.json not found / unreadable at {session_dir}"
        return m
    m.has_state_json = True
    try:
        m.cumulative_gain_validated_pct = float(
            state.get("cumulative_gain_validated", 0.0)
        )
    except (TypeError, ValueError):
        m.cumulative_gain_validated_pct = 0.0
    m.wall_clock_min = _wall_clock_min(state)

    stack = state.get("optimization_stack") or []
    m.optimization_stack = [s for s in stack if isinstance(s, dict)]
    pruned = state.get("pruned_families") or []
    m.pruned_families_raw = [p for p in pruned if isinstance(p, dict)]
    m.prune_branch_count_orchestration = _count_orchestration_prune_branch(state)

    # roofline action counter — derived from attempts_history (the
    # roofline composite action joins the standard audit-attempts
    # mechanism via cli registration).
    m.roofline_action_count = _count_action_attempts(state, "roofline")
    m.profile_action_count = _count_action_attempts(state, "profile")

    # Snapshot signal
    cached_ta = state.get("last_trace_analyze") or {}
    if isinstance(cached_ta, dict):
        snap_raw = cached_ta.get("roofline_snapshot_id")
        if isinstance(snap_raw, int):
            m.snapshot_id = snap_raw
        m.analysis_md_text = str(cached_ta.get("analysis_md_text") or "")

    # Decision quality
    m.discovered_flag_names = _flatten_discovered_flag_names(state)
    proposed = _extract_proposed_flags(state)
    if m.discovered_flag_names:
        m.hallucinated_flag_count = sum(
            1 for f in proposed if f not in m.discovered_flag_names
        )
    m.analysis_md_referenced_count = _count_analysis_md_references(state)

    # Cache metrics — the backend surfaces these to backend.calls
    # per-call. Coordinator does not aggregate to SharedState. For now
    # read pre-aggregated values if a future hook writes them; otherwise
    # leave at 0.
    cache_metrics = state.get("tick_cache_metrics") or {}
    if isinstance(cache_metrics, dict):
        m.cache_creation_input_tokens = int(
            cache_metrics.get("cache_creation_input_tokens") or 0
        )
        m.cache_read_input_tokens = int(
            cache_metrics.get("cache_read_input_tokens") or 0
        )
        m.input_tokens = int(cache_metrics.get("input_tokens") or 0)
        m.output_tokens = int(cache_metrics.get("output_tokens") or 0)
    return m


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _cache_hit_rate(m: SessionMetrics) -> float:
    total = m.cache_creation_input_tokens + m.cache_read_input_tokens
    if total <= 0:
        return 0.0
    return m.cache_read_input_tokens / total


def _format_action_seq(stack: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in stack[:12]:
        kind = str(entry.get("kind") or entry.get("action") or "?")
        variant = entry.get("variant_name") or entry.get("variant_id") or ""
        parts.append(f"{kind}:{variant}" if variant else kind)
    suffix = "" if len(stack) <= 12 else f" ...(+{len(stack)-12} more)"
    return ", ".join(parts) + suffix if parts else "(empty)"


def render(baseline: SessionMetrics, exp: SessionMetrics) -> str:
    delta_gain = (
        exp.cumulative_gain_validated_pct - baseline.cumulative_gain_validated_pct
    )
    delta_wall = exp.wall_clock_min - baseline.wall_clock_min
    exp_cache = _cache_hit_rate(exp)
    base_cache = _cache_hit_rate(baseline)

    out: list[str] = []
    out.append("=" * 72)
    out.append("Roofline-v2 N7 baseline vs experiment verification")
    out.append("=" * 72)
    out.append("")
    out.append(f"  baseline session_dir: {baseline.session_dir}")
    out.append(f"  exp      session_dir: {exp.session_dir}")
    out.append("")
    if baseline.error:
        out.append(f"  baseline ERROR: {baseline.error}")
    if exp.error:
        out.append(f"  exp      ERROR: {exp.error}")
    out.append("")
    out.append("  " + "-" * 68)
    out.append(
        f"  {'metric':36s} {'baseline':>12s} {'exp':>12s} {'delta':>8s}"
    )
    out.append("  " + "-" * 68)
    rows: list[tuple[str, str, str, str]] = [
        (
            "cumulative_gain_validated_pct",
            f"{baseline.cumulative_gain_validated_pct:>11.3f}%",
            f"{exp.cumulative_gain_validated_pct:>11.3f}%",
            f"{delta_gain:>+7.3f}%",
        ),
        (
            "wall_clock_min",
            f"{baseline.wall_clock_min:>11.2f} ",
            f"{exp.wall_clock_min:>11.2f} ",
            f"{delta_wall:>+7.2f} ",
        ),
        (
            "roofline_action_count",
            f"{baseline.roofline_action_count:>12d}",
            f"{exp.roofline_action_count:>12d}",
            f"{exp.roofline_action_count - baseline.roofline_action_count:>+8d}",
        ),
        (
            "profile_action_count",
            f"{baseline.profile_action_count:>12d}",
            f"{exp.profile_action_count:>12d}",
            f"{exp.profile_action_count - baseline.profile_action_count:>+8d}",
        ),
        (
            "snapshot_id (last)",
            f"{baseline.snapshot_id:>12d}",
            f"{exp.snapshot_id:>12d}",
            f"{exp.snapshot_id - baseline.snapshot_id:>+8d}",
        ),
        (
            "prune_branch (orchestration)",
            f"{baseline.prune_branch_count_orchestration:>12d}",
            f"{exp.prune_branch_count_orchestration:>12d}",
            f"{exp.prune_branch_count_orchestration - baseline.prune_branch_count_orchestration:>+8d}",
        ),
        (
            "optimization_stack_size",
            f"{len(baseline.optimization_stack):>12d}",
            f"{len(exp.optimization_stack):>12d}",
            f"{len(exp.optimization_stack) - len(baseline.optimization_stack):>+8d}",
        ),
        (
            "cache_hit_rate",
            f"{base_cache*100:>11.1f}%",
            f"{exp_cache*100:>11.1f}%",
            f"{(exp_cache-base_cache)*100:>+7.1f}%",
        ),
        (
            "analysis_md_referenced_count",
            f"{baseline.analysis_md_referenced_count:>12d}",
            f"{exp.analysis_md_referenced_count:>12d}",
            f"{exp.analysis_md_referenced_count - baseline.analysis_md_referenced_count:>+8d}",
        ),
        (
            "hallucinated_flag_count",
            f"{baseline.hallucinated_flag_count:>12d}",
            f"{exp.hallucinated_flag_count:>12d}",
            f"{exp.hallucinated_flag_count - baseline.hallucinated_flag_count:>+8d}",
        ),
    ]
    for metric, b, e, d in rows:
        out.append(f"  {metric:36s} {b} {e} {d}")
    out.append("  " + "-" * 68)
    out.append("")
    out.append("  action_seq (baseline):")
    out.append(f"    {_format_action_seq(baseline.optimization_stack)}")
    out.append("  action_seq (exp):")
    out.append(f"    {_format_action_seq(exp.optimization_stack)}")
    out.append("")

    # Verdict — §10.2 main target
    out.append("  " + "=" * 68)
    if delta_gain >= 5.0:
        verdict = "PASS — delta gain meets §10.2 hard target (≥ +5.0%)"
    elif delta_gain > 0:
        verdict = (
            f"PARTIAL — delta gain {delta_gain:+.3f}% positive but below "
            "the +5% target"
        )
    else:
        verdict = (
            f"FAIL — delta gain {delta_gain:+.3f}% non-positive "
            "(experiment regression / no improvement)"
        )
    out.append(f"  VERDICT: {verdict}")
    out.append("  " + "=" * 68)

    # §10.3 v2-specific quality criteria — informational only
    out.append("")
    out.append("  §10.3 v2 quality criteria (informational):")
    qual: list[tuple[str, bool, str]] = [
        (
            "cache_hit_rate ≥ 50%",
            exp_cache >= 0.50,
            f"{exp_cache*100:.1f}%",
        ),
        (
            "analysis_md_referenced_count ≥ 3",
            exp.analysis_md_referenced_count >= 3,
            str(exp.analysis_md_referenced_count),
        ),
        (
            "hallucinated_flag_count = 0",
            exp.hallucinated_flag_count == 0,
            str(exp.hallucinated_flag_count),
        ),
        (
            "roofline_action_count ≥ 1",
            exp.roofline_action_count >= 1,
            str(exp.roofline_action_count),
        ),
    ]
    for name, passed, value in qual:
        mark = "OK  " if passed else "MISS"
        out.append(f"    [{mark}] {name:38s} = {value}")
    return "\n".join(out)


def _exit_code_for_delta(delta_gain: float) -> int:
    if delta_gain >= 5.0:
        return 0
    if delta_gain > 0:
        return 2
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a baseline vs experiment Hyperloom session "
                    "for Roofline-v2 verification (design §10.5).",
    )
    p.add_argument("--baseline", required=True, type=Path,
                   help="Baseline session_dir (contains state.json)")
    p.add_argument("--exp", required=True, type=Path,
                   help="Experiment session_dir (contains state.json)")
    p.add_argument("--json", action="store_true",
                   help="Also emit a JSON summary on stderr (CI consumption)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline = extract(args.baseline)
    exp = extract(args.exp)
    print(render(baseline, exp))
    if args.json:
        summary = {
            "baseline": {
                "session_dir": str(baseline.session_dir),
                "cumulative_gain_validated_pct": baseline.cumulative_gain_validated_pct,
                "wall_clock_min": baseline.wall_clock_min,
                "roofline_action_count": baseline.roofline_action_count,
                "cache_hit_rate": _cache_hit_rate(baseline),
            },
            "exp": {
                "session_dir": str(exp.session_dir),
                "cumulative_gain_validated_pct": exp.cumulative_gain_validated_pct,
                "wall_clock_min": exp.wall_clock_min,
                "roofline_action_count": exp.roofline_action_count,
                "cache_hit_rate": _cache_hit_rate(exp),
                "analysis_md_referenced_count": exp.analysis_md_referenced_count,
                "hallucinated_flag_count": exp.hallucinated_flag_count,
                "snapshot_id": exp.snapshot_id,
                "prune_branch_count_orchestration": exp.prune_branch_count_orchestration,
            },
            "delta": {
                "cumulative_gain_validated_pct": (
                    exp.cumulative_gain_validated_pct
                    - baseline.cumulative_gain_validated_pct
                ),
                "wall_clock_min": exp.wall_clock_min - baseline.wall_clock_min,
            },
        }
        sys.stderr.write(json.dumps(summary, indent=2) + "\n")
    return _exit_code_for_delta(
        exp.cumulative_gain_validated_pct
        - baseline.cumulative_gain_validated_pct
    )


if __name__ == "__main__":
    sys.exit(main())
