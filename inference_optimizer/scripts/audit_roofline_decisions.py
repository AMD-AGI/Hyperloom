#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline-v2 N7: per-session decision-quality audit per design §10.5.

Reads one Hyperloom session_dir's ``state.json`` and prints a forensic
view of how well the main Orchestration LLM consumed the v2 features:

* **roofline action audit**       — when did each roofline run trigger,
  what was the cumulative gain at the time, did it succeed.
* **prune_branch audit**           — every pruned family with source +
  reason; cross-reference against analysis.md keywords so the operator
  can tell which prunes were report-grounded vs. main-llm-only.
* **discovered_flags audit**       — how many of the LLM's proposed
  flags matched / hallucinated against the discovered_flags namespace,
  and which untested flags remain.
* **analysis.md reference rate**   — how often the LLM's PRUNE reasons
  + propose notes quoted analysis.md keywords (a proxy for §10.3's
  "analysis_md_referenced_count" criterion).
* **cache hit rate** (if surfaced) — N6's per-call usage metrics
  aggregated to a session-level rate; § 10.3 criterion ≥ 50%.

Output is plain text by default; ``--json`` dumps a machine-readable
summary for programmatic consumption.

Read-only. Pure stdlib (argparse / json / pathlib / dataclasses /
datetime / re). Runs anywhere a session_dir is mounted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_ANALYSIS_MD_KEYWORDS = (
    "analysis.md", "saturated", "comm-bound", "memory-bound",
    "compute-bound", "efficiency", "Top Operations",
    "Executive Summary", "Recommendations", "snapshot",
    "rcclAllreduce", "bottleneck",
)

_FLAG_PATTERN = re.compile(r"--[a-z][a-z0-9_-]+")


# ---------------------------------------------------------------------------
# Audit report dataclass
# ---------------------------------------------------------------------------
@dataclass
class AuditReport:
    """Decision-quality audit summary derived from one session's state.

    Attributes:
        session_dir (Path): Audited session directory.
        has_state_json (bool): Whether ``state.json`` was found and parsed.
        error (str): Error description when the state could not be read.
        cumulative_gain_validated_pct (float): Validated cumulative gain.
        snapshot_id (int): Last roofline snapshot id observed.
        roofline_attempts (list[dict[str, Any]]): Recorded roofline attempts.
        pruned_families_raw (list[dict[str, Any]]): Pruned-family dict entries.
        discovered_flag_names (set[str]): Flags in the discovered namespace.
        proposed_flag_names (list[str]): Flags the LLM proposed (with dupes).
        hallucinated_flag_names (set[str]): Proposed flags absent from the
            discovered namespace.
        untested_flag_names (set[str]): Discovered flags never proposed.
        analysis_md_referenced_count (int): Reasons/notes grounded on
            analysis.md keywords.
        cache_creation_input_tokens (int): Aggregated cache-creation tokens.
        cache_read_input_tokens (int): Aggregated cache-read tokens.
    """

    session_dir: Path
    has_state_json: bool = False
    error: str = ""

    cumulative_gain_validated_pct: float = 0.0
    snapshot_id: int = 0

    # Per-action roofline timeline
    roofline_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Prune audit
    pruned_families_raw: list[dict[str, Any]] = field(default_factory=list)

    # Flag audit
    discovered_flag_names: set[str] = field(default_factory=set)
    proposed_flag_names: list[str] = field(default_factory=list)
    hallucinated_flag_names: set[str] = field(default_factory=set)
    untested_flag_names: set[str] = field(default_factory=set)

    # Decision quality
    analysis_md_referenced_count: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _safe_load_state(session_dir: Path) -> dict[str, Any] | None:
    """Load ``state.json`` from a session directory if present and valid.

    Args:
        session_dir (Path): Session directory expected to contain
            ``state.json``.

    Returns:
        dict[str, Any] | None: Parsed state mapping, or ``None`` when missing
        or unreadable.
    """
    p = session_dir / "state.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _flatten_discovered_flag_names(state: dict[str, Any]) -> set[str]:
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


def _extract_proposed_flag_names(state: dict[str, Any]) -> list[str]:
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
                found.extend(_FLAG_PATTERN.findall(args))
    sub = state.get("explore_search") or {}
    tested = sub.get("tested") if isinstance(sub, dict) else None
    if isinstance(tested, dict):
        for snap in tested.values():
            if isinstance(snap, dict):
                args = str(snap.get("extra_server_args") or "")
                if args:
                    found.extend(_FLAG_PATTERN.findall(args))
    return found


def _count_analysis_md_references(state: dict[str, Any]) -> int:
    """Count prune reasons and explore notes grounded on analysis.md.

    Args:
        state (dict[str, Any]): Parsed ``state.json`` mapping.

    Returns:
        int: Number of ``pruned_families`` reasons plus ``explore_attempts``
        notes that contain at least one analysis.md grounding keyword.
    """
    count = 0
    pruned = state.get("pruned_families") or []
    if isinstance(pruned, list):
        for entry in pruned:
            if isinstance(entry, dict):
                reason = str(entry.get("reason") or "").lower()
                if any(kw.lower() in reason for kw in _ANALYSIS_MD_KEYWORDS):
                    count += 1
    attempts = state.get("explore_attempts") or []
    if isinstance(attempts, list):
        for entry in attempts:
            if isinstance(entry, dict):
                notes = str(entry.get("notes") or "").lower()
                if any(kw.lower() in notes for kw in _ANALYSIS_MD_KEYWORDS):
                    count += 1
    return count


def extract(session_dir: Path) -> AuditReport:
    """Read a session directory and build its decision-quality audit.

    Loads ``state.json`` and populates an :class:`AuditReport` with gain,
    snapshot, roofline-attempt, prune, flag, and cache fields. A missing or
    unreadable state is recorded in the report's ``error`` field.

    Args:
        session_dir (Path): Session directory to audit.

    Returns:
        AuditReport: Populated audit report for the session.
    """
    r = AuditReport(session_dir=session_dir)
    state = _safe_load_state(session_dir)
    if state is None:
        r.error = f"state.json not found / unreadable at {session_dir}"
        return r
    r.has_state_json = True
    try:
        r.cumulative_gain_validated_pct = float(
            state.get("cumulative_gain_validated", 0.0)
        )
    except (TypeError, ValueError):
        pass
    cached_ta = state.get("last_trace_analyze") or {}
    if isinstance(cached_ta, dict):
        snap_raw = cached_ta.get("roofline_snapshot_id")
        if isinstance(snap_raw, int):
            r.snapshot_id = snap_raw

    roofline_attempts = state.get("roofline_attempts") or []
    if isinstance(roofline_attempts, list):
        r.roofline_attempts = [a for a in roofline_attempts if isinstance(a, dict)]

    pruned = state.get("pruned_families") or []
    if isinstance(pruned, list):
        r.pruned_families_raw = [p for p in pruned if isinstance(p, dict)]

    r.discovered_flag_names = _flatten_discovered_flag_names(state)
    r.proposed_flag_names = _extract_proposed_flag_names(state)
    if r.discovered_flag_names:
        r.hallucinated_flag_names = {
            f for f in r.proposed_flag_names if f not in r.discovered_flag_names
        }
        proposed_set = set(r.proposed_flag_names)
        r.untested_flag_names = r.discovered_flag_names - proposed_set
    r.analysis_md_referenced_count = _count_analysis_md_references(state)

    cache_metrics = state.get("tick_cache_metrics") or {}
    if isinstance(cache_metrics, dict):
        r.cache_creation_input_tokens = int(
            cache_metrics.get("cache_creation_input_tokens") or 0
        )
        r.cache_read_input_tokens = int(
            cache_metrics.get("cache_read_input_tokens") or 0
        )
    return r


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _cache_hit_rate(r: AuditReport) -> float:
    """Compute the prompt-cache read hit rate for an audit report.

    Args:
        r (AuditReport): Audit report carrying cache token counters.

    Returns:
        float: ``cache_read / (cache_creation + cache_read)`` in [0, 1], or
        ``0.0`` when no cache tokens were recorded.
    """
    total = r.cache_creation_input_tokens + r.cache_read_input_tokens
    if total <= 0:
        return 0.0
    return r.cache_read_input_tokens / total


def _format_roofline_timeline(attempts: list[dict[str, Any]]) -> str:
    """Render roofline action attempts as an indented text table.

    Args:
        attempts (list[dict[str, Any]]): Recorded roofline attempt entries.

    Returns:
        str: Multi-line table (timestamp/status/task_id), or a placeholder
        line when there are no attempts.
    """
    if not attempts:
        return "    (no roofline action attempts recorded)"
    lines: list[str] = []
    lines.append(
        f"    {'#':>2s} {'ts':<28s} {'status':>10s} {'task_id':>14s}"
    )
    lines.append("    " + "-" * 64)
    for i, entry in enumerate(attempts, start=1):
        ts = str(entry.get("ts") or "?")[:27]
        status = str(entry.get("status") or "?")
        task_id = str(entry.get("task_id") or "?")[:14]
        lines.append(f"    {i:>2d} {ts:<28s} {status:>10s} {task_id:>14s}")
    return "\n".join(lines)


def _format_pruned_families(pruned: list[dict[str, Any]]) -> str:
    """Render pruned families as a table flagging analysis.md grounding.

    Args:
        pruned (list[dict[str, Any]]): Pruned-family dict entries.

    Returns:
        str: Multi-line table (family/source/grounded/reason), or a
        placeholder line when there are no pruned families.
    """
    if not pruned:
        return "    (no pruned_families)"
    lines: list[str] = []
    lines.append(
        f"    {'family':<28s} {'source':<14s} {'analysis-md-grounded':<22s} {'reason'}"
    )
    lines.append("    " + "-" * 90)
    for entry in pruned:
        fam = str(entry.get("family") or "?")
        source = str(entry.get("source") or "?")
        reason = str(entry.get("reason") or "")
        grounded = any(
            kw.lower() in reason.lower() for kw in _ANALYSIS_MD_KEYWORDS
        )
        tag = "yes" if grounded else "no"
        # Truncate reason to 100 chars for readability
        reason_short = reason if len(reason) <= 100 else reason[:97] + "..."
        lines.append(
            f"    {fam:<28s} {source:<14s} {tag:<22s} {reason_short}"
        )
    return "\n".join(lines)


def _format_flag_audit(r: AuditReport, *, max_show: int = 10) -> str:
    """Render the flag-namespace audit (discovered/proposed/hallucinated).

    Args:
        r (AuditReport): Audit report holding the flag sets.
        max_show (int): Maximum hallucinated flag names to list in the sample.

    Returns:
        str: Multi-line summary of flag namespace coverage.
    """
    lines: list[str] = []
    lines.append(
        f"    discovered_flags namespace : {len(r.discovered_flag_names)} flags"
    )
    lines.append(
        f"    flags proposed             : {len(set(r.proposed_flag_names))} unique"
    )
    lines.append(
        f"    hallucinated (not in namespace): {len(r.hallucinated_flag_names)}"
    )
    if r.hallucinated_flag_names:
        sample = sorted(r.hallucinated_flag_names)[:max_show]
        lines.append(f"      sample: {sample}")
    lines.append(
        f"    untested (in namespace, never proposed): {len(r.untested_flag_names)}"
    )
    return "\n".join(lines)


def render(r: AuditReport) -> str:
    """Build the human-readable decision-audit report for a session.

    Combines the roofline timeline, prune grounding table, flag audit, and
    the §10.3 decision-quality criteria into a single text block.

    Args:
        r (AuditReport): Audit report to render.

    Returns:
        str: Multi-line report text suitable for printing to stdout.
    """
    out: list[str] = []
    out.append("=" * 72)
    out.append("Roofline-v2 N7 decision audit")
    out.append("=" * 72)
    out.append("")
    out.append(f"  session_dir: {r.session_dir}")
    if r.error:
        out.append(f"  ERROR: {r.error}")
        return "\n".join(out)
    out.append(
        f"  cumulative_gain_validated_pct : {r.cumulative_gain_validated_pct:.3f}%"
    )
    out.append(f"  last snapshot_id              : {r.snapshot_id}")
    out.append(f"  roofline_attempts             : {len(r.roofline_attempts)}")
    out.append(f"  pruned_families               : {len(r.pruned_families_raw)}")
    out.append("")
    out.append("  --- roofline action timeline ---")
    out.append(_format_roofline_timeline(r.roofline_attempts))
    out.append("")
    out.append("  --- pruned_families (analysis-md grounding) ---")
    out.append(_format_pruned_families(r.pruned_families_raw))
    out.append("")
    out.append("  --- flag audit ---")
    out.append(_format_flag_audit(r))
    out.append("")
    out.append("  --- decision quality criteria (§10.3) ---")
    cache = _cache_hit_rate(r)
    quality: list[tuple[str, bool, str]] = [
        (
            "cache_hit_rate ≥ 50%",
            cache >= 0.50,
            f"{cache*100:.1f}%",
        ),
        (
            "analysis_md_referenced_count ≥ 3",
            r.analysis_md_referenced_count >= 3,
            str(r.analysis_md_referenced_count),
        ),
        (
            "hallucinated_flag_count = 0",
            len(r.hallucinated_flag_names) == 0,
            str(len(r.hallucinated_flag_names)),
        ),
        (
            "roofline_action_count ≥ 1",
            len(r.roofline_attempts) >= 1,
            str(len(r.roofline_attempts)),
        ),
    ]
    for name, passed, value in quality:
        mark = "OK  " if passed else "MISS"
        out.append(f"    [{mark}] {name:38s} = {value}")
    out.append("=" * 72)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the decision-audit CLI.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv`` when ``None``.

    Returns:
        argparse.Namespace: Parsed arguments with ``session`` and ``json``
        attributes.
    """
    p = argparse.ArgumentParser(
        description="Audit Roofline-v2 decisions in a single session_dir."
    )
    p.add_argument("--session", required=True, type=Path,
                   help="Session directory containing state.json")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of pretty text")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the decision audit for one session and return an exit code.

    Extracts the audit report and prints it as JSON (``--json``) or pretty
    text.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv`` when ``None``.

    Returns:
        int: ``0`` when ``state.json`` was read successfully, otherwise ``1``.
    """
    args = _parse_args(argv)
    report = extract(args.session)
    if args.json:
        payload = {
            "session_dir": str(report.session_dir),
            "has_state_json": report.has_state_json,
            "error": report.error,
            "cumulative_gain_validated_pct": report.cumulative_gain_validated_pct,
            "snapshot_id": report.snapshot_id,
            "roofline_attempts_count": len(report.roofline_attempts),
            "pruned_families_count": len(report.pruned_families_raw),
            "pruned_families": report.pruned_families_raw,
            "discovered_flag_count": len(report.discovered_flag_names),
            "proposed_unique_flag_count": len(set(report.proposed_flag_names)),
            "hallucinated_flag_count": len(report.hallucinated_flag_names),
            "hallucinated_flag_names": sorted(report.hallucinated_flag_names),
            "untested_flag_count": len(report.untested_flag_names),
            "analysis_md_referenced_count": report.analysis_md_referenced_count,
            "cache_creation_input_tokens": report.cache_creation_input_tokens,
            "cache_read_input_tokens": report.cache_read_input_tokens,
            "cache_hit_rate": _cache_hit_rate(report),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render(report))
    return 0 if report.has_state_json else 1


if __name__ == "__main__":
    sys.exit(main())
