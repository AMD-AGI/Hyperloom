#!/usr/bin/env python3
"""Audit a single Hyperloom session for Roofline-v2 decision quality.

Usage::

    python -m inference_optimizer.scripts.audit_roofline_decisions \\
        --session /tmp/roofline-v2/qwen3-exp

Reads ``state.json`` and produces a per-snapshot summary of what the
roofline sub-agent recommended vs what the main Orchestration LLM
actually did:

* Cached ``last_roofline_analysis`` — the most recent sub-agent
  output (snapshot_id, primary bottleneck, suggested prunes /
  next-actions, degraded flag, error).
* ``pruned_families`` — every prune currently in effect plus the
  ``source`` (orchestration vs robustness) and reason recorded when
  Coordinator wrote it. Cross-referenced against the sub-agent's
  ``suggested_prunes`` so the operator can see which advice the
  main LLM consumed vs ignored.
* Action sequence walk — list the ``optimization_stack`` entries
  in chronological order so the operator can eyeball whether the
  sequence reflects the sub-agent's ``suggested_next_actions``
  (e.g. did ``params`` with comm-overlap flags actually get tried
  after a comm-bottleneck analysis?).

Output is plain text by default; pass ``--json`` to dump a
machine-readable summary for programmatic consumption.

The script is read-only and pure-Python (no extra dependencies)
so it can run anywhere a session_dir is mounted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditReport:
    session_dir: Path
    has_state_json: bool = False
    error: str = ""

    cumulative_gain_validated_pct: float = 0.0
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    pruned_families_raw: list[dict[str, Any]] = field(default_factory=list)
    last_roofline_analysis: dict[str, Any] = field(default_factory=dict)
    last_select_kernels: dict[str, Any] = field(default_factory=dict)

    # Derived
    advice_consumed_count: int = 0
    advice_ignored_count: int = 0
    next_action_followed_count: int = 0
    next_action_ignored_count: int = 0


def _safe_load_state_json(session_dir: Path) -> dict[str, Any] | None:
    p = session_dir / "state.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extract(session_dir: Path) -> AuditReport:
    report = AuditReport(session_dir=session_dir)
    state = _safe_load_state_json(session_dir)
    if state is None:
        report.error = f"state.json not found / unreadable at {session_dir}"
        return report
    report.has_state_json = True
    try:
        report.cumulative_gain_validated_pct = float(
            state.get("cumulative_gain_validated", 0.0),
        )
    except (TypeError, ValueError):
        report.cumulative_gain_validated_pct = 0.0

    stack = state.get("optimization_stack") or []
    report.optimization_stack = [s for s in stack if isinstance(s, dict)]
    pruned = state.get("pruned_families") or []
    report.pruned_families_raw = [p for p in pruned if isinstance(p, dict)]
    rfa = state.get("last_roofline_analysis") or {}
    if isinstance(rfa, dict):
        report.last_roofline_analysis = rfa
    sk = state.get("last_select_kernels") or {}
    if isinstance(sk, dict):
        report.last_select_kernels = sk

    # Derive advice consumption stats. The sub-agent's
    # ``suggested_prunes`` lists action families it recommends
    # pruning; check how many of those appear in ``pruned_families``.
    suggested = [
        str(entry.get("family") or "").strip()
        for entry in (rfa.get("suggested_prunes") or [])
        if isinstance(entry, dict)
    ]
    suggested_set = {s for s in suggested if s}
    pruned_set = {
        str((p.get("family") if isinstance(p, dict) else p) or "")
        for p in pruned
    }
    pruned_set.discard("")
    report.advice_consumed_count = len(suggested_set & pruned_set)
    report.advice_ignored_count = len(suggested_set - pruned_set)

    # Next-action consumption: ``suggested_next_actions`` lists kinds
    # the sub-agent wants the LLM to propose; check how many appear
    # in the optimization_stack post-snapshot.
    next_suggested = [
        str(entry.get("kind") or "").strip()
        for entry in (rfa.get("suggested_next_actions") or [])
        if isinstance(entry, dict)
    ]
    next_suggested_set = {n for n in next_suggested if n}
    stack_kinds = {
        str(e.get("kind") or e.get("action") or "").strip()
        for e in report.optimization_stack
    }
    stack_kinds.discard("")
    report.next_action_followed_count = len(next_suggested_set & stack_kinds)
    report.next_action_ignored_count = len(next_suggested_set - stack_kinds)

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _format_action_stack(stack: list[dict[str, Any]]) -> str:
    if not stack:
        return "    (empty)"
    out: list[str] = []
    for idx, entry in enumerate(stack, start=1):
        kind = str(entry.get("kind") or entry.get("action") or "?")
        variant = entry.get("variant_name") or entry.get("variant_id") or ""
        gain = entry.get("gain_pct")
        gain_str = (
            f" gain={gain:+.2f}%"
            if isinstance(gain, (int, float)) else ""
        )
        suffix = f":{variant}" if variant else ""
        out.append(f"    {idx:>2d}. {kind}{suffix}{gain_str}")
    return "\n".join(out)


def _format_prune_table(pruned_raw: list[dict[str, Any]],
                        suggested_prunes: list[dict[str, Any]] | None) -> str:
    if not pruned_raw:
        return "    (no families pruned)"
    suggested_index = {
        str(entry.get("family") or "").strip(): entry
        for entry in (suggested_prunes or [])
        if isinstance(entry, dict)
    }
    out: list[str] = []
    out.append(
        f"    {'family':28s} {'source':14s} {'reason'}"
    )
    out.append("    " + "-" * 70)
    for entry in pruned_raw:
        fam = str(entry.get("family") or "?")
        source = str(entry.get("source") or "?")
        reason = str(entry.get("reason") or "")
        suggested = suggested_index.get(fam)
        if suggested is not None:
            conf = str(suggested.get("confidence") or "?").upper()
            tag = f"[from-analyzer:{conf}]"
        else:
            tag = "[main-llm-only]"
        out.append(f"    {fam:28s} {source:14s} {tag} {reason}")
    return "\n".join(out)


def _format_roofline_block(rfa: dict[str, Any]) -> str:
    if not rfa:
        return "    (not yet run / cache empty)"
    out: list[str] = []
    snap = rfa.get("snapshot_id", "?")
    ts = rfa.get("analyzed_at_iso") or "?"
    gain = rfa.get("analyzed_at_gain_pct", 0.0)
    out.append(f"    snapshot_id={snap}  analyzed_at_gain={gain:.2f}%  ts={ts}")
    if rfa.get("error"):
        out.append(f"    DEGRADED error={rfa.get('error')}")
    primary = rfa.get("primary_bottleneck") or "unknown"
    dist = rfa.get("bottleneck_distribution") or {}
    dist_str = ", ".join(
        f"{k}={float(v) * 100:.0f}%"
        for k, v in sorted(
            dist.items(), key=lambda kv: -float(kv[1] or 0.0),
        )
        if isinstance(v, (int, float))
    ) or "unavailable"
    out.append(f"    primary={primary}  distribution=[{dist_str}]")
    sp = rfa.get("suggested_prunes") or []
    if sp:
        out.append("    suggested_prunes:")
        for entry in sp:
            if isinstance(entry, dict):
                out.append(
                    f"      - {(entry.get('confidence') or '?').upper():4s} "
                    f"{entry.get('family','?'):28s} {entry.get('reason','')}"
                )
    sn = rfa.get("suggested_next_actions") or []
    if sn:
        out.append("    suggested_next_actions:")
        for entry in sn:
            if isinstance(entry, dict):
                out.append(
                    f"      - {(entry.get('priority') or '?').upper():4s} "
                    f"{entry.get('kind','?'):28s} {entry.get('rationale','')}"
                )
    rr = rfa.get("reprofile_recommended")
    if rr:
        out.append(
            f"    reprofile_recommended=true  reason={rfa.get('reprofile_reason','')}"
        )
    return "\n".join(out)


def render(report: AuditReport) -> str:
    out: list[str] = []
    out.append("=" * 72)
    out.append("Roofline-v2 decision audit")
    out.append("=" * 72)
    out.append("")
    out.append(f"  session_dir: {report.session_dir}")
    if report.error:
        out.append(f"  ERROR: {report.error}")
        return "\n".join(out)
    out.append(
        f"  cumulative_gain_validated_pct: "
        f"{report.cumulative_gain_validated_pct:.3f}%"
    )
    out.append(f"  optimization_stack size: {len(report.optimization_stack)}")
    out.append(f"  pruned_families:         {len(report.pruned_families_raw)}")
    out.append("")
    out.append("  --- last_roofline_analysis ---")
    out.append(_format_roofline_block(report.last_roofline_analysis))
    out.append("")
    out.append("  --- pruned_families (with analyzer cross-reference) ---")
    out.append(_format_prune_table(
        report.pruned_families_raw,
        report.last_roofline_analysis.get("suggested_prunes"),
    ))
    out.append("")
    out.append("  --- optimization_stack (chronological) ---")
    out.append(_format_action_stack(report.optimization_stack))
    out.append("")
    out.append("  --- advice consumption (vs last_roofline_analysis) ---")
    out.append(
        f"    prune advice consumed: {report.advice_consumed_count}/"
        f"{report.advice_consumed_count + report.advice_ignored_count}"
    )
    out.append(
        f"    next-action advice followed: {report.next_action_followed_count}/"
        f"{report.next_action_followed_count + report.next_action_ignored_count}"
    )
    out.append("=" * 72)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Roofline-v2 decisions in a single session_dir.",
    )
    p.add_argument("--session", required=True, type=Path,
                   help="Session directory containing state.json")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of pretty text")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = extract(args.session)
    if args.json:
        payload = {
            "session_dir": str(report.session_dir),
            "has_state_json": report.has_state_json,
            "error": report.error,
            "cumulative_gain_validated_pct":
                report.cumulative_gain_validated_pct,
            "optimization_stack_size": len(report.optimization_stack),
            "pruned_families_count": len(report.pruned_families_raw),
            "pruned_families": report.pruned_families_raw,
            "last_roofline_analysis": report.last_roofline_analysis,
            "advice_consumed_count": report.advice_consumed_count,
            "advice_ignored_count": report.advice_ignored_count,
            "next_action_followed_count": report.next_action_followed_count,
            "next_action_ignored_count": report.next_action_ignored_count,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render(report))
    return 0 if report.has_state_json else 1


if __name__ == "__main__":
    sys.exit(main())
