#!/usr/bin/env python3
"""Compare a baseline vs experiment Hyperloom session for Roofline-v2.

Usage::

    python -m inference_optimizer.scripts.verify_roofline_v2 \\
        --baseline /tmp/roofline-v2/qwen3-baseline \\
        --exp      /tmp/roofline-v2/qwen3-exp

Reads ``state.json`` from each session and produces a side-by-side
comparison of the metrics relevant to design/roofline-v2.md §10.2:

* ``cumulative_gain_validated_pct`` (main hard target ≥ +5%)
* ``wall_clock_min`` derived from the session ts range
* ``profile_count`` / ``roofline_action_count`` (from attempts /
  optimization_stack / last_roofline_analysis)
* ``prune_branch_count`` (from pruned_families length)
* ``action_seq`` (compressed `kind:variant_name` list)
* ``last_roofline_analysis`` snapshot delta (primary_bottleneck +
  suggestion counts so the operator can eyeball whether the
  experiment actually consumed roofline advice)

Exit code 0 when ``delta cumulative_gain_validated_pct >= +5.0``;
exit code 2 when delta is positive but below the +5% threshold;
exit code 1 when delta is non-positive (regression / no improvement).

The script is read-only and pure-Python (no pandas / numpy
dependencies) so it can run anywhere a session_dir is mounted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Session metric extraction
# ---------------------------------------------------------------------------
@dataclass
class SessionMetrics:
    """Subset of SharedState we surface in the comparison table."""

    session_dir: Path
    cumulative_gain_validated_pct: float = 0.0
    wall_clock_min: float = 0.0
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    pruned_families: list[str] = field(default_factory=list)
    last_roofline_analysis: dict[str, Any] = field(default_factory=dict)
    profile_attempts: int = 0
    select_kernels_attempts: int = 0
    has_state_json: bool = False
    error: str = ""


def _safe_load_state_json(session_dir: Path) -> dict[str, Any] | None:
    state_path = session_dir / "state.json"
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_iso_or_none(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_clock_min(state: dict[str, Any]) -> float:
    """Approximate session wall-clock from cumulative_gain_validated_ts
    and the earliest baseline attempt ts. Both are stored as ISO-8601
    strings on SharedState."""
    candidates: list[datetime] = []
    earliest_keys = ("baseline_ts", "session_started_ts")
    for k in earliest_keys:
        dt = _parse_iso_or_none(state.get(k))
        if dt is not None:
            candidates.append(dt)
    history = state.get("baseline_attempts") or []
    for entry in history:
        if isinstance(entry, dict):
            dt = _parse_iso_or_none(entry.get("ts"))
            if dt is not None:
                candidates.append(dt)
                break
    end_dt = _parse_iso_or_none(state.get("cumulative_gain_validated_ts"))
    if end_dt is None:
        # Fall back to the latest attempt across the 6 audit actions.
        for action in (
            "validate_stack", "params", "backends",
            "sweep", "profile", "baseline",
        ):
            attempts = state.get(f"{action}_attempts") or []
            if attempts:
                last = attempts[-1]
                if isinstance(last, dict):
                    dt = _parse_iso_or_none(last.get("ts"))
                    if dt is not None and (end_dt is None or dt > end_dt):
                        end_dt = dt
    if not candidates or end_dt is None:
        return 0.0
    start = min(candidates)
    return max(0.0, (end_dt - start).total_seconds() / 60.0)


def extract(session_dir: Path) -> SessionMetrics:
    metrics = SessionMetrics(session_dir=session_dir)
    state = _safe_load_state_json(session_dir)
    if state is None:
        metrics.error = f"state.json not found / unreadable at {session_dir}"
        return metrics
    metrics.has_state_json = True
    try:
        metrics.cumulative_gain_validated_pct = float(
            state.get("cumulative_gain_validated", 0.0),
        )
    except (TypeError, ValueError):
        metrics.cumulative_gain_validated_pct = 0.0
    metrics.wall_clock_min = _wall_clock_min(state)
    stack = state.get("optimization_stack") or []
    metrics.optimization_stack = [
        s for s in stack if isinstance(s, dict)
    ]
    pruned = state.get("pruned_families") or []
    metrics.pruned_families = [
        str(p["family"]) if isinstance(p, dict) and p.get("family") else str(p)
        for p in pruned
    ]
    rfa = state.get("last_roofline_analysis") or {}
    if isinstance(rfa, dict):
        metrics.last_roofline_analysis = rfa
    metrics.profile_attempts = len(state.get("profile_attempts") or [])
    # select_kernels attempts are not in the per-action audit list (it's
    # a request kind, not an action); approximate via roofline_snapshot_id
    # since each select_kernels increments the snapshot counter.
    select_state = state.get("last_select_kernels") or {}
    if isinstance(select_state, dict):
        snap = select_state.get("roofline_snapshot_id")
        if isinstance(snap, int):
            metrics.select_kernels_attempts = snap
    return metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _format_action_seq(stack: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in stack[:12]:  # cap so terminal output stays readable
        kind = str(entry.get("kind") or entry.get("action") or "?")
        variant = entry.get("variant_name") or entry.get("variant_id") or ""
        if variant:
            parts.append(f"{kind}:{variant}")
        else:
            parts.append(kind)
    suffix = "" if len(stack) <= 12 else f" ...(+{len(stack)-12} more)"
    return ", ".join(parts) + suffix if parts else "(empty)"


def _format_roofline_summary(rfa: dict[str, Any]) -> str:
    if not rfa:
        return "(not yet run / cache empty)"
    snap = rfa.get("snapshot_id", "?")
    primary = rfa.get("primary_bottleneck", "?")
    n_prunes = len(rfa.get("suggested_prunes") or [])
    n_next = len(rfa.get("suggested_next_actions") or [])
    return (
        f"snapshot={snap} primary={primary} "
        f"suggested_prunes={n_prunes} suggested_next={n_next}"
    )


def render(baseline: SessionMetrics, exp: SessionMetrics) -> str:
    delta_gain = (
        exp.cumulative_gain_validated_pct
        - baseline.cumulative_gain_validated_pct
    )
    delta_wall = exp.wall_clock_min - baseline.wall_clock_min
    out: list[str] = []
    out.append("=" * 72)
    out.append("Roofline-v2 baseline vs experiment comparison")
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
        f"  {'metric':32s} {'baseline':>14s} {'exp':>14s} {'delta':>10s}"
    )
    out.append("  " + "-" * 68)
    out.append(
        f"  {'cumulative_gain_validated_pct':32s} "
        f"{baseline.cumulative_gain_validated_pct:>13.3f}% "
        f"{exp.cumulative_gain_validated_pct:>13.3f}% "
        f"{delta_gain:>+9.3f}%"
    )
    out.append(
        f"  {'wall_clock_min':32s} "
        f"{baseline.wall_clock_min:>13.2f}  "
        f"{exp.wall_clock_min:>13.2f}  "
        f"{delta_wall:>+9.2f}"
    )
    out.append(
        f"  {'optimization_stack_size':32s} "
        f"{len(baseline.optimization_stack):>14d} "
        f"{len(exp.optimization_stack):>14d} "
        f"{len(exp.optimization_stack) - len(baseline.optimization_stack):>+10d}"
    )
    out.append(
        f"  {'pruned_families_count':32s} "
        f"{len(baseline.pruned_families):>14d} "
        f"{len(exp.pruned_families):>14d} "
        f"{len(exp.pruned_families) - len(baseline.pruned_families):>+10d}"
    )
    out.append(
        f"  {'profile_attempts':32s} "
        f"{baseline.profile_attempts:>14d} "
        f"{exp.profile_attempts:>14d} "
        f"{exp.profile_attempts - baseline.profile_attempts:>+10d}"
    )
    out.append(
        f"  {'select_kernels_attempts':32s} "
        f"{baseline.select_kernels_attempts:>14d} "
        f"{exp.select_kernels_attempts:>14d} "
        f"{exp.select_kernels_attempts - baseline.select_kernels_attempts:>+10d}"
    )
    out.append("  " + "-" * 68)
    out.append("")
    out.append("  action_seq (baseline):")
    out.append(f"    {_format_action_seq(baseline.optimization_stack)}")
    out.append("  action_seq (exp):")
    out.append(f"    {_format_action_seq(exp.optimization_stack)}")
    out.append("")
    out.append(f"  pruned_families (baseline): {baseline.pruned_families or '(none)'}")
    out.append(f"  pruned_families (exp):      {exp.pruned_families or '(none)'}")
    out.append("")
    out.append(f"  last_roofline_analysis (baseline): "
               f"{_format_roofline_summary(baseline.last_roofline_analysis)}")
    out.append(f"  last_roofline_analysis (exp):      "
               f"{_format_roofline_summary(exp.last_roofline_analysis)}")
    out.append("")
    # Verdict
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
                    "for Roofline-v2 verification (design §10.2).",
    )
    p.add_argument("--baseline", required=True, type=Path,
                   help="Baseline session_dir (contains state.json)")
    p.add_argument("--exp", required=True, type=Path,
                   help="Experiment session_dir (contains state.json)")
    p.add_argument("--json", action="store_true",
                   help="Also emit a machine-readable JSON summary on stderr")
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
                "cumulative_gain_validated_pct":
                    baseline.cumulative_gain_validated_pct,
                "wall_clock_min": baseline.wall_clock_min,
                "optimization_stack_size": len(baseline.optimization_stack),
                "pruned_families": baseline.pruned_families,
                "roofline_summary":
                    _format_roofline_summary(baseline.last_roofline_analysis),
            },
            "exp": {
                "session_dir": str(exp.session_dir),
                "cumulative_gain_validated_pct":
                    exp.cumulative_gain_validated_pct,
                "wall_clock_min": exp.wall_clock_min,
                "optimization_stack_size": len(exp.optimization_stack),
                "pruned_families": exp.pruned_families,
                "roofline_summary":
                    _format_roofline_summary(exp.last_roofline_analysis),
            },
            "delta": {
                "cumulative_gain_validated_pct":
                    exp.cumulative_gain_validated_pct
                    - baseline.cumulative_gain_validated_pct,
                "wall_clock_min":
                    exp.wall_clock_min - baseline.wall_clock_min,
            },
        }
        sys.stderr.write(json.dumps(summary, indent=2) + "\n")

    return _exit_code_for_delta(
        exp.cumulative_gain_validated_pct
        - baseline.cumulative_gain_validated_pct,
    )


if __name__ == "__main__":
    sys.exit(main())
