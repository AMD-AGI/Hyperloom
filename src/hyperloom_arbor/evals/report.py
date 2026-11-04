"""report.py — per-session eval report.

``EVAL_REPORT.md`` is the verdict, not the record: it holds the numbers and
points at ``reports/trace/SESSION_LOG.md`` for the narrative, so the two do not
drift into two half-true accounts of the same run.

Adherence checks print the instruction they test. A number that says
``agent_dispatch_rate 40%`` is only actionable next to the line of
hyperloom-optimize.md it violates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute  # noqa: E402

#: Each adherence key paired with the instruction it tests, so a failing check
#: names the prompt line to go edit.
_ADHERENCE_RULES = [
    ("agent_dispatch_rate_pct",
     "Every iteration dispatches a specialist via the Agent tool",
     "hyperloom-optimize.md:165", lambda v: v is None or v >= 90),
    ("config_tier_exit_ok",
     "Config tier is abandoned by iteration 3",
     "hyperloom-optimize.md:70", lambda v: v is not False),
    ("research_protocol_rate_pct",
     "Framework/kernel changes are preceded by research",
     "hyperloom-optimize.md:304-319", lambda v: v is None or v >= 90),
    ("iterations_with_multiple_gates",
     "One change per gate",
     "hyperloom-optimize.md:254", lambda v: not v),
    ("gates_unverified",
     "Every gate's number is corroborated by a bench artifact",
     "evals/collect.py:load_artifacts", lambda v: not v),
    ("benchmark_harness_edits",
     "bench.sh / correctness.sh never modified",
     "README.md", lambda v: not v),
    ("benchmark_harness_edits_after_first_gate",
     "...and never mid-run (would break within-run comparison)",
     "README.md", lambda v: not v),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, dict):
        return ", ".join(f"{k}×{n}" for k, n in list(v.items())[:6]) or "—"
    return str(v)


def _table(title: str, data: dict[str, Any]) -> list[str]:
    lines = [f"### {title}\n", "| Metric | Value |", "|---|---|"]
    for k, v in data.items():
        lines.append(f"| {k.replace('_', ' ')} | {_fmt(v)} |")
    lines.append("")
    return lines


def render_report(session_dir: Path, price_in: float | None = None,
                  price_out: float | None = None) -> Path:
    m = compute(session_dir, price_in, price_out)
    out: list[str] = []

    out.append(f"# Arbor eval report — `{m['session']}`\n")
    header = [
        ("Model optimized", m.get("model")), ("Hardware", m.get("hardware")),
        ("Framework", m.get("framework")),
        # The model doing the optimizing — the A/B lever for a harness change,
        # and what determines whether reasoning text is capturable at all.
        ("Orchestrator", m.get("orchestrator_model")),
        ("Arbor SHA", m.get("arbor_git_sha")),
        ("Steps", m.get("steps")),
    ]
    out.append("| | |")
    out.append("|---|---|")
    for k, v in header:
        if v not in (None, ""):
            out.append(f"| **{k}** | {v} |")
    out.append("")
    out.append("Full narrative: [`reports/trace/SESSION_LOG.md`](reports/trace/SESSION_LOG.md)\n")

    o = m["outcome"]
    gain = o.get("total_gain_pct")
    if gain is not None:
        out.append(
            f"## Verdict: **{gain:+.1f}%** "
            f"({_fmt(o.get('baseline_tput'))} → {_fmt(o.get('best_tput'))} tok/s) "
            f"over {o.get('iterations_observed')} iterations\n"
        )
    else:
        out.append("## Verdict: no measured throughput change recorded\n")

    out += _table("Outcome", o)
    out += _table("Efficiency", m["efficiency"])

    out.append("### Protocol adherence\n")
    out.append("| Check | Instruction | Value | Pass |")
    out.append("|---|---|---|---|")
    a = m["adherence"]
    for key, desc, where, ok in _ADHERENCE_RULES:
        v = a.get(key, m['outcome'].get(key))
        passed = ok(v)
        # An unmeasurable check reads "n/a", never "fail" — a session without
        # marks has not violated anything, it simply was not instrumented.
        mark = "n/a" if v is None else ("✅" if passed else "❌")
        out.append(f"| {desc} | `{where}` | {_fmt(v)} | {mark} |")
    out.append("")
    out += _table("Adherence detail", a)
    out += _table("Pathology", m["pathology"])
    branches = m.get("branches") or []
    if branches:
        out.append("### Branches (Arbor's search unit)\n")
        out.append("| branch_id | tier | iter | steps | sub | wall s | out tok | think tok | gate | delta |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in branches:
            out.append(
                f"| `{b['branch_id']}` | {b['tier'] or '—'} | {b['iteration'] if b['iteration'] is not None else '—'} "
                f"| {b['steps']} | {b['subagent_steps']} | {b['wall_seconds']:,.0f} "
                f"| {b['output_tokens']:,} | {b.get('thinking_tokens', 0):,} | {b['decision'] or '—'} "
                f"| {b['delta_pct'] if b['delta_pct'] is not None else '—'} |"
            )
        out.append("")
        out.append("Hypothesis vs. what the gate actually tested — divergence means the "
                   "branch mark went stale (the agent pivoted without re-marking):\n")
        for b in branches:
            if not b["hypothesis"] and not b["gated_change"]:
                continue
            out.append(f"- **`{b['branch_id']}`**")
            if b["hypothesis"]:
                out.append(f"  - planned: {b['hypothesis']}")
            if b["gated_change"]:
                out.append(f"  - gated:   {b['gated_change']}")
        out.append("")

    out += _table("Tool usage", m["tools"])

    path = session_dir / "EVAL_REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    args = parser.parse_args(argv)

    path = render_report(Path(args.session_dir).resolve(), args.price_in, args.price_out)
    print(f"[arbor-trace] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
