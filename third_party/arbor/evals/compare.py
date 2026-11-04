"""compare.py — rank many sessions side by side.

The campaign table in docs/archive/configs-mi355x-README.md compares four models on one
number each. This produces the same table with the process columns attached,
so two runs that both reached +30% can be told apart by what they spent and
whether they followed the protocol to get there.

Grouping is by whatever you ask for — ``--group-by model`` for the campaign
table, ``--group-by arbor_git_sha`` to A/B a prompt edit across the same model,
``--group-by orchestrator_model`` to A/B the Claude model doing the driving.
Note that ``model`` is the model being *optimized* (Qwen3-32B) while
``orchestrator_model`` is the one *doing* the optimizing (claude-opus-5).
Because ``meta.json`` records the Arbor commit that produced each session, a
prompt change is a groupable variable rather than something you have to
remember you made.

Hyperloom sessions can be included directly: they already write
``reports/trace/llm_calls.jsonl`` in the schema collect.py emits, so a
directory from either harness is a valid input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute  # noqa: E402

#: Columns for the default table: outcome first, then what it cost, then
#: whether the run behaved. Kept short enough to read in a terminal.
_COLUMNS = [
    ("session", lambda m: m.get("label") or m.get("session")),
    ("model", lambda m: m.get("model") or "—"),
    ("sha", lambda m: (m.get("arbor_git_sha") or "—")[:7]),
    ("gain %", lambda m: m["outcome"].get("total_gain_pct")),
    ("iters", lambda m: m["outcome"].get("iterations_observed")),
    ("keep %", lambda m: m["outcome"].get("keep_rate_pct")),
    ("hours", lambda m: (m["efficiency"].get("wall_clock_seconds") or 0) / 3600 or None),
    ("Mtok", lambda m: (m["efficiency"].get("total_tokens") or 0) / 1e6 or None),
    ("gain/hr", lambda m: _ratio(m["outcome"].get("total_gain_pct"),
                                 (m["efficiency"].get("wall_clock_seconds") or 0) / 3600)),
    ("gain/Mtok", lambda m: _ratio(m["outcome"].get("total_gain_pct"),
                                   (m["efficiency"].get("total_tokens") or 0) / 1e6)),
    ("agent %", lambda m: m["adherence"].get("agent_dispatch_rate_pct")),
    ("revert t%", lambda m: m["efficiency"].get("reverted_time_share_pct")),
]


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return round(num / den, 2)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def discover_sessions(roots: list[str]) -> list[Path]:
    """Find every collected session under the given roots.

    A session is any directory holding ``trajectory.jsonl``, so pointing this
    at a campaign root picks up every model without enumerating them.
    """
    found: list[Path] = []
    for root in roots:
        rp = Path(root).resolve()
        if (rp / "trajectory.jsonl").exists():
            found.append(rp)
            continue
        found.extend(sorted(p.parent for p in rp.rglob("trajectory.jsonl")))
    return found


def build_table(sessions: list[Path], price_in: float | None,
                price_out: float | None) -> tuple[list[dict[str, Any]], str]:
    metrics = []
    for s in sessions:
        try:
            metrics.append(compute(s, price_in, price_out))
        except Exception as exc:
            print(f"[arbor-trace] skipped {s}: {exc}", file=sys.stderr)

    # Session directories are UTC timestamps, so a campaign root that ran four
    # models the same night yields four identical labels. Qualify with the
    # parent only when the bare name is ambiguous.
    counts: dict[str, int] = {}
    for m in metrics:
        counts[m["session"]] = counts.get(m["session"], 0) + 1
    for m in metrics:
        if counts[m["session"]] > 1:
            m["label"] = f"{Path(m['session_dir']).parent.name}/{m['session']}"

    header = "| " + " | ".join(c[0] for c in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    rows = [header, sep]
    for m in metrics:
        rows.append("| " + " | ".join(_fmt(fn(m)) for _, fn in _COLUMNS) + " |")
    return metrics, "\n".join(rows)


def group_summary(metrics: list[dict[str, Any]], key: str) -> str:
    """Mean gain and mean cost per group — the A/B answer."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for m in metrics:
        groups.setdefault(m.get(key) or "—", []).append(m)

    lines = [f"\n### Grouped by `{key}`\n",
             "| group | n | mean gain % | mean hours | mean Mtok | mean agent % |",
             "|---|---|---|---|---|---|"]
    for gk, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        def mean(fn):
            vals = [v for v in (fn(m) for m in group) if isinstance(v, (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None
        lines.append(
            f"| {gk} | {len(group)} "
            f"| {_fmt(mean(lambda m: m['outcome'].get('total_gain_pct')))} "
            f"| {_fmt(mean(lambda m: (m['efficiency'].get('wall_clock_seconds') or 0) / 3600))} "
            f"| {_fmt(mean(lambda m: (m['efficiency'].get('total_tokens') or 0) / 1e6))} "
            f"| {_fmt(mean(lambda m: m['adherence'].get('agent_dispatch_rate_pct')))} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+",
                        help="Session dirs, or roots to search recursively")
    parser.add_argument("--group-by", default=None,
                        choices=["model", "hardware", "framework", "arbor_git_sha",
                                 "orchestrator_model"])
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    parser.add_argument("--out", default=None, help="Write markdown here")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sessions = discover_sessions(args.roots)
    if not sessions:
        print("[arbor-trace] no collected sessions found "
              "(looked for trajectory.jsonl)", file=sys.stderr)
        return 1

    metrics, table = build_table(sessions, args.price_in, args.price_out)

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
        return 0

    doc = f"# Arbor session comparison\n\n{len(sessions)} sessions\n\n{table}\n"
    if args.group_by:
        doc += group_summary(metrics, args.group_by) + "\n"

    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
        print(f"[arbor-trace] wrote {args.out}")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
