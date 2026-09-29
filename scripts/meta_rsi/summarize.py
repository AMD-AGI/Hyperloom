#!/usr/bin/env python3
"""Aggregate ``analyze.py`` results of many sessions into markdown evidence tables.

python scripts/meta_rsi/summarize.py RESULTS_DIR [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(results_dir: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def _m(x: float) -> str:
    return f"{x / 1e6:.1f}M"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _merge_anatomy(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call-weighted merge of per-session section tables."""
    tok: Counter[str] = Counter()
    same: Counter[str] = Counter()
    calls = 0
    total_tok = 0.0
    sessions: Counter[str] = Counter()
    for a in items:
        n = a.get("calls") or 0
        if not n:
            continue
        calls += n
        total_tok += (a.get("avg_tok") or 0) * n
        for s in a.get("sections") or []:
            t = s["avg_tok"] * n
            tok[s["section"]] += t
            same[s["section"]] += t * s["unchanged_vs_prev"]
            sessions[s["section"]] += 1
    rows = []
    for sec, t in tok.most_common():
        rows.append(
            {
                "section": sec,
                "avg_tok": t / calls if calls else 0,
                "share": t / total_tok if total_tok else 0,
                "unchanged": same[sec] / t if t else 0,
                "sessions": sessions[sec],
            }
        )
    return rows


def summarize(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    add = lines.append
    n = len(results)
    comp: dict[str, Counter[str]] = defaultdict(Counter)
    comp_phase: dict[str, Counter[str]] = defaultdict(Counter)
    grand: Counter[str] = Counter()
    for r in results:
        t = r["ledger"]["tokens"]
        grand.update({k: t["total"][k] for k in ("billed", "weighted", "calls")})
        for k, v in t["by_component"].items():
            comp[k].update({x: v[x] for x in ("billed", "weighted", "calls")})
        for k, v in t["by_component_phase"].items():
            comp_phase[k].update({x: v[x] for x in ("billed", "weighted", "calls")})

    add(f"# Meta-RSI session mining ({n} sessions)\n")
    add(
        f"Total: {grand['calls']} LLM calls, {_m(grand['billed'])} billed tokens, "
        f"{_m(grand['weighted'])} cost-weighted (cache read x0.1, cache write x1.25, output x5).\n"
    )

    add("## Tokens by component\n")
    add("| component | calls | billed | share | weighted | share |")
    add("|---|---:|---:|---:|---:|---:|")
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]["weighted"]):
        add(
            f"| {k} | {v['calls']} | {_m(v['billed'])} | {_pct(v['billed'] / grand['billed'])} | "
            f"{_m(v['weighted'])} | {_pct(v['weighted'] / grand['weighted'])} |"
        )

    add("\n## Top component x phase (by weighted)\n")
    add("| component@phase | calls | billed | weighted | share |")
    add("|---|---:|---:|---:|---:|")
    for k, v in sorted(comp_phase.items(), key=lambda kv: -kv[1]["weighted"])[:12]:
        add(
            f"| {k} | {v['calls']} | {_m(v['billed'])} | {_m(v['weighted'])} | {_pct(v['weighted'] / grand['weighted'])} |"
        )

    # orchestration anatomy per phase, merged across sessions
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for ph, a in (r.get("orchestration") or {}).get("anatomy_by_phase", {}).items():
            by_phase[ph].append(a)
    for ph in sorted(by_phase, key=lambda p: -sum(a.get("calls", 0) for a in by_phase[p])):
        items = by_phase[ph]
        calls = sum(a.get("calls", 0) for a in items)
        if calls < 5:
            continue
        avg = sum((a.get("avg_tok") or 0) * a.get("calls", 0) for a in items) / calls
        add(
            f"\n## Orchestration prompt anatomy: {ph} ({calls} calls in {len(items)} sessions, avg {avg / 1e3:.1f}k tok)\n"
        )
        add("| section | avg tok | share | unchanged vs prev call | sessions |")
        add("|---|---:|---:|---:|---:|")
        for row in _merge_anatomy(items)[:14]:
            add(
                f"| `{row['section'][:70]}` | {row['avg_tok']:.0f} | {_pct(row['share'])} | {_pct(row['unchanged'])} | {row['sessions']} |"
            )

    add("\n## Orchestration history overhead (context beyond this turn's prompt + system prompt)\n")
    add("| session | phase | history share p50 | history tok p50 | calls |")
    add("|---|---|---:|---:|---:|")
    for r in sorted(
        results, key=lambda r: -r["ledger"]["tokens"]["by_component"].get("orchestration", {}).get("billed", 0)
    )[:12]:
        o = r.get("orchestration") or {}
        for ph, share in sorted(o.get("history_share_p50", {}).items()):
            calls = (o.get("repetition") or {}).get(ph, {}).get("calls", 0)
            if calls >= 3:
                add(
                    f"| {Path(r['session_dir']).name} | {ph} | {_pct(share)} | {o['history_tok_p50'].get(ph, 0)} | {calls} |"
                )

    add("\n## Orchestration repetition\n")
    add(
        "| session | phase | calls | consecutive repeats | re-send replies | escalations | max denials | ctx tok in re-send replies |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    rep_rows = []
    for r in results:
        for ph, v in ((r.get("orchestration") or {}).get("repetition") or {}).items():
            rep_rows.append((Path(r["session_dir"]).name, ph, v))
    for name, ph, v in sorted(rep_rows, key=lambda x: -x[2]["ctx_tok_in_resend_replies"])[:14]:
        add(
            f"| {name} | {ph} | {v['calls']} | {v['consecutive_repeat']} | {v['resend_replies']} | {v['escalations']} | "
            f"{v['max_denials_reported']} | {_m(v['ctx_tok_in_resend_replies'])} |"
        )

    add("\n## Critic prompt anatomy\n")
    rows = _merge_anatomy([(r.get("critic") or {}).get("anatomy") or {} for r in results])
    add("| section | avg tok | share | unchanged vs prev | sessions |")
    add("|---|---:|---:|---:|---:|")
    for row in rows[:10]:
        add(
            f"| `{row['section'][:70]}` | {row['avg_tok']:.0f} | {_pct(row['share'])} | {_pct(row['unchanged'])} | {row['sessions']} |"
        )

    add("\n## Specialist initial prompt anatomy (per task)\n")
    rows = _merge_anatomy([(r.get("specialist") or {}).get("initial_prompt_anatomy") or {} for r in results])
    add("| section | avg tok | share | sessions |")
    add("|---|---:|---:|---:|")
    for row in rows[:16]:
        add(f"| `{row['section'][:70]}` | {row['avg_tok']:.0f} | {_pct(row['share'])} | {row['sessions']} |")

    add("\n## Specialist task spend by journal verdict\n")
    spend: Counter[str] = Counter()
    tasks = 0
    growth = []
    turns = []
    for r in results:
        s = r.get("specialist") or {}
        spend.update(s.get("billed_by_verdict") or {})
        tasks += s.get("tasks") or 0
        if s.get("tasks"):
            growth.append(s.get("ctx_growth_per_turn_p50") or 0)
            turns.append(s.get("turns_p50") or 0)
    tot = sum(spend.values()) or 1
    add(
        f"{tasks} specialist tasks; per-session median turns {sorted(turns)[len(turns) // 2] if turns else 0}, "
        f"median context growth per turn {sorted(growth)[len(growth) // 2] if growth else 0:.0f} tok.\n"
    )
    add("| verdict | billed | share |")
    add("|---|---:|---:|")
    for k, v in spend.most_common():
        add(f"| {k} | {_m(v)} | {_pct(v / tot)} |")

    add("\n## Forge\n")
    add("| session | calls | billed | largest contexts |")
    add("|---|---:|---:|---|")
    for r in sorted(results, key=lambda r: -(r.get("forge") or {}).get("billed", 0))[:6]:
        f = r.get("forge") or {}
        if f.get("calls"):
            add(
                f"| {Path(r['session_dir']).name} | {f['calls']} | {_m(f['billed'])} | {', '.join(_m(x) for x in f['ctx'][:3])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results_dir")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    text = summarize(_load(Path(args.results_dir)))
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
