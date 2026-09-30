"""Replay orchestration ticks: how many calls saw nothing new since the previous call.

A call is "unchanged" when its prompt, normalized by the rules below, equals the prompt of
the previous orchestration call in the same run. Normalization drops what changes on every
tick without carrying information: clock and budget counters, the tick number, the model's
own previous summary (current_action), inbox sequence numbers and message ids; repeated
identical inbox payloads collapse to one.

For each heartbeat H (minutes), a gate that skips unchanged calls but lets one through at
least every H minutes is replayed; saved cost uses the ledger row with the same call_id.
A skipped call whose recorded reply reports new work counts as a miss.
"""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from an_global import cost, price_family, weighted

BUNDLES = Path(os.environ.get("PULSE_BUNDLES", "/root/pulse15d/02_bundles"))
OUT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d")) / "analysis"
HEARTBEATS = (0, 5, 15, 30)

DROP_LINE = re.compile(r"^\s*(budget\s*:|reloop\s*:|time\s*:|elapsed=|current_action=)")
SUBS = [
    (re.compile(r"\btick[= ]\d+"), "tick=N"),
    (re.compile(r"\bseq=\d+\s+msg_id=[0-9a-f]+"), "seq=N msg_id=X"),
    (re.compile(r"\bmsg_id=[0-9a-f]{16,}"), "msg_id=X"),
    (re.compile(r"\d+(\.\d+)?\s*(min|sec|s)\b"), "T"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?"), "TS"),
]
ACTION = re.compile(
    r"\b(dispatch(ed)?|emitted|launched|started (a|the|two|three)|requested|queued|kicked off|submitted|integrat(ed|ing))\b",
    re.I,
)
IDLE = re.compile(
    r"(no change|nothing (new|changed)|still (blocked|waiting|running)|didn't start|did not start|no new work|no action|waiting for)",
    re.I,
)


def normalize(prompt: str) -> str:
    out, seen = [], set()
    for line in prompt.splitlines():
        if DROP_LINE.match(line):
            continue
        for rx, rep in SUBS:
            line = rx.sub(rep, line)
        if "msg_id=X" in line:
            if line in seen:
                continue
            seen.add(line)
        out.append(line)
    return "\n".join(out)


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def ledger_rows(run_dir: Path) -> dict:
    by_call = {}
    path = run_dir / "reports/trace/llm_calls.jsonl"
    if not path.exists():
        return by_call
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("component") == "orchestration" and r.get("call_id"):
            by_call[r["call_id"]] = r
    return by_call


def main() -> None:
    joined = {}
    for line in open(OUT / "runs_joined.jsonl"):
        j = json.loads(line)
        joined[(j["name"], j["run_dir"])] = j
    totals = defaultdict(Counter)
    per_run = []
    for conv in sorted(BUNDLES.glob("*/*/**/reports/trace/conversations.jsonl")):
        run_dir = conv.parent.parent.parent
        archive = conv.relative_to(BUNDLES).parts[1]
        rel = str(run_dir.relative_to(BUNDLES / conv.relative_to(BUNDLES).parts[0] / archive))
        meta = joined.get((archive, rel), {})
        led = ledger_rows(run_dir)
        rows = []
        for line in conv.open(errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("component") == "orchestration" and r.get("prompt"):
                rows.append(r)
        if len(rows) < 3:
            continue
        rows.sort(key=lambda r: r.get("ts") or "")
        digests = [hashlib.sha1(normalize(r["prompt"]).encode()).hexdigest() for r in rows]
        stats = Counter()
        for hb in HEARTBEATS:
            last_sent_ts = None
            prev_digest = None
            for r, dg in zip(rows, digests):
                t = ts(r["ts"]) if r.get("ts") else 0.0
                led_row = led.get(r.get("call_id"), {})
                c = cost(led_row, price_family(led_row.get("model") or r.get("model")))
                w = weighted(led_row)
                unchanged = prev_digest is not None and dg == prev_digest
                due = hb > 0 and last_sent_ts is not None and (t - last_sent_ts) >= hb * 60
                if unchanged and not due:
                    stats[f"skip_{hb}"] += 1
                    stats[f"skip_cost_{hb}"] += c
                    stats[f"skip_w_{hb}"] += w
                    if ACTION.search(r.get("response") or "") and not IDLE.search(r.get("response") or ""):
                        stats[f"miss_{hb}"] += 1
                else:
                    last_sent_ts = t
                prev_digest = dg
                if hb == 0:
                    stats["calls"] += 1
                    stats["cost"] += c
                    stats["w"] += w
                    stats["phase_" + str(r.get("phase"))] += 1
        era = meta.get("era", "?")
        per_run.append({"name": archive, "run_dir": rel, "era": era, **stats})
        for k, v in stats.items():
            totals[era][k] += v
    with open(OUT / "idle_replay.jsonl", "w") as fh:
        for r in per_run:
            fh.write(json.dumps(r) + "\n")
    lines = []
    for era, t in sorted(totals.items()):
        lines.append(
            f"## era={era} runs={sum(1 for r in per_run if r['era'] == era)} orchestration calls={t['calls']:,} cost=${t['cost']:,.0f}"
        )
        for hb in HEARTBEATS:
            n = t[f"skip_{hb}"]
            lines.append(
                f"  heartbeat={hb:>2} min: skip {n:,} calls ({100 * n / max(1, t['calls']):.1f}%), "
                f"${t[f'skip_cost_{hb}']:,.0f} ({100 * t[f'skip_cost_{hb}'] / max(1e-9, t['cost']):.1f}% of orchestration cost), "
                f"misses {t[f'miss_{hb}']} ({100 * t[f'miss_{hb}'] / max(1, n):.1f}% of skipped)"
            )
    top = sorted((r for r in per_run if r["era"] == "recent"), key=lambda r: -r.get("skip_cost_15", 0))[:10]
    lines.append("## recent runs with the largest savings at heartbeat=15")
    for r in top:
        lines.append(
            f"  {r['name'][:55]:55s} calls={r['calls']:5d} skip15={r.get('skip_15', 0):5d} ${r.get('skip_cost_15', 0):7.1f} of ${r['cost']:7.1f} misses={r.get('miss_15', 0)}"
        )
    text = "\n".join(lines)
    (OUT / "idle_replay_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
