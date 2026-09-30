"""Summarize local session ledgers (this cluster's history) for scenario selection.

Scans PULSE_LOCAL_SESSIONS (colon-separated roots) for run directories modified since
--since, and writes one record per run with its breakdown metadata, token usage and cost.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from an_era import run_cost
from an_ledger import load_rows, meta_from_breakdown, run_root, summarize

ROOT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d"))
LOCAL = os.environ.get("PULSE_LOCAL_SESSIONS", "/wekafs/csl/Hyperloom-Sessions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-01", help="only runs whose ledger changed on or after this date")
    ap.add_argument("--max-depth", type=int, default=7)
    args = ap.parse_args()
    cutoff = time.mktime(time.strptime(args.since, "%Y-%m-%d"))
    out = []
    for base in (Path(p) for p in LOCAL.split(":") if p):
        for ledger in base.glob("**/reports/trace/llm_calls.jsonl"):
            if "pulse15d" in ledger.parts or len(ledger.relative_to(base).parts) > args.max_depth:
                continue
            if ledger.stat().st_mtime < cutoff:
                continue
            root = run_root(ledger)
            rows = load_rows(ledger)
            if not rows:
                continue
            rec = {"path": str(root), **meta_from_breakdown(root / "session_breakdown.json"), **summarize(rows)}
            rec["cost"], rec["cost_by_comp"] = run_cost(rec)
            rec["orch_by_phase"] = {}
            for r in rows:
                if r.get("component") == "orchestration":
                    ph = r.get("phase")
                    rec["orch_by_phase"][ph] = rec["orch_by_phase"].get(ph, 0) + 1
            out.append(rec)
    dest = ROOT / "analysis" / "weka_sep_runs.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    print(f"{len(out)} local runs -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
