# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Summarize local session ledgers (this cluster's own history) for scenario selection.

Scans PULSE_LOCAL_SESSIONS (colon-separated roots) for run directories whose ledger changed on
or after --since (default PULSE_RECENT_SINCE) and writes one record per run with its breakdown
metadata, token usage and cost to ``analysis/local_runs.jsonl``. Ledgers under the round
directory itself are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from an_era import run_cost
from an_ledger import load_rows, meta_from_breakdown, run_root, summarize
from round_env import env_value, recent_since, round_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="only runs whose ledger changed on or after this date (YYYY-MM-DD)")
    ap.add_argument("--max-depth", type=int, default=7)
    args = ap.parse_args()
    root = round_dir()
    cutoff = time.mktime(time.strptime(args.since or recent_since(), "%Y-%m-%d"))
    out = []
    for base in (Path(p) for p in env_value("PULSE_LOCAL_SESSIONS").split(":") if p):
        for ledger in base.glob("**/reports/trace/llm_calls.jsonl"):
            if root in ledger.parents or len(ledger.relative_to(base).parts) > args.max_depth:
                continue
            if ledger.stat().st_mtime < cutoff:
                continue
            run = run_root(ledger)
            rows = load_rows(ledger)
            if not rows:
                continue
            rec = {"path": str(run), **meta_from_breakdown(run / "session_breakdown.json"), **summarize(rows)}
            rec["cost"], rec["cost_by_comp"] = run_cost(rec)
            rec["orch_by_phase"] = {}
            for r in rows:
                if r.get("component") == "orchestration":
                    ph = r.get("phase")
                    rec["orch_by_phase"][ph] = rec["orch_by_phase"].get(ph, 0) + 1
            out.append(rec)
    dest = root / "analysis" / "local_runs.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    print(f"{len(out)} local runs -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
