# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-run token accounting from the fetched Pulse bundles (ledger + session_breakdown).

One archive can hold several run directories (resumes, retries); each run directory
with its own ledger becomes one record.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from metrics import TOKEN_FIELDS, weighted
from round_env import analysis_dir, bundles_dir


def run_root(ledger: Path) -> Path:
    parent = ledger.parent
    if parent.name == "trace" and parent.parent.name == "reports":
        return parent.parent.parent
    return parent


def meta_from_breakdown(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    md = d.get("metadata") or {}
    sess = md.get("session") or {}
    tc = md.get("task_config") or {}
    oc = d.get("outcome") or {}
    fin = oc.get("final") or {}
    val = oc.get("validation") or {}
    return {
        "code_revision": sess.get("code_revision"),
        "user": sess.get("sandbox_user_id"),
        "elapsed_min": sess.get("elapsed_minutes"),
        "max_min": sess.get("max_minutes"),
        "ticks": sess.get("tick_count"),
        "created": sess.get("created_at_utc"),
        "model_name": tc.get("model_name"),
        "model_path": tc.get("model_path"),
        "framework": tc.get("framework_name"),
        "gpu": tc.get("gpu_type"),
        "tp": tc.get("tp"),
        "conc": tc.get("conc"),
        "isl": tc.get("isl"),
        "osl": tc.get("osl"),
        "precision": tc.get("precision"),
        "stage_reached": oc.get("stage_reached"),
        "status": oc.get("status"),
        "stop_reason": oc.get("stop_reason"),
        "gain_pct": fin.get("gain_pct"),
        "validated_gain_pct": val.get("validated_total_gain_pct"),
        "benchmark_mode": (md.get("grading") or {}).get("benchmark_mode"),
        "objective": (md.get("grading") or {}).get("objective"),
        "critic_iterations": len((d.get("critic") or {}).get("iterations") or []),
    }


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open(errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(Counter)
    models = defaultdict(Counter)
    phases = defaultdict(Counter)
    calls = Counter()
    first_ts, last_ts = None, None
    for r in rows:
        comp = str(r.get("component") or "?")
        calls[comp] += 1
        for f in TOKEN_FIELDS:
            by[comp][f] += r.get(f) or 0
        w = weighted(r)
        by[comp]["weighted"] += w
        models[comp][str(r.get("model"))] += 1
        phases[comp][str(r.get("phase"))] += w
        if r.get("status") not in (None, "ok"):
            by[comp]["errors"] += 1
        ts = r.get("ts")
        if ts:
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
    return {
        "calls": dict(calls),
        "tokens": {c: dict(v) for c, v in by.items()},
        "models": {c: dict(v) for c, v in models.items()},
        "phase_weighted": {c: dict(v) for c, v in phases.items()},
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def main() -> None:
    out = analysis_dir()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for shard in sorted(p for p in bundles_dir().iterdir() if p.is_dir()):
        for archive in sorted(p for p in shard.iterdir() if p.is_dir()):
            for ledger in sorted(archive.rglob("llm_calls.jsonl")):
                root = run_root(ledger)
                rows = load_rows(ledger)
                if not rows:
                    continue
                records.append(
                    {
                        "name": archive.name,
                        "run_dir": str(root.relative_to(archive)),
                        **meta_from_breakdown(root / "session_breakdown.json"),
                        **summarize(rows),
                    }
                )
    with open(out / "runs_ledger.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(
        len(records),
        "run directories with a ledger from",
        len({r["name"] for r in records}),
        "archives",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
