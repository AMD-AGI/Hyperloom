"""Per-run token accounting from the fetched Pulse bundles (ledger + session_breakdown).

One archive can hold several run directories (resumes, retries); each run directory
with its own ledger becomes one record.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BUNDLES = Path(os.environ.get("PULSE_BUNDLES", "/root/pulse15d/02_bundles"))
OUT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d")) / "analysis"
TOKEN_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def weighted(row: dict) -> float:
    """Anthropic-relative units: cache write 1.25x, cache read 0.1x, output 5x of plain input."""
    return (
        (row.get("input_tokens") or 0)
        + 1.25 * (row.get("cache_creation_input_tokens") or 0)
        + 0.1 * (row.get("cache_read_input_tokens") or 0)
        + 5 * (row.get("output_tokens") or 0)
    )


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
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for shard in sorted(p for p in BUNDLES.iterdir() if p.is_dir()):
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
    with open(OUT / "runs_ledger.jsonl", "w") as fh:
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
