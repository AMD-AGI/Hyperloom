"""scripts/monitor.py — cross-platform port of monitor.sh.

Usage::

    python -m inference_optimizer.scripts.monitor                        # snapshot
    python -m inference_optimizer.scripts.monitor --watch 5              # tail
    python -m inference_optimizer.scripts.monitor --per-agent --per-lane

Exits ``0`` when the DB looks healthy and ``1`` when there are stale
running tasks or excessive cursor lag.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


__all__ = ["main", "snapshot"]


def _resolve_db(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("INFERENCE_OPTIMIZER_DB_PATH")
    if env:
        return Path(env)
    sd = os.environ.get("SESSION_DIR")
    if sd:
        return Path(sd) / "storage" / "conductor.db"
    raise SystemExit(
        "no DB resolved — pass --db, or set INFERENCE_OPTIMIZER_DB_PATH "
        "or SESSION_DIR"
    )


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("  (empty)")
        return
    keys = list(rows[0].keys())
    widths = [
        max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows))
        for k in keys
    ]
    line = "  ".join(k.ljust(w) for k, w in zip(keys, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)))


def snapshot(
    db_path: Path,
    *,
    per_agent: bool = False,
    per_lane: bool = False,
    top_events: int = 0,
    lag_threshold: int = 10,
) -> int:
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 3
    print(f"== {datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
          f"db={db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            {
                "what": "events",
                "n": conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"],
                "latest": conn.execute("SELECT MAX(ts) AS t FROM events").fetchone()["t"],
            },
            {
                "what": "in-flight tasks",
                "n": conn.execute(
                    "SELECT COUNT(*) AS n FROM tasks WHERE state IN ('queued','running')"
                ).fetchone()["n"],
                "latest": "",
            },
            {
                "what": "active leases",
                "n": conn.execute(
                    "SELECT COUNT(*) AS n FROM leases WHERE expires_at > ?",
                    (_iso_now(),),
                ).fetchone()["n"],
                "latest": "",
            },
        ]
        max_seq_row = conn.execute("SELECT MAX(seq) AS s FROM events").fetchone()
        max_seq = max_seq_row["s"] if max_seq_row and max_seq_row["s"] is not None else 0
        min_cur_row = conn.execute(
            "SELECT MIN(last_processed_seq) AS s FROM cursors"
        ).fetchone()
        min_cur = (min_cur_row["s"] if min_cur_row and min_cur_row["s"] is not None
                   else 0)
        lag = max_seq - min_cur
        rows.append({"what": "cursors lag", "n": lag, "latest": ""})
        _print_table(rows)

        if per_agent:
            print("\n-- per-agent cursor lag --")
            curs = conn.execute("SELECT * FROM cursors").fetchall()
            agent_rows = [
                {
                    "agent": c["agent"],
                    "cursor_seq": c["last_processed_seq"],
                    "lag": max_seq - c["last_processed_seq"],
                }
                for c in curs
            ]
            agent_rows.sort(key=lambda r: r["lag"], reverse=True)
            _print_table(agent_rows)

        if per_lane:
            print("\n-- active leases --")
            leases = conn.execute(
                "SELECT lane, holder_id, action, expires_at FROM leases "
                "WHERE expires_at > ? ORDER BY expires_at DESC",
                (_iso_now(),),
            ).fetchall()
            _print_table([dict(r) for r in leases])

        if top_events:
            print(f"\n-- last {top_events} events --")
            evs = conn.execute(
                "SELECT seq, from_agent, to_agent, topic, ts "
                "FROM events ORDER BY seq DESC LIMIT ?",
                (int(top_events),),
            ).fetchall()
            _print_table([dict(r) for r in evs])

        # health gate
        zombies = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks "
            "WHERE state='running' AND updated_at < ?",
            (_iso_minutes_ago(30),),
        ).fetchone()["n"]
        if lag > lag_threshold or zombies > 0:
            print(
                f"STATUS=degraded (lag={lag} threshold={lag_threshold} "
                f"zombies={zombies})",
                file=sys.stderr,
            )
            return 1
    finally:
        conn.close()
    return 0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _iso_minutes_ago(minutes: int) -> str:
    from datetime import timedelta
    t = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return t.isoformat(timespec="microseconds")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="monitor")
    p.add_argument("--db")
    p.add_argument("--watch", type=float, default=0.0)
    p.add_argument("--per-agent", action="store_true")
    p.add_argument("--per-lane", action="store_true")
    p.add_argument("--top-events", type=int, default=0)
    p.add_argument("--lag-threshold", type=int, default=10)
    args = p.parse_args(argv)

    db = _resolve_db(args.db)
    if args.watch <= 0:
        return snapshot(
            db,
            per_agent=args.per_agent,
            per_lane=args.per_lane,
            top_events=args.top_events,
            lag_threshold=args.lag_threshold,
        )
    while True:
        snapshot(
            db,
            per_agent=args.per_agent,
            per_lane=args.per_lane,
            top_events=args.top_events,
            lag_threshold=args.lag_threshold,
        )
        try:
            time.sleep(float(args.watch))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
