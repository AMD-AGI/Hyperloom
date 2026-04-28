"""Quick inspector for an inference-optimizer session DB.

Usage:
    python scripts/inspect_session.py [session_dir]

If no path is given, picks the newest under
``$INFERENCE_OPTIMIZER_SESSION_ROOT`` (or ``$TEMP/io-smoke``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path


def _find_session_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    root = os.environ.get(
        "INFERENCE_OPTIMIZER_SESSION_ROOT",
        os.path.join(os.environ.get("TEMP", "/tmp"), "io-smoke"),
    )
    candidates = sorted(
        Path(root).glob("*/storage/conductor.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.exit(f"no conductor.db under {root!r}")
    return candidates[0].parent.parent


def main() -> None:
    session_dir = _find_session_dir()
    db_path = session_dir / "storage" / "conductor.db"
    print(f"== session_dir: {session_dir}")
    print(f"== db_path:     {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"\nevents total: {n_events}")
    print("\nevents by (topic, from, to):")
    rows = conn.execute(
        "SELECT topic, from_agent, to_agent, COUNT(*) AS n "
        "FROM events GROUP BY topic, from_agent, to_agent ORDER BY n DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r['n']:5d}  {r['topic']:<20} {r['from_agent']} -> {r['to_agent']}")

    print("\ncursors:")
    for r in conn.execute("SELECT * FROM cursors").fetchall():
        print(f"  {dict(r)}")

    print("\nlast 5 events:")
    for r in conn.execute(
        "SELECT seq, topic, from_agent, to_agent, payload FROM events "
        "ORDER BY seq DESC LIMIT 5"
    ).fetchall():
        body = ""
        try:
            payload = json.loads(r["payload"])
            body = payload.get("body_md") or payload.get("kind") or ""
        except Exception:
            pass
        print(f"  seq={r['seq']:<4} {r['topic']:<18} {r['from_agent']:<10} -> "
              f"{r['to_agent']:<10}  {body}")

    print(f"\ntasks rows: {conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]}")
    print(f"leases rows: {conn.execute('SELECT COUNT(*) FROM leases').fetchone()[0]}")

    state_path = session_dir / "state.json"
    if state_path.exists():
        print(f"\nstate.json keys: {list(json.loads(state_path.read_text()).keys())}")


if __name__ == "__main__":
    main()
