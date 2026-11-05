"""A parallel run's status as one JSON object: the dashboard's payload (`arbor.parallel status --json`).

Standard library only, and no import from this package, so that the dashboard
server can send this file inside its command (arbor/dashboard/ship.mjs) to the
host where the store is, as it sends the realtime collector:

    python3 -c '<this file>' --store <store>/<run_id>

It reads the store and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

TAIL_EVENTS = 40


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _text(path, limit=4000):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()[:limit]
    except OSError:
        return None


def _tail_jsonl(path, n):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 65536))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def live_clocks(rec, now):
    """A lane's clocks as of `now`, the interval in progress counted (lanes.py keeps them)."""
    c = rec.get("clocks") or {}
    span = now - c["since"] if c.get("since") is not None else 0.0
    out = {k: c.get(k) for k in ("created", "deps_met", "queued", "first_submitted", "started", "closed")}
    waiting_deps = rec.get("state") == "WAITING_DEPS" and c.get("created") is not None
    out["wait_deps_s"] = c.get("wait_deps_s", round(now - c["created"], 1) if waiting_deps else 0.0)
    out["wait_node_s"] = round(c.get("wait_node_s", 0.0) + (span if c.get("counting") == "wait" else 0.0), 1)
    out["on_node_s"] = round(c.get("on_node_s", 0.0) + (span if c.get("counting") == "run" else 0.0), 1)
    out["node_s"] = c.get("node_s", round(out["on_node_s"] * int(rec.get("nodes") or 1), 1))
    out["chunks"] = c.get("chunks", len(rec.get("jobs") or []))
    return out


def _items(rec):
    """The lane's items, live from its session's map (closed ones from LANE_CLOSED.json)."""
    doc = _read(os.path.join(rec.get("session_root") or "", "directions.json"), {}) or {}
    return [{k: d.get(k) for k in ("id", "stage", "status", "why", "wall_s", "worked", "keeps", "closed_at")}
            for d in doc.get("directions") or []]


def _lane(rd, name, now):
    rec = _read(os.path.join(rd, "lanes", name, "lane.json"), {}) or {}
    keep = ("lane", "direction", "kind", "title", "rank", "level", "nodes", "depends_on", "state", "head", "node",
            "jobs", "exclude", "resubmits", "outcome", "closing", "session_root")
    out = {k: rec.get(k) for k in keep}
    out["clocks"] = live_clocks(rec, now)
    out["failures"] = (rec.get("failures") or [])[-3:]
    out["start"] = {k: (rec.get("start") or {}).get(k) for k in ("base", "deps")}
    out["kept"] = (rec.get("final") or {}).get("kept")
    out["items"] = _items(rec)
    return out


def _round(root, n, now, full):
    rd = os.path.join(root, "rounds", f"r{n:02d}")
    lanes_dir = os.path.join(rd, "lanes")
    names = sorted(os.listdir(lanes_dir)) if os.path.isdir(lanes_dir) else []
    out = {"round": n, "map": _read(os.path.join(rd, "directions.json")),
           "map_step": _read(os.path.join(rd, "map", "map.json")),
           "integration": _read(os.path.join(rd, "integrate", "integration.json")),
           "summary": _read(os.path.join(rd, "round.json"), {}).get("integration") if not full else None}
    out["lanes"] = [_lane(rd, x, now) for x in names if os.path.isfile(os.path.join(lanes_dir, x, "lane.json"))]
    if not full:
        for x in out["lanes"]:
            x["items"] = []
    return out


def collect(root, now=None):
    root = str(root)
    now = time.time() if now is None else now
    state = _read(os.path.join(root, "state.json"), {}) or {}
    meta = _read(os.path.join(root, "run.json"), {}) or {}
    cfg = meta.get("config") or {}
    n = int(state.get("round") or 0)
    rows = _tail_jsonl(os.path.join(root, "metrics.jsonl"), 1)
    return {
        "collected_at": now, "store": root, "run_id": meta.get("run_id"), "arbor_sha": meta.get("arbor_sha"),
        "max_lanes": cfg.get("max_lanes", 4), "max_rounds": cfg.get("max_rounds", 0),
        "state": state, "host": _read(os.path.join(root, "HOST")), "base": _read(os.path.join(root, "bases", "current.json")),
        "approvals": _read(os.path.join(root, "approvals.json"), {}),
        "metrics": rows[-1] if rows else None,
        "rounds": [_round(root, m, now, full=(m == n)) for m in range(1, n + 1)],
        "attention": _text(os.path.join(root, "ATTENTION.md")), "stop_md": _text(os.path.join(root, "STOP.md")),
        "switches": {s: os.path.exists(os.path.join(root, s)) for s in ("STOP", "PAUSE")},
        "events": _tail_jsonl(os.path.join(root, "events.jsonl"), TAIL_EVENTS),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="a parallel run's status, as JSON")
    ap.add_argument("--store", required=True, help="the run's directory: <store>/<run_id>")
    a = ap.parse_args(argv)
    json.dump(collect(a.store), sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
