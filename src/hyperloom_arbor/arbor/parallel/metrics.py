"""One row per tick in metrics.jsonl: where the run is, what its lanes cost, what it has gained.

    {"ts", "phase", "round", "round_wall_s", "nodes_busy", "max_nodes",
     "lanes": {state: count}, "lane_clocks": {lane: {wait_deps_s, wait_node_s, on_node_s, node_s, chunks}},
     "kept": n, "compounded_gain_pct"}

The compounded gain multiplies every finished round's claim (its kept
additions' same-node deltas): the run's own measured gain over base_000.
"""

from __future__ import annotations

from arbor.parallel import integrate, scheduler
from arbor.parallel import lanes as lanes_mod


def compounded(store, upto: int) -> tuple[float, int]:
    g, kept = 1.0, 0
    for m in range(1, upto + 1):
        res = (integrate.load(store, m) or {}).get("result") or {}
        g *= 1.0 + float(res.get("claim_pct") or 0.0) / 100.0
        kept += int(res.get("kept") or 0)
    return round((g - 1.0) * 100.0, 3), kept


def row(store, state: dict, max_nodes: int, now: float) -> dict:
    n = int(state.get("round") or 0)
    recs = lanes_mod.load_all(store, n) if n else []
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.state] = counts.get(r.state, 0) + 1
    gain, kept = compounded(store, n)
    started = state.get("round_started")
    return {"ts": now, "phase": state.get("phase"), "round": n,
            "round_wall_s": round(now - started, 1) if started else None,
            "nodes_busy": scheduler.busy([r.slot() for r in recs], max_nodes), "max_nodes": max_nodes,
            "lanes": counts,
            "lane_clocks": {r.lane: {k: v for k, v in lanes_mod.live_clocks(r, now).items() if k.endswith(("_s", "chunks"))}
                            for r in recs},
            "kept": kept, "compounded_gain_pct": gain}
