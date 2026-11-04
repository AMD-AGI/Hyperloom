"""A round's record: rounds/rNN/round.json, and ROUND.md, which the next round's dossier A reads.

It holds the map (each direction, its lane and its outcome), every lane's
clocks and items, the integration table, the round's claim, and what the next
round should look at: the drops and conflicts it may reopen, the directions
the lanes proposed, architectural trades for the operator, and the rewrites
waiting for an approval. STOP.md is written the same way when the run stops.
"""

from __future__ import annotations

from pathlib import Path

from arbor import directions, stages
from arbor.parallel import integrate, kinds, lanes as lanes_mod
from arbor.parallel.store import atomic_write_json, atomic_write_text, read_json

dur = directions.dur


def _trades(rec) -> list[dict]:
    """An architectural lane's ARM tables that trade one band for another (for the operator)."""
    if rec.kind != "architectural" or not rec.session_root:
        return []
    st = stages.load(rec.session_root) or {"items": {}}
    out = []
    for iid, it in st["items"].items():
        for a in it.get("attempts") or []:
            pts = (a.get("measured") or {}).get("points")
            if isinstance(pts, list) and kinds.classify_points(pts, 0.5, 1.0) == "trade":
                out.append({"lane": rec.lane, "item": iid, "iteration": a.get("iteration"), "points": pts[:12]})
    return out


def awaiting_approval(store, n: int) -> list[str]:
    doc = read_json(store.round_dir(n) / "directions.json", {}) or {}
    approved = store.approvals()
    return [d["id"] for d in doc.get("directions") or []
            if d.get("kind") == "rewrite" and d.get("status") in ("open", "plan") and d["id"] not in approved]


def round_record(store, n: int, now: float) -> dict:
    doc = read_json(store.round_dir(n) / "directions.json", {}) or {}
    integ = integrate.load(store, n) or {}
    recs = lanes_mod.load_all(store, n)
    lanes = [{"lane": r.lane, "direction": r.direction, "kind": r.kind, "state": r.state, "nodes": r.nodes,
              "depends_on": r.depends_on, "clocks": lanes_mod.live_clocks(r, now), "outcome": r.outcome,
              "start": {k: (r.start or {}).get(k) for k in ("base", "deps")}, "failures": r.failures[-3:]}
             for r in recs]
    reopenable = [{"direction": a["direction"], "lane": a["lane"], "state": a["state"], "why": a.get("why", "")}
                  for a in integ.get("additions") or [] if a.get("reopenable")]
    started = (doc.get("started") or now)
    return {"round": n, "uk": doc.get("uk"), "source": doc.get("source"), "base": integ.get("base"),
            "wall_s": round(now - started, 1), "map": doc.get("directions") or [], "lanes": lanes,
            "integration": {k: integ.get(k) for k in ("order", "additions", "unchanged", "result", "failed")},
            "reopenable": reopenable,
            "proposals": [p | {"lane": r.lane} for r in recs for p in (r.outcome or {}).get("proposals") or []],
            "trades": [t for r in recs for t in _trades(r)], "awaiting_approval": awaiting_approval(store, n)}


def render(rec: dict) -> str:
    res = (rec["integration"] or {}).get("result") or {}
    out = [f"# Round {rec['round']} (dossier u{rec.get('uk')})", "",
           f"Base {rec.get('base')} -> {res.get('base_next', rec.get('base'))}; kept {res.get('kept', 0)} "
           f"addition(s); claim {res.get('claim_pct', 0.0):+.2f} % (the same-node A/Bs, compounded); "
           f"round wall time {dur(rec['wall_s']) or '-'}. Map: {rec.get('source')}.", "",
           "## Lanes", "",
           "| lane | direction | kind | outcome | KEEPs | wait deps | wait node | on node | chunks |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in rec["lanes"]:
        c, o = x["clocks"], x["outcome"] or {}
        out.append(f"| {x['lane']} | {x['direction']} | {x['kind']} | {o.get('status', x['state'])} | "
                   f"{', '.join(map(str, o.get('keeps') or [])) or '-'} | {dur(c.get('wait_deps_s')) or '-'} | "
                   f"{dur(c.get('wait_node_s')) or '-'} | {dur(c.get('on_node_s')) or '-'} | {c.get('chunks', 0)} |")
    out += ["", "## Items", ""]
    for x in rec["lanes"]:
        for i in (x["outcome"] or {}).get("items") or []:
            out.append(f"- `{i['id']}` {i.get('stage') or ''} **{i['status']}** "
                       f"({dur(i.get('wall_s')) or '-'}): {(i.get('why') or '')[:200]}")
    out += ["", "## Integration", "", "| k | lane | direction | state | delta | spread | node | why |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in (rec["integration"] or {}).get("additions") or []:
        ab = a.get("ab") or {}
        out.append(f"| {a['k']} | {a['lane']} | {a['direction']} | {a['state']} | "
                   f"{ab.get('delta_pct') if ab.get('delta_pct') is not None else '-'} | "
                   f"{ab.get('base_spread_pct') if ab.get('base_spread_pct') is not None else '-'} | "
                   f"{ab.get('node') or '-'} | {(a.get('why') or '')[:160]} |")
    for u in (rec["integration"] or {}).get("unchanged") or []:
        out.append(f"| - | {u['lane']} | {u['direction']} | nothing to add | - | - | - | {u['why'][:160]} |")
    if (rec["integration"] or {}).get("failed"):
        out += ["", f"**The integration failed:** {rec['integration']['failed']}"]
    out += ["", "## For the next round", ""]
    out += [f"- reopenable: {r['direction']} ({r['state']}): {r['why'][:200]}" for r in rec["reopenable"]]
    out += [f"- proposed by {p['lane']}: {p.get('id')} ({p.get('kind')}): {p.get('title', '')[:160]}"
            for p in rec["proposals"]]
    out += [f"- trade for the operator: {t['lane']} {t['item']} at iteration {t['iteration']}" for t in rec["trades"]]
    out += [f"- rewrite awaiting approval: {d}" for d in rec["awaiting_approval"]]
    return "\n".join(out) + "\n"


def write_round(store, n: int, now: float) -> dict:
    rec = round_record(store, n, now)
    atomic_write_json(store.round_dir(n) / "round.json", rec)
    atomic_write_text(store.round_dir(n) / "ROUND.md", render(rec))
    return rec


def write_stop(store, state: dict, now: float) -> str:
    """STOP.md: why the run stopped, its rounds, lanes and clocks, and what waits for the operator."""
    n = int(state.get("round") or 0)
    stop = state.get("stop") or {}
    out = [f"# Run stopped: {stop.get('why', '?')}", "",
           f"Rule: {stop.get('rule', '?')}. Round {n}, phase {stop.get('phase', '?')}. "
           + ("`arbor.parallel resume` continues it." if stop.get("resume_to") else "It does not resume."), "",
           "## Rounds", ""]
    for m in range(1, n + 1):
        integ = integrate.load(store, m) or {}
        res = integ.get("result") or {}
        recs = lanes_mod.load_all(store, m)
        out.append(f"- r{m:02d}: {len(recs)} lane(s), kept {res.get('kept', 0)}, claim "
                   f"{res.get('claim_pct', 0.0):+.2f} %" + (f"; failed: {integ['failed']}" if integ.get("failed") else ""))
        for r in recs:
            c = lanes_mod.live_clocks(r, now)
            out.append(f"  - {r.lane} {r.direction} ({r.kind}): {r.state}, {(r.outcome or {}).get('status', '-')}; "
                       f"waited {dur(c['wait_deps_s']) or '0s'} on deps, {dur(c['wait_node_s']) or '0s'} for a node, "
                       f"{dur(c['on_node_s']) or '0s'} on a node")
    waiting = awaiting_approval(store, n) if n else []
    if waiting:
        out += ["", "## Rewrites awaiting approval", ""] + [
            f"- {d}: `python3 -m arbor.parallel approve {d} --why ... --config <run.json>`" for d in waiting]
    text = "\n".join(out) + "\n"
    atomic_write_text(Path(store.root) / "STOP.md", text)
    return text
