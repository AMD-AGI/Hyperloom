"""The drained round's integration: each lane's addition stacked on the base, one same-node A/B each.

Order: the lanes that kept something, by (level, rank, id), the order their
dependencies allow. Each lane is one addition, its final tree: landed.patch
is cumulative, so the lane's commit on its own start tree (final/manifest.json)
holds everything it kept, and a cherry-pick of it carries only its own work
onto the integrated tree, whatever the lane started from.

For addition k: merge it onto integrated k-1 (a 3-way merge, flags merged per
key). A conflict drops it, reopenable by the next round's dossier, with the
paths. Otherwise both trees are exported and `ab_model_m.sbatch` measures them on
one node, B C C B, under the pinned gates; `gatecheck.evaluate`, with the
lane's strongest booked intent choosing the bar, decides:

- KEEP: integrated k is the merge;
- REVERT: the addition is dropped, reopenable, with its numbers;
- NOT_MEASURED: once more on another node; twice is an integration failure,
  and the run stops for the operator (the measurement, not the addition, is
  broken).

When every addition is decided, base_{n+1} is published from the integrated
tree (if anything was kept), and the round's claim is the product of the kept
additions' same-node deltas. Everything is in integrate/integration.json, and
every A/B attempt in its own directory with a DONE marker.
"""

from __future__ import annotations

import json
from pathlib import Path

from arbor.parallel import gatecheck, graph
from arbor.parallel.cluster import track_poll, track_submit
from arbor.parallel.gatecheck import KEEP, NOT_MEASURED, ArmRun, GateRecord
from arbor.parallel.store import atomic_write_json, read_json
from arbor.parallel.trees import Conflict, FlagConflict, flags_to_workspace, merge_flag_sets

MAX_AB = 2
DECIDED = ("KEEP", "DROPPED", "CONFLICT")


def path(store, n: int) -> Path:
    return store.round_dir(n) / "integrate" / "integration.json"


def load(store, n: int) -> dict | None:
    return read_json(path(store, n))


def _save(ctx, n: int, doc: dict) -> None:
    atomic_write_json(path(ctx.store, n), doc)


# -- bases -----------------------------------------------------------------------------


def publish_base(ctx, index: int, sha: str, flags: dict, provenance: dict, aiter: Path | None = None) -> dict:
    """bases/base_NNN/: what the launcher deploys (base.diff, base.blobs) and the flags; then current.json."""
    d = ctx.store.base_dir(index)
    ctx.trees.tag(f"base_{index:03d}", sha)
    info = ctx.trees.export(sha, d / "base")
    atomic_write_json(d / "flags.json", flags)
    atomic_write_json(d / "workspace_config.json", flags_to_workspace(flags))
    if aiter is not None and aiter.is_file() and not (d / "landed_aiter.py").exists():
        (d / "landed_aiter.py").write_bytes(aiter.read_bytes())
    rec = {"index": index, "sha": sha, "files": info["files"], "prefix": str(d / "base"), "flags": flags,
           "ts": ctx.clock(), **provenance}
    atomic_write_json(d / "base.json", rec)
    atomic_write_json(ctx.store.bases / "current.json", rec)
    ctx.store.event("base.published", index=index, sha=sha, **{k: v for k, v in provenance.items()
                                                                if k in ("round", "source", "kept")})
    return rec


# -- the plan --------------------------------------------------------------------------


def plan(ctx, n: int, lanes: list) -> dict:
    """integration.json's first form: the additions in order, the integrated tree at the base."""
    base = ctx.store.current_base()
    kept = [r for r in lanes if (r.final or {}).get("kept")]
    entries = [{"id": r.lane, "rank": r.rank, "depends_on": r.depends_on} for r in lanes]
    order = [x for x in graph.order(entries, graph.deps(entries)) if x in {r.lane for r in kept}]
    by = {r.lane: r for r in lanes}
    doc = {"round": n, "base": f"base_{base['index']:03d}", "base_sha": base["sha"], "order": order,
           "additions": [{"k": k, "lane": x, "direction": by[x].direction, "state": "PENDING",
                          "intent": by[x].final.get("intent") or "WIN", "attempts": []}
                         for k, x in enumerate(order, 1)],
           "unchanged": [{"lane": r.lane, "direction": r.direction, "why": (r.final or {}).get("why", "")}
                         for r in lanes if r not in kept],
           "integrated": {"sha": base["sha"], "flags": base.get("flags") or {}},
           "result": None, "failed": None}
    _save(ctx, n, doc)
    ctx.store.event("integrate.planned", round=n, order=order)
    return doc


# -- one addition ----------------------------------------------------------------------


def _attempt_dir(ctx, n: int, a: dict) -> Path:
    return ctx.store.round_dir(n) / "integrate" / "ab" / str(a["k"]) / f"attempt-{len(a['attempts'])}"


def _prepare(ctx, n: int, doc: dict, a: dict, lane) -> bool:
    """Merge the addition onto the integrated tree; export both arms. False on a conflict."""
    f, cur = lane.final, doc["integrated"]
    merged = ctx.trees.merge(cur["sha"], f["commit"], message=f"r{n:02d} + {lane.lane}") if f.get("commit") \
        else cur["sha"]
    flags = merge_flag_sets(lane.start.get("flags") or {}, cur["flags"], f.get("flags") or {})
    if isinstance(merged, Conflict) or isinstance(flags, FlagConflict):
        a["paths"] = list(merged.paths if isinstance(merged, Conflict) else ()) + \
            [f"flag:{k}" for k in (flags.keys if isinstance(flags, FlagConflict) else ())]
        a.update(state="CONFLICT", reopenable=True, why="conflicts with the integrated tree in " + ", ".join(a["paths"]))
        ctx.store.event("integrate.conflict", round=n, lane=lane.lane, paths=a["paths"])
        return False
    a["merged"], a["flags"] = merged, flags
    return True


def _submit(ctx, n: int, doc: dict, a: dict, exclude: list[str]) -> None:
    cfg = ctx.cfg
    a["attempts"].append({"name": f"par-{cfg.run_id}-r{n:02d}-ab{a['k']}-a{len(a['attempts']) + 1}",
                          "state": "SUBMITTING", "exclude": list(exclude)})
    rec, d = a["attempts"][-1], _attempt_dir(ctx, n, a)
    d.mkdir(parents=True, exist_ok=True)
    ctx.trees.export(doc["integrated"]["sha"], d / "base")
    ctx.trees.export(a["merged"], d / "cand")
    atomic_write_json(d / "base_workspace_config.json", flags_to_workspace(doc["integrated"]["flags"]))
    atomic_write_json(d / "cand_workspace_config.json", flags_to_workspace(a["flags"]))
    atomic_write_json(d / "spec.json", {"round": n, "k": a["k"], "lane": a["lane"], "base_sha": doc["integrated"]["sha"],
                                        "cand_sha": a["merged"], "intent": a["intent"]})
    rec["dir"] = str(d)
    a["state"] = "AB"
    _save(ctx, n, doc)                      # the job's name, before the job
    base = ctx.store.current_base()
    aiter = ctx.store.base_dir(base["index"]) / "landed_aiter.py"
    env = {"PAR_AB_DIR": str(d), "PAR_BASE_VMU": str(d / "base"), "PAR_CAND_VMU": str(d / "cand"),
           "PAR_BASE_FLAGS": str(d / "base_workspace_config.json"),
           "PAR_CAND_FLAGS": str(d / "cand_workspace_config.json"),
           "PAR_GATES": str(ctx.store.gates_json), "PAR_GATES_SHA": ctx.store.meta()["gates_sha256"],
           "PAR_TEMPLATE": str(cfg.template_config), "PAR_AITER": str(aiter) if aiter.is_file() else "",
           "PAR_MARKER": "", "CAMPAIGN_ARBOR": str(cfg.arbor_root)}
    # TODO(phase 4): PAR_POINTS, an architectural addition measured at every point of its table.
    track_submit(ctx.cluster, rec, str(cfg.path(cfg.cluster.ab_script)), env=env, exclude=exclude,
                 time_limit=cfg.cluster.time_limit,
                 output=f"{cfg.cluster.log_dir}/{rec['name']}.out" if cfg.cluster.log_dir else str(d / "job.log"))
    _save(ctx, n, doc)
    ctx.store.event("integrate.ab_submitted", round=n, k=a["k"], lane=a["lane"], job=rec.get("id"),
                    state=rec["state"])


def collect(ctx, d: Path, gates: dict, intent: str) -> GateRecord:
    """What the A/B job left, decided by gatecheck; a job that left nothing is NOT_MEASURED."""
    checked = (d / "gates_sha256").read_text().strip() if (d / "gates_sha256").exists() else ""
    runs = d / "runs.jsonl"
    if checked != ctx.store.meta()["gates_sha256"]:
        rec = GateRecord(NOT_MEASURED, "the A/B job did not verify the pinned gates", intent)
    elif not runs.exists():
        rec = GateRecord(NOT_MEASURED, "the A/B job wrote no runs", intent)
    else:
        rows = [ArmRun.from_dict(json.loads(x)) for x in runs.read_text().splitlines() if x.strip()]
        rec = gatecheck.evaluate(rows, gates, intent)
    atomic_write_json(d / "gate_result.json", rec.to_dict())
    return rec


def _poll(ctx, n: int, doc: dict, a: dict, gates: dict) -> None:
    rec = a["attempts"][-1]
    d = Path(rec["dir"])
    if not (d / "DONE").exists():
        track_poll(ctx.cluster, rec)
        if rec["state"] == "LOST":
            a["attempts"].pop()
            _submit(ctx, n, doc, a, rec.get("exclude") or [])
            return
        if rec["state"] in ("SUBMITTING", "PENDING", "RUNNING"):
            _save(ctx, n, doc)
            return
    node = (d / "node").read_text().strip() if (d / "node").exists() else rec.get("node") or ""
    g = collect(ctx, d, gates, a["intent"])
    rec.update(state="DECIDED", node=node, verdict=g.verdict)
    a["ab"] = {"job": rec.get("id"), "node": node, "order": gates.get("order", "B C C B"),
               "delta_pct": _r(g.delta_pct), "base_spread_pct": _r(g.base_spread_pct), "booked": g.booked,
               "verdict": g.verdict, "reason": g.reason, "base_mean": _r(g.base_mean, 3),
               "cand_mean": _r(g.cand_mean, 3)}
    if g.verdict == NOT_MEASURED:
        if len(a["attempts"]) >= MAX_AB:
            a.update(state="FAILED", why=f"NOT_MEASURED twice: {g.reason}")
            doc["failed"] = f"addition {a['k']} ({a['lane']}) was not measured twice, on " + \
                ", ".join(x.get("node") or "?" for x in a["attempts"])
            ctx.store.event("integrate.failed", round=n, k=a["k"], why=doc["failed"])
            _save(ctx, n, doc)
            return
        ctx.store.event("integrate.not_measured", round=n, k=a["k"], node=node, why=g.reason)
        _submit(ctx, n, doc, a, sorted(set((rec.get("exclude") or []) + ([node] if node else []))))
        return
    if g.verdict == KEEP:
        doc["integrated"] = {"sha": a["merged"], "flags": a["flags"]}
        a.update(state="KEEP")
    else:
        a.update(state="DROPPED", reopenable=True,
                 why=f"REVERT {_r(g.delta_pct)} % ({g.reason})"[:300])
    ctx.store.event("integrate.decided", round=n, k=a["k"], lane=a["lane"], state=a["state"],
                    delta_pct=_r(g.delta_pct))
    _save(ctx, n, doc)


def _r(v, digits: int = 2):
    return round(v, digits) if isinstance(v, (int, float)) else None


# -- the step --------------------------------------------------------------------------


def step(ctx, n: int, lanes: list, gates: dict) -> dict | None:
    """One action of the integration; the finished document once every addition is decided.

    A document with `failed` set is finished too: the run stops for the operator.
    """
    doc = load(ctx.store, n) or plan(ctx, n, lanes)
    if doc.get("result") or doc.get("failed"):
        return doc
    by = {r.lane: r for r in lanes}
    for a in doc["additions"]:
        if a["state"] == "PENDING":
            if not _prepare(ctx, n, doc, a, by[a["lane"]]):
                _save(ctx, n, doc)
                continue
            if a["merged"] == doc["integrated"]["sha"] and a["flags"] == doc["integrated"]["flags"]:
                a.update(state="DROPPED", reopenable=False, why="adds nothing to the integrated tree")
                _save(ctx, n, doc)
                continue
            _submit(ctx, n, doc, a, [])
            return None
        if a["state"] == "AB":
            _poll(ctx, n, doc, a, gates)
            if doc.get("failed"):
                return doc
            if a["state"] not in DECIDED:
                return None
    return _finish(ctx, n, doc)


def _finish(ctx, n: int, doc: dict) -> dict:
    kept = [a for a in doc["additions"] if a["state"] == "KEEP"]
    claim = 1.0
    for a in kept:
        claim *= 1.0 + (a["ab"]["delta_pct"] or 0.0) / 100.0
    base = ctx.store.current_base()
    nxt = base
    if kept:
        nxt = publish_base(ctx, base["index"] + 1, doc["integrated"]["sha"], doc["integrated"]["flags"],
                           {"round": n, "parent": base["index"], "source": f"round {n}'s integration",
                            "kept": [a["lane"] for a in kept]},
                           aiter=ctx.store.base_dir(base["index"]) / "landed_aiter.py")
    doc["result"] = {"base_next": f"base_{nxt['index']:03d}", "kept": len(kept),
                     "claim_pct": round((claim - 1.0) * 100.0, 3)}
    _save(ctx, n, doc)
    ctx.store.event("integrate.done", round=n, **doc["result"])
    return doc


def live_job(doc: dict | None) -> dict | None:
    """The A/B job in flight, if any (status and STOP use it)."""
    for a in (doc or {}).get("additions") or []:
        if a["state"] == "AB" and a["attempts"] and a["attempts"][-1]["state"] in ("SUBMITTING", "PENDING", "RUNNING"):
            return a["attempts"][-1]
    return None
