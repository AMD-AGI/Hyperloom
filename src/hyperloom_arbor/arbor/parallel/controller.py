"""The controller: one stateless pass over the run's store.

`tick()` reads everything it needs from the store, acts, writes back and
returns; the keeper runs it every minute, and killing it between ticks loses
nothing. In order:

 1. the HOST lock and its heartbeat (another host's ticks are refused);
 2. the switches: STOP stops every lane and ends the run once their chains
    have; PAUSE, or a BROKEN lane, pauses it (lanes are still observed);
 3. every lane's record moved to what its jobs and session say (an Unknown
    job state never triggers anything);
 4. the phase:
      INIT -> MAP -> (a valid map with a lane) FANOUT -> (nothing waiting or queued) DRAIN
      -> (every lane CLOSED) INTEGRATE -> CLOSED -> the next round's MAP, on the integrated tree;
    and STOPPED, for the operator, when the map has no open increment and no
    approved rewrite, when no valid map comes after two attempts of a job,
    after `no_keep_rounds` rounds in a row without an integrated KEEP, when an
    integration fails, at `max_rounds`, or on the operator's STOP;
 5. the schedule: lanes whose dependencies closed join the queue, and the
    queue starts while `max_lanes` nodes allow;
 6. the integration's next action;
 7. a metrics row.

Each stop writes STOP.md. A stop on a map without increments, on dry rounds or
at max_rounds can be resumed (`arbor.parallel resume`) once the operator has
acted (an approval, say); the operator's STOP, a failed map step and a failed
integration cannot.
"""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from arbor import directions
from arbor.parallel import graph, integrate, kinds, lanes, mapstep, metrics, plans, report, scheduler
from arbor.parallel.config import ConfigError, RunConfig
from arbor.parallel.store import Store, append_jsonl, atomic_write_json, atomic_write_text, read_json
from arbor.parallel.trees import Conflict, Trees, flags_from_workspace

PHASES = ("INIT", "MAP", "FANOUT", "DRAIN", "INTEGRATE", "CLOSED", "PAUSED", "STOPPING", "STOPPED")
HUNG_S = 3 * 3600


@dataclass
class Ctx:
    cfg: RunConfig
    store: Store
    trees: Trees
    cluster: Any
    host: Any
    clock: Callable[[], float] = time.time


def make(cfg: RunConfig, *, cluster=None, host=None, clock=time.time, hostname: str | None = None) -> Ctx:
    if cluster is None:
        from arbor.parallel.cluster import SlurmCluster
        c = cfg.cluster
        cluster = SlurmCluster(partition=c.partition or None, account=c.account or None, host=c.host or None)
    if host is None:
        from arbor.parallel.hostjobs import SystemdHost
        host = SystemdHost()
    store = Store(cfg.root, clock, hostname)
    return Ctx(cfg, store, Trees(store.trees), cluster, host, clock)


@contextmanager
def _local_lock(store: Store):
    """One tick at a time on this host (the HOST file keeps other hosts out)."""
    store.root.mkdir(parents=True, exist_ok=True)
    with open(store.root / ".tick.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# -- the phase -------------------------------------------------------------------------


def _phase(ctx: Ctx, st: dict, phase: str, **why: Any) -> None:
    ctx.store.event("run.phase", round=st.get("round"), phase=phase, was=st.get("phase"), **why)
    st["phase"], st["since"] = phase, ctx.clock()
    ctx.store.set_state(st)


def _stop(ctx: Ctx, st: dict, rule: str, why: str, *, resume_to: str | None = None) -> None:
    st["stop"] = {"rule": rule, "why": why, "phase": st.get("phase"), "resume_to": resume_to, "ts": ctx.clock()}
    _phase(ctx, st, "STOPPED", rule=rule, why=why)
    report.write_stop(ctx.store, st, ctx.clock())


def tick(ctx: Ctx, *, takeover: bool = False) -> dict:
    with _local_lock(ctx.store):
        return _tick(ctx, takeover)


def _tick(ctx: Ctx, takeover: bool) -> dict:
    store = ctx.store
    st = store.state()
    if st["phase"] == "STOPPED":
        return {"phase": "STOPPED", "round": st.get("round"), "stop": st.get("stop")}
    store.take_host(takeover=takeover)
    try:
        gates = store.gates()
    except ConfigError as exc:
        atomic_write_text(store.pause_file, f"{exc}\n")
        gates = None
    if store.stopped() and st["phase"] != "STOPPING":
        _begin_stopping(ctx, st)
    n = int(st.get("round") or 0)
    paused = store.paused() or any(r.state == "BROKEN" for r in lanes.load_all(store, n))
    mode = "stopping" if st["phase"] == "STOPPING" else "paused" if paused or st["phase"] == "PAUSED" else "run"
    for rec in lanes.load_all(store, n):
        lanes.reconcile(ctx, rec, mode=mode)
    if st["phase"] == "STOPPING":
        _stopping(ctx, st)
    elif store.paused() or any(r.state == "BROKEN" for r in lanes.load_all(store, n)):
        if st["phase"] != "PAUSED":
            st["paused_from"] = st["phase"]
            _phase(ctx, st, "PAUSED", why=_pause_why(ctx, n))
    else:
        if st["phase"] == "PAUSED":
            _resume_lanes(ctx, n)
            _phase(ctx, st, st.pop("paused_from", "MAP") or "MAP", why="resumed")
        _advance(ctx, st, gates)
    _attention(ctx, st)
    st = store.state()
    append_jsonl(store.metrics, metrics.row(store, st, ctx.cfg.max_lanes, ctx.clock()))
    return {"phase": st["phase"], "round": st.get("round"), "stop": st.get("stop")}


def _pause_why(ctx: Ctx, n: int) -> str:
    broken = [r.lane for r in lanes.load_all(ctx.store, n) if r.state == "BROKEN"]
    if broken:
        return f"BROKEN lane(s) {', '.join(broken)}: resubmit or close them, then resume"
    return (ctx.store.pause_file.read_text(errors="replace").strip() or "the operator's PAUSE")[:300]


def _resume_lanes(ctx: Ctx, n: int) -> None:
    for rec in lanes.load_all(ctx.store, n):
        if rec.state == "FAILED":
            lanes.retry(ctx, rec)


def _advance(ctx: Ctx, st: dict, gates: dict | None) -> None:
    """Every phase change this tick can make, in order (a round can move several steps at once)."""
    for _ in range(len(PHASES)):
        phase = st["phase"]
        {"INIT": _init, "MAP": _map, "FANOUT": _fanout, "DRAIN": _drain, "INTEGRATE": _integrate,
         "CLOSED": _next_round}.get(phase, lambda c, s, g: None)(ctx, st, gates)
        if st["phase"] == phase:
            return


# -- INIT ------------------------------------------------------------------------------


def import_base(ctx: Ctx) -> dict:
    """base_000: the image's sources (or a repository's commit) plus the base diff, every blob checked."""
    cfg, trees = ctx.cfg, ctx.trees
    if ctx.store.current_base():
        return ctx.store.current_base()
    tmpl = cfg.template()
    flags = flags_from_workspace(tmpl)
    if cfg.base_repo:
        b = cfg.base_repo
        trees.init_from_repo(cfg.path(b["repo"]), image_ref=b["image_ref"], paths=list(b.get("paths") or ["vllm"]))
        sha, source = trees.rev(b["base_ref"]), f"{b['base_ref']} over {b['image_ref']}"
    else:
        if not cfg.image_src:
            raise ConfigError("the run names image_src (the image's stock vllm/ on this host) or base_repo")
        sha = trees.init({k: cfg.path(v) for k, v in cfg.image_src.items()})
        vmu, source = cfg.base_vmu or str(tmpl.get("base_vmu") or ""), "the image"
        if vmu:
            r = trees.apply_patch(sha, Path(vmu + ".diff").read_text(), directory="", message=f"base_000: {vmu}")
            if isinstance(r, Conflict):
                raise ConfigError(f"the base {vmu}.diff does not apply to the image: {', '.join(r.paths)}")
            bad = trees.check_blobs(r, vmu + ".blobs")
            if bad:
                raise ConfigError(f"the base {vmu} imports with other blobs than recorded: {', '.join(bad[:5])}")
            sha, source = r, vmu
    aiter = Path(cfg.path(cfg.base_aiter)) if cfg.base_aiter else None
    return integrate.publish_base(ctx, 0, sha, flags, {"round": 0, "source": source}, aiter=aiter)


def _init(ctx: Ctx, st: dict, gates) -> None:
    try:
        import_base(ctx)
    except (ConfigError, OSError) as exc:          # the run's config is pinned: a new run fixes it
        _stop(ctx, st, "init", f"base_000 could not be built: {exc}"[:600])
        return
    st.update(round=1, round_started=ctx.clock(), no_keep=0)
    _phase(ctx, st, "MAP")


# -- MAP -------------------------------------------------------------------------------


def gets_lane(e: dict, approvals: dict) -> str | None:
    """Why a direction gets no lane this round, or None when it does."""
    kind = directions.lane_of(e)
    if e.get("kind") == "increment":
        if e.get("status") != "open":
            return f"{e.get('status')}"
    elif e["id"] not in approvals:
        return "a rewrite, not approved"
    elif e.get("status") not in ("open", "plan"):
        return f"{e.get('status')}"
    if kinds.needs_plan(kind) and not e.get("plan"):
        return f"a {kind} lane with no plan"
    return None


def _map(ctx: Ctx, st: dict, gates) -> None:
    n = st["round"]
    r = mapstep.step(ctx, n)
    if r is None:
        return
    if not r["ok"]:
        _stop(ctx, st, "map", "no valid map: " + "; ".join(r.get("errors") or [])[:600])
        return
    if not build_round(ctx, st, r):
        approvals = ctx.store.approvals()
        why = "; ".join(f"{e['id']}: {gets_lane(e, approvals)}" for e in r["entries"])
        _stop(ctx, st, "no-open-increment", f"the map has no open increment and no approved rewrite ({why})"[:600],
              resume_to="MAP")
        return
    _phase(ctx, st, "FANOUT", lanes=len(lanes.load_all(ctx.store, n)))


def build_round(ctx: Ctx, st: dict, r: dict) -> bool:
    """The round's map (directions.json, scope run) and its lanes; False when no direction gets one.
    Idempotent: a tick that died half way through creates only the lanes still missing."""
    n, store = st["round"], ctx.store
    entries, where, approvals = r["entries"], Path(r["where"]), store.approvals()
    chosen = [e for e in entries if gets_lane(e, approvals) is None]
    g = graph.deps(entries, {e["id"] for e in chosen})
    lv, order = graph.levels(g), graph.order(entries, g)
    lane_ids = {d: f"r{n:02d}-l{i}" for i, d in enumerate(order, 1)}
    by = {e["id"]: e for e in entries}
    pins = {d: plans.pin(plans.resolve(by[d]["plan"], where), store.round_dir(n) / "plans")
            for d in order if by[d].get("plan")}
    atomic_write_json(store.round_dir(n) / "directions.json", {
        "version": directions.VERSION, "scope": "run", "run": ctx.cfg.run_id, "round": n,
        "uk": mapstep.state(store, n).get("uk"), "source": r.get("source"), "started": st.get("round_started"),
        "directions": [{**e, "lane": directions.lane_of(e), "lane_id": lane_ids.get(e["id"]),
                        "no_lane": gets_lane(e, approvals), "outcome": None} for e in entries]})
    if not chosen:
        return False
    for d in order:
        if store.lane(n, lane_ids[d]) is None:
            lanes.create(ctx, n, by[d], lane_ids[d], kind=directions.lane_of(by[d]), level=lv[d],
                         depends_on=[lane_ids[x] for x in g[d]], plan=pins.get(d))
    return True


# -- FANOUT and DRAIN ------------------------------------------------------------------


def schedule(ctx: Ctx, n: int) -> list[str]:
    """Promote what the dependencies release, then start what the cap allows; the lanes started."""
    recs = lanes.load_all(ctx.store, n)
    by = {r.lane: r for r in recs}
    for lid in scheduler.promote([r.slot() for r in recs]):
        lanes.promote(ctx, by[lid])
    started = []
    for lid in scheduler.plan_starts([r.slot() for r in recs], ctx.cfg.max_lanes):
        rec = by[lid]
        try:
            lanes.prepare(ctx, rec, by, [r.direction for r in recs if r.lane != lid])
        except Exception as exc:  # noqa: BLE001 - a lane that cannot be set up is the operator's
            rec.failures.append({"ts": ctx.clock(), "why": f"setting it up failed: {exc}"[:400]})
            lanes._to(ctx, rec, "BROKEN", kind="lane.broken", why=str(exc)[:300])
            continue
        lanes.submit(ctx, rec)
        started.append(lid)
    _sync_outcomes(ctx, n)
    return started


def _sync_outcomes(ctx: Ctx, n: int) -> None:
    p = ctx.store.round_dir(n) / "directions.json"
    doc = read_json(p)
    if not doc:
        return
    out = {r.direction: {"lane": r.lane, "state": r.state, **(r.outcome or {})} for r in lanes.load_all(ctx.store, n)}
    changed = False
    for d in doc["directions"]:
        new = out.get(d["id"])
        if new is not None and d.get("outcome") != new:
            d["outcome"], changed = new, True
    if changed:
        atomic_write_json(p, doc)


def _fanout(ctx: Ctx, st: dict, gates) -> None:
    n = st["round"]
    schedule(ctx, n)
    if not any(r.state in ("WAITING_DEPS", "QUEUED") for r in lanes.load_all(ctx.store, n)):
        _phase(ctx, st, "DRAIN")


def _drain(ctx: Ctx, st: dict, gates) -> None:
    n = st["round"]
    schedule(ctx, n)
    if all(r.state == "CLOSED" for r in lanes.load_all(ctx.store, n)):
        _phase(ctx, st, "INTEGRATE")


# -- INTEGRATE and the next round ------------------------------------------------------


def _integrate(ctx: Ctx, st: dict, gates) -> None:
    if gates is None:
        return
    n = st["round"]
    doc = integrate.step(ctx, n, lanes.load_all(ctx.store, n), gates)
    if doc is None:
        return
    _sync_outcomes(ctx, n)
    _integration_outcomes(ctx, n, doc)
    report.write_round(ctx.store, n, ctx.clock())
    if doc.get("failed"):
        _stop(ctx, st, "integration", f"the integration failed: {doc['failed']}")
        return
    st["no_keep"] = 0 if doc["result"]["kept"] else int(st.get("no_keep") or 0) + 1
    if st["no_keep"] >= ctx.cfg.no_keep_rounds:
        _stop(ctx, st, "no-keep", f"{st['no_keep']} rounds in a row without an integrated KEEP", resume_to="NEXT")
    elif ctx.cfg.max_rounds and n >= ctx.cfg.max_rounds:
        _stop(ctx, st, "max-rounds", f"max_rounds ({ctx.cfg.max_rounds}) reached", resume_to="NEXT")
    else:
        _phase(ctx, st, "CLOSED", kept=doc["result"]["kept"], claim_pct=doc["result"]["claim_pct"])


def _integration_outcomes(ctx: Ctx, n: int, doc: dict) -> None:
    p = ctx.store.round_dir(n) / "directions.json"
    rd = read_json(p)
    if not rd:
        return
    by = {a["direction"]: a for a in doc.get("additions") or []}
    for d in rd["directions"]:
        a = by.get(d["id"])
        if a and d.get("outcome") is not None:
            d["outcome"]["integration"] = {k: a.get(k) for k in ("k", "state", "why", "reopenable", "ab")}
    atomic_write_json(p, rd)


def _next_round(ctx: Ctx, st: dict, gates) -> None:
    st.update(round=int(st["round"]) + 1, round_started=ctx.clock())
    st.pop("stop", None)
    _phase(ctx, st, "MAP", base=ctx.store.current_base()["index"])


# -- STOP ------------------------------------------------------------------------------


def _begin_stopping(ctx: Ctx, st: dict) -> None:
    n = int(st.get("round") or 0)
    why = (ctx.store.stop_file.read_text(errors="replace").strip() or "the operator's STOP")[:300]
    for rec in lanes.load_all(ctx.store, n):
        if rec.state in lanes.ACTIVE:
            lanes.stop(ctx, rec, why)
        elif rec.state in ("WAITING_DEPS", "QUEUED", "FAILED", "BROKEN"):
            lanes._to(ctx, rec, "STOPPED", kind="lane.stopped", why=why)
    ms = mapstep.state(ctx.store, n) if n else {"jobs": {}}
    for job, rec in (ms.get("jobs") or {}).items():
        if rec.get("id") and rec.get("state") in ("PENDING", "RUNNING"):
            (ctx.cluster if job == "profile" else ctx.host).cancel(rec["id"])
    ab = integrate.live_job(integrate.load(ctx.store, n)) if n else None
    if ab and ab.get("id"):
        ctx.cluster.cancel(ab["id"])
    st["stop_from"] = st.get("phase")
    _phase(ctx, st, "STOPPING", why=why)


def _stopping(ctx: Ctx, st: dict) -> None:
    n = int(st.get("round") or 0)
    if any(r.state in lanes.ACTIVE for r in lanes.load_all(ctx.store, n)):
        return
    _stop(ctx, st, "operator", (ctx.store.stop_file.read_text(errors="replace").strip()
                                or "the operator's STOP")[:300])


def resume(ctx: Ctx) -> str:
    """Remove PAUSE; a resumable stop goes back to its phase. What happened, in words."""
    st = ctx.store.state()
    ctx.store.pause_file.unlink(missing_ok=True)
    stop = st.get("stop") or {}
    if st["phase"] != "STOPPED":
        ctx.store.event("run.resume_requested", phase=st["phase"])
        return f"PAUSE removed; the run is {st['phase']} and the next tick continues it"
    if not stop.get("resume_to"):
        return f"the run stopped on {stop.get('rule')} and does not resume: {stop.get('why')}"
    if stop["resume_to"] == "NEXT":
        st["no_keep"] = 0
        st["phase"] = "CLOSED"
    else:
        st["phase"] = stop["resume_to"]             # MAP: the same map, read again with the approvals
    st.pop("stop", None)
    ctx.store.set_state(st)
    ctx.store.event("run.resumed", phase=st["phase"], rule=stop.get("rule"))
    return f"resumed: {st['phase']}"


# -- ATTENTION -------------------------------------------------------------------------


def _attention(ctx: Ctx, st: dict) -> None:
    """ATTENTION.md: what a person should look at now (it never triggers an action)."""
    n, now = int(st.get("round") or 0), ctx.clock()
    lines = []
    for r in lanes.load_all(ctx.store, n):
        if r.state == "BROKEN":
            lines.append(f"- {r.lane} ({r.direction}) is BROKEN: {(r.failures[-1]['why'] if r.failures else '')[:300]}. "
                         f"`arbor.parallel resubmit {r.lane} [--exclude NODE]` or `close-lane {r.lane} --as killed`.")
        if r.state == "RUNNING":
            s = Path(r.session_root) / "session.json"
            age = now - s.stat().st_mtime if s.exists() else None
            if age is not None and age > HUNG_S:
                lines.append(f"- {r.lane} ({r.direction}) has run with no new iteration for {age / 3600:.1f} h "
                             f"(job {r.head} on {r.node}). Nothing is done automatically.")
    if st.get("phase") == "PAUSED":
        lines.insert(0, f"- the run is PAUSED: {_pause_why(ctx, n)}")
    if st.get("phase") == "STOPPED":
        lines.insert(0, f"- the run STOPPED ({(st.get('stop') or {}).get('rule')}): see STOP.md")
    if lines:
        atomic_write_text(ctx.store.attention, "# Attention\n\n" + "\n".join(lines) + "\n")
    else:
        ctx.store.attention.unlink(missing_ok=True)


# -- what the next tick would do (tick --dry-run) --------------------------------------


def explain_next(ctx: Ctx) -> dict:
    """Read-only: the phase, and what the next tick would start. Writes nothing."""
    st = ctx.store.state()
    n = int(st.get("round") or 0)
    out: dict = {"phase": st["phase"], "round": n}
    if st["phase"] in ("INIT", "MAP") and ctx.cfg.seed and n <= 1:
        r = mapstep.seeded_preview(ctx)
        out["map"] = {"ok": r["ok"], "errors": r["errors"], "source": r.get("source")}
        if r["ok"]:
            approvals = ctx.store.approvals()
            chosen = [e for e in r["entries"] if gets_lane(e, approvals) is None]
            g = graph.deps(r["entries"], {e["id"] for e in chosen})
            lv, order = graph.levels(g), graph.order(r["entries"], g)
            by = {e["id"]: e for e in r["entries"]}
            ids = {d: f"r{max(n, 1):02d}-l{i}" for i, d in enumerate(order, 1)}
            slots = [scheduler.Slot(ids[d], "QUEUED" if not g[d] else "WAITING_DEPS", lv[d],
                                    by[d]["rank"] if isinstance(by[d].get("rank"), int) else 999,
                                    kinds.nodes(directions.lane_of(by[d]), by[d]), tuple(ids[x] for x in g[d]))
                     for d in order]
            out["lanes"] = [{"lane": ids[d], "direction": d, "kind": directions.lane_of(by[d]),
                             "level": lv[d], "depends_on": g[d]} for d in order]
            out["would_start"] = scheduler.plan_starts(slots, ctx.cfg.max_lanes)
            out["no_lane"] = {e["id"]: gets_lane(e, approvals) for e in r["entries"] if gets_lane(e, approvals)}
        return out
    if st["phase"] in ("FANOUT", "DRAIN"):
        slots = [r.slot() for r in lanes.load_all(ctx.store, n)]
        promoted = scheduler.promote(slots)
        slots = [scheduler.Slot(s.id, "QUEUED", s.level, s.rank, s.nodes, s.depends_on) if s.id in promoted else s
                 for s in slots]
        out.update(promote=promoted, would_start=scheduler.plan_starts(slots, ctx.cfg.max_lanes),
                   lanes={s.id: s.state for s in slots})
    return out


def status(ctx: Ctx) -> dict:
    """The dashboard's payload (`arbor.parallel status --json`)."""
    from arbor.parallel import status as status_mod
    return status_mod.collect(ctx.store.root, now=ctx.clock())


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)
