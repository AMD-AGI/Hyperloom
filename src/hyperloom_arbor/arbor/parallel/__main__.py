"""python3 -m arbor.parallel <command> --config <run.json> ...

    init                         create the run's store: pin the config, the gates and the Arbor commit
    tick [--dry-run] [--takeover]
                                 one pass of the controller (--dry-run: what it would do; writes nothing)
    run [--interval 60] [--takeover]
                                 tick until the run stops (the keeper runs this)
    status [--json]              where the run is (--json: the dashboard's payload)
    explain <lane|rNN>           a lane's record, clocks and items; or a round's
    graph [--round N]            the round's lanes, their dependencies and levels
    approve m-<id> --why ...     let a rewrite into the next map's lanes
    pause [--why ...]  resume    pause the run (lanes keep running; nothing new starts); go on
    stop [--why ...]             stop every lane at its next boundary, then the run
    resubmit <lane> [--exclude NODE]
                                 a FAILED or BROKEN lane's chain again (never a CLOSED one)
    close-lane <lane> --as killed --why ...
                                 the operator ends a lane; nothing of it is integrated
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time


from arbor.parallel import config as config_mod
from arbor.parallel import controller, lanes, report, status
from arbor.parallel.store import HostLocked, atomic_write_json, atomic_write_text, init_run


def _ctx(args):
    return controller.make(config_mod.load(args.config))


def _find_lane(ctx, name: str):
    for n in range(int(ctx.store.state().get("round") or 0), 0, -1):
        rec = lanes.load(ctx.store, n, name)
        if rec:
            return rec
    raise SystemExit(f"no lane {name} in this run")


def _status_text(s: dict) -> str:
    st = s["state"]
    out = [f"run {s['run_id']}: {st.get('phase')} in round {st.get('round')}"
           + (f" ({(st.get('stop') or {}).get('rule')}: {(st.get('stop') or {}).get('why')})" if st.get("stop") else ""),
           f"base {(s.get('base') or {}).get('index')}, host {(s.get('host') or {}).get('host')}, "
           f"max {s['max_lanes']} nodes"]
    m = s.get("metrics") or {}
    if m:
        out.append(f"nodes busy {m.get('nodes_busy')}, kept {m.get('kept')}, compounded {m.get('compounded_gain_pct')} %")
    for r in (s["rounds"][-1:] if s["rounds"] else []):
        for x in r["lanes"]:
            c = x["clocks"]
            out.append(f"  {x['lane']:<8} {x['direction']:<28} {x['kind']:<13} {x['state']:<12} "
                       f"deps {report.dur(c['wait_deps_s']) or '-':>6}  node wait {report.dur(c['wait_node_s']) or '-':>6}  "
                       f"on node {report.dur(c['on_node_s']) or '-':>6}  chunks {c['chunks']}"
                       + (f"  {x['outcome']['status']}" if x.get("outcome") else ""))
    if s.get("attention"):
        out.append(s["attention"].strip())
    return "\n".join(out)


def cmd_init(args) -> int:
    cfg = config_mod.load(args.config)
    store = init_run(cfg)
    print(f"run {cfg.run_id}: {store.root} (Arbor {store.meta()['arbor_sha'][:12]}, gates "
          f"{store.meta()['gates_sha256'][:12]})")
    return 0


def cmd_tick(args) -> int:
    ctx = _ctx(args)
    if args.dry_run:
        print(controller.dumps(controller.explain_next(ctx)))
        return 0
    try:
        print(controller.dumps(controller.tick(ctx, takeover=args.takeover)))
    except HostLocked as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    return 0


def cmd_run(args) -> int:
    ctx = _ctx(args)
    while True:
        try:
            r = controller.tick(ctx, takeover=args.takeover)
        except HostLocked as exc:
            print(f"[parallel] refused: {exc}", flush=True)
            atomic_write_text(ctx.store.root / "keeper" / "STOP", f"another host holds the run: {exc}\n")
            return 3
        except Exception as exc:  # noqa: BLE001 - one bad tick (NFS, a slow command) is not the end of a run
            ctx.store.event("tick.error", error=f"{type(exc).__name__}: {exc}"[:500])
            print(f"[parallel] {time.strftime('%H:%M:%S')} tick failed: {exc}", flush=True)
            r = {}
        else:
            print(f"[parallel] {time.strftime('%H:%M:%S')} {r.get('phase')} round {r.get('round')}", flush=True)
        if r.get("phase") == "STOPPED":
            atomic_write_text(ctx.store.root / "keeper" / "STOP", f"the run stopped: {(r.get('stop') or {}).get('why')}\n")
            return 0
        time.sleep(args.interval)


def cmd_status(args) -> int:
    cfg = config_mod.load(args.config)
    s = status.collect(cfg.root)
    print(json.dumps(s, default=str) if args.json else _status_text(s))
    return 0


def cmd_explain(args) -> int:
    ctx = _ctx(args)
    if args.what.startswith("r") and args.what[1:].isdigit():
        print(controller.dumps(report.round_record(ctx.store, int(args.what[1:]), ctx.clock())))
        return 0
    rec = _find_lane(ctx, args.what)
    print(controller.dumps({**rec.to_dict(), "live_clocks": lanes.live_clocks(rec, ctx.clock()),
                            "items": status._items(rec.to_dict())}))
    return 0


def cmd_graph(args) -> int:
    ctx = _ctx(args)
    n = args.round or int(ctx.store.state().get("round") or 0)
    recs = lanes.load_all(ctx.store, n)
    if not recs:
        print(f"round {n} has no lanes")
        return 0
    for r in sorted(recs, key=lambda r: (r.level, r.rank, r.lane)):
        print(f"{'  ' * r.level}{r.lane} {r.direction} ({r.kind}, rank {r.rank}, {r.nodes} node(s)) {r.state}"
              + (f" <- {', '.join(r.depends_on)}" if r.depends_on else ""))
    return 0


def cmd_approve(args) -> int:
    ctx = _ctx(args)
    a = ctx.store.approvals()
    a[args.direction] = {"why": args.why, "by": getpass.getuser(), "ts": time.time()}
    atomic_write_json(ctx.store.approvals_json, a)
    ctx.store.event("run.approved", direction=args.direction, why=args.why)
    print(f"approved {args.direction}: it gets a lane in the next map that lists it (status open or plan)")
    return 0


def cmd_pause(args) -> int:
    ctx = _ctx(args)
    atomic_write_text(ctx.store.pause_file, (args.why or "the operator's PAUSE") + "\n")
    print("PAUSE written: the next tick pauses the run; running lanes go on")
    return 0


def cmd_resume(args) -> int:
    print(controller.resume(_ctx(args)))
    return 0


def cmd_stop(args) -> int:
    ctx = _ctx(args)
    atomic_write_text(ctx.store.stop_file, (args.why or "the operator's STOP") + "\n")
    print("STOP written: the next tick stops every lane; the run stops once their chains have ended")
    return 0


def cmd_resubmit(args) -> int:
    ctx = _ctx(args)
    rec = _find_lane(ctx, args.lane)
    if rec.state not in ("FAILED", "BROKEN"):
        raise SystemExit(f"{rec.lane} is {rec.state}: only a FAILED or BROKEN lane is resubmitted")
    lanes.resubmit(ctx, rec, exclude_node=args.exclude, by="operator")
    print(f"{rec.lane}: {rec.state} (excluding {', '.join(rec.exclude) or 'nothing'})")
    return 0


def cmd_close_lane(args) -> int:
    ctx = _ctx(args)
    rec = lanes.operator_close(ctx, _find_lane(ctx, args.lane), args.as_, args.why)
    print(f"{rec.lane}: {rec.state}" + (" (closing at its next boundary)" if rec.state != "CLOSED" else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m arbor.parallel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--config", required=True, help="the run's config (JSON)")
        p.set_defaults(fn=fn)
        return p

    add("init", cmd_init)
    p = add("tick", cmd_tick)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--takeover", action="store_true")
    p = add("run", cmd_run)
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--takeover", action="store_true")
    p = add("status", cmd_status)
    p.add_argument("--json", action="store_true")
    p = add("explain", cmd_explain)
    p.add_argument("what", help="a lane (r01-l2) or a round (r01)")
    p = add("graph", cmd_graph)
    p.add_argument("--round", type=int)
    p = add("approve", cmd_approve)
    p.add_argument("direction")
    p.add_argument("--why", required=True)
    p = add("pause", cmd_pause)
    p.add_argument("--why")
    add("resume", cmd_resume)
    p = add("stop", cmd_stop)
    p.add_argument("--why")
    p = add("resubmit", cmd_resubmit)
    p.add_argument("lane")
    p.add_argument("--exclude")
    p = add("close-lane", cmd_close_lane)
    p.add_argument("lane")
    p.add_argument("--as", dest="as_", required=True, choices=["killed"])
    p.add_argument("--why", required=True)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except config_mod.ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
