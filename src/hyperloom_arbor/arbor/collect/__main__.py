"""python3 -m arbor.collect check | project | realtime | knowledge

    check      can this host reach the share and upload to it (ssh, the share root, an upload)
    project    the project, user and campaign a session would be filed under
    realtime   snapshot a campaign's live state to <out>/status.json, and push it; loops
    knowledge  stage what a campaign learned under <out>/<type>/, and push it

Every command that pushes runs `check` first and, when it fails, keeps the
local copy and says what to fix. Exit status: 0 when everything asked for was
done, 1 when the share could not be reached or written (local output still
written), 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import time
from pathlib import Path

from arbor.collect import knowledge, realtime
from arbor.collect.project import component, resolve_project
from arbor.collect.share import Share

RECHECK_EVERY = 10   # intervals between pushes while the share is unreachable


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def identify(args: argparse.Namespace) -> dict:
    """Project, user and campaign for a session: the share path's three names."""
    session = Path(args.session)
    config = _load_json(Path(args.config) if args.config else None)
    state = _load_json(session / "session.json")
    project, source = resolve_project(
        args.project or os.environ.get("ARBOR_PROJECT") or config.get("project"),
        [("config model_name", config.get("model_name")),
         ("pack", state.get("pack_name") or config.get("pack")),
         ("session", session.name),
         ("identity", state.get("identity") or config.get("identity"))])
    return {
        "project": component(project),
        "project_source": source,
        "user": component(args.user or os.environ.get("ARBOR_USER") or getpass.getuser()),
        "campaign": component(args.campaign or session.name),
        "host": socket.gethostname(),
    }


def _print_check(result: dict) -> None:
    for s in result["steps"]:
        print(f"[collect] share {s['step']}: {'ok' if s['ok'] else 'FAILED'} ({s['detail']})")
    if not result["ok"]:
        print(f"[collect] share unavailable: {result.get('fix', '')}")


def cmd_check(args: argparse.Namespace) -> int:
    share = Share()
    result = share.check(component(args.user or os.environ.get("ARBOR_USER") or getpass.getuser()))
    _print_check(result)
    return 0 if result["ok"] else 1


def cmd_project(args: argparse.Namespace) -> int:
    ident = identify(args)
    share = Share()
    print(json.dumps({**ident, "share_dir": share.campaign_dir(ident["project"], ident["user"],
                                                               ident["campaign"])}, indent=1))
    return 0


def cmd_realtime(args: argparse.Namespace) -> int:
    session = Path(args.session)
    ident = identify(args)
    share = Share()
    remote = share.campaign_dir(ident["project"], ident["user"], ident["campaign"]) + "/realtime"
    out = Path(args.out) if args.out else session / "collector" / "realtime"
    config = _load_json(Path(args.config) if args.config else None)
    if args.stdout:
        # The dashboard's local mode: one snapshot on stdout, and nothing
        # written, so a dashboard polling beside a running collector never
        # replaces the collector's status.json between its write and its push.
        snap = realtime.snapshot(session, Path(args.log) if args.log else None, config,
                                 Path(args.results) if args.results else None,
                                 Path(args.events) if args.events else None, args.slurm_job)
        snap["share"] = {**ident, "share_host": None, "destination": None,
                         "share_ok": None, "pushed_at": None, "last_error": None}
        print(json.dumps(snap, default=str))
        return 0
    push = not args.no_push
    state: dict = {"share_ok": None, "pushed_at": None, "last_error": None}
    if push:
        result = share.check(ident["user"])
        _print_check(result)
        state["share_ok"] = result["ok"]
        state["last_error"] = None if result["ok"] else result.get("fix")
    print(f"[collect] realtime: {ident['project']}/{ident['user']}/{ident['campaign']} -> "
          f"{out}{' and ' + share.host + ':' + remote if push else ''}")
    n, rc = 0, 0
    while True:
        snap = realtime.snapshot(session, Path(args.log) if args.log else None, config,
                                 Path(args.results) if args.results else None,
                                 Path(args.events) if args.events else None, args.slurm_job)
        snap["share"] = {**ident, "share_host": share.host if push else None,
                         "destination": remote if push else None, **state}
        meta = {**ident, "session_dir": str(session), "collector_version": realtime.COLLECTOR_VERSION,
                "updated_at": snap["collected_at"], "pushed_at": state["pushed_at"]}
        realtime.write_json(out / "status.json", snap)
        realtime.write_json(out / "meta.json", meta)
        if push and (state["share_ok"] or n % RECHECK_EVERY == 0):
            ok, err = share.push([(out / "status.json", remote + "/status.json"),
                                  (out / "meta.json", remote + "/meta.json")])
            if ok != state["share_ok"]:
                print(f"[collect] realtime push {'ok' if ok else 'FAILED: ' + err}", flush=True)
            state.update(share_ok=ok, last_error=None if ok else err,
                         pushed_at=realtime.now_iso() if ok else state["pushed_at"])
        rc = 0 if (not push or state["share_ok"]) else 1
        if args.once:
            return rc
        n += 1
        time.sleep(args.interval)


def cmd_knowledge(args: argparse.Namespace) -> int:
    session = Path(args.session)
    ident = identify(args)
    share = Share()
    out = Path(args.out) if args.out else session / "collector" / "knowledge"
    types = tuple(t.strip() for t in args.types.split(",")) if args.types else knowledge.TYPES
    unknown = [t for t in types if t not in knowledge.TYPES]
    if unknown:
        print(f"[collect] unknown types {unknown}; known: {list(knowledge.TYPES)}", file=sys.stderr)
        return 2
    manifest = knowledge.stage(session, out, ident, Path(args.dossier) if args.dossier else None,
                               [Path(p) for p in args.profiles or []], types)
    by_type: dict[str, int] = {}
    for f in manifest["files"]:
        by_type[f["path"].split("/")[0]] = by_type.get(f["path"].split("/")[0], 0) + 1
    print(f"[collect] knowledge staged in {out}: {by_type}; skipped {len(manifest['skipped'])}")
    if args.no_push:
        return 0
    result = share.check(ident["user"])
    _print_check(result)
    if not result["ok"]:
        return 1
    remote = share.campaign_dir(ident["project"], ident["user"], ident["campaign"]) + "/knowledge"
    ok, err = share.push(knowledge.files_for_push(out, remote))
    print(f"[collect] knowledge push to {share.host}:{remote}: {'ok' if ok else 'FAILED: ' + err}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m arbor.collect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def naming(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--session", required=True, help="the campaign's session directory")
        sp.add_argument("--config", help="the campaign config (names the model, states the target)")
        sp.add_argument("--project", help="override the project (still normalized)")
        sp.add_argument("--user", help="override the user (default: ARBOR_USER, then the login name)")
        sp.add_argument("--campaign", help="override the campaign (default: the session directory's name)")
        sp.add_argument("--out", help="local output directory (default: <session>/collector/<kind>)")
        sp.add_argument("--no-push", action="store_true", help="collect locally only")

    c = sub.add_parser("check", help="can this host reach the share and upload to it")
    c.add_argument("--user")
    naming(sub.add_parser("project", help="the names a session would be filed under"))
    r = sub.add_parser("realtime", help="snapshot a campaign's live state and push it")
    naming(r)
    r.add_argument("--log", help="the campaign's stdout log ([arbor] lines)")
    r.add_argument("--results", help="a directory of *baseline*.json records")
    r.add_argument("--events", help="an event log to tail")
    r.add_argument("--slurm-job", help="the campaign's Slurm job name, when its driver runs in one: "
                   "the job's state, not a local process, says whether it runs")
    r.add_argument("--interval", type=float, default=60.0, help="seconds between snapshots (default 60)")
    r.add_argument("--once", action="store_true", help="one snapshot, then exit")
    r.add_argument("--stdout", action="store_true",
                   help="print one snapshot as JSON on stdout and exit, writing and pushing "
                        "nothing (the dashboard's local mode)")
    k = sub.add_parser("knowledge", help="stage what a campaign learned and push it")
    naming(k)
    k.add_argument("--dossier", help="the dossier directory to share")
    k.add_argument("--profiles", action="append", help="a profile directory (repeatable; "
                   "<session>/profiles/* are always included)")
    k.add_argument("--types", help=f"comma-separated subset of {','.join(knowledge.TYPES)}")
    args = p.parse_args(argv)
    try:
        return {"check": cmd_check, "project": cmd_project, "realtime": cmd_realtime,
                "knowledge": cmd_knowledge}[args.cmd](args)
    except ValueError as exc:
        print(f"[collect] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
