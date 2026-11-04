"""What the dashboard's global mode runs on the share host, in one ssh command each.

    python3 share_query.py index    <root>                       every campaign
    python3 share_query.py campaign <root> <project> <user> <campaign>   one campaign
    python3 share_query.py file     <root> <project> <user> <campaign> <path>   one file, base64

``server.mjs`` sends this file compressed and base64-encoded inside the command
(``python3 -c 'import base64,zlib;exec(zlib.decompress(base64.b64decode("...")))' index <root>``),
because the platform's gateway takes ssh commands but not piped stdin; nothing
is installed on the share host. Compressed, the command stays under the 8 KB
that Git for Windows' ssh passes (``arbor/tests/test_dashboard.py`` holds it
there). Standard library only, Python 3.8 or later.

The answer is always one JSON object on stdout, ``{"error": "..."}`` when the
query failed: through the gateway, every command exits 0 and its stderr
arrives on stdout (tested 2026-10-07), so neither can say that it failed.

The layout is the collectors' (``arbor/collect/share.py``):
``<root>/projects/<project>/<user>/<campaign>/{realtime,knowledge}/``. A
campaign is listed when either directory exists, so a campaign that only ever
pushed its knowledge still shows.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone

# The collectors' path components (arbor/collect/project.py, SAFE_COMPONENT).
COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Inlined in a campaign's payload; everything else is listed and fetched on demand.
INLINE = re.compile(r"^knowledge/(summary/SUMMARY\.md|outcomes/OUTCOMES\.md|findings/FINDINGS\.md"
                    r"|profiles/[^/]+/(PROFILE|profile)\.md|dossier/README\.md)$")
MAX_INLINE_BYTES = 512_000
MAX_INLINE_TOTAL = 4_000_000
MAX_FILE_BYTES = 8_000_000


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: str, errors: list) -> dict | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        errors.append(f"{os.path.basename(path)} unreadable: {exc}")
        return None


def _mtime(path: str) -> str | None:
    try:
        return _iso(os.stat(path).st_mtime)
    except OSError:
        return None


def summarize(status: dict | None) -> dict | None:
    """The handful of fields the index shows, from a realtime status.json."""
    if not status:
        return None
    c = status.get("campaign") or {}
    log = status.get("log") or {}
    sup = log.get("supervisor") or {}
    return {
        "collected_at": status.get("collected_at"),
        "host": status.get("host"),
        "identity": c.get("identity"),
        "pack": c.get("pack"),
        "metric_name": c.get("metric_name"),
        "metric_unit": c.get("metric_unit"),
        "baseline": c.get("baseline"),
        "best_metric": c.get("best_metric"),
        "gain_pct": c.get("gain_pct"),
        "iteration": c.get("iteration"),
        "current_iteration": log.get("current_iteration"),
        "counts": c.get("counts"),
        "phase": (log.get("phase") or {}).get("label"),
        "running": sup.get("running"),
        "terminal": sup.get("terminal"),
        "stop_file": c.get("stop_file"),
        "aborted_file": c.get("aborted_file"),
        "target": (status.get("target") or {}).get("value"),
    }


def _knowledge_brief(manifest: dict | None) -> dict | None:
    if not manifest:
        return None
    by_type: dict = {}
    for f in manifest.get("files") or []:
        kind = str(f.get("path", "")).split("/")[0]
        by_type[kind] = by_type.get(kind, 0) + 1
    return {"collected_at": manifest.get("collected_at"), "files": by_type,
            "skipped": len(manifest.get("skipped") or [])}


def campaign_dirs(root: str):
    projects = os.path.join(root, "projects")
    for project in sorted(_subdirs(projects)):
        for user in sorted(_subdirs(os.path.join(projects, project))):
            for campaign in sorted(_subdirs(os.path.join(projects, project, user))):
                d = os.path.join(projects, project, user, campaign)
                if os.path.isdir(os.path.join(d, "realtime")) or os.path.isdir(os.path.join(d, "knowledge")):
                    yield project, user, campaign, d


def _subdirs(path: str) -> list:
    try:
        return [e.name for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")]
    except OSError:
        return []


def index(root: str) -> dict:
    out = {"root": root, "host": socket.gethostname(), "listed_at": _iso(datetime.now().timestamp()),
           "projects_present": os.path.isdir(os.path.join(root, "projects")), "campaigns": []}
    for project, user, campaign, d in campaign_dirs(root):
        errors: list = []
        status = _load(os.path.join(d, "realtime", "status.json"), errors)
        out["campaigns"].append({
            "project": project, "user": user, "campaign": campaign,
            "meta": _load(os.path.join(d, "realtime", "meta.json"), errors),
            "status_mtime": _mtime(os.path.join(d, "realtime", "status.json")),
            "summary": summarize(status),
            "knowledge": _knowledge_brief(_load(os.path.join(d, "knowledge", "summary", "manifest.json"), errors)),
            "errors": errors,
        })
    return out


def _campaign_dir(root: str, names: list) -> str:
    if len(names) != 3 or not all(COMPONENT.match(n) for n in names):
        raise ValueError(f"not a project, user and campaign: {names}")
    return os.path.join(root, "projects", *names)


def campaign(root: str, names: list) -> dict:
    d = _campaign_dir(root, names)
    errors: list = []
    manifest = _load(os.path.join(d, "knowledge", "summary", "manifest.json"), errors)
    listed = {f.get("path") for f in (manifest or {}).get("files") or []}
    files, texts, total = [], {}, 0
    kroot = os.path.join(d, "knowledge")
    for base, dirs, names_ in os.walk(kroot):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for name in sorted(names_):
            if name.startswith("."):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, d).replace(os.sep, "/")
            try:
                st = os.stat(path)
            except OSError:
                continue
            files.append({"path": rel, "bytes": st.st_size, "mtime": _iso(st.st_mtime),
                          # A file the manifest does not list is left from an earlier collection.
                          "in_manifest": rel[len("knowledge/"):] in listed or name == "manifest.json"})
            if INLINE.match(rel) and st.st_size <= MAX_INLINE_BYTES and total + st.st_size <= MAX_INLINE_TOTAL:
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        texts[rel] = fh.read()
                    total += st.st_size
                except OSError as exc:
                    errors.append(f"{rel} unreadable: {exc}")
    return {
        "project": names[0], "user": names[1], "campaign": names[2], "dir": d,
        "meta": _load(os.path.join(d, "realtime", "meta.json"), errors),
        "status": _load(os.path.join(d, "realtime", "status.json"), errors),
        "status_mtime": _mtime(os.path.join(d, "realtime", "status.json")),
        "knowledge": {"manifest": manifest, "files": files, "texts": texts} if files else None,
        "errors": errors,
    }


def read_file(root: str, names: list, rel: str) -> bytes:
    d = _campaign_dir(root, names)
    parts = rel.split("/")
    if parts[0] not in ("realtime", "knowledge") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"not a campaign file: {rel}")
    path = os.path.realpath(os.path.join(d, *parts))
    if not path.startswith(os.path.realpath(d) + os.sep):
        raise ValueError(f"outside the campaign: {rel}")
    size = os.stat(path).st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{rel} is {size} bytes, over the {MAX_FILE_BYTES}-byte limit; "
                         f"copy it from the share instead")
    with open(path, "rb") as fh:
        return fh.read()


def main(argv: list) -> int:
    try:
        if len(argv) == 2 and argv[0] == "index":
            out = index(argv[1])
        elif len(argv) == 5 and argv[0] == "campaign":
            out = campaign(argv[1], argv[2:5])
        elif len(argv) == 6 and argv[0] == "file":
            data = read_file(argv[1], argv[2:5], argv[5])
            out = {"path": argv[5], "bytes": len(data), "b64": base64.b64encode(data).decode("ascii")}
        else:
            print(json.dumps({"error": f"usage: {' | '.join(l.strip() for l in __doc__.splitlines()[2:5])}"}))
            return 2
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": f"share_query: {exc}"}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
