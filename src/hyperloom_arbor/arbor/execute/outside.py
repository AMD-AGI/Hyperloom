"""What a KEEP's served tree needs from outside it, and how that is rebuilt (CORE-01 part B).

Model M day-0 c0's landed.patch (2026-10-24) is 208 symlinks out of 220 entries: its agent pointed
AITER_JIT_DIR at arbor/aiter_jit/ and linked every AITER JIT module there into AITER's install
directory, so the modules were saved nowhere and a fresh container need not have them. The
operator's choice: rebuild them, do not store them.

On every KEEP the driver records each link in the served tree that leaves the tree -- its path,
its target, whether the target is there, and the sha256 and size of what it serves -- in
<session>/outside.json; a component snapshot carries the same file. No binary is stored.

A module (a link ending in .so) is rebuilt by its JIT at the first call: AITER finds no module
behind a dangling link (ModuleNotFoundError), builds it from the AITER in the image, and installs
it over the link in AITER_JIT_DIR (aiter/jit/core.py, atomic_copy). That puts a binary where git
tracks a link, and the next `git add -A` would commit it into every patch after. So `prepare`,
which the driver runs at every start, marks those links skip-worktree: git keeps the link at HEAD
and in the patches, and never stages the module built in its place. `check` compares a tree with
a record.

    python -m arbor.execute.outside record  --container C --workspace W [--config CFG] --out FILE
    python -m arbor.execute.outside prepare --container C --workspace W [--config CFG]
    python -m arbor.execute.outside check   --container C --workspace W --record FILE [--config CFG]

check exits 1 when a recorded module is neither at its target nor rebuilt in the tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from arbor.execute.workspace import git, landed_pathspec, run_in

RECORD = "outside.json"

REBUILD = ("A module (kind 'module') is rebuilt by its JIT (AITER) at its first call when its "
           "target is missing: start the server with this KEEP's config.json env on the same "
           "image and GPU arch. Run `python -m arbor.execute.outside prepare` first, so git never "
           "stages the rebuilt module, and `check` afterwards. Any other link's target comes "
           "from the image.")

# Runs where the workspace is (in the container, or here for @host): one round trip, the links
# read from the index (HEAD after a commit), so a link a rebuild replaced is still known.
# argv: workspace, "1" to hash what each link serves, then the served pathspec.
_SCAN = r'''
import hashlib, json, os, subprocess, sys
ws, digests, specs = sys.argv[1], sys.argv[2] == "1", sys.argv[3:]
def git(*a, inp=None):
    return subprocess.run(["git", "-C", ws, *a], input=inp, capture_output=True, check=True).stdout
links, skip = [], set()
for rec in git("ls-files", "-s", "-z", "--", *specs).split(b"\0"):
    meta, _, path = rec.partition(b"\t")
    if path and meta.split(b" ")[0] == b"120000":
        links.append((path.decode(errors="surrogateescape"), meta.split(b" ")[1].decode()))
for rec in git("ls-files", "-v", "-z", "--", *specs).split(b"\0"):
    if rec[:2] in (b"S ", b"s "):
        skip.add(rec[2:].decode(errors="surrogateescape"))
targets = {}
if links:
    out, i = git("cat-file", "--batch", inp="".join(b + "\n" for _, b in links).encode()), 0
    for path, _ in links:
        nl = out.index(b"\n", i)
        size = int(out[i:nl].split()[2])
        targets[path] = out[nl + 1:nl + 1 + size].decode(errors="surrogateescape")
        i = nl + 1 + size + 1
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
root, entries = os.path.realpath(ws), []
for path, _ in links:
    target = targets[path]
    resolved = target if os.path.isabs(target) else os.path.normpath(
        os.path.join(root, os.path.dirname(path), target))
    if resolved == root or resolved.startswith(root + os.sep):
        continue
    wt = os.path.join(ws, path)
    in_tree = ("link" if os.path.islink(wt) else "file" if os.path.isfile(wt)
               else "dir" if os.path.isdir(wt) else "absent")
    present = os.path.exists(resolved)
    kind = ("module" if path.endswith(".so") else "dir" if os.path.isdir(resolved)
            else "file" if present else "link")
    serving = (resolved if present and os.path.isfile(resolved) and in_tree == "link"
               else wt if in_tree == "file" else None)
    entries.append({"path": path, "target": target, "kind": kind, "present": present,
                    "in_tree": in_tree, "skip_worktree": path in skip,
                    "served_by": None if serving is None else ("target" if serving == resolved else "tree"),
                    "size": os.path.getsize(serving) if serving else None,
                    "sha256": sha(serving) if serving and digests else None})
print(json.dumps(entries))
'''


def scan(container: str, workspace: str, config: dict | None = None, *,
         digests: bool = True) -> list[dict]:
    """Every link in the served tree that leaves it, and what it serves now."""
    p = run_in(container, ["python3", "-c", _SCAN, workspace, "1" if digests else "0",
                           *landed_pathspec(config)],
               capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"outside scan failed: {(p.stderr or p.stdout).strip()[-400:]}")
    return json.loads(p.stdout or "[]")


def _missing(e: dict) -> bool:
    """Neither at its target nor rebuilt in the tree."""
    return not e["present"] and e["in_tree"] != "file"


def record(container: str, workspace: str, config: dict | None, out_path: str | Path) -> dict:
    """Write the KEEP's outside.json (atomically) and return it."""
    entries = scan(container, workspace, config)
    doc = {"head": git(container, workspace, "rev-parse", "HEAD"), "ts": time.time(),
           "rebuild": REBUILD,
           "counts": {"links": len(entries),
                      "modules": sum(e["kind"] == "module" for e in entries),
                      "rebuilt_in_tree": sum(e["in_tree"] == "file" for e in entries),
                      "missing": sum(_missing(e) for e in entries)},
           "entries": entries}
    out = Path(out_path)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, out)
    return doc


def prepare(container: str, workspace: str, config: dict | None = None) -> list[str]:
    """Mark skip-worktree every module link a JIT will rebuild, or has rebuilt, in its place.

    Those whose target is missing (the JIT installs the module over the link) and those the
    tree already holds as a file. A module whose target is there loads through its link and
    is left alone. Returns the paths marked now.
    """
    todo = [e["path"] for e in scan(container, workspace, config, digests=False)
            if e["kind"] == "module" and not e["skip_worktree"]
            and (not e["present"] or e["in_tree"] != "link")]
    if todo:
        run = run_in(container, ["git", "update-index", "--skip-worktree", "--", *todo],
                     workdir=workspace, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"git update-index --skip-worktree failed: {run.stderr.strip()[-300:]}")
    return todo


def check(container: str, workspace: str, recorded: dict, config: dict | None = None) -> dict:
    """Compare the tree with a record: same, rebuilt (or another build), missing, new, gone."""
    now = {e["path"]: e for e in scan(container, workspace, config)}
    was = {e["path"]: e for e in recorded.get("entries", [])}
    out: dict[str, list[str]] = {"same": [], "rebuilt": [], "missing": [], "new": [], "gone": []}
    for path, e in now.items():
        if path not in was:
            out["new"].append(path)
        elif _missing(e):
            out["missing"].append(path)
        elif e["sha256"] == was[path].get("sha256") and e["in_tree"] == was[path].get("in_tree"):
            out["same"].append(path)                     # a directory: there, and no digest
        else:
            out["rebuilt"].append(path)
    out["gone"] = sorted(set(was) - set(now))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arbor.execute.outside", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "prepare", "check"):
        p = sub.add_parser(name)
        p.add_argument("--container", required=True)
        p.add_argument("--workspace", required=True)
        p.add_argument("--config", help="the campaign config (its scratch_paths)")
        if name == "record":
            p.add_argument("--out", required=True)
        if name == "check":
            p.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    config = json.loads(Path(a.config).read_text()) if a.config else None
    if a.cmd == "record":
        doc = record(a.container, a.workspace, config, a.out)
        print(f"outside: {doc['counts']} -> {a.out}")
        return 0
    if a.cmd == "prepare":
        marked = prepare(a.container, a.workspace, config)
        print(f"outside: {len(marked)} module link(s) left to their JIT, git told to skip them")
        return 0
    res = check(a.container, a.workspace, json.loads(Path(a.record).read_text()), config)
    print("outside: " + ", ".join(f"{k} {len(v)}" for k, v in res.items()))
    for path in res["missing"]:
        print(f"  missing: {path}")
    return 1 if res["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
