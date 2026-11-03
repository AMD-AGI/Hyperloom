"""The source trees a parallel run measures, as commits in one git repository on shared storage.

The image's stock `vllm/` (and later `aiter/`) is the root commit. Every
integrated base is a commit on top of it; every candidate is a commit whose
parent is the base it was made on. Merging a candidate onto a newer base is a
cherry-pick: a 3-way merge with the candidate's own base as the ancestor. A
conflict is a result, not an error: the addition is dropped, reopenable by the
next round's dossier (no re-landing inside a run).

A base leaves the store in the form the campaign launcher already deploys
(`base_vmu`): `<prefix>.diff`, the whole difference from the image applied
with `git apply -p1` at site-packages, and `<prefix>.blobs`, one
"<blob id> <path>" line per file the diff writes, which the launcher checks
with `git hash-object` after applying. A tree that does not check is never
run on.

All git work happens on the host, in throwaway worktrees; nothing here runs
on a node. Ported from arbor/mesh3/trees.py (<fork>/mesh3-confirm a691edc) with its
names changed only.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

AUTHOR = {"GIT_AUTHOR_NAME": "parallel", "GIT_AUTHOR_EMAIL": "parallel@local",
          "GIT_COMMITTER_NAME": "parallel", "GIT_COMMITTER_EMAIL": "parallel@local"}


class TreeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Conflict:
    paths: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class FlagConflict:
    keys: tuple[str, ...]


class Trees:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- plumbing ----------------------------------------------------------------
    def git(self, *args: str, cwd: Path | None = None, input: str | None = None,
            check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ, **AUTHOR, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
        p = subprocess.run(["git", *args], cwd=str(cwd or self.root), input=input, text=True,
                           capture_output=True, env=env)
        if check and p.returncode != 0:
            raise TreeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[:500]}")
        return p

    def _scratch(self, sha: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix="par-wt-")) / "wt"
        self.git("worktree", "add", "--detach", "-q", str(d), sha)
        return d

    def _drop(self, wt: Path) -> None:
        self.git("worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(wt.parent, ignore_errors=True)
        self.git("worktree", "prune", check=False)

    def rev(self, ref: str) -> str:
        return self.git("rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()

    def tag(self, name: str, sha: str) -> None:
        self.git("tag", "-f", name, sha)

    # -- building ----------------------------------------------------------------
    def init(self, sources: dict[str, str | Path]) -> str:
        """The root commit: each `prefix -> directory` copied in as it is. Idempotent."""
        if (self.root / ".git").exists() or (self.root / "HEAD").exists():
            return self.rev("image")
        self.root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "image")
        for key, value in (("core.autocrlf", "false"), ("core.fileMode", "false"),
                           ("gc.auto", "0"), ("advice.detachedHead", "false")):
            self.git("config", key, value)
        for prefix, src in sources.items():
            shutil.copytree(src, self.root / prefix, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "image: stock sources")
        sha = self.rev("HEAD")
        self.tag("image", sha)
        # Keep the main checkout out of the way: all later work is in scratch worktrees.
        self.git("checkout", "-q", "--detach")
        return sha

    def init_from_repo(self, src: str | Path, *, image_ref: str, paths: list[str] = ()) -> str:
        """The store as a clone of the project's repository (or a bundle of it), the image's
        tree being `image_ref`, a commit in it. Bases are then that repository's commits, and
        exports cover only `paths`. Idempotent."""
        if (self.root / ".git").exists():
            return self.rev("image")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--no-checkout", str(src), str(self.root)],
                       check=True, capture_output=True)
        for key, value in (("core.autocrlf", "false"), ("core.fileMode", "false"),
                           ("gc.auto", "0"), ("advice.detachedHead", "false")):
            self.git("config", key, value)
        if paths:
            self.git("config", "parallel.paths", " ".join(paths))
        sha = self.rev(image_ref)
        self.tag("image", sha)
        return sha

    def apply_patch(self, made_on: str, patch: str, *, directory: str, message: str) -> str | Conflict:
        """A commit of `patch` on `made_on`: a lane's landed patch (paths relative to `directory`),
        or, with directory="", a base diff made at site-packages (paths already prefixed)."""
        if not patch.strip():
            return self.rev(made_on)
        wt = self._scratch(made_on)
        try:
            where = [f"--directory={directory}"] if directory else []
            p = self.git("apply", "--index", "--whitespace=nowarn", *where, "-p1",
                         "-", cwd=wt, input=patch if patch.endswith("\n") else patch + "\n", check=False)
            if p.returncode != 0:
                return Conflict(_apply_failures(p.stderr), p.stderr.strip()[:500])
            self.git("commit", "-q", "--allow-empty", "-m", message, cwd=wt)
            return self.rev_in(wt)
        finally:
            self._drop(wt)

    def rev_in(self, wt: Path) -> str:
        return self.git("rev-parse", "HEAD", cwd=wt).stdout.strip()

    def merge(self, onto: str, candidate: str, *, message: str) -> str | Conflict:
        """Cherry-pick a candidate (made on its parent) onto another base."""
        onto, candidate = self.rev(onto), self.rev(candidate)
        parent = self.rev(f"{candidate}^")
        if parent == onto:
            return candidate
        wt = self._scratch(onto)
        try:
            p = self.git("cherry-pick", "--no-commit", candidate, cwd=wt, check=False)
            if p.returncode != 0:
                unmerged = self.git("diff", "--name-only", "--diff-filter=U", cwd=wt, check=False)
                paths = tuple(sorted(l for l in unmerged.stdout.splitlines() if l))
                self.git("cherry-pick", "--abort", cwd=wt, check=False)
                return Conflict(paths, (p.stderr or p.stdout).strip()[:500])
            self.git("commit", "-q", "--allow-empty", "-m", message, cwd=wt)
            return self.rev_in(wt)
        finally:
            self._drop(wt)

    # -- reading -----------------------------------------------------------------
    def paths(self) -> list[str]:
        """The paths a base consists of (`parallel.paths`; empty: the whole tree)."""
        out = self.git("config", "--get", "parallel.paths", check=False).stdout.split()
        return out

    def changed(self, a: str, b: str) -> list[str]:
        out = self.git("diff", "--name-only", "--no-renames", a, b, "--", *self.paths()).stdout
        return [l for l in out.splitlines() if l]

    def show(self, sha: str, path: str) -> str:
        return self.git("show", f"{sha}:{path}").stdout

    def export(self, sha: str, prefix: str | Path, *, image: str = "image") -> dict:
        """`<prefix>.diff` and `<prefix>.blobs` for the launcher's base_vmu.

        The diff is byte-for-byte what `git diff --binary <image> <sha> -- <paths>` prints,
        the command the operator's vmu files were made with, so an export can be checked
        against them by md5.
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = self.paths()
        diff = subprocess.run(["git", "diff", "--binary", "--no-renames", image, sha, "--", *paths],
                              cwd=str(self.root), capture_output=True, check=True).stdout
        written = self.git("diff", "--name-only", "--no-renames", "--diff-filter=AMT", image, sha,
                           "--", *paths).stdout.split()
        blobs = []
        for path in written:
            line = self.git("ls-tree", sha, "--", path).stdout.split()
            blobs.append(f"{line[2]} {path}")
        tmp = f".{uuid.uuid4().hex}.tmp"
        for suffix, data in ((".diff", diff), (".blobs", ("\n".join(blobs) + ("\n" if blobs else "")).encode())):
            t = prefix.with_name(prefix.name + suffix + tmp)
            t.write_bytes(data)
            os.replace(t, prefix.with_name(prefix.name + suffix))
        return {"sha": sha, "files": len(blobs), "deleted": len(self.changed(image, sha)) - len(blobs)}

    def check_blobs(self, sha: str, blobs_path: str | Path) -> list[str]:
        """Paths whose recorded blob differs from the commit's (empty: the export is the commit)."""
        bad = []
        for line in Path(blobs_path).read_text().splitlines():
            blob, _, path = line.partition(" ")
            got = self.git("ls-tree", sha, "--", path).stdout.split()
            if len(got) < 3 or got[2] != blob:
                bad.append(path)
        return bad


# Bytecode a server compiles beside the sources is never part of a tree. Arbor's landed.patch
# carries it when the workspace's baseline commit tracked it: 46 of b5m c003's 56 files were
# __pycache__/*.pyc, binary hunks no source tree can take (2026-10-14).
_ARTIFACT_DIRS = ("__pycache__/",)
_ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def strip_artifacts(patch: str) -> tuple[str, list[str]]:
    """`patch` without the file sections of compiled bytecode, and the paths it dropped."""
    out: list[str] = []
    dropped: list[str] = []
    keep = True
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = line.rstrip("\n").split(" b/", 1)[-1]
            keep = not (any(d in path for d in _ARTIFACT_DIRS) or path.endswith(_ARTIFACT_SUFFIXES))
            if not keep:
                dropped.append(path)
        if keep:
            out.append(line)
    return "".join(out), dropped


def _apply_failures(stderr: str) -> tuple[str, ...]:
    """Paths from `git apply` errors: "patch failed: <path>:<line>", "<path>: does not exist in index"."""
    paths = set()
    for line in stderr.splitlines():
        if line.startswith("error: patch failed: "):
            paths.add(line[len("error: patch failed: "):].rsplit(":", 1)[0])
        elif line.startswith("error: ") and ": " in line[7:]:
            head = line[7:].split(": ", 1)[0]
            if "/" in head or "." in head:
                paths.add(head)
    return tuple(sorted(paths))


def merge_flags(base: dict, ours: dict, theirs: dict) -> dict | FlagConflict:
    """3-way merge of a flat flag map (server_flags or server_env). None means absent."""
    out, clash = {}, []
    for key in sorted(set(base) | set(ours) | set(theirs)):
        b, o, t = base.get(key), ours.get(key), theirs.get(key)
        if t == b:
            v = o
        elif o == b or o == t:
            v = t
        else:
            clash.append(key)
            continue
        if v is not None:
            out[key] = v
    return FlagConflict(tuple(clash)) if clash else out


def parse_server_flags(text: str) -> dict:
    """A server_flags string as {flag: value}; a bare flag is True. Order is kept."""
    tokens, out, i = shlex.split(text or ""), {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            key, _, value = tok.partition("=")
            out[key] = value
        elif tok.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and not nxt.startswith("--"):
                out[tok], i = nxt, i + 1
            else:
                out[tok] = True
        else:
            out[f"_pos{i}"] = tok
        i += 1
    return out


def render_server_flags(flags: dict) -> str:
    parts = []
    for key, value in flags.items():
        if key.startswith("_pos"):
            parts.append(shlex.quote(str(value)))
        elif value is True:
            parts.append(key)
        elif value is not None and value is not False:
            parts += [key, shlex.quote(str(value))]
    return " ".join(parts)


def flags_from_workspace(ws: dict) -> dict:
    """arbor/config.json ({server_env, server_flags: str}) as the maps merges work on."""
    flags = ws.get("server_flags") or ""
    return {"server_env": {k: str(v) for k, v in (ws.get("server_env") or {}).items()},
            "server_flags": parse_server_flags(flags) if isinstance(flags, str) else dict(flags)}


def flags_to_workspace(flags: dict) -> dict:
    """The maps back into what a campaign config and arbor/config.json hold."""
    return {"server_env": dict(flags.get("server_env") or {}),
            "server_flags": render_server_flags(flags.get("server_flags") or {})}


def workspace_from_session(doc) -> dict | None:
    """A session's workspace_config.json as {server_env, server_flags}, or None.

    The driver's mirror wraps it, {"workspace_config": {...}, "seed_digest", "job", "at"}
    (arbor/cluster/handoff.py), while the test fakes wrote it bare; freeze read the wrapper
    as the flags themselves, so a real lane's kept flags came out empty (CORE-01, 2026-10-27).
    """
    if not isinstance(doc, dict):
        return None
    inner = doc.get("workspace_config") if "workspace_config" in doc else doc
    if isinstance(inner, dict) and ("server_env" in inner or "server_flags" in inner):
        return inner
    return None


def merge_flag_sets(base: dict, ours: dict, theirs: dict) -> dict | FlagConflict:
    """flags.json holds {"server_flags": {...}, "server_env": {...}}; merged per map."""
    out, clash = {}, []
    for part in sorted(set(base) | set(ours) | set(theirs)):
        r = merge_flags(base.get(part) or {}, ours.get(part) or {}, theirs.get(part) or {})
        if isinstance(r, FlagConflict):
            clash += [f"{part}.{k}" for k in r.keys]
        else:
            out[part] = r
    return FlagConflict(tuple(clash)) if clash else out
