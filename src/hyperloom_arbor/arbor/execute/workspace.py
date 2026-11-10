"""Git-based workspace isolation inside a Docker container.

Every candidate is a commit. REVERT undoes it. Patches are extractable for replay.

KernelForge pattern: the campaign's mutable config (server_env, server_flags) lives
at arbor/config.json *inside* workspace_root so it is tracked by the same git repo
as source patches. git add -A captures config and code changes in one commit;
git revert rolls both back atomically.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ARBOR_CONFIG_RELPATH = "arbor/config.json"

# `"docker_container": "@host"` puts the workspace in a directory on this host:
# git, the validators and the integrity digests run here, not through
# `docker exec`. For a campaign with no server to run, a `document` workload.
# Docker container names cannot contain "@".
HOST = "@host"


def exec_argv(container: str, argv: list[str],
              workdir: str | None = None, stdin: bool = False) -> tuple[list[str], str | None]:
    """The command that runs ``argv`` where the workspace lives, and its cwd. ``stdin``: the command reads what it is
    given, which ``docker exec`` passes on only with ``-i``."""
    if container == HOST:
        return list(argv), workdir
    return ["docker", "exec", *(["-i"] if stdin else []), *(["-w", workdir] if workdir else []), container,
            *argv], None


def run_in(container: str, argv: list[str], workdir: str | None = None,
           **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` of ``argv`` in the container, or here for HOST. With ``input``, the container's command reads
    it: without ``-i``, ``docker exec`` gives it an empty stdin (``git apply -`` then applies nothing)."""
    cmd, cwd = exec_argv(container, argv, workdir, stdin="input" in kwargs)
    return subprocess.run(cmd, cwd=cwd, **kwargs)


MOUNT_PROBE = ".arbor-mount-probe"


def session_mount_problem(container: str, session_dir: str | Path, session_mount: str,
                          tries: int = 3) -> str | None:
    """Why the container does not see the session at ``<session_mount>/<session name>``, or None.

    The bench writes ``<session_mount>/<session name>/iter_N/bench_output.json`` in the container,
    and the harness reads ``<session>/iter_N`` here, so ``session_mount`` must be the container's
    path of the session's parent directory. A mount of the session itself puts every file one
    level deeper, where the harness never reads it. A token written here and read back there
    proves the two paths are one directory; a directory that only exists there does not.
    None too when it cannot be checked (no container, the host itself, docker not answering):
    a broken container fails on its own soon enough.
    """
    if not container or container == HOST or not session_mount:
        return None
    session_dir = Path(session_dir)
    probe = session_dir / MOUNT_PROBE
    there = f"{session_mount.rstrip('/')}/{session_dir.name}/{MOUNT_PROBE}"
    token = f"{os.getpid()}-{time.time_ns()}"
    try:
        probe.write_text(token)
    except OSError:
        return None
    try:
        seen = None
        for attempt in range(tries):
            try:
                r = run_in(container, ["cat", there], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return None
            if r.returncode == 0 and r.stdout.strip() == token:
                return None
            said = (r.stderr or "") + (r.stdout or "")
            if r.returncode != 0 and "No such file" not in said:
                return None                # no container, no daemon, no permission: not this check's to judge
            seen = (r.stdout or r.stderr or "").strip()[:120]
            if attempt + 1 < tries:
                time.sleep(1)              # a shared filesystem can lag behind a fresh write
        return (f"the container {container!r} does not see this session at {there.rsplit('/', 1)[0]} "
                f"(read {seen or 'nothing'!r} there): session_mount ({session_mount!r}) must be the container's "
                f"path of the session's parent directory, {session_dir.parent}. Otherwise the bench writes "
                f"its results one level deeper, where the harness never reads them")
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def write_workspace_config(
    container: str, workspace: str, server_env: dict, server_flags: str
) -> None:
    """Write arbor/config.json into the workspace inside the container.

    Called at campaign init and never again by the harness — Claude edits it
    for config-tier changes. The harness reads it at bench time.
    """
    payload = json.dumps(
        {"server_env": server_env, "server_flags": server_flags}, indent=2
    )
    dest = f"{workspace}/{ARBOR_CONFIG_RELPATH}"
    # Write via python3 inside the container — avoids shell quoting/stdin issues.
    script = (
        f"import os, json; "
        f"os.makedirs('{workspace}/arbor', exist_ok=True); "
        f"open('{dest}', 'w').write({payload!r})"
    )
    run_in(container, ["python3", "-c", script], check=True)


def read_workspace_config(container: str, workspace: str) -> dict:
    """Read arbor/config.json from inside the container. Returns {} on missing."""
    r = run_in(container, ["cat", f"{workspace}/{ARBOR_CONFIG_RELPATH}"],
               capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}


def config_at(container: str, workspace: str, rev: str) -> dict | None:
    """arbor/config.json as committed at ``rev``; None when it is not there."""
    r = run_in(container, ["git", "show", f"{rev}:{ARBOR_CONFIG_RELPATH}"], workdir=workspace,
               capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def ensure_workspace_config(
    container: str, workspace: str, server_env: dict, server_flags: str
) -> bool:
    """Re-seed arbor/config.json when it is missing or empty. True if written.

    The seed is written once at campaign init -- but a REVERT can delete it. On
    a fresh container the repo is re-initialised WITHOUT the seed, so the next
    iteration's ``commit()`` creates the file itself (as an empty ``{}``), and
    reverting that iteration removes it outright.

    Such a revert leaves the workspace with no server_env and no server_flags
    at all. ``read_workspace_config`` returns {} on a missing file rather than
    raising, so the next iteration would start its server on defaults -- none
    of the campaign's parallelism, quantization or bench duration -- and report
    the number as a comparable measurement.

    Called at the top of every iteration: cheap, idempotent, and it closes the
    whole class rather than the one path that produced it.
    """
    existing = read_workspace_config(container, workspace)
    if existing.get("server_env") or existing.get("server_flags"):
        return False
    write_workspace_config(container, workspace, server_env, server_flags)
    return True


def git(container: str, workspace: str, *args: str) -> str:
    """Run a git command inside the container. Raises RuntimeError on failure."""
    r = run_in(container, ["git", *args], workdir=workspace,
               capture_output=True, text=True)
    if r.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(
            f"git {cmd} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()}"
        )
    return r.stdout.strip()


GIT_EXCLUDE_PATTERNS = ("__pycache__/", "*.pyc", "*.pyo")

# Python regenerates these inside the workspace on every server start, so they
# are never part of a candidate and must never make the tree look dirty.
_IGNORED_STATUS_SUFFIXES = (".pyc", ".pyo")
_IGNORED_STATUS_FRAGMENTS = ("__pycache__/",)


def ensure_git_excludes(container: str, workspace: str) -> bool:
    """Add __pycache__/*.pyc to .git/info/exclude. True if written.

    Without it, a probe whose whole deliverable is out-of-tree (an evaluator
    plus ledger records) leaves the workspace holding nothing but regenerated
    .pyc files. ``has_changes`` counts untracked files and says dirty; ``commit``
    stages only tracked files plus arbor/ and stages nothing; ``git commit``
    then exits 1 with "nothing to commit" and the harness discards the entire
    iteration -- including the ledger records, which are written downstream of
    the commit.

    There is already a correct path for a probe that changes no files (it
    records the PROBE and persists its findings); the .pyc noise was the only
    thing keeping the loop out of it. Uses .git/info/exclude rather than
    .gitignore so nothing is added to the tracked tree.
    """
    missing = []
    r = run_in(container, ["cat", f"{workspace}/.git/info/exclude"],
               capture_output=True, text=True)
    existing = r.stdout if r.returncode == 0 else ""
    for pat in GIT_EXCLUDE_PATTERNS:
        if not any(line.strip() == pat for line in existing.splitlines()):
            missing.append(pat)
    if not missing:
        return False
    # `printf '%s\n' a b c` writes one pattern per line. The earlier form passed
    # a Python repr to `printf %s`, which emits a LITERAL backslash-n: the file
    # got one unmatchable line, the patterns never took effect, and because the
    # membership test below never found them it re-appended that garbage on
    # every single iteration.
    quoted = " ".join(shlex.quote(p) for p in missing)
    run_in(container, ["bash", "-c",
                       f"mkdir -p {shlex.quote(workspace)}/.git/info && "
                       f"printf '%s\\n' {quoted} >> {shlex.quote(workspace)}/.git/info/exclude"],
           capture_output=True)
    return True


def has_changes(container: str, workspace: str) -> bool:
    """True if the workspace has uncommitted changes (staged or unstaged).

    Ignores regenerated bytecode: ``commit`` would not stage it anyway, so
    counting it as a change produces a tree that is "dirty" but uncommittable.
    Raises RuntimeError if the docker/git command fails.
    """
    r = run_in(container, ["git", "status", "--porcelain"], workdir=workspace,
               capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git status failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()}"
        )
    for line in r.stdout.splitlines():
        path = line[3:].strip() if len(line) > 3 else ""
        if not path:
            continue
        if path.endswith(_IGNORED_STATUS_SUFFIXES):
            continue
        if any(frag in path for frag in _IGNORED_STATUS_FRAGMENTS):
            continue
        return True
    return False


def commit(container: str, workspace: str, message: str) -> str:
    """Stage all changes and commit. Returns the new SHA.

    Uses `git add -A`: tracked modifications AND new files. The earlier form was
    `git add -u`, which stages tracked files ONLY, so a NEW file an agent created
    outside arbor/ was silently dropped from the commit, from iter_N/patch.diff,
    and from landed.patch -- while still sitting on the container filesystem, so
    the bench measured it and the campaign booked the number.

    The failure: an iteration adds a tile table under `configs/` alongside a
    tuned-tile selector, books a win, and only the selector enters the commit.
    Iterations later the table is gone, the selector finds no tuned tiles and
    every tile falls back to the defaults -- so the booked best points at a tree
    that no archived patch can rebuild.

    Safe here because the workspace has no submodules (checked) and
    ensure_git_excludes keeps regenerated bytecode out of the index.
    """
    git(container, workspace, "add", "-A")
    # arbor/ may not exist yet on first commit — create it so the add never fails
    run_in(container, ["mkdir", "-p", f"{workspace}/arbor"], capture_output=True)
    run_in(container, ["bash", "-c",
                       f"[ -f {workspace}/arbor/config.json ] || echo '{{}}' > {workspace}/arbor/config.json"],
           capture_output=True)
    try:
        git(container, workspace, "add", "arbor/")
    except RuntimeError:
        pass  # arbor/ may be empty — not fatal
    try:
        git(container, workspace, "commit", "-m", message)
    except RuntimeError as e:
        # "nothing to commit" must never destroy an iteration. The caller has
        # already decided this iteration exists; an empty commit keeps the
        # iter-N marker in history so revert/patch bookkeeping stays uniform,
        # and the real deliverable may be entirely out-of-tree (an evaluator,
        # ledger records) for a probe.
        if "nothing to commit" not in str(e) and "no changes added" not in str(e):
            raise
        git(container, workspace, "commit", "--allow-empty", "-m", message)
    return git(container, workspace, "rev-parse", "HEAD")


def revert(container: str, workspace: str) -> None:
    """Revert the last commit. An empty one (a probe's, or a change made only in another editable tree,
    arbor/execute/trees.py) has nothing to undo, which ``git revert`` refuses: its revert is an empty commit too,
    so the history still says the iteration was reverted."""
    r = run_in(container, ["git", "diff", "--quiet", "HEAD~1", "HEAD"], workdir=workspace, capture_output=True)
    if r.returncode == 0:
        subject = git(container, workspace, "log", "-1", "--format=%s")
        git(container, workspace, "commit", "--allow-empty", "-m", f'Revert "{subject}"')
        return
    git(container, workspace, "revert", "--no-edit", "HEAD")


def diff_stat(container: str, workspace: str) -> str:
    """Summary of what changed in the last commit."""
    return git(container, workspace, "diff", "--stat", "HEAD~1", "HEAD")


def changed_files(container: str, workspace: str) -> list[str]:
    """List of files changed in the last commit."""
    out = git(container, workspace, "diff", "--name-only", "HEAD~1", "HEAD")
    return [f for f in out.splitlines() if f.strip()]


def save_patch(container: str, workspace: str, out_path: str | Path) -> None:
    """Export the last commit as a patch file on the host."""
    patch = git(container, workspace, "format-patch", "--stdout", "HEAD~1")
    Path(out_path).write_text(patch)


# What landed.patch carries is what the server runs, and nothing else: not the agent's probe
# scripts under arbor/eval/ and not bytecode, which a downstream baseline would otherwise
# have to split out by hand. Everything else
# under arbor/ IS served and must stay: TunableOp tables (arbor/tunableop/*.csv), a tuned
# fused-MoE table and its AITER JIT links and sources.
HARNESS_FILES = ("arbor/config.json",)
# The agent's own instruments: evaluators and probes the server never imports. They still
# carry across containers, in scratch.patch, because the agent works with them; they never
# reach an export, an integration or another tree. A config's `scratch_paths` adds more.
# `eval` is the packs' probe space at the workspace's top, which the loop already treats as
# read-only (READONLY_PROBE_PREFIXES): agents keep their logs, probes and even core dumps
# there, and none of it may ride in a KEEP's patches. Logs are never served, wherever
# the agent leaves one.
SCRATCH_PATHS = ("arbor/eval", "eval", "*.log")
# A file this large is no source change. Such a file stays in the tree and out of every patch
# (landed.patch, scratch.patch and a KEEP's), each named where its patch is written (`oversize`).
OVERSIZE_BYTES = 10 * 1024 * 1024
# Pathspecs match with fnmatch, where `*` crosses `/`: these catch bytecode at any depth,
# the package root's own __pycache__/ included, which the baseline commit tracks whenever the
# image shipped it.
BYTECODE_EXCLUDES = (":(exclude)*__pycache__/*", ":(exclude)*.pyc", ":(exclude)*.pyo")


def scratch_paths(config: dict | None = None) -> tuple[str, ...]:
    """The scratch paths: the defaults plus the config's ``scratch_paths``, without trailing /."""
    extra = (config or {}).get("scratch_paths") or ()
    if isinstance(extra, str):
        extra = (extra,)
    out: list[str] = []
    for p in (*SCRATCH_PATHS, *extra):
        p = str(p).strip().rstrip("/")
        if p and p not in out:
            out.append(p)
    return tuple(out)


def landed_pathspec(config: dict | None = None) -> list[str]:
    """The pathspec of the served tree: what landed.patch holds and what every check compares."""
    return [".", *(f":(exclude){p}" for p in HARNESS_FILES),
            *(f":(exclude){p}" for p in scratch_paths(config)), *BYTECODE_EXCLUDES]


def scratch_pathspec(config: dict | None = None) -> list[str]:
    """The pathspec of the agent's scratch: what scratch.patch holds."""
    return [*scratch_paths(config), *BYTECODE_EXCLUDES]


def oversize(container: str, workspace: str, base: str, head: str, spec: list[str]) -> list[dict]:
    """The files ``spec`` covers that differ between ``base`` and ``head`` and are over
    OVERSIZE_BYTES at ``head``, as ``{"path", "bytes"}``, for a patch to leave out (``without``)."""
    names = [n for n in git(container, workspace, "diff", "--name-only", "--no-renames", "-z",
                             base, head, "--", *spec).split("\0") if n.strip()]
    if not names:
        return []
    big = []
    for entry in git(container, workspace, "ls-tree", "-r", "-l", "-z", head, "--", *names).split("\0"):
        meta, _, path = entry.partition("\t")
        size = meta.split()[-1] if meta.split() else ""
        if path and size.isdigit() and int(size) > OVERSIZE_BYTES:
            big.append({"path": path, "bytes": int(size)})
    return big


def without(spec: list[str], files: list[dict]) -> list[str]:
    """``spec`` with each of ``files`` (``oversize``'s) excluded by its literal path."""
    return [*spec, *(f":(exclude,literal){f['path']}" for f in files)]


def _guarded(container: str, workspace: str, base: str, spec: list[str], what: str) -> list[str]:
    """``spec`` without the files too large for a patch, each named on stderr as left out of ``what``."""
    big = oversize(container, workspace, base, "HEAD", spec)
    for f in big:
        print(f"[arbor] {what} leaves out {f['path']} ({f['bytes']} bytes, over {OVERSIZE_BYTES}): "
              "it stays in the tree", file=sys.stderr)
    return without(spec, big)


def _root_commit(container: str, workspace: str) -> str:
    root = git(container, workspace, "rev-list", "--max-parents=0", "HEAD").strip()
    root = root.splitlines()[0].strip() if root else ""
    if not root:
        raise RuntimeError("could not resolve the repository root commit")
    return root


def base_commit(container: str, workspace: str, config: dict | None = None) -> str:
    """The commit the served tree is measured against: what the patches diff from.

    Standalone, the repository's first commit: the launcher git-inits the image's tree.
    In component mode a host may hand over a worktree with history behind it (MB's is its
    experiment repository at the branch's start commit, with the session root and earlier
    merges before that), so ``component.root`` names the start commit, and the first commit
    is only the fallback. A ``component.root`` that is not a commit of
    the workspace raises: the driver refuses such a config at start.
    """
    config = config or {}
    root = None
    if config.get("mode") == "component" and isinstance(config.get("component"), dict):
        root = config["component"].get("root")
    if not root:
        return _root_commit(container, workspace)
    r = run_in(container, ["git", "rev-parse", "--verify", "--quiet", f"{root}^{{commit}}"],
               workdir=workspace, capture_output=True, text=True)
    sha = r.stdout.strip() if r.returncode == 0 else ""
    if not sha:
        raise RuntimeError(f"component.root {root} is not a commit of {workspace}")
    return sha


def _write_diff(container: str, workspace: str, out_path: str | Path, *args: str) -> None:
    """`git diff *args`, written byte for byte as git produced it.

    Never through git(), which strips the output. `git apply` REQUIRES the
    terminating newline: with it stripped, every landed.patch was rejected as
    "corrupt patch at line <last>", and the restore block fell back to replaying
    per-iteration patches. Adding back one newline was not
    enough either: a binary hunk ends with a blank line, so when a binary file
    came last the patch still lost it, and `git apply` reported "corrupt binary
    patch" yet exited 0 having skipped the file. Bytes,
    not text, so a source file in another encoding cannot fail the write.
    """
    r = run_in(container, ["git", "diff", *args], workdir=workspace, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode(errors="replace").strip()
        raise RuntimeError(f"git diff failed (exit {r.returncode}): {err}")
    Path(out_path).write_bytes(r.stdout)


def save_landed_patch(container: str, workspace: str, out_path: str | Path,
                      config: dict | None = None) -> None:
    """Export the WHOLE served tree's diff against the campaign's root commit.

    The repo lives inside the container, so it dies with the container -- which
    means every landed KEEP dies with it. A recreated container silently loses
    them, and later iterations then measure a stock tree against a bar that
    only the enhanced tree could reach.

    Per-iteration patches under ``iter_N/patch.diff`` are not sufficient to
    rebuild from: a retroactively corrected verdict never archived one, and
    replaying N patches in
    sequence multiplies the chance one fails to apply. One cumulative diff
    against the root commit is a single, always-current artifact on shared
    storage, and it is exactly what a fresh container needs.

    Written on every KEEP, overwriting the previous copy. Raises on a git
    failure; the caller treats that as bookkeeping that must not be able to
    overturn a verdict.

    Served files only (``landed_pathspec``). arbor/config.json is HARNESS-WRITTEN,
    not landed work: it is seeded with different content by every fresh git-init,
    so including it makes the patch fail to apply against a new repo, and `git
    apply` is atomic: that one file takes the real source changes down with it.
    Excluding all of arbor/ would be wrong the other way: it dropped a KEEP's
    own files there (TunableOp tables, which a kept
    PYTORCH_TUNABLEOP_FILENAME points at), so the next container ran
    without them. The agent's scratch goes to scratch.patch instead
    (``save_scratch_patch``), and bytecode goes nowhere.
    `--binary` so a patch is never silently lossy when a binary appears.
    The base is ``base_commit``: the first commit, or a host's ``component.root``.
    """
    root = base_commit(container, workspace, config)
    spec = _guarded(container, workspace, root, landed_pathspec(config), "landed.patch")
    _write_diff(container, workspace, out_path, "--binary", root, "HEAD", "--", *spec)


def save_scratch_patch(container: str, workspace: str, out_path: str | Path,
                       config: dict | None = None) -> None:
    """Export the agent's scratch (``scratch_paths``) against the root commit.

    The probes and evaluators a pack tells the agent to keep in the tree must
    survive a fresh container, as they did while they rode in landed.patch, but
    they are not served code. Written on every KEEP and PROBE (the verdicts that
    keep their commit); the launcher applies it after landed.patch. An empty file
    when there is no scratch.
    """
    root = base_commit(container, workspace, config)
    spec = _guarded(container, workspace, root, scratch_pathspec(config), "scratch.patch")
    _write_diff(container, workspace, out_path, "--binary", root, "HEAD", "--", *spec)


def recover_orphaned_commit(
    container: str, workspace: str, recorded_iteration: int
) -> bool:
    """Revert all iter-N commits whose N exceeds recorded_iteration.

    Happens when the process is killed after commit() but before state.save().
    Handles multiple stacked orphans (e.g. killed run left 2 commits).
    Returns True if any revert was performed.

    An iteration already reverted is not an orphan: its ``Revert "iter-N: ..."`` commit has
    undone it, and reverting that revert would put the change back. Earlier the walk matched
    the revert too, so a retried iteration re-applied the change it had just undone, and the
    tree held a patch the session recorded as reverted. The walk goes down from HEAD: a
    revert marks its iteration undone, an ``iter-N:`` commit above the record that is not
    undone is an orphan, and anything else ends the walk. The orphans are reverted newest
    first, and what the recovery changed is logged.
    """
    try:
        log = git(container, workspace, "log", "-60", "--format=%H%x09%s")
    except RuntimeError:
        return False
    undone: set[int] = set()
    orphans: list[tuple[str, str]] = []
    for line in log.splitlines():
        sha, _, subject = line.partition("\t")
        m = re.match(r'Revert "iter-(\d+):', subject)
        if m:
            undone.add(int(m.group(1)))
            continue
        m = re.match(r"iter-(\d+):", subject)
        if not m or int(m.group(1)) <= recorded_iteration:
            break
        if int(m.group(1)) not in undone:
            orphans.append((sha, subject))
    if not orphans:
        return False
    before = git(container, workspace, "rev-parse", "HEAD")
    try:
        git(container, workspace, "checkout", "--", ".")
        git(container, workspace, "clean", "-fd")
    except RuntimeError:
        # A tree that cannot be cleaned is still reverted below: `git revert` refuses a dirty tree on its own, and
        # that error is the one the caller sees.
        pass
    for sha, subject in orphans:
        print(f"[arbor] Orphaned commit detected ({subject.split(':', 1)[0]} > recorded "
              f"iter-{recorded_iteration}) — reverting: {subject[:60]}")
        empty = run_in(container, ["git", "diff", "--quiet", f"{sha}~1", sha], workdir=workspace,
                       capture_output=True).returncode == 0
        if empty:
            git(container, workspace, "commit", "--allow-empty", "-m", f'Revert "{subject}"')
        else:
            git(container, workspace, "revert", "--no-edit", sha)
    print(f"[arbor] Orphan recovery changed: "
          f"{git(container, workspace, 'diff', '--stat', before, 'HEAD').strip() or 'nothing in the tree'}")
    return True
