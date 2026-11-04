"""Git-based workspace isolation inside a Docker container.

Every candidate is a commit. REVERT undoes it. Patches are extractable for replay.

KernelForge pattern: the campaign's mutable config (server_env, server_flags) lives
at arbor/config.json *inside* workspace_root so it is tracked by the same git repo
as source patches. git add -A captures config and code changes in one commit;
git revert rolls both back atomically.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ARBOR_CONFIG_RELPATH = "arbor/config.json"

# `"docker_container": "@host"` puts the workspace in a directory on this host:
# git, the validators and the integrity digests run here, not through
# `docker exec`. For a campaign with no server to run, a `document` workload.
# Docker container names cannot contain "@".
HOST = "@host"


def exec_argv(container: str, argv: list[str],
              workdir: str | None = None) -> tuple[list[str], str | None]:
    """The command that runs ``argv`` where the workspace lives, and its cwd."""
    if container == HOST:
        return list(argv), workdir
    return ["docker", "exec", *(["-w", workdir] if workdir else []), container, *argv], None


def run_in(container: str, argv: list[str], workdir: str | None = None,
           **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` of ``argv`` in the container, or here for HOST."""
    cmd, cwd = exec_argv(container, argv, workdir)
    return subprocess.run(cmd, cwd=cwd, **kwargs)


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

    One campaign's iteration did exactly this: the revert left the workspace with
    no server_env and no server_flags at all. ``read_workspace_config`` returns
    {} on a missing file rather than raising, so the next iteration would have
    started its server on defaults -- no TP=4, no FP4 quantization, and no
    DURATION=600 -- and reported the number as a comparable measurement.

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

    One campaign's iteration lost a complete decode-profile probe to this. The
    agent's whole deliverable was out-of-tree (an evaluator plus ledger
    records), so the workspace held nothing but regenerated .pyc files.
    ``has_changes`` counts untracked files and said dirty; ``commit`` stages
    only tracked files plus arbor/ and staged nothing; ``git commit`` then
    exited 1 with "nothing to commit" and the harness discarded the entire
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

    v2 iteration 5 is the worked example: it added `configs/mxfp8_linear_tiles.json`
    alongside a tuned-tile selector in rocm_native.py, booked it as a class-B
    win, and only rocm_native.py entered the commit. Two iterations later the
    table was gone, `_load_tuned_tiles()` was returning {} and all 99 tiles had
    fallen back to the MiniMax-M3 defaults -- so the booked best pointed at a tree
    that no archived patch could rebuild.

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
    """Revert the last commit."""
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


# What landed.patch carries is what the server runs, and nothing else. Model M day-0's records
# (2026-10-24) were mostly something else: 35 of b0's 86 file diffs were the agent's probe
# scripts under arbor/eval/ and 29 were bytecode, so the CI/CD baseline had to be split by
# hand. Everything else under arbor/ IS served and must stay: B3's TunableOp tables
# (arbor/tunableop/*.csv), c0's tuned fused-MoE table and its AITER JIT links and sources.
HARNESS_FILES = ("arbor/config.json",)
# The agent's own instruments: evaluators and probes the server never imports. They still
# carry across containers, in scratch.patch, because the agent works with them; they never
# reach an export, an integration or another tree. A config's `scratch_paths` adds more.
# `eval` is the packs' probe space at the workspace's top, which the loop already treats as
# read-only (READONLY_PROBE_PREFIXES): under Hyperloom MB the agent kept its logs, probes and
# a GPU core dump there, and they rode in a KEEP's patches. Logs are never served, wherever
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
    per-iteration patches, the path that cross-applied the OTHER campaign's work
    into this container (2026-09-17 11:21). Adding back one newline was not
    enough either: a binary hunk ends with a blank line, so when a binary file
    came last the patch still lost it, and `git apply` reported "corrupt binary
    patch" yet exited 0 having skipped the file (CORE-01, 2026-10-27). Bytes,
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
    means every landed KEEP dies with it. One campaign recreated the container
    and silently lost three of them, then spent four iterations
    measuring a stock tree against a bar that only the enhanced tree could
    reach.

    Per-iteration patches under ``iter_N/patch.diff`` are not sufficient to
    rebuild from: a retroactively corrected verdict never archived one (that is
    how iteration 1's fix became unrecoverable), and replaying N patches in
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
    Excluding all of arbor/ would be wrong the other way: it dropped
    a KEEP's own files there (campaign B3 iteration 22's TunableOp tables, which
    its kept PYTORCH_TUNABLEOP_FILENAME points at), so the next container ran
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
    """
    reverted_any = False
    for _ in range(20):
        try:
            last_msg = git(container, workspace, "log", "-1", "--format=%s")
        except RuntimeError:
            break
        # Match iter-N commits AND their reverts (Revert "iter-N: ...")
        m = re.match(r"(?:Revert \")?iter-(\d+):", last_msg)
        if m and int(m.group(1)) > recorded_iteration:
            print(
                f"[arbor] Orphaned commit detected (iter-{m.group(1)} > recorded "
                f"iter-{recorded_iteration}) — reverting: {last_msg[:60]}"
            )
            try:
                git(container, workspace, "checkout", "--", ".")
                git(container, workspace, "clean", "-fd")
            except RuntimeError:
                pass
            revert(container, workspace)
            reverted_any = True
        else:
            break
    return reverted_any
