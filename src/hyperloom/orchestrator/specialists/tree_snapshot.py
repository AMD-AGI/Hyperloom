# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session-owned git history of a shared framework tree.

It is what a specialist's worktree branches off when the tree has no git of its
own, and the baseline every patch specialist's run is held to: the tree as the
session last accepted it.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.session.session_paths import fs_safe_id

if TYPE_CHECKING:
    from ..state.shared_state import SharedState

log = logging.getLogger(__name__)

#: Build output and caches: the specialist edits source, and a worktree carrying
#: a framework's shared objects would cost gigabytes on every dispatch. Neither
#: is held to the baseline, so a build or an import never reads as a trespass.
_EXCLUDES: tuple[str, ...] = ("__pycache__/", "*.pyc", "*.pyo", "*.so", "*.so.*")

_IDENTITY: tuple[str, ...] = ("-c", "user.name=hyperloom", "-c", "user.email=hyperloom@localhost")

#: Where a snapshot repository records the tree it holds, so a refresh can find it.
_TREE_KEY = "hyperloom.tree"

_GIT_TIMEOUT_SEC = 600.0

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@dataclass(frozen=True)
class TreeRestore:
    """What :func:`restore_tree` found in the tree and whether it put it back.

    Attributes:
        changed: Paths, relative to the tree, that differed from the snapshot.
        error: Why the tree could not be read or put back; empty when it was.
    """

    changed: tuple[str, ...] = ()
    error: str = ""

    @property
    def clean(self) -> bool:
        """Whether the tree matched its snapshot."""
        return not self.changed and not self.error

    @property
    def reason(self) -> str:
        """What a run that left the tree like this did, for the error it is discarded with."""
        failed = f"; restore failed: {self.error}" if self.error else ""
        return f"wrote the shared tree at {', '.join(self.changed[:8]) or '?'}{failed}"


def _lock(repo: Path) -> threading.Lock:
    """The lock serialising snapshots into ``repo``; specialists dispatch concurrently."""
    with _locks_guard:
        return _locks.setdefault(str(repo), threading.Lock())


def _repo(session_dir: Path, tree: Path) -> Path:
    """The bare repository holding ``tree``'s snapshot for this session."""
    return session_dir / "tree_snapshots" / f"{fs_safe_id(str(tree))}.git"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git with ``args``."""
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_SEC, check=False)


def _git_dir(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git against ``repo`` alone, for what needs no work tree."""
    return _run("--git-dir", str(repo), *args)


def _git(repo: Path, tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git against ``repo`` with ``tree`` as its work tree.

    ``core.autocrlf`` is pinned off so the history holds the tree's bytes as
    they are, and a restore writes back exactly those bytes.
    """
    return _run(
        "-c",
        f"safe.directory={repo}",
        "-c",
        f"safe.directory={tree}",
        "-c",
        "core.autocrlf=false",
        "--git-dir",
        str(repo),
        "--work-tree",
        str(tree),
        *args,
    )


def _failure(action: str, cp: subprocess.CompletedProcess[str]) -> str:
    """Describe a failed git step."""
    return f"git {action} rc={cp.returncode}: {(cp.stderr or cp.stdout).strip()[:400]}"


def _record_tree(repo: Path, tree: Path) -> str:
    """Name ``tree`` in ``repo`` so a refresh can find it; return the error, empty on success."""
    cp = _git(repo, tree, "config", _TREE_KEY, str(tree))
    return "" if cp.returncode == 0 else _failure("record tree", cp)


def _init(repo: Path, tree: Path, session_dir: Path) -> str:
    """Create ``repo`` for ``tree``; return the error, empty on success."""
    repo.parent.mkdir(parents=True, exist_ok=True)
    cp = _run("init", "--bare", "-q", str(repo))
    if cp.returncode != 0:
        return _failure("init", cp)
    excludes = list(_EXCLUDES)
    # A session kept inside the tree writes there all run long; none of it is the framework's.
    if session_dir.resolve().is_relative_to(tree.resolve()):
        excludes.append(f"/{session_dir.resolve().relative_to(tree.resolve()).as_posix()}/")
    (repo / "info").mkdir(parents=True, exist_ok=True)
    (repo / "info" / "exclude").write_text("".join(f"{p}\n" for p in excludes), encoding="utf-8")
    return _record_tree(repo, tree)


def _commit(repo: Path, tree: Path) -> str:
    """Commit ``tree`` as it is now into ``repo``; return the error, empty on success."""
    cp = _git(repo, tree, "add", "-A", "--", ".")
    if cp.returncode != 0:
        return _failure("add", cp)
    fresh = not _has_commit(repo)
    staged = _git(repo, tree, "diff", "--cached", "--quiet")
    if fresh or staged.returncode != 0:
        cp = _git(repo, tree, *_IDENTITY, "commit", "-q", "--no-verify", "--allow-empty", "-m", "snapshot")
        if cp.returncode != 0:
            return _failure("commit", cp)
    return ""


def _has_commit(repo: Path) -> bool:
    """Whether ``repo`` holds a snapshot; an init interrupted before its commit does not."""
    if not (repo / "HEAD").is_file():
        return False
    return _git_dir(repo, "rev-parse", "--verify", "--quiet", "HEAD^{commit}").returncode == 0


def held_snapshot(session_dir: Path, tree: Path) -> Path | None:
    """The repository holding ``tree``'s snapshot, or ``None`` before the first one."""
    repo = _repo(session_dir, tree)
    return repo if _has_commit(repo) else None


def _drop(repo: Path) -> str:
    """Drop the snapshot ``repo`` holds; return why it is still held, empty once it is not.

    A snapshot that could not follow its tree would hold the next run to a tree
    the session no longer has, and a legitimate write would be reverted as that
    run's. Without one the run goes unchecked, and the next snapshot starts over.
    A worktree already branched off it keeps its own branch.
    """
    try:
        if not _has_commit(repo):
            return ""
        cp = _git_dir(repo, "update-ref", "-d", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"; not dropped: {exc!r}"
    return "" if cp.returncode == 0 else f"; {_failure('drop', cp)}"


def snapshot_tree(session_dir: Path, tree: Path) -> tuple[Path | None, str]:
    """Commit ``tree`` as it is now into a bare repository under ``session_dir``.

    The repository lives outside ``tree`` and nothing is written into the tree,
    so it keeps reading as the plain directory every other consumer takes it
    for. A worktree branched off it holds the tree's files at the same relative
    paths, so a diff harvested from it applies to ``tree`` itself. A snapshot
    that cannot be moved to the tree is dropped.

    Args:
        session_dir: The session the snapshot belongs to.
        tree: The directory to snapshot.

    Returns:
        ``(repo, "")`` once the repository holds the tree as it is now, else
        ``(None, error)``.
    """
    repo = _repo(session_dir, tree)
    with _lock(repo):
        try:
            # A snapshot a session took before it recorded its tree is named
            # here, or no refresh could ever find it again.
            error = _record_tree(repo, tree) if _has_commit(repo) else _init(repo, tree, session_dir)
            error = error or _commit(repo, tree)
        except (OSError, subprocess.SubprocessError) as exc:
            error = f"snapshot of {tree} failed: {exc!r}"
        if error:
            error += _drop(repo)
    return (None, error) if error else (repo, "")


def refresh_snapshots(session_dir: Path) -> list[str]:
    """Commit every tree this session snapshots as it is now, once an integrate has settled it.

    A snapshot that cannot be moved is dropped and the failure logged as an
    error; the session carries on.

    Args:
        session_dir: The session whose snapshots to move.

    Returns:
        One error per snapshot that could not be moved; empty when all were.
    """
    errors: list[str] = []
    for repo in sorted((session_dir / "tree_snapshots").glob("*.git")):
        with _lock(repo):
            try:
                tree = _git_dir(repo, "config", "--get", _TREE_KEY).stdout.strip()
                error = _commit(repo, Path(tree)) if tree else "names no tree to refresh from"
            except (OSError, subprocess.SubprocessError) as exc:
                error = repr(exc)
            if error:
                error += _drop(repo)
        if error:
            log.error("tree snapshot %s not moved to the settled tree, dropped: %s", repo.name, error)
            errors.append(f"{repo.name}: {error}")
    return errors


def _drift(repo: Path, tree: Path) -> tuple[str, ...]:
    """Paths in ``tree`` that differ from the snapshot, excludes and nested checkouts aside.

    Raises:
        OSError: When git cannot read the tree.
    """
    cp = _git(
        repo,
        tree,
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=all",
        "--no-renames",
    )
    if cp.returncode != 0:
        raise OSError(_failure("status", cp))
    return tuple(sorted(entry[3:] for entry in cp.stdout.split("\0") if entry))


def restore_tree(session_dir: Path, tree: Path) -> TreeRestore:
    """Put ``tree`` back to its snapshot: revert what changed, delete what was added.

    Excluded build output is neither reported nor touched. Without a snapshot
    there is no baseline to hold the tree to, and nothing is done.

    Args:
        session_dir: The session the snapshot belongs to.
        tree: The tree to restore.

    Returns:
        What differed, and the error when the tree could not be read or still
        differs after the restore.
    """
    repo = held_snapshot(session_dir, tree)
    if repo is None:
        return TreeRestore()
    changed: tuple[str, ...] = ()
    with _lock(repo):
        try:
            changed = _drift(repo, tree)
            if not changed:
                return TreeRestore()
            for args in (("reset", "--hard"), ("clean", "-ffd")):
                cp = _git(repo, tree, *args)
                if cp.returncode != 0:
                    return TreeRestore(changed, _failure(args[0], cp))
            left = _drift(repo, tree)
        except (OSError, subprocess.SubprocessError) as exc:
            return TreeRestore(changed, f"restore of {tree} failed: {exc}")
    return TreeRestore(changed, f"{tree} still differs after the restore: {', '.join(left[:8])}" if left else "")


def integrate_in_flight(session_dir: Path) -> bool:
    """Whether an integrate has mutated the framework tree without deciding on it yet."""
    from ..state.shared_state import SharedState

    try:
        return bool(SharedState.load_or_init(session_dir).pending_integrate)
    except (OSError, ValueError):
        # An unreadable state cannot vouch for the tree.
        log.warning("cannot read pending_integrate; treating the framework tree as mid-integrate", exc_info=True)
        return True


def baseline_snapshot(session_dir: Path | None, tree: Path) -> tuple[Path | None, str]:
    """The snapshot a patch specialist's worktree branches off and its run is held to.

    Moved to the tree as it is on every dispatch: the check after the last run
    put back whatever that run wrote, so anything else the tree has gained
    since, such as a kernel KEEP or a replayed prelude, was written
    legitimately. Mid-integrate the held snapshot stands as it is: the tree
    holds a candidate no decision has kept, neither for the specialist to
    build on nor for the check after its run to accept.

    Args:
        session_dir: The session the snapshot belongs to; none means no snapshot.
        tree: The shared tree the specialist works beside.

    Returns:
        ``(repo, "")`` with the snapshot, else ``(None, error)``.
    """
    if session_dir is None:
        snapshot, error = None, f"{tree} has no session dir to snapshot it into"
    elif not integrate_in_flight(session_dir):
        snapshot, error = snapshot_tree(session_dir, tree)
    else:
        snapshot = held_snapshot(session_dir, tree)
        error = "" if snapshot is not None else f"{tree} is being integrated into and has no snapshot yet"
    if snapshot is None:
        log.warning("no snapshot of %s, so no specialist run is checked against it: %s", tree, error)
    return snapshot, error


async def restore_shared_tree(
    session_dir: Path | None, tree: Path | None, state: SharedState | None, task_id: str
) -> TreeRestore:
    """Hold the shared tree a patch specialist ran beside to its snapshot.

    The specialist runs with every permission, so its worktree is a
    convention it can step outside of. An integrate in flight writes the tree
    legitimately, so the tree is not judged then. A tree that cannot be put
    back stops the session: every later measurement would run on code no
    decision kept. The git work runs off the event loop; the stop is recorded
    on it, where the live state is written.

    Args:
        session_dir: The session the snapshot belongs to.
        tree: The shared tree, or ``None`` when the run had none.
        state: The live session state the stop is recorded on.
        task_id: The specialist task the run belongs to.

    Returns:
        What the restore found; ``clean`` when the run left the tree alone or
        was not judged.
    """
    if session_dir is None or tree is None:
        return TreeRestore()
    if await asyncio.to_thread(integrate_in_flight, session_dir):
        log.info("specialist %s: %s not checked, an integrate is writing it", task_id, tree)
        return TreeRestore()
    restore = await asyncio.to_thread(restore_tree, session_dir, tree)
    if restore.clean:
        return restore
    log.error(
        "specialist %s wrote the shared tree %s at %d path(s): %s",
        task_id,
        tree,
        len(restore.changed),
        list(restore.changed[:50]),
    )
    if restore.error:
        log.critical("specialist %s: %s could not be put back, stopping the session: %s", task_id, tree, restore.error)
        if state is not None:
            state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
            state.save(session_dir)
    return restore


__all__ = [
    "TreeRestore",
    "baseline_snapshot",
    "held_snapshot",
    "integrate_in_flight",
    "refresh_snapshots",
    "restore_shared_tree",
    "restore_tree",
    "snapshot_tree",
]
