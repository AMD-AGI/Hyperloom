# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session-owned git history of a shared framework tree.

It is what a specialist's worktree branches off when the tree has no git of its
own, and the baseline every patch specialist's run is held to: the tree as the
session last accepted it, and the HEAD of every dependency checkout beside it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.common.io import atomic_write_json
from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.framework_paths import enclosing_checkout
from hyperloom.inference_optimizer.session.session_paths import fs_safe_id

from ..bringup.trees import head_commit

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

#: Beside the snapshots: the HEAD each dependency checkout was at when the
#: session last accepted it, and the hold a stop leaves until the trees are back.
_HEADS = "dependency_heads.json"
_UNRESTORED = "unrestored.json"

_GIT_TIMEOUT_SEC = 600.0

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@dataclass(frozen=True)
class TreeRestore:
    """What the check after a patch specialist's run found, and whether the trees are back.

    Attributes:
        changed: Paths, relative to the tree, that differed from the snapshot.
        error: Why the tree could not be put back; empty when it was.
        moved: Dependency checkouts whose HEAD moved, with the old and new HEAD.
    """

    changed: tuple[str, ...] = ()
    error: str = ""
    moved: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether the trees were as the session last accepted them."""
        return not (self.changed or self.error or self.moved)

    @property
    def stops(self) -> bool:
        """Whether a tree is still not as the session last accepted it, so the session cannot go on."""
        return bool(self.error or self.moved)

    @property
    def reason(self) -> str:
        """What a run that left the trees like this did, for the error it is discarded with."""
        found = [f"wrote the shared tree at {', '.join(self.changed[:8])}"] if self.changed else []
        found += [f"moved the HEAD of {moved}" for moved in self.moved]
        return "; ".join(found + ([f"restore failed: {self.error}"] if self.error else []))


def _lock(repo: Path) -> threading.Lock:
    """The lock serialising snapshots into ``repo``; specialists dispatch concurrently."""
    with _locks_guard:
        return _locks.setdefault(str(repo), threading.Lock())


def _snapshots(session_dir: Path) -> Path:
    """Where the session keeps its snapshots, the dependency HEADs and the hold."""
    return session_dir / "tree_snapshots"


def _repo(session_dir: Path, tree: Path) -> Path:
    """The bare repository holding ``tree``'s snapshot for this session."""
    return _snapshots(session_dir) / f"{fs_safe_id(str(tree))}.git"


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


def _held(session_dir: Path) -> bool:
    """Whether a stop holds the session on shared trees no restore has put back yet."""
    return (_snapshots(session_dir) / _UNRESTORED).is_file()


def _head(checkout: str) -> str:
    """The commit ``checkout`` sits at, or ``""`` when it cannot be read."""
    try:
        return head_commit(checkout)
    except (OSError, subprocess.SubprocessError) as exc:
        log.error("cannot read the HEAD of %s: %r", checkout, exc)
        return ""


def _dependency_checkouts(tree: Path, roots: Iterable[str]) -> set[str]:
    """The git checkouts holding a Section 7 root, other than any that holds ``tree`` itself."""
    own = tree.resolve()
    checkouts = {checkout.resolve() for checkout in map(enclosing_checkout, roots) if checkout is not None}
    return {str(checkout) for checkout in checkouts if not own.is_relative_to(checkout)}


def _record_heads(session_dir: Path, checkouts: Iterable[str]) -> None:
    """Record the HEAD each of ``checkouts`` sits at now; one that cannot be read goes unchecked."""
    heads = {checkout: head for checkout in sorted(checkouts) if (head := _head(checkout))}
    try:
        atomic_write_json(_snapshots(session_dir) / _HEADS, heads)
    except OSError as exc:
        log.error("dependency HEADs not recorded for %s: %r", session_dir, exc)


def _recorded_heads(session_dir: Path) -> dict[str, str]:
    """The HEAD recorded for each dependency checkout; none before the first record or when it cannot be read."""
    path = _snapshots(session_dir) / _HEADS
    try:
        return dict(json.loads(path.read_bytes())) if path.is_file() else {}
    except (OSError, ValueError) as exc:
        log.error("dependency HEADs recorded at %s cannot be read, so none is checked: %r", path, exc)
        return {}


def _moved_heads(session_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The dependency checkouts whose HEAD moved off the recorded one, and those whose HEAD cannot be read."""
    recorded = _recorded_heads(session_dir)
    heads = {checkout: _head(checkout) for checkout in recorded}
    moved = tuple(f"{c} from {recorded[c][:12]} to {h[:12]}" for c, h in heads.items() if h and h != recorded[c])
    return moved, tuple(checkout for checkout, head in heads.items() if not head)


def refresh_snapshots(session_dir: Path) -> list[str]:
    """Move every snapshot and dependency HEAD this session records to the trees an integrate has settled.

    A snapshot that cannot be moved is dropped and the failure logged as an
    error; the session carries on. A session held on trees no restore has put
    back moves nothing, so what the stop found is not taken as accepted.

    Args:
        session_dir: The session whose snapshots to move.

    Returns:
        One error per snapshot that could not be moved; empty when all were.
    """
    if _held(session_dir):
        log.error("tree snapshots of %s not moved: the session is held on unrestored shared trees", session_dir)
        return [f"{_UNRESTORED}: the session is held on unrestored shared trees"]
    errors: list[str] = []
    for repo in sorted(_snapshots(session_dir).glob("*.git")):
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
    recorded = _recorded_heads(session_dir)
    if recorded:
        _record_heads(session_dir, recorded)
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
        What differed, and the error when the tree still differs after the
        restore or the restore failed.

    Raises:
        OSError: When the tree or its snapshot cannot be read.
        subprocess.SubprocessError: When git wedges reading them.
    """
    repo = held_snapshot(session_dir, tree)
    if repo is None:
        return TreeRestore()
    with _lock(repo):
        changed = _drift(repo, tree)
        if not changed:
            return TreeRestore()
        try:
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


def baseline_snapshot(session_dir: Path | None, tree: Path, roots: Iterable[str]) -> tuple[Path | None, str]:
    """The snapshot a patch specialist's worktree branches off and its run is held to.

    Moved to the tree as it is on every dispatch, and the HEAD of every
    dependency checkout among ``roots`` recorded beside it: the check after the
    last run put back whatever that run wrote, so anything else the tree has
    gained since, such as a kernel KEEP or a replayed prelude, was written
    legitimately. Both stand as they are mid-integrate, when the tree holds a
    candidate no decision has kept, and while a stop holds the session on
    trees no restore has put back, which the run's guard retries first.

    Args:
        session_dir: The session the snapshot belongs to; none means no snapshot.
        tree: The shared tree the specialist works beside.
        roots: The Section 7 source roots the specialist is shown.

    Returns:
        ``(repo, "")`` with the snapshot, else ``(None, error)``.
    """
    if session_dir is None:
        snapshot, error = None, f"{tree} has no session dir to snapshot it into"
    elif _held(session_dir) or integrate_in_flight(session_dir):
        snapshot = held_snapshot(session_dir, tree)
        error = "" if snapshot is not None else f"{tree} is held mid-integrate or unrestored and has no snapshot yet"
    else:
        snapshot, error = snapshot_tree(session_dir, tree)
        _record_heads(session_dir, _dependency_checkouts(tree, roots))
    if snapshot is None:
        log.warning("no snapshot of %s, so no specialist run is checked against it: %s", tree, error)
    return snapshot, error


def _check(session_dir: Path, tree: Path, task_id: str) -> TreeRestore:
    """Put ``tree`` back to its snapshot and compare the dependency HEADs with the recorded ones.

    What cannot be read is logged and not held against the run: not being able
    to check a run is no evidence that it trespassed.
    """
    try:
        restore = restore_tree(session_dir, tree)
    except (OSError, subprocess.SubprocessError) as exc:
        log.error("specialist %s: %s cannot be read against its snapshot, so it is not checked: %r", task_id, tree, exc)
        restore = TreeRestore()
    moved, unread = _moved_heads(session_dir)
    if unread:
        log.error("specialist %s: the HEAD of %s cannot be read, so it is not checked", task_id, ", ".join(unread))
    return replace(restore, moved=moved)


def _retry(session_dir: Path, tree: Path) -> TreeRestore:
    """Put back the trees a stop left unrestored, and lift the hold once they are all back."""
    try:
        error = restore_tree(session_dir, tree).error
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"{tree} cannot be read against its snapshot: {exc!r}"
    moved, unread = _moved_heads(session_dir)
    error = error or (f"the HEAD of {', '.join(unread)} cannot be read" if unread else "")
    if not (error or moved):
        try:
            (_snapshots(session_dir) / _UNRESTORED).unlink(missing_ok=True)
        except OSError as exc:
            error = f"the hold cannot be lifted: {exc!r}"
    return TreeRestore(error=error, moved=moved)


@dataclass(frozen=True)
class SharedTreeGuard:
    """Holds a patch specialist's run to the shared trees as the session last accepted them.

    The specialist runs with every permission, so its worktree is a convention
    it can step outside of. After the run the tree the session optimises is put
    back to its snapshot, and every dependency checkout beside it must still
    sit at the HEAD recorded at dispatch; a dependency is only ever read, never
    put back. An integrate in flight writes the trees legitimately, so they are
    not judged then. A tree that is not back stops the session, as every later
    measurement would run on code no decision kept, and holds it: a resumed
    session puts the trees back before its next run rather than taking them as
    that run's baseline. The git work runs off the event loop; the stop is
    recorded on it, where the live state is written.

    Attributes:
        session_dir: The session the snapshot belongs to.
        tree: The tree the session optimises, or ``None`` when the run had none.
        state: The live session state a stop is recorded on.
        task_id: The specialist task the run belongs to.
    """

    session_dir: Path | None
    tree: Path | None
    state: SharedState | None
    task_id: str

    async def resume(self) -> TreeRestore:
        """Before the run, put back the trees a stop left unrestored, or stop the session again.

        Returns:
            ``clean`` when nothing was held or the trees are back; otherwise
            why the session stopped again, and the run must not start.
        """
        if self.session_dir is None or self.tree is None or not _held(self.session_dir):
            return TreeRestore()
        if await asyncio.to_thread(integrate_in_flight, self.session_dir):
            restore = TreeRestore(error="an integrate is writing the tree, so its restore cannot be retried")
        else:
            restore = await asyncio.to_thread(_retry, self.session_dir, self.tree)
        if restore.clean:
            log.warning("specialist %s: the shared trees are back as the session last accepted them", self.task_id)
        else:
            self._stop(self.session_dir, restore)
        return restore

    async def check(self) -> TreeRestore:
        """After the run, put the tree back and compare the dependency HEADs; stop when a tree is not back.

        A tree that cannot be read is logged and the run kept, so nothing
        raised here masks the run's own outcome.

        Returns:
            What the check found; ``clean`` when the run left the trees alone
            or was not judged.
        """
        if self.session_dir is None or self.tree is None:
            return TreeRestore()
        if await asyncio.to_thread(integrate_in_flight, self.session_dir):
            log.info("specialist %s: %s not checked, an integrate is writing it", self.task_id, self.tree)
            return TreeRestore()
        restore = await asyncio.to_thread(_check, self.session_dir, self.tree, self.task_id)
        if not restore.clean:
            log.error("specialist %s %s; paths: %s", self.task_id, restore.reason, list(restore.changed[:50]))
        if restore.stops:
            self._stop(self.session_dir, restore)
        return restore

    def _stop(self, session_dir: Path, restore: TreeRestore) -> None:
        """Stop the session and hold it until the shared trees are back."""
        hold = _snapshots(session_dir) / _UNRESTORED
        log.critical(
            "specialist %s: stopping the session, the shared trees are not back (%s); a resume retries before "
            "its next run, and %s holds the session until then",
            self.task_id,
            restore.reason,
            hold,
        )
        if self.state is not None:
            self.state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
        try:
            atomic_write_json(hold, {"task_id": self.task_id, "reason": restore.reason})
            if self.state is not None:
                self.state.save(session_dir)
        except OSError as exc:
            log.critical("specialist %s: the stop could not be recorded on disk: %r", self.task_id, exc)


__all__ = [
    "SharedTreeGuard",
    "TreeRestore",
    "baseline_snapshot",
    "held_snapshot",
    "integrate_in_flight",
    "refresh_snapshots",
    "restore_tree",
    "snapshot_tree",
]
