# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lend wheel-installed framework packages a Git history for the rewrite controller's run.

The rewrite campaign names its base, keeps and reverts through Git, and a
package pip installed into site-packages has no history. Its Python and Triton
sources are still the code the server imports, so a rewrite of them is as real
as one in a checkout; the tree only lacks a commit to name.

The history lives in the session directory. The package holds a ``.git`` file
naming it and nothing else, and only while the controller runs: the KERNEL entry
withdraws the pointer before integration. Every other lane decides how to apply,
replay and revert a patch by whether the tree is a checkout, and a wheel-installed
framework's earlier KEEPs were recorded as backed-up ``patch`` runs, so the tree
has to read as the plain directory it was whenever anything but the controller
looks at it. Integration lands the controller's patches on it the same way.

The next entry re-attaches the same history, so what the session kept carries
forward into the base it seals.

Compiled extensions and code objects stay out of the index. They are gigabytes
no rewrite edits, and an ignored path is also outside every restore the
campaign runs, so a rebuilt extension is never deleted as campaign residue.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from hyperloom.inference_optimizer.framework_paths import framework_apply_tree
from kernelforge.utils.editable_repo import installed_package_path
from kernelforge.utils.git import git_admin_dir
from kernelforge.workflows.rewrite_controller import reclaim_campaign_branch

log = logging.getLogger(__name__)

_HISTORIES_DIRNAME = "wheel_repositories"

#: Names, inside a history, the package directory it was lent to.
_WORK_TREE_FILENAME = "hyperloom-lent-work-tree"

_EXCLUDES: tuple[str, ...] = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.so.*",
    "*.co",
    "*.hsaco",
    "*.a",
    "*.o",
)

_IDENTITY: tuple[str, ...] = ("-c", "user.name=hyperloom", "-c", "user.email=hyperloom@localhost")

_GIT_TIMEOUT_SEC = 600


def _git(work_tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(work_tree), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=True,
    )


def _installed_package_dir(name: str) -> Path | None:
    """The directory pip installed ``name`` into, or ``None`` when it is not installed that way.

    A package some enclosing checkout already tracks has a history of its own.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    package_dir = Path(spec.origin).resolve().parent
    if not installed_package_path(str(package_dir)):
        return None
    tree = framework_apply_tree(str(package_dir))
    tracked_elsewhere = tree is not None and tree.checkout and tree.root.resolve() != package_dir
    return None if tracked_elsewhere else package_dir


def _histories_root(session_dir: Path) -> Path:
    return (Path(session_dir) / _HISTORIES_DIRNAME).resolve()


def _pointer_target(package_dir: Path) -> Path | None:
    """The history ``package_dir/.git`` names, or ``None`` when it is not a pointer file."""
    pointer = package_dir / ".git"
    if not pointer.is_file():
        return None
    return git_admin_dir(package_dir)


def _point_at(package_dir: Path, history: Path) -> None:
    (package_dir / ".git").write_text(f"gitdir: {history}\n", encoding="utf-8")


def _return_stale_history(package_dir: Path, stale: Path) -> None:
    """Hand back a tree an earlier session's campaign was killed holding, before it is snapshotted.

    A snapshot commits the tree as it stands, so a rewrite nobody measured
    would otherwise become this session's baseline. Only the earlier history
    knows which commit to return to, and it is gone once the pointer moves.
    """
    if not stale.is_dir():
        log.warning(
            "%s names a history that no longer exists (%s); snapshotting the tree as it stands",
            package_dir,
            stale,
        )
        return
    branch = reclaim_campaign_branch(package_dir)
    if branch:
        log.warning("returned %s from campaign branch %s an earlier session left behind", package_dir, branch)


def _create_history(package_dir: Path, history: Path) -> None:
    """Snapshot ``package_dir`` into a new history, leaving nothing behind when that fails.

    A history without its first commit would be re-attached by the next entry
    as if it were complete, and nothing could be sealed against it.
    """
    history.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(package_dir, "init", "-q", f"--separate-git-dir={history}")
        (history / "info").mkdir(parents=True, exist_ok=True)
        (history / "info" / "exclude").write_text("".join(f"{pattern}\n" for pattern in _EXCLUDES), encoding="utf-8")
        (history / _WORK_TREE_FILENAME).write_text(f"{package_dir}\n", encoding="utf-8")
        _git(package_dir, "add", "-A", "--", ".")
        _git(package_dir, *_IDENTITY, "commit", "-q", "--no-verify", "-m", f"hyperloom: snapshot of {package_dir.name}")
    except (OSError, subprocess.SubprocessError):
        (package_dir / ".git").unlink(missing_ok=True)
        shutil.rmtree(history, ignore_errors=True)
        raise


def _attach(package_dir: Path, history: Path) -> bool:
    """Give ``package_dir`` the session's history, creating it on the first entry."""
    pointer = package_dir / ".git"
    if pointer.is_dir():
        return False
    current = _pointer_target(package_dir)
    if current is not None and current != history:
        _return_stale_history(package_dir, current)
        # ``init --separate-git-dir`` over an existing pointer moves the history it
        # names, which belongs to whoever left it.
        pointer.unlink()
    if history.is_dir():
        _point_at(package_dir, history)
    else:
        _create_history(package_dir, history)
    return True


def attach_wheel_histories(session_dir: Path, packages: Iterable[str]) -> tuple[Path, ...]:
    """Lend each of ``packages`` that pip installed this session's history; return the packages lent one.

    Best-effort per package: a package that cannot take one costs its own
    operators, which the controller then skips for having no repository.
    """
    attached: list[Path] = []
    for name in packages:
        package_dir = _installed_package_dir(name)
        if package_dir is None:
            continue
        history = _histories_root(session_dir) / f"{name}.git"
        try:
            if _attach(package_dir, history):
                attached.append(package_dir)
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("could not lend %s a history at %s: %s", package_dir, history, error)
    return tuple(attached)


def lent_work_trees(session_dir: Path) -> frozenset[Path]:
    """Every package directory this session has lent a history, whether or not it holds the pointer now."""
    root = _histories_root(session_dir)
    if not root.is_dir():
        return frozenset()
    trees: set[Path] = set()
    for record in root.glob(f"*.git/{_WORK_TREE_FILENAME}"):
        text = record.read_text(encoding="utf-8").strip()
        if text:
            trees.add(Path(text))
    return frozenset(trees)


def detach_wheel_histories(session_dir: Path) -> None:
    """Remove the pointers this session's histories left in installed packages."""
    root = _histories_root(session_dir)
    for package_dir in lent_work_trees(session_dir):
        target = _pointer_target(package_dir)
        if target is None or target.parent != root:
            continue
        try:
            (package_dir / ".git").unlink()
        except OSError as error:
            log.warning("could not remove the history pointer in %s: %s", package_dir, error)


__all__ = ["attach_wheel_histories", "detach_wheel_histories", "lent_work_trees"]
