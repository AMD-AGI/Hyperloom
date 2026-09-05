# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pinned identity of the source trees a session may observe and patch.

Resolved once at session start into ``reports/bringup/trees.json``. Frames are
normalised against these roots, so a failure digest is comparable only between
two attempts that agreed on them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.git_safety import repo_root, safe_directory_args
from hyperloom.common.io import atomic_write_json
from hyperloom.inference_optimizer.session import paths as session_paths
from hyperloom.inference_optimizer.session import session_paths as session_layout
from hyperloom.orchestrator.framework.paths import resolve_kernel_search_roots

#: ``vcs`` value for a tree whose directory is itself a git working tree.
VCS_GIT = "git_checkout"

#: ``vcs`` value for a tree with no version control of its own -- an installed
#: wheel, whose files carry no history to diff a patch against.
VCS_NONE = "none"

#: Seconds allowed for the commit probe run against a candidate tree.
_GIT_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class TreeIdentity:
    """One pinned source tree.

    Attributes:
        tree_id: Stable identifier derived from ``root``, the same across
            sessions on one host.
        root: Absolute root of the tree.
        package_dirs: Absolute package directories covered by ``root``.
        vcs: :data:`VCS_GIT` or :data:`VCS_NONE`. Delivery dispatches on this:
            a git tree diffs through its own objects, one without through a
            recorded content manifest.
        head_commit: For a git tree, the commit ``root`` was pinned at; ``""``
            otherwise.
    """

    tree_id: str
    root: str
    package_dirs: tuple[str, ...]
    vcs: str
    head_commit: str = ""

    @property
    def is_git(self) -> bool:
        """Whether this tree's root is itself a git work tree."""
        return self.vcs == VCS_GIT

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, with ``package_dirs`` as a list."""
        return {
            "tree_id": self.tree_id,
            "root": self.root,
            "package_dirs": list(self.package_dirs),
            "vcs": self.vcs,
            "head_commit": self.head_commit,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TreeIdentity":
        """Rebuild a tree from :meth:`to_dict` output.

        Raises:
            KeyError: When a field :meth:`to_dict` writes is absent.
        """
        return cls(
            tree_id=str(raw["tree_id"]),
            root=str(raw["root"]),
            package_dirs=tuple(str(d) for d in raw["package_dirs"]),
            vcs=str(raw["vcs"]),
            head_commit=str(raw["head_commit"]),
        )


def path_slug(path: str, *, fallback: str = "tree") -> str:
    """Return ``<basename>-<8 hex>`` for an absolute path.

    The hash disambiguates two paths whose directories share a name.

    Args:
        path: The absolute path to name.
        fallback: Basename used when the path has none.

    Returns:
        str: The slug.
    """
    digest = hashlib.sha256(path.encode("utf-8", "replace")).hexdigest()[:8]
    name = path.rstrip("/").rsplit("/", 1)[-1] or fallback
    return f"{name}-{digest}"


def tree_id_for(root: str) -> str:
    """Return the identity a tree is pinned, recorded and compared under."""
    return path_slug(root)


def head_commit(root: str | Path) -> str:
    """Return the commit ``root`` currently sits at, or ``""``.

    Args:
        root: A git work-tree root. An enclosing repository's commit is not
            this tree's, so callers pass the root they mean.

    Returns:
        str: The commit, or ``""`` when the tree has no commit yet.

    Raises:
        subprocess.TimeoutExpired: When the probe wedges, rather than recording
            ``""`` as a diff base.
    """
    args = safe_directory_args(["-C", str(root), "rev-parse", "HEAD"])
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def tree_kind(directory: Path | str) -> str:
    """Return :data:`VCS_GIT` or :data:`VCS_NONE` for a directory used as a root.

    The enclosing checkout counts only when it *is* the directory asked about:
    the search walks upwards, and a wheel under an unrelated checkout is not
    that repo.
    """
    path = Path(directory).expanduser().resolve()
    root = repo_root(path)
    return VCS_GIT if root is not None and Path(root) == path else VCS_NONE


def resolve_trees(roots: Sequence[str] | None = None) -> tuple[TreeIdentity, ...]:
    """Pin the source trees present on this host.

    Args:
        roots: Package directories to pin, defaulting to the framework source
            roots in the order PolicyGate and patch application resolve them.

    Returns:
        tuple[TreeIdentity, ...]: One entry per distinct directory, in discovery
        order; a candidate that is not a directory here is skipped.
    """
    candidates = tuple(roots) if roots is not None else resolve_kernel_search_roots()
    pinned: list[TreeIdentity] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.strip().rstrip("/")
        if not text:
            continue
        package_dir = Path(text)
        if not package_dir.is_dir():
            continue
        root = str(package_dir)
        if root in seen:
            continue
        seen.add(root)
        vcs = tree_kind(package_dir)
        pinned.append(
            TreeIdentity(
                tree_id=tree_id_for(root),
                root=root,
                package_dirs=(root,),
                vcs=vcs,
                head_commit=head_commit(root) if vcs == VCS_GIT else "",
            )
        )
    return tuple(pinned)


def tree_roots(trees: Sequence[TreeIdentity]) -> tuple[str, ...]:
    """Return every root and package directory the trees cover.

    Both, because a frame may name either.

    Returns:
        tuple[str, ...]: De-duplicated absolute directories, longest first so a
        nested package directory wins over its enclosing root.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tree in trees:
        for value in (tree.root, *tree.package_dirs):
            text = value.strip().rstrip("/")
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return tuple(sorted(out, key=len, reverse=True))


def trees_path(session_dir: Path | None = None) -> Path:
    """Return ``<session_dir>/reports/bringup/trees.json``.

    Args:
        session_dir: Session root; defaults to the current session.
    """
    sd = session_dir if session_dir is not None else session_paths.session_dir()
    return session_layout.reports_dir(sd) / "bringup" / "trees.json"


def write_trees(trees: Sequence[TreeIdentity], *, session_dir: Path | None = None) -> Path:
    """Write the pinned trees to their session artifact, returning its path.

    Args:
        trees: Pinned trees to record.
        session_dir: Session root; defaults to the current session.
    """
    target = trees_path(session_dir)
    atomic_write_json(target, {"trees": [t.to_dict() for t in trees]}, trailing_newline=True)
    return target


def read_trees(session_dir: Path | None = None) -> tuple[TreeIdentity, ...]:
    """Read the pinned trees back, empty when no session pinned any.

    Args:
        session_dir: Session root; defaults to the current session.

    Raises:
        ValueError: When the artifact exists and does not decode; reading it as
            no trees would re-key every digest taken against it.
        KeyError: When it decodes without the fields :func:`write_trees` writes.
    """
    target = trees_path(session_dir)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    return tuple(TreeIdentity.from_dict(e) for e in raw["trees"])


__all__ = [
    "VCS_GIT",
    "VCS_NONE",
    "TreeIdentity",
    "head_commit",
    "read_trees",
    "resolve_trees",
    "path_slug",
    "tree_id_for",
    "tree_kind",
    "tree_roots",
    "write_trees",
]
