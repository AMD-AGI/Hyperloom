# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a source tree held before a round, and what the round left it holding.

For a git tree the base commit is the pre-image; for a tree with no git the
baseline is an explicit content manifest, per declared target, in which
absent-before is a state of its own. A patch's validated post-state exists only
in the worktree that produced it, so it is hashed there and frozen beside the
patch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.git_safety import safe_directory_args
from hyperloom.common.io import atomic_write_json
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from hyperloom.orchestrator.bringup.trees import VCS_GIT, head_commit
from hyperloom.orchestrator.source_snapshot import _safe_rel

log = logging.getLogger(__name__)

#: Seconds allowed for a single git probe against a tree.
_GIT_TIMEOUT_SEC = 30.0

#: Recorded in place of a hash when the path did not exist pre-round.
ABSENT = ""

#: Mode recorded for a path that did not exist pre-round.
ABSENT_MODE = -1


def file_digest(path: Path) -> str:
    """Return the lowercase sha256 of ``path``'s bytes, :data:`ABSENT` if unreadable.

    Args:
        path: File to hash.

    Returns:
        str: Hex digest, or :data:`ABSENT`.
    """
    try:
        with Path(path).open("rb") as fh:
            digest = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ABSENT
    return digest.hexdigest()


@dataclass(frozen=True)
class FileEntry:
    """One declared target's pre-round state.

    Attributes:
        rel: Path relative to the tree root, posix-separated.
        existed: Whether the path was a regular file before the round.
        sha256: Content hash, or :data:`ABSENT` when ``existed`` is False.
        mode: Permission bits, or :data:`ABSENT_MODE` when ``existed`` is False.
    """

    rel: str
    existed: bool
    sha256: str = ABSENT
    mode: int = ABSENT_MODE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"rel": self.rel, "existed": self.existed, "sha256": self.sha256, "mode": self.mode}


@dataclass(frozen=True)
class TreeBaseline:
    """What a tree looked like before a round touched it.

    Attributes:
        tree_id: The pinned tree this baseline belongs to.
        root: Absolute tree root.
        kind: The tree's vcs discriminant; every revert and drift check
            dispatches on it.
        base_commit: Pre-round commit for a git tree, ``""`` otherwise.
        entries: Per-target pre-image, one per declared target.
    """

    tree_id: str
    root: str
    kind: str
    base_commit: str = ""
    entries: tuple[FileEntry, ...] = ()

    @property
    def is_git(self) -> bool:
        """Whether reverts and diffs for this tree go through git."""
        return self.kind == VCS_GIT

    def entry(self, rel: str) -> FileEntry | None:
        """Return the recorded pre-image for tree-relative ``rel``, else ``None``."""
        for item in self.entries:
            if item.rel == rel:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "tree_id": self.tree_id,
            "root": self.root,
            "kind": self.kind,
            "base_commit": self.base_commit,
            "entries": [e.to_dict() for e in self.entries],
        }


def capture_baseline(
    *,
    tree_id: str,
    root: Path | str,
    kind: str,
    targets: Iterable[str] = (),
) -> TreeBaseline:
    """Record what ``root`` looks like before the round mutates it.

    Args:
        tree_id: The pinned tree's id.
        root: Absolute tree root.
        kind: The tree's vcs discriminant.
        targets: Tree-relative paths the round declares it will touch; only
            these are hashed.

    Returns:
        TreeBaseline: The pre-round record, plus the base commit for a git tree.
    """
    root_path = Path(root)
    entries: list[FileEntry] = []
    for raw in sorted({str(t) for t in targets}):
        rel = _safe_rel(raw)
        if rel is None:
            log.warning("delivery: declared target %r escapes %s; it gets no pre-image", raw, root_path)
            continue
        target = root_path / rel
        if target.is_file():
            entries.append(
                FileEntry(rel=rel, existed=True, sha256=file_digest(target), mode=target.stat().st_mode & 0o7777)
            )
        else:
            entries.append(FileEntry(rel=rel, existed=False))
    return TreeBaseline(
        tree_id=tree_id,
        root=str(root_path),
        kind=kind,
        base_commit=head_commit(root_path) if kind == VCS_GIT else "",
        entries=tuple(entries),
    )


def baseline_path(session_dir: Path, round_key: str, tree_id: str) -> Path:
    """Return the session artifact path for one round's tree baseline.

    Args:
        session_dir: The session root directory.
        round_key: Identifier of the round the baseline was taken for.
        tree_id: The tree the baseline covers.

    Returns:
        Path: ``<session_dir>/reports/delivery/<round_key>/<tree_id>.json``.
    """
    safe_round = round_key.replace("/", "_") or "round"
    safe_tree = tree_id.replace("/", "_") or "tree"
    return reports_dir(session_dir) / "delivery" / safe_round / f"{safe_tree}.json"


def write_baseline(baseline: TreeBaseline, *, session_dir: Path, round_key: str) -> str:
    """Persist a baseline as a session artifact.

    Args:
        baseline: The record to write.
        session_dir: The session root directory.
        round_key: Identifier of the round the baseline was taken for.

    Returns:
        str: The written path.
    """
    target = baseline_path(session_dir, round_key, baseline.tree_id)
    atomic_write_json(target, baseline.to_dict(), trailing_newline=True)
    return str(target)


def drifted_paths(baseline: TreeBaseline, *, targets: Sequence[str] = ()) -> tuple[str, ...]:
    """Return the declared paths whose current state differs from the baseline.

    A git tree is asked what its diff against the base commit touches; a tree
    with no git is re-hashed against the recorded manifest.

    Args:
        baseline: The pre-round record.
        targets: Optional pathspec to scope a git tree's diff. Ignored for a
            non-git tree, whose manifest already names what was declared.

    Returns:
        tuple[str, ...]: Tree-relative paths that moved, sorted.

    Raises:
        RuntimeError: When a git tree cannot be diffed against its base commit,
            or was recorded without one.
    """
    root = Path(baseline.root)
    if baseline.is_git:
        if not baseline.base_commit:
            raise RuntimeError(f"git tree {root} has no recorded base commit to check drift against")
        argv = ["-C", str(root), "diff", "--name-only", baseline.base_commit, "--"]
        argv.extend(str(t) for t in targets or (".",))
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *safe_directory_args(argv)],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git diff against {baseline.base_commit} failed in {root}: {proc.stderr.strip()}")
        return tuple(sorted(line.strip() for line in proc.stdout.splitlines() if line.strip()))
    moved: list[str] = []
    for entry in baseline.entries:
        target = root / entry.rel
        if not target.is_file():
            if entry.existed:
                moved.append(entry.rel)
            continue
        if not entry.existed or file_digest(target) != entry.sha256:
            moved.append(entry.rel)
    return tuple(sorted(moved))


#: Sidecar written beside a harvested patch, holding its post-image digests.
POST_IMAGE_SUFFIX = ".post_images.json"


def post_images_path(patch_path: Path | str) -> Path:
    """Return the sidecar path carrying ``patch_path``'s frozen post-images."""
    patch = Path(patch_path)
    return patch.with_name(patch.name + POST_IMAGE_SUFFIX)


def post_images_from_diff(diff_text: str, root: Path | str) -> dict[str, str]:
    """Hash every path a ``-p1`` diff writes, as it stands in ``root``.

    Called where the work was validated, with ``root`` still holding the
    specialist's output.

    Args:
        diff_text: The harvested unified diff.
        root: Tree the diff was taken from, holding the post-state now.

    Returns:
        dict[str, str]: Tree-relative path to content hash; a path the diff
        deletes is omitted, having no post-image.
    """
    from hyperloom.orchestrator.specialists.patch_safety import patch_file_targets

    root_path = Path(root)
    out: dict[str, str] = {}
    for _old, new in patch_file_targets(diff_text):
        if new == "/dev/null":
            continue
        rel = _safe_rel(new.split("/", 1)[1] if "/" in new else new)
        if rel is None:
            continue
        digest = file_digest(root_path / rel)
        if digest:
            out[rel] = digest
    return out


def write_post_images(patch_path: Path | str, images: Mapping[str, str]) -> str:
    """Freeze a patch's post-image digests beside it, before transport.

    Args:
        patch_path: The patch the digests describe.
        images: Tree-relative path to content hash.

    Returns:
        str: The written path, or ``""`` when there was nothing to freeze.
    """
    if not images:
        return ""
    target = post_images_path(patch_path)
    atomic_write_json(target, dict(images), trailing_newline=True)
    return str(target)


def read_post_images(patch_path: Path | str) -> dict[str, str]:
    """Read back the digests frozen by :func:`write_post_images`.

    Args:
        patch_path: The patch the digests describe.

    Returns:
        dict[str, str]: The frozen digests, empty when the patch carries no
        sidecar.
    """
    try:
        raw = json.loads(post_images_path(patch_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {str(k): str(v) for k, v in raw.items() if str(k) and str(v)}


__all__ = [
    "ABSENT",
    "FileEntry",
    "TreeBaseline",
    "baseline_path",
    "capture_baseline",
    "drifted_paths",
    "file_digest",
    "post_images_from_diff",
    "read_post_images",
    "write_baseline",
    "write_post_images",
]
