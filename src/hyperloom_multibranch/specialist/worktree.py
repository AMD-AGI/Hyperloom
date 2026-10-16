# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The code a specialist edits: the installed framework packages as a git repository, one worktree per branch.

The toolbox applies a recipe's patches to the packages installed in the image, so a specialist writes its change
against those same files. The session's source base commits them laid out as their upstream repositories lay them out,
and a branch's worktree is that commit with the branch head's patches applied; a change is the worktree's diff
against its head, a patch the toolbox applies where it applies every other.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    ComponentLayout,
    PackageLocator,
    UnsupportedComponentError,
    locate_installed_package,
)

#: Built files a text patch can never change.
SKIPPED = ("__pycache__", "*.pyc", "*.pyo", "*.so", "*.a", "*.o")
BASE_BRANCH = "installed"
HEAD_SUBJECT = "head "
GIT_TIMEOUT_S = 300.0
# The commits are the session's own, whoever runs it, and the bytes must reach the patch as they are on disk.
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "hyperloom-multibranch",
    "GIT_AUTHOR_EMAIL": "multibranch@hyperloom.invalid",
    "GIT_COMMITTER_NAME": "hyperloom-multibranch",
    "GIT_COMMITTER_EMAIL": "multibranch@hyperloom.invalid",
}
_FIXED_CONFIG = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "diff.noprefix=false")
_STAGE = ("add", "--all", "--")


class WorktreeError(RuntimeError):
    """A git command on the source base or a worktree failed."""


def _git(cwd: Path, *args: str, stdin: bytes | None = None) -> str:
    # An inherited GIT_* (GIT_DIR inside a hook) would aim the command at another repository.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    done = subprocess.run(
        ["git", *_FIXED_CONFIG, *args],
        cwd=cwd,
        env={**env, **_GIT_IDENTITY},
        input=stdin,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        raise WorktreeError(f"git {args[0]} in {cwd} exited {done.returncode}: {done.stderr.decode(errors='replace')}")
    return done.stdout.decode()


def _commit_all(cwd: Path, message: str) -> None:
    _git(cwd, *_STAGE)
    _git(cwd, "commit", "--quiet", "--allow-empty", "-m", message)


class SourceBase:
    """The session's base repository at ``path``; its ``installed`` branch holds every component's installed package."""

    def __init__(
        self,
        path: Path,
        *,
        locator: PackageLocator = locate_installed_package,
        layouts: Mapping[str, ComponentLayout] = COMPONENT_LAYOUTS,
    ) -> None:
        self.path = path
        self._locator = locator
        self._layouts = layouts

    def build(self, components: Sequence[str]) -> None:
        """Copy each component's installed package to its upstream path and commit it; a session builds one base."""
        self.path.mkdir(parents=True)
        _git(self.path, "init", f"--initial-branch={BASE_BRANCH}")
        for component in components:
            layout = self._layout(component)
            package = self._locator(layout)
            shutil.copytree(package.path, self.path / layout.upstream_prefix, ignore=shutil.ignore_patterns(*SKIPPED))
        _commit_all(self.path, BASE_BRANCH)

    def worktree(self, branch: str, path: Path) -> BranchWorktree:
        """The worktree of ``branch`` at ``path``, added at the installed packages the first time."""
        if not path.exists():
            _git(self.path, "worktree", "add", "-b", f"mb/{branch}", str(path), BASE_BRANCH)
        return BranchWorktree(path, self._layouts)

    def _layout(self, component: str) -> ComponentLayout:
        layout = self._layouts.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout; its source cannot be laid out")
        return layout


@dataclass(frozen=True)
class Changes:
    """The files a worktree's change writes and deletes, relative to the worktree's root."""

    written: tuple[str, ...]
    deleted: tuple[str, ...]


class BranchWorktree:
    """One branch's checkout of the source base; the experiment commit its HEAD holds is named in HEAD's subject."""

    def __init__(self, path: Path, layouts: Mapping[str, ComponentLayout] = COMPONENT_LAYOUTS) -> None:
        self.path = path
        self._layouts = layouts

    def holds(self, component: str) -> bool:
        """Whether the source base laid out ``component``; a recipe's patches to any other component are not here."""
        layout = self._layouts.get(component)
        return layout is not None and (self.path / layout.upstream_prefix).is_dir()

    def head(self) -> str | None:
        """The experiment commit HEAD holds, or ``None`` while it holds the installed packages alone."""
        subject = _git(self.path, "log", "-1", "--format=%s").strip()
        return subject[len(HEAD_SUBJECT) :] if subject.startswith(HEAD_SUBJECT) else None

    def sync(self, head: str, patches: Sequence[bytes]) -> None:
        """Make the worktree ``head``: the installed packages with its ``patches`` applied in order.

        Uncommitted edits are dropped, so a change made against an older head never reaches a diff against this one.
        """
        if self.head() == head:
            self._reset("HEAD")
            return
        self._reset(BASE_BRANCH)
        with tempfile.TemporaryDirectory(prefix="mb-sync-") as scratch:
            for index, patch in enumerate(patches):
                patch_file = Path(scratch) / f"{index:04d}.patch"
                patch_file.write_bytes(patch)
                _git(self.path, "apply", "--whitespace=nowarn", str(patch_file))
        _commit_all(self.path, f"{HEAD_SUBJECT}{head}")

    def diff(self, component: str) -> str:
        """The worktree's change to ``component`` against its head, as the patch the toolbox applies; empty when none."""
        return self._staged_diff(component, "--no-color", "--no-ext-diff")

    def changed(self, component: str) -> Changes:
        """The files the worktree's change to ``component`` writes and deletes against its head."""
        listing = self._staged_diff(component, "--no-renames", "--name-status")
        rows = [line.split("\t", 1) for line in listing.splitlines() if line.strip()]
        return Changes(
            written=tuple(path for status, path in rows if status != "D"),
            deleted=tuple(path for status, path in rows if status == "D"),
        )

    def _reset(self, commit: str) -> None:
        _git(self.path, "reset", "--hard", commit)
        _git(self.path, "clean", "-fdx", "--quiet")

    def _staged_diff(self, component: str, *options: str) -> str:
        layout = self._layouts.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout")
        _git(self.path, *_STAGE, layout.upstream_prefix)
        return _git(self.path, "diff", "--cached", *options, "HEAD", "--", layout.upstream_prefix)


__all__ = ["BASE_BRANCH", "HEAD_SUBJECT", "SKIPPED", "BranchWorktree", "Changes", "SourceBase", "WorktreeError"]
