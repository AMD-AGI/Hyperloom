# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The code a specialist edits: the installed packages, with the source beside them they build from (aiter's csrc) and,
where the host builds native code, the tree that build compiles (sgl-kernel's C++), as a git repository, one worktree
per branch.

The toolbox applies a recipe's patches to the packages installed in the image, so a specialist writes its change
against those same files. The session's source base commits them laid out as their upstream repositories lay them out,
and a branch's worktree is that commit with the branch head's patches applied; a change is the worktree's diff
against its head, a patch the toolbox applies where it applies every other.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    ComponentLayout,
    PackageBase,
    PackageLocator,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot, meta_sections, source_roots

#: Built files a text patch can never change.
SKIPPED = ("__pycache__", "*.pyc", "*.pyo", "*.so", "*.a", "*.o", "*.co", "*.hsaco")
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
_HEAD = "HEAD"
_QUIET = "--quiet"
_DIFF = "diff"
_CACHED = "--cached"
#: A diff as the toolbox applies it, whatever colour or external diff driver the user's git config asks for.
_PLAIN = ("--no-color", "--no-ext-diff")


class WorktreeError(RuntimeError):
    """A git command on the source base or a worktree failed."""


#: The commit a source build's version names, as setuptools-scm writes it: ``0.5.21.dev20261008+g11b1fb21de``.
_BUILT_FROM = re.compile(r"\+g([0-9a-f]{7,40})")


@dataclass(frozen=True)
class Upstream:
    """The repository and the full commit an installed package was built from."""

    repo: str
    commit: str


def installed_upstream(package: PackageBase) -> Upstream | None:
    """Where ``package`` was built from: the commit its version names, resolved in the checkout it is installed from,
    and that checkout's origin; ``None`` when the version names no commit or no checkout holds it.

    A wheel has no checkout: its version names an abbreviated commit, which no full commit can be read from.
    """
    named = _BUILT_FROM.search(package.dist_version or "")
    if named is None:
        return None
    try:
        commit = _git(package.path, "rev-parse", "--verify", _QUIET, f"{named.group(1)}^{{commit}}").strip()
        repo = _git(package.path, "remote", "get-url", "origin").strip()
    except WorktreeError:
        return None
    return Upstream(repo, commit) if commit and repo else None


def _git(cwd: Path, *args: str, stdin: bytes | None = None, index: Path | None = None) -> str:
    # An inherited GIT_* (GIT_DIR inside a hook) would aim the command at another repository.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
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
    _git(cwd, "commit", _QUIET, "--allow-empty", "-m", message)


class SourceBase:
    """The session's base repository at ``path``; its ``installed`` branch holds every component's installed package,
    and the native tree of each component in ``native``."""

    def __init__(
        self,
        path: Path,
        *,
        locator: PackageLocator = locate_installed_package,
        layouts: Mapping[str, ComponentLayout] = COMPONENT_LAYOUTS,
        native: Mapping[str, NativeTree] | None = None,
    ) -> None:
        self.path = path
        self._locator = locator
        self._layouts = layouts
        self._native = dict(native or {})

    def build(self, components: Sequence[str]) -> None:
        """Copy each component's installed package and meta trees to their upstream paths and commit them; a session
        builds one base.

        Built files stay out of the base and, through ``info/exclude``, out of every worktree's stage.

        Raises:
            UnsupportedComponentError: For a component with meta trees installed without a checkout that holds the
                commit its version names (aiter from a wheel).
        """
        self.path.mkdir(parents=True)
        _git(self.path, "init", f"--initial-branch={BASE_BRANCH}")
        excluded = list(SKIPPED)
        for component in components:
            layout = self._layout(component)
            package, *metas = source_roots(layout, _with_checkout(component, self._locator(layout), layout))
            shutil.copytree(package.path, self.path / package.prefix, ignore=_ignore(package.path, layout.generated))
            excluded += [f"/{package.prefix}/{built}/" for built in layout.generated]
            for meta in (meta for meta in metas if meta.path.is_dir()):
                shutil.copytree(meta.path, self.path / meta.prefix, ignore=_ignore(meta.path, ()))
            native = self._native.get(component)
            if native is not None:
                self._lay_native(native.root, (package, *metas))
        (self.path / ".git" / "info").mkdir(exist_ok=True)
        (self.path / ".git" / "info" / "exclude").write_text("".join(f"{line}\n" for line in excluded))
        _commit_all(self.path, BASE_BRANCH)

    def worktree(self, branch: str, path: Path) -> BranchWorktree:
        """The worktree of ``branch`` at ``path``, added at the installed packages the first time."""
        if not path.exists():
            _git(self.path, "worktree", "add", "-b", f"mb/{branch}", str(path), BASE_BRANCH)
        return BranchWorktree(path, self._layouts, self._native)

    def _lay_native(self, native: SourceRoot, laid: Sequence[SourceRoot]) -> None:
        """Copy ``native`` to its upstream path, but for what ``laid`` already put there: the package a test loads is
        the installed one, whatever the checkout holds at its path."""
        inside = PurePosixPath(native.prefix)
        held = [PurePosixPath(root.prefix) for root in laid]
        kept = [path.relative_to(inside).as_posix() for path in held if inside in path.parents]
        shutil.copytree(native.path, self.path / native.prefix, ignore=_ignore(native.path, kept), dirs_exist_ok=True)

    def _layout(self, component: str) -> ComponentLayout:
        layout = self._layouts.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout; its source cannot be laid out")
        return layout


def _ignore(root: Path, built: Sequence[str]) -> Callable[[str, list[str]], set[str]]:
    """The copy filter that leaves out :data:`SKIPPED` and the directories ``built`` under ``root``."""
    skipped = shutil.ignore_patterns(*SKIPPED)
    built_dirs = {root / path for path in built}

    def ignore(directory: str, names: list[str]) -> set[str]:
        return set(skipped(directory, names)) | {name for name in names if Path(directory, name) in built_dirs}

    return ignore


def _with_checkout(component: str, package: PackageBase, layout: ComponentLayout) -> Path:
    """Where ``package`` is installed, refused for a component whose meta trees the base lays out when it is installed
    without a checkout: a recipe pins a component to a full upstream commit, which only a checkout can resolve."""
    if meta_sections(layout) and installed_upstream(package) is None:
        raise UnsupportedComponentError(
            f"{component} at {package.path} is installed with no git checkout that holds the commit its version "
            f"{package.dist_version!r} names (a wheel names only an abbreviated commit), so no recipe can pin where its "
            f"sources came from; the source base lays {component} out only from a develop install"
        )
    return package.path


@dataclass(frozen=True)
class Changes:
    """The files a worktree's change writes and deletes, relative to the worktree's root."""

    written: tuple[str, ...]
    deleted: tuple[str, ...]


class BranchWorktree:
    """One branch's checkout of the source base; the experiment commit its HEAD holds is named in HEAD's subject."""

    def __init__(
        self,
        path: Path,
        layouts: Mapping[str, ComponentLayout] = COMPONENT_LAYOUTS,
        native: Mapping[str, NativeTree] | None = None,
    ) -> None:
        self.path = path
        self._layouts = layouts
        self._native = dict(native or {})

    def holds(self, component: str) -> bool:
        """Whether the source base laid out ``component``; a recipe's patches to any other component are not here."""
        layout = self._layouts.get(component)
        return layout is not None and (self.path / layout.upstream_prefix).is_dir()

    def native_upstream(self, component: str) -> Upstream | None:
        """The commit ``component``'s native tree was laid out from, which a change to it is a diff against; ``None``
        when the base laid none."""
        native = self._native.get(component)
        return None if native is None else Upstream(native.upstream_repo, native.upstream_ref)

    def head(self) -> str | None:
        """The experiment commit HEAD holds, or ``None`` while it holds the installed packages alone."""
        subject = _git(self.path, "log", "-1", "--format=%s").strip()
        return subject[len(HEAD_SUBJECT) :] if subject.startswith(HEAD_SUBJECT) else None

    def sync(self, head: str, patches: Sequence[bytes]) -> None:
        """Make the worktree ``head``: the installed packages with its ``patches`` applied in order.

        Uncommitted edits are dropped, so a change made against an older head never reaches a diff against this one.
        """
        if self.head() == head:
            self._reset(_HEAD)
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
        return self._staged_diff(component, *_PLAIN)

    def commit(self) -> str:
        """The git commit the worktree's HEAD is at."""
        return _git(self.path, "rev-parse", _HEAD).strip()

    def diff_from(self, commit: str, component: str) -> str:
        """The worktree's files of ``component`` against its git commit ``commit``, whatever was committed on top of it
        since, as the patch the toolbox applies; empty when none.

        It is read through an index of its own: whoever edits the worktree (a third-party agent) may be using the
        worktree's index, and a stage there would race its commits.
        """
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-diff-") as scratch:
            index = Path(scratch) / "index"
            _git(self.path, "read-tree", commit, index=index)
            _git(self.path, *_STAGE, *paths, index=index)
            return _git(self.path, _DIFF, _CACHED, *_PLAIN, commit, "--", *paths, index=index)

    def diff_between(self, commit: str, before: Sequence[str], after: str, component: str) -> str:
        """The patch that takes ``component``'s files from ``commit`` with the patches ``before`` applied in order to
        ``commit`` with the patch ``after`` applied, as the patch the toolbox applies; empty when both are the same.

        Both are built in indexes of their own, so the worktree's files and index are left as they are.
        """
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-between-") as scratch:
            start = self._tree(commit, before, Path(scratch) / "before")
            end = self._tree(commit, (after,), Path(scratch) / "after")
            return _git(self.path, _DIFF, *_PLAIN, start, end, "--", *paths)

    def _tree(self, commit: str, patches: Sequence[str], scratch: Path) -> str:
        """The git tree of ``commit`` with ``patches`` applied in order, written through the index under ``scratch``."""
        scratch.mkdir()
        index = scratch / "index"
        _git(self.path, "read-tree", commit, index=index)
        for number, patch in enumerate(text for text in patches if text):
            patch_file = scratch / f"{number:04d}.patch"
            patch_file.write_text(patch, encoding="utf-8")
            _git(self.path, "apply", _CACHED, "--whitespace=nowarn", str(patch_file), index=index)
        return _git(self.path, "write-tree", index=index).strip()

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
        _git(self.path, "clean", "-fdx", _QUIET)

    def _staged_diff(self, component: str, *options: str) -> str:
        paths = self._paths(component)
        _git(self.path, *_STAGE, *paths)
        return _git(self.path, _DIFF, _CACHED, *options, _HEAD, "--", *paths)

    def _paths(self, component: str) -> tuple[str, ...]:
        """``component``'s package, the meta trees the source base laid out beside it, and its native tree."""
        layout = self._layouts.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout")
        native = self._native.get(component)
        trees = [section.prefix for section in meta_sections(layout)] + ([native.root.prefix] if native else [])
        # A pathspec that names nothing fails the stage, and an install may ship no meta tree.
        held = _git(self.path, "ls-tree", "--name-only", BASE_BRANCH, "--", *trees).splitlines() if trees else []
        return (layout.upstream_prefix, *held)


__all__ = [
    "BASE_BRANCH",
    "HEAD_SUBJECT",
    "SKIPPED",
    "BranchWorktree",
    "Changes",
    "SourceBase",
    "Upstream",
    "WorktreeError",
    "installed_upstream",
]
