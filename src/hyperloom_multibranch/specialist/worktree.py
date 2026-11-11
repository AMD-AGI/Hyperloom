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

import contextlib
import http.client
import importlib.metadata
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    ComponentLayout,
    PackageBase,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom.measure.pins import ImagePinError, active_image
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot, meta_sections, source_roots
from hyperloom_common.git_safety import (
    app_git,
    check_shared_repository,
    git_dir_umask,
    share_objects_and_refs,
    trusted_admin_dir,
)
from hyperloom_common.shared_storage import SHARED_CHILD_UMASK, parent_is_shared
from hyperloom_multibranch.repo.isolated_git import (
    INIT,
    NO_EXCLUDES_FILE,
    blank_driver_programs,
    git_argv,
    git_env,
    leading_options,
    trusting,
)

logger = logging.getLogger(__name__)

#: The file in a base's git directory that names the paths each worktree links rather than holds.
LINKED_PATHS = "hyperloom-third-party"
#: A Python extension module (``_C.abi3.so``, ``ops.cpython-312-x86_64-linux-gnu.so``): what importing the package
#: from a worktree needs of its build. Other libraries (aiter's JIT modules) stay out, so a patched source is rebuilt.
_EXTENSION = re.compile(r"\.(abi3|cpython-[^.]+)\.so$")
#: Where in a base's git directory the copies of those trees go.
THIRD_PARTY_COPY = "hyperloom-third-party-trees"
_EXCLUDE = ":(exclude)"
_DIFF_HEADER = re.compile(r'^diff --git "?a/(?P<old>.+?)"? "?b/(?P<new>.+?)"?\r?\n?$')
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
_FIXED_CONFIG = (
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.safecrlf=false",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
)
_ADD = "add"
_STAGE = (_ADD, "--all", "--")
#: Paths per ``add -N``: at PATH_MAX each, 1 MiB, inside the kernel's 2 MiB for arguments.
_ADD_BATCH = 256
#: A repository nested in a worktree, tracked or not, is no part of a change: a diff never looks into one.
_NO_NESTED = "--ignore-submodules=all"
_HEAD = "HEAD"
#: The commit the worktree's last sync made, kept apart for each worktree as ``refs/worktree/*`` is: an agent may commit
#: its change on top, and the change is everything since.
_SYNCED = "refs/worktree/hyperloom/synced"
_REV_PARSE = "rev-parse"
_GIT_DIR = ".git"
_OWNER_DIR, _OWNER_FILE = 0o755, 0o644
_QUIET = "--quiet"
_DIFF = "diff"
_CACHED = "--cached"
#: A diff as the toolbox applies it, whatever colour, external diff driver or text conversion a config asks for.
_PLAIN = ("--no-color", "--no-ext-diff", "--no-textconv")


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
    and that checkout's origin; for a wheel, which has no checkout and ships no full commit, see
    :func:`_wheel_upstream`. ``None`` when the version names no commit or neither resolves it."""
    named = _BUILT_FROM.search(package.dist_version or "")
    if named is None:
        return None
    abbreviated = named.group(1)
    return _checkout_upstream(package.path, abbreviated) or _wheel_upstream(
        package.path, package.dist_name, abbreviated
    )


def _checkout_upstream(path: Path, abbreviated: str) -> Upstream | None:
    # The image's checkout may belong to another user, and the system or global config that trusted it is not read.
    trusted = ("-c", "safe.directory=*")
    try:
        commit = _git(path, *trusted, _REV_PARSE, "--verify", _QUIET, f"{abbreviated}^{{commit}}").strip()
        repo = _git(path, *trusted, "remote", "get-url", "origin").strip()
    except WorktreeError:
        return None
    return Upstream(repo, commit) if commit and repo else None


#: A full commit standing alone in an image reference (``vllm-openai-rocm:nightly-rocm100-<commit>``), not inside a
#: longer hex run such as a digest.
_FULL_COMMIT = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_GITHUB_REPO = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+")
#: The first line of a commit's ``.patch`` on GitHub, as ``git format-patch`` writes it.
_PATCH_FROM = re.compile(r"From (?P<commit>[0-9a-f]{40}) ")
PATCH_TIMEOUT_S = 10.0
_PATCH_HEAD_BYTES = 128


#: Each wheel's upstream once resolved, by :func:`_wheel_upstream`'s arguments.
_RESOLVED: dict[tuple[Path, str | None, str], Upstream] = {}


def _wheel_upstream(path: Path, dist_name: str | None, abbreviated: str) -> Upstream | None:
    """The upstream of the wheel installed at ``path``: the GitHub repository its metadata declares, at the full commit
    the active image's reference names when it starts with ``abbreviated``, else the one that repository resolves
    ``abbreviated`` to. Kept once resolved, as the image and the wheel stay the same for the process and the lookup is
    a network call; one that failed is made again at the next call, as GitHub may answer it then."""
    key = (path, dist_name, abbreviated)
    if key in _RESOLVED:
        return _RESOLVED[key]
    repo = _declared_repo(path, dist_name)
    if repo is None:
        return None
    commit = _image_commit(abbreviated) or _github_commit(repo, abbreviated)
    if commit is None:
        return None
    _RESOLVED[key] = Upstream(repo, commit)
    return _RESOLVED[key]


def _declared_repo(path: Path, dist_name: str | None) -> str | None:
    """The GitHub repository the distribution ``dist_name`` beside ``path`` declares as a project URL."""
    if dist_name is None:
        return None
    found = next(iter(importlib.metadata.distributions(name=dist_name, path=[str(path.parent)])), None)
    if found is None:
        return None
    declared = [entry.split(",", 1)[-1].strip().rstrip("/") for entry in found.metadata.get_all("Project-URL") or []]
    return next((url for url in declared if _GITHUB_REPO.fullmatch(url)), None)


def _image_commit(abbreviated: str) -> str | None:
    try:
        reference = active_image().ref
    except ImagePinError:
        return None
    return next((commit for commit in _FULL_COMMIT.findall(reference) if commit.startswith(abbreviated)), None)


def _github_commit(repo: str, abbreviated: str) -> str | None:
    named = _PATCH_FROM.match(_patch_head(f"{repo}/commit/{abbreviated}.patch") or "")
    return named["commit"] if named and named["commit"].startswith(abbreviated) else None


def _patch_head(url: str) -> str | None:
    """The first line of the patch at ``url``, ``None`` when it cannot be read."""
    try:
        with urllib.request.urlopen(url, timeout=PATCH_TIMEOUT_S) as reply:
            return reply.readline(_PATCH_HEAD_BYTES).decode("ascii", "replace")
    except (OSError, http.client.HTTPException) as error:
        logger.warning("cannot read %s, so its commit stays unresolved: %s", url, error)
        return None


def source_unkept(component: str, recipe: Recipe) -> str:
    """Why no change to ``component``'s source can land on ``recipe``, ``""`` when one can: a component new on the
    recipe lands with the upstream commit its installed package was built from, which the rule gate requires."""

    if any(item.component == component for item in recipe.components):
        return ""
    try:
        package = locate_installed_package(COMPONENT_LAYOUTS[component])
    except UnsupportedComponentError as error:
        return f"the recipe does not carry {component}, and its upstream commit cannot be read: {error}"
    if installed_upstream(package) is not None:
        return ""
    version = package.dist_version or "(no version)"
    return (
        f"the recipe does not carry {component}, and the image's {component} {version} names no upstream commit a"
        " checkout holds, and neither the image's reference nor its declared GitHub repository resolves the abbreviated"
        " one a wheel's version names, so the rule gate refuses a source change for lacking its upstream repository and"
        " commit"
    )


def _git(cwd: Path, *args: str, index: Path | None = None, common: Path | None = None, umask: int = -1) -> str:
    """``git <args>`` in ``cwd``.

    With ``common``, ``cwd`` is a branch's worktree, which the agent writes, made from the repository whose git
    directory is ``common``: git runs by :func:`~hyperloom_common.git_safety.app_git`, against the worktree's admin
    directory only once it is proven the app's and ``common``'s, never against one the worktree's gitfile was pointed
    at. Otherwise ``cwd`` is the source base or an image's checkout: every driver program its config defines is blanked
    (:func:`~hyperloom_multibranch.repo.isolated_git.blank_driver_programs`) and the tree is trusted whoever owns it
    (``safe.directory``), as an earlier leg that made it may have been another account; when ``cwd`` holds a shared
    repository (:func:`~hyperloom_common.git_safety.check_shared_repository`), what git reads besides objects and refs
    must be its maker's alone, or nothing runs, and git is pinned to that git directory (``GIT_COMMON_DIR``) and runs
    under the shared child umask, so the admin directory of a worktree it adds is the group's. Either way no program
    the config names runs. ``umask`` is git's (``-1``: this process's, or the shared child umask there)."""
    pinned = _GIT_IDENTITY if index is None else {**_GIT_IDENTITY, "GIT_INDEX_FILE": str(index)}
    if common is not None:
        return _worktree_git(cwd, common, args, pinned)
    try:
        if _shared_git_dir(cwd):
            pinned = {**pinned, "GIT_COMMON_DIR": str(cwd / _GIT_DIR)}
            umask = SHARED_CHILD_UMASK if umask == -1 else umask
    except OSError as error:
        raise WorktreeError(f"git in {cwd} was refused: {error}") from error
    env = git_env(**pinned)
    trusted = trusting(cwd)
    try:
        blanked = blank_driver_programs(cwd, env, *trusted, *leading_options(args))
    except (OSError, subprocess.SubprocessError) as error:
        raise WorktreeError(f"git config in {cwd} cannot be read to blank its filter drivers: {error}") from error
    done = subprocess.run(
        git_argv(*trusted, *_FIXED_CONFIG, *blanked, *args),
        cwd=cwd,
        env=env,
        umask=umask,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        raise WorktreeError(f"git {args[0]} in {cwd} exited {done.returncode}: {done.stderr.decode(errors='replace')}")
    return done.stdout.decode()


def _shared_git_dir(cwd: Path) -> bool:
    """Whether ``cwd`` holds the git directory of a shared repository, checked before git runs there
    (:func:`~hyperloom_common.git_safety.check_shared_repository`).

    Raises:
        OSError: It holds state another account could have written.
    """
    try:
        info = (cwd / _GIT_DIR).lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(info.st_mode) and check_shared_repository(cwd / _GIT_DIR) is not None


def _worktree_git(worktree: Path, common: Path, args: Sequence[str], pinned: Mapping[str, str]) -> str:
    """``git <args>`` in the branch worktree ``worktree`` by ``app_git``; raises :class:`WorktreeError` when it fails
    or is refused, the worktree's git metadata not being the app's. ``app_git`` allows no transport, so a promisor
    remote's lazy fetch starts nothing either."""
    env = {**pinned, "LC_ALL": "C"}
    try:
        admin = trusted_admin_dir(worktree, common)
        done = app_git(worktree, *NO_EXCLUDES_FILE, *_FIXED_CONFIG, *args, timeout=GIT_TIMEOUT_S, env=env, admin=admin)
    except (OSError, subprocess.SubprocessError) as error:
        raise WorktreeError(f"git {args[0]} in {worktree} was refused: {error}") from error
    if done.returncode != 0:
        raise WorktreeError(
            f"git {args[0]} in {worktree} exited {done.returncode}: {done.stderr.strip()}{_blanked(worktree, common)}"
        )
    return done.stdout


def _blanked(worktree: Path, common: Path) -> str:
    """What a failed command in ``worktree`` was not allowed to run: the driver programs the repository's config names,
    which ``app_git`` ran as empty, so a command that needs one (a required filter) fails; ``""`` when it names none."""
    try:
        admin = trusted_admin_dir(worktree, common)
        pairs = blank_driver_programs(
            worktree, git_env(GIT_COMMON_DIR=str(admin.parent.parent)), "--git-dir", str(admin)
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    keys = sorted({setting.partition("=")[0] for setting in pairs[1::2]})
    if not keys:
        return ""
    return f" (the app runs no program a repository's config names; it ran these as empty: {', '.join(keys)})"


def _write_owner_file(path: Path, text: str, umask: int) -> None:
    """Write ``text`` to ``path``, ``0644`` as ``umask`` (:func:`~hyperloom_common.git_safety.git_dir_umask`) masks it:
    read by the next account, written by this one alone."""
    mode = _OWNER_FILE & ~umask
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as written:
        os.fchmod(descriptor, mode)
        written.write(text)


def _commit_all(cwd: Path, message: str) -> None:
    """Commit the whole tree at ``cwd``: the source base as it builds it, before any agent writes there."""
    _git(cwd, *_STAGE)
    _git(cwd, "commit", _QUIET, "--allow-empty", "-m", message)


class SourceBase:
    """The session's base repository at ``path``; its ``installed`` branch holds every component's installed package,
    and the native tree of each component in ``native``."""

    def __init__(
        self,
        path: Path,
        *,
        native: Mapping[str, NativeTree] | None = None,
    ) -> None:
        self.path = path
        self._native = dict(native or {})
        self._worktrees: dict[Path, BranchWorktree] = {}

    def build(self, components: Sequence[str]) -> None:
        """Copy each component's installed package and meta trees to their upstream paths and commit them; a session
        builds one base.

        Built files stay out of the base and, through ``info/exclude``, out of every worktree's stage.

        Raises:
            UnsupportedComponentError: For a component with meta trees whose upstream commit
                :func:`installed_upstream` cannot resolve.
        """
        # Judged before anything is made: below shared storage the run's next leg may be another account of the group,
        # which writes the objects and refs and nothing else. The git directory is made writable by this account alone
        # whatever the umask, and so is the base directory there, so no account renames the git directory.
        shared = parent_is_shared(self.path)
        umask = git_dir_umask()
        self.path.mkdir(_OWNER_DIR, parents=True)
        if shared:
            self.path.chmod(0o2755)
        _git(self.path, *INIT, *(("--shared=group",) if shared else ()), f"--initial-branch={BASE_BRANCH}", umask=umask)
        excluded = list(SKIPPED)
        links: list[tuple[str, str]] = []
        for component in components:
            layout = self._layout(component)
            package, *metas = source_roots(layout, _with_checkout(component, locate_installed_package(layout), layout))
            target = self.path / package.prefix
            # An earlier component's tree may hold this one's (SGLang's checkout holds sgl-kernel's); the package a test
            # loads is the installed one.
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(package.path, target, ignore=_ignore(package.path, layout.generated))
            excluded += [f"/{package.prefix}/{built}/" for built in layout.generated]
            links += _extensions(package, layout.generated)
            for meta in (meta for meta in metas if meta.path.is_dir()):
                shutil.copytree(meta.path, self.path / meta.prefix, ignore=_ignore(meta.path, ()))
            third_party = _third_party(metas)
            if third_party is not None:
                prefix, installed = third_party
                copy = self.path / _GIT_DIR / THIRD_PARTY_COPY / prefix
                shutil.copytree(installed, copy, symlinks=True, ignore=_ignore(installed, ()))
                links.append((prefix.as_posix(), str(copy)))
                excluded.append(f"/{prefix.as_posix()}")
            native = self._native.get(component)
            if native is not None:
                self._lay_native(native.root, (package, *metas))
        git_dir = self.path / _GIT_DIR
        (git_dir / "info").mkdir(_OWNER_DIR, exist_ok=True)
        _write_owner_file(git_dir / "info" / "exclude", "".join(f"{line}\n" for line in excluded), umask)
        _write_owner_file(git_dir / LINKED_PATHS, "".join(f"{prefix}\t{target}\n" for prefix, target in links), umask)
        if shared:
            share_objects_and_refs(git_dir)
        _commit_all(self.path, BASE_BRANCH)

    def worktree(self, branch: str, path: Path) -> BranchWorktree:
        """The worktree of ``branch`` at ``path``, added at the installed packages the first time; every call for
        ``path`` returns the same one, so all its callers share its :attr:`~BranchWorktree.lock`."""
        if not path.exists():
            _git(self.path, "worktree", _ADD, "-b", f"mb/{branch}", str(path), BASE_BRANCH)
        common = self.path / _GIT_DIR
        for prefix, target in linked_paths(common):
            _link(path, PurePosixPath(prefix), target)
        return self._worktrees.setdefault(path, BranchWorktree(path, self._native, common=common))

    def _lay_native(self, native: SourceRoot, laid: Sequence[SourceRoot]) -> None:
        """Copy ``native`` to its upstream path, but for what ``laid`` already put there: the package a test loads is
        the installed one, whatever the checkout holds at its path."""
        inside = PurePosixPath(native.prefix)
        held = [PurePosixPath(root.prefix) for root in laid]
        kept = [path.relative_to(inside).as_posix() for path in held if inside in path.parents]
        shutil.copytree(native.path, self.path / native.prefix, ignore=_ignore(native.path, kept), dirs_exist_ok=True)

    def _layout(self, component: str) -> ComponentLayout:
        layout = COMPONENT_LAYOUTS.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout; its source cannot be laid out")
        return layout


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _link(worktree: Path, prefix: PurePosixPath, target: str) -> None:
    """Link ``target`` at ``prefix`` in ``worktree`` unless something is there, making the directories on the way.

    The agent writes the worktree and may have made a directory on the way a symlink to somewhere else: each directory
    is opened without following one, and the next made and opened relative to it, so nothing is made outside.

    Raises:
        WorktreeError: A directory on the way is a symlink or no directory.
    """
    fd = os.open(worktree, _DIRECTORY_FLAGS)
    try:
        _link_below(fd, worktree / prefix, prefix.parts[:-1], target)
    finally:
        os.close(fd)


def _link_below(dir_fd: int, link: Path, parts: tuple[str, ...], target: str) -> None:
    """Link ``target`` at ``link``, whose directories still to make and open below ``dir_fd`` are ``parts``.

    Raises:
        WorktreeError: A directory on the way is a symlink or no directory.
    """
    if not parts:
        with contextlib.suppress(FileExistsError):
            os.symlink(target, link.name, dir_fd=dir_fd)
        return
    part = parts[0]
    with contextlib.suppress(FileExistsError):
        os.mkdir(part, dir_fd=dir_fd)
    try:
        fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=dir_fd)
    except OSError as error:
        raise WorktreeError(f"{link} cannot be linked: {part} is no directory: {error}") from error
    try:
        _link_below(fd, link, parts[1:], target)
    finally:
        os.close(fd)


def _file_sections(patch: str) -> list[tuple[tuple[str, ...], str]]:
    """``patch`` cut at each ``diff --git`` header: each file's paths (old, new) and its text, line ends kept."""
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    for line in patch.splitlines(keepends=True):
        header = _DIFF_HEADER.match(line)
        if header:
            sections.append(((header["old"], header["new"]), [line]))
        elif sections:
            sections[-1][1].append(line)
    return [(names, "".join(lines)) for names, lines in sections]


def _within(path: str, trees: Sequence[str]) -> bool:
    return any(path == tree or path.startswith(f"{tree}/") for tree in trees)


def linked_paths(common: Path) -> list[tuple[str, str]]:
    """The paths each worktree of the source base whose git directory is ``common`` links rather than holds, as the
    base recorded them: ``3rdparty`` trees from the base's copies (:func:`_third_party`) and compiled extensions from
    the installed package (:func:`_extensions`). Read from the base, never from where a worktree's gitfile points."""
    links = common / LINKED_PATHS
    lines = links.read_text().splitlines() if links.exists() else []
    return [(prefix, target) for prefix, target in (line.split("\t") for line in lines)]


def _extensions(package: SourceRoot, generated: Sequence[str]) -> list[tuple[str, str]]:
    """``package``'s compiled Python extensions, ``(path in a worktree, installed file)``: the worktree holds none, as a
    patch never carries a build, and without them its package cannot be imported (vLLM's ``_C``). They are binaries a
    branch never edits, so each links the installed file."""
    built = [package.path / path for path in generated]
    return [
        (f"{package.prefix}/{path.relative_to(package.path).as_posix()}", str(path))
        for path in sorted(package.path.rglob("*.so"))
        if _EXTENSION.search(path.name) and not any(root in path.parents for root in built)
    ]


def _third_party(metas: Sequence[SourceRoot]) -> tuple[PurePosixPath, Path] | None:
    """The checkout's ``3rdparty`` tree (CK and its kind), the headers a JIT build of the meta trees includes:
    ``(path in a worktree, installed tree)``, or ``None`` without one. A worktree links a copy the base keeps, never
    the installed tree, which every test builds against: an edit through the link must stay in the session. It is
    linked, not laid out, as it is large, never patched, and a patch must not see it."""
    if not metas:
        return None
    tree = metas[0].path.parent / "3rdparty"
    return (PurePosixPath(metas[0].prefix).parent / "3rdparty", tree) if tree.is_dir() else None


def _ignore(root: Path, built: Sequence[str]) -> Callable[[str, list[str]], set[str]]:
    """The copy filter that leaves out :data:`SKIPPED` and the directories ``built`` under ``root``."""
    skipped = shutil.ignore_patterns(*SKIPPED)
    built_dirs = {root / path for path in built}

    def ignore(directory: str, names: list[str]) -> set[str]:
        return set(skipped(directory, names)) | {name for name in names if Path(directory, name) in built_dirs}

    return ignore


def _with_checkout(component: str, package: PackageBase, layout: ComponentLayout) -> Path:
    """Where ``package`` is installed, refused for a component whose meta trees the base lays out when its upstream
    commit cannot be resolved (:func:`installed_upstream`): a recipe pins a component to a full upstream commit."""
    if meta_sections(layout) and installed_upstream(package) is None:
        raise UnsupportedComponentError(
            f"{component} at {package.path} is installed with no git checkout that holds the commit its version "
            f"{package.dist_version!r} names, and neither the image's reference nor its declared GitHub repository "
            "resolves it, so no recipe can pin where its sources came from"
        )
    return package.path


@dataclass(frozen=True)
class Changes:
    """The files a worktree's change writes and deletes, relative to the worktree's root."""

    written: tuple[str, ...]
    deleted: tuple[str, ...]


class BranchWorktree:
    """One branch's checkout of the source base; the experiment commit it was last synced to is named in the subject of
    the commit that sync made.

    Whoever runs git in it or reads its files as a change holds :attr:`lock` for the whole of that work: a sync's
    reset, apply and commit, and a diff's marking of new files, all go through the worktree's one tree and index.

    The agent writes the worktree, its gitfile included, so git runs in it only against an admin directory of the
    source base's repository, ``common`` (:func:`_git`), and never stages it whole: an ``add`` of a path the index
    tracks runs git in a repository nested there, under that repository's own config.
    """

    def __init__(
        self,
        path: Path,
        native: Mapping[str, NativeTree] | None = None,
        *,
        common: Path,
    ) -> None:
        self.path = path
        self._native = dict(native or {})
        self._common = common
        self.lock = threading.Lock()

    def _git(self, *args: str, index: Path | None = None) -> str:
        return _git(self.path, *args, index=index, common=self._common)

    def holds(self, component: str) -> bool:
        """Whether the source base laid out ``component``; a recipe's patches to any other component are not here."""
        layout = COMPONENT_LAYOUTS.get(component)
        return layout is not None and (self.path / layout.upstream_prefix).is_dir()

    def native_upstream(self, component: str) -> Upstream | None:
        """The commit ``component``'s native tree was laid out from, which a change to it is a diff against; ``None``
        when the base laid none."""
        native = self._native.get(component)
        return None if native is None else Upstream(native.upstream_repo, native.upstream_ref)

    def head(self) -> str | None:
        """The experiment commit the worktree was last synced to, or ``None`` before its first sync."""
        synced = self._synced()
        return None if synced is None else self._git("log", "-1", "--format=%s", synced).strip()[len(HEAD_SUBJECT) :]

    def sync(self, head: str, patches: Sequence[bytes]) -> tuple[str, ...]:
        """Make the worktree ``head``: the installed packages with its ``patches`` applied in order; returns the
        commits made in the worktree since its last sync, which it drops, the oldest first.

        Uncommitted edits are dropped too, so a change made against an older head never reaches a diff against this one.
        """
        dropped = tuple(self._git("rev-list", "--reverse", f"{self._since()}..{_HEAD}").split())
        if self.head() == head:
            self._reset(_SYNCED)
            return dropped
        self._reset(BASE_BRANCH)
        with tempfile.TemporaryDirectory(prefix="mb-sync-") as scratch:
            for index, patch in enumerate(patches):
                patch_file = Path(scratch) / f"{index:04d}.patch"
                patch_file.write_bytes(patch)
                self._git("apply", "--index", "--whitespace=nowarn", str(patch_file))
        # The patches went to the index as they were applied: nothing is staged from the tree, which may hold a
        # repository the agent nested in it.
        self._git("commit", _QUIET, "--allow-empty", "-m", f"{HEAD_SUBJECT}{head}")
        self._git("update-ref", _SYNCED, _HEAD)
        return dropped

    def diff(self, component: str) -> str:
        """The worktree's change to ``component`` against its head, committed in the worktree or not, as the patch the
        toolbox applies; empty when none."""
        return self._diff_since(component, *_PLAIN)

    def commit(self) -> str:
        """The git commit the worktree's HEAD is at."""
        return self._git(_REV_PARSE, _HEAD).strip()

    def diff_from(self, commit: str, component: str) -> str:
        """The worktree's files of ``component`` against its git commit ``commit``, whatever was committed on top of it
        since, as the patch the toolbox applies; empty when none.

        It is read through an index of its own: whoever edits the worktree (a third-party agent) may be using the
        worktree's index, and a stage there would race its commits.
        """
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-diff-") as scratch:
            index = Path(scratch) / "index"
            self._git("read-tree", commit, index=index)
            return self._change(commit, paths, *_PLAIN, index=index)

    def diff_between(self, commit: str, before: Sequence[str], after: str, component: str) -> str:
        """The patch that takes ``component``'s files from ``commit`` with the patches ``before`` applied in order to
        ``commit`` with the patch ``after`` applied, as the patch the toolbox applies; empty when both are the same.

        Both are built in indexes of their own, so the worktree's files and index are left as they are.
        """
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-between-") as scratch:
            start = self._tree(commit, before, Path(scratch) / "before")
            end = self._tree(commit, (after,), Path(scratch) / "after")
            return self._git(_DIFF, *_PLAIN, start, end, "--", *paths)

    def diff_applied(self, commit: str, patches: Sequence[str], component: str) -> str:
        """The patch that takes ``component``'s files from ``commit`` to ``commit`` with ``patches`` applied in order,
        as the patch the toolbox applies; empty when they change nothing. The worktree's files and index are left as
        they are."""
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-applied-") as scratch:
            end = self._tree(commit, patches, Path(scratch) / "after")
            return self._git(_DIFF, *_PLAIN, commit, end, "--", *paths)

    def _tree(self, commit: str, patches: Sequence[str], scratch: Path) -> str:
        """The git tree of ``commit`` with ``patches`` applied in order, written through the index under ``scratch``."""
        scratch.mkdir()
        index = scratch / "index"
        self._git("read-tree", commit, index=index)
        for number, patch in enumerate(text for text in patches if text):
            patch_file = scratch / f"{number:04d}.patch"
            patch_file.write_text(patch, encoding="utf-8")
            self._git("apply", _CACHED, "--whitespace=nowarn", str(patch_file), index=index)
        return self._git("write-tree", index=index).strip()

    def changed(self, component: str) -> Changes:
        """The files the worktree's change to ``component`` writes and deletes against its head."""
        listing = self._diff_since(component, "--no-renames", "--name-status")
        rows = [line.split("\t", 1) for line in listing.splitlines() if line.strip()]
        return Changes(
            written=tuple(path for status, path in rows if status != "D"),
            deleted=tuple(path for status, path in rows if status == "D"),
        )

    def _reset(self, commit: str) -> None:
        self._git("reset", "--hard", commit)
        kept = [f"--exclude=/{prefix}" for prefix, _target in linked_paths(self._common)]
        # Twice forced: a repository the agent nested in the tree goes too, or it would outlive the sync.
        self._git("clean", "-ffdx", _QUIET, *kept)

    def source_only(self, component: str, patch: str) -> tuple[str, tuple[str, ...]]:
        """``patch`` less its files outside ``component``'s paths, and the paths it left out: an agent's scratch (logs,
        probes, core dumps) committed beside its change never reaches a recipe, nor the apply that builds one."""
        paths = self._paths(component)
        own = [path for path in paths if not path.startswith(_EXCLUDE)]
        nested = [path.removeprefix(_EXCLUDE) for path in paths if path.startswith(_EXCLUDE)]
        kept, outside = [], []
        for names, section in _file_sections(patch):
            if any(_within(name, own) and not _within(name, nested) for name in names):
                kept.append(section)
            else:
                outside.append(names[-1])
        return "".join(kept), tuple(outside)

    def _diff_since(self, component: str, *options: str) -> str:
        return self._change(self._since(), self._paths(component), *options)

    def _change(self, since: str, paths: Sequence[str], *options: str, index: Path | None = None) -> str:
        """The diff of the worktree's files under ``paths`` against ``since``, a new file shown as created: each
        untracked one is marked intent-to-add by name (:data:`_ADD_BATCH` at a time), never by an ``add`` over a path
        the index tracks. A repository nested in the tree is neither marked (``ls-files`` lists one as its directory)
        nor looked into by the diff (:data:`_NO_NESTED`)."""
        listed = self._git("ls-files", "--others", "--exclude-standard", "-z", "--", *paths, index=index)
        untracked = [path for path in listed.split("\0") if path and not path.endswith("/")]
        for start in range(0, len(untracked), _ADD_BATCH):
            batch = untracked[start : start + _ADD_BATCH]
            self._git("--literal-pathspecs", _ADD, "-N", "--", *batch, index=index)
        return self._git(_DIFF, *options, _NO_NESTED, since, "--", *paths, index=index)

    def _synced(self) -> str | None:
        return self._git("for-each-ref", "--format=%(objectname)", _SYNCED).strip() or None

    def _since(self) -> str:
        """The commit a change in the worktree is made on: its last sync's, or the installed packages before one."""
        return self._synced() or BASE_BRANCH

    def _paths(self, component: str) -> tuple[str, ...]:
        """``component``'s package, the meta trees the source base laid out beside it, and its native tree, less the
        trees of other components inside them."""
        layout = COMPONENT_LAYOUTS.get(component)
        if layout is None:
            raise UnsupportedComponentError(f"{component} has no package layout")
        native = self._native.get(component)
        trees = [section.prefix for section in meta_sections(layout)] + ([native.root.prefix] if native else [])
        # A pathspec that names nothing fails the stage, and an install may ship no meta tree.
        held = self._git("ls-tree", "--name-only", BASE_BRANCH, "--", *trees).splitlines() if trees else []
        own = (layout.upstream_prefix, *held)
        return (*own, *(f"{_EXCLUDE}{tree}" for tree in self._nested(component, own)))

    def _nested(self, component: str, own: Sequence[str]) -> list[str]:
        """The trees of the other components that lie inside ``own``: a change there is theirs."""
        others = [other.upstream_prefix for name, other in COMPONENT_LAYOUTS.items() if name != component]
        others += [tree.root.prefix for name, tree in self._native.items() if name != component]
        inside = [PurePosixPath(path) for path in own]
        return [tree for tree in others if any(root in PurePosixPath(tree).parents for root in inside)]


__all__ = [
    "BASE_BRANCH",
    "HEAD_SUBJECT",
    "SKIPPED",
    "THIRD_PARTY_COPY",
    "LINKED_PATHS",
    "BranchWorktree",
    "Changes",
    "SourceBase",
    "Upstream",
    "WorktreeError",
    "installed_upstream",
    "source_unkept",
    "linked_paths",
]
