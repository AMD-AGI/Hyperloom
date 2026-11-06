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

import functools
import http.client
import importlib.metadata
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.error
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
from hyperloom_common.shared_storage import SHARED_CHILD_UMASK, SHARED_DIR_MODE, make_shared_dirs, share_new_directory
from hyperloom_multibranch.repo.isolated_git import INIT, blank_filter_drivers, git_argv, git_env

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
_STAGE = ("add", "--all", "--")
_HEAD = "HEAD"
#: The commit the worktree's last sync made, kept apart for each worktree as ``refs/worktree/*`` is: an agent may commit
#: its change on top, and the change is everything since.
_SYNCED = "refs/worktree/hyperloom/synced"
_REV_PARSE = "rev-parse"
_GIT_DIR = ".git"
_QUIET = "--quiet"
_DIFF = "diff"
_CACHED = "--cached"
#: A diff as the toolbox applies it, whatever colour, external diff driver or text conversion a git config asks for.
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


@functools.cache
def _wheel_upstream(path: Path, dist_name: str | None, abbreviated: str) -> Upstream | None:
    """The upstream of the wheel installed at ``path``: the GitHub repository its metadata declares, at the full commit
    the active image's reference names when it starts with ``abbreviated``, else the one that repository resolves
    ``abbreviated`` to. Cached, as the image and the wheel stay the same for the process and the lookup is a network
    call."""
    repo = _declared_repo(path, dist_name)
    if repo is None:
        return None
    commit = _image_commit(abbreviated) or _github_commit(repo, abbreviated)
    return None if commit is None else Upstream(repo, commit)


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
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as error:
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


def _git(cwd: Path, *args: str, stdin: bytes | None = None, index: Path | None = None) -> str:
    """``git <args>`` in ``cwd``, a tree the run's sandboxes write: every filter driver its config defines is blanked
    (:func:`~hyperloom_multibranch.repo.isolated_git.blank_filter_drivers`), as the add, diff, apply and reset here
    would run one on the tree's files. git runs under umask ``002``: what it makes -- a worktree, its files, the
    repository's objects and refs -- is the group's, as the dev sandboxes and the run's next leg write it."""
    pinned = _GIT_IDENTITY if index is None else {**_GIT_IDENTITY, "GIT_INDEX_FILE": str(index)}
    env = git_env(**pinned)
    try:
        # The command's own options (a safe.directory) decide which repository's config git reads, so the list does too.
        blanked = blank_filter_drivers(cwd, env, *_leading_options(args))
    except (OSError, subprocess.SubprocessError) as error:
        raise WorktreeError(f"git config in {cwd} cannot be read to blank its filter drivers: {error}") from error
    done = subprocess.run(
        git_argv(*_FIXED_CONFIG, *blanked, *args),
        cwd=cwd,
        env=env,
        umask=SHARED_CHILD_UMASK,
        input=stdin,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        raise WorktreeError(f"git {args[0]} in {cwd} exited {done.returncode}: {done.stderr.decode(errors='replace')}")
    return done.stdout.decode()


def _leading_options(args: Sequence[str]) -> list[str]:
    """The ``-c <setting>`` pairs ``args`` starts with, before its subcommand."""
    count = 0
    while count + 1 < len(args) and args[count] == "-c":
        count += 2
    return list(args[:count])


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
            UnsupportedComponentError: For a component with meta trees installed without a checkout that holds the
                commit its version names (aiter from a wheel).
        """
        make_shared_dirs(self.path.parent)
        self.path.mkdir(SHARED_DIR_MODE)
        share_new_directory(self.path)
        _git(self.path, *INIT, f"--initial-branch={BASE_BRANCH}")
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
        (git_dir / "info").mkdir(exist_ok=True)
        (git_dir / "info" / "exclude").write_text("".join(f"{line}\n" for line in excluded))
        (git_dir / LINKED_PATHS).write_text("".join(f"{prefix}\t{target}\n" for prefix, target in links))
        _commit_all(self.path, BASE_BRANCH)

    def worktree(self, branch: str, path: Path) -> BranchWorktree:
        """The worktree of ``branch`` at ``path``, added at the installed packages the first time; every call for
        ``path`` returns the same one, so all its callers share its :attr:`~BranchWorktree.lock`."""
        if not path.exists():
            _git(self.path, "worktree", "add", "-b", f"mb/{branch}", str(path), BASE_BRANCH)
        for prefix, target in linked_paths(path):
            if not (path / prefix).exists():
                make_shared_dirs((path / prefix).parent)
                (path / prefix).symlink_to(target)
        return self._worktrees.setdefault(path, BranchWorktree(path, self._native))

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


def linked_paths(worktree: Path) -> list[tuple[str, str]]:
    """The paths the worktree at ``worktree`` links rather than holds, as its source base recorded them: ``3rdparty``
    trees from the base's copies (:func:`_third_party`) and compiled extensions from the installed package
    (:func:`_extensions`)."""
    common = Path(_git(worktree, _REV_PARSE, "--path-format=absolute", "--git-common-dir").strip())
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
    reset, apply and commit, and a diff's stage, all go through the worktree's one tree and index.
    """

    def __init__(
        self,
        path: Path,
        native: Mapping[str, NativeTree] | None = None,
    ) -> None:
        self.path = path
        self._native = dict(native or {})
        self.lock = threading.Lock()

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
        return (
            None if synced is None else _git(self.path, "log", "-1", "--format=%s", synced).strip()[len(HEAD_SUBJECT) :]
        )

    def sync(self, head: str, patches: Sequence[bytes]) -> tuple[str, ...]:
        """Make the worktree ``head``: the installed packages with its ``patches`` applied in order; returns the
        commits made in the worktree since its last sync, which it drops, the oldest first.

        Uncommitted edits are dropped too, so a change made against an older head never reaches a diff against this one.
        """
        dropped = tuple(_git(self.path, "rev-list", "--reverse", f"{self._since()}..{_HEAD}").split())
        if self.head() == head:
            self._reset(_SYNCED)
            return dropped
        self._reset(BASE_BRANCH)
        with tempfile.TemporaryDirectory(prefix="mb-sync-") as scratch:
            for index, patch in enumerate(patches):
                patch_file = Path(scratch) / f"{index:04d}.patch"
                patch_file.write_bytes(patch)
                _git(self.path, "apply", "--whitespace=nowarn", str(patch_file))
        _commit_all(self.path, f"{HEAD_SUBJECT}{head}")
        _git(self.path, "update-ref", _SYNCED, _HEAD)
        return dropped

    def diff(self, component: str) -> str:
        """The worktree's change to ``component`` against its head, committed in the worktree or not, as the patch the
        toolbox applies; empty when none."""
        return self._staged_diff(component, *_PLAIN)

    def commit(self) -> str:
        """The git commit the worktree's HEAD is at."""
        return _git(self.path, _REV_PARSE, _HEAD).strip()

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

    def diff_applied(self, commit: str, patches: Sequence[str], component: str) -> str:
        """The patch that takes ``component``'s files from ``commit`` to ``commit`` with ``patches`` applied in order,
        as the patch the toolbox applies; empty when they change nothing. The worktree's files and index are left as
        they are."""
        paths = self._paths(component)
        with tempfile.TemporaryDirectory(prefix="mb-applied-") as scratch:
            end = self._tree(commit, patches, Path(scratch) / "after")
            return _git(self.path, _DIFF, *_PLAIN, commit, end, "--", *paths)

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
        kept = [f"--exclude=/{prefix}" for prefix, _target in linked_paths(self.path)]
        _git(self.path, "clean", "-fdx", _QUIET, *kept)

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

    def _staged_diff(self, component: str, *options: str) -> str:
        paths = self._paths(component)
        _git(self.path, *_STAGE, *paths)
        return _git(self.path, _DIFF, _CACHED, *options, self._since(), "--", *paths)

    def _synced(self) -> str | None:
        return _git(self.path, "for-each-ref", "--format=%(objectname)", _SYNCED).strip() or None

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
        held = _git(self.path, "ls-tree", "--name-only", BASE_BRANCH, "--", *trees).splitlines() if trees else []
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
