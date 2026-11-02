# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The image's installed frameworks kept as installed on a host that cannot mount them read-only: what a sandbox
command wrote into them is put back after it.

Every test lays its patches over the installed packages, so a file a tool wrote there in place (a Forge fusion
defaulting to the installed tree) fails every branch's tests, and every server imports it. Where a sandbox command
cannot get a mount namespace of its own (:func:`~hyperloom_multibranch.specialist.agent_sandbox.framework_protection`),
a :class:`FrameworkGuard` takes a pristine copy of each package the first time it checks, on this host's local disk,
and every later check compares the package with it: a file changed, removed or replaced is copied back, a file or
directory added is removed. A file whose size, modification time and mode are as copied is not read, so a check of
an untouched package costs a walk of its directories. Bytecode caches are no part of a package: Python writes them
beside the sources it imports, and validates them against those sources.

A guard serves every sandbox of its process, which run commands at once, so its checks hold one lock. It holds the
commands running and those that ended since its last check, by holder and command, and names them in what a check
restored, in the session's log and in each of their own logs: a command that wrote while it ran is one of them. A
command counts as running until it and every process it started ended, which its sandbox says (:meth:`ended`). While
any runs, the guard checks every :attr:`~FrameworkGuard.patrol_seconds` on a thread of its own, so what a process
writes while it runs for hours is restored before the tests that follow measure it; and it checks once more when its
process exits, removing its copy after, and logs as an error the commands still running then, whose later writes
nothing restores.

The toolbox patches the installed frameworks in place itself, with TraceLens's profiling patch, before it profiles
(``hyperloom.orchestrator.actions.executors._server_patcher``). A changed or added file that is the copy with one of
the patches under ``profile_patches`` (TraceLens's checkout) applied is that patch: it is kept and becomes the file's
copy, so the next profile finds it applied; anything else written to the same file is restored to it.

A check races the commands that write while it runs: a file or directory that disappears under it was removed
meanwhile, and is left to the next check rather than ending this one before it restored the rest.

When the copy cannot be taken, :meth:`FrameworkGuard.check` raises :class:`FrameworksUnguarded`, and a command that
asked is refused rather than run over packages nothing restores.
"""

from __future__ import annotations

import atexit
import filecmp
import itertools
import logging
import os
import shutil
import stat
import tempfile
import threading
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PRISTINE_PREFIX = "hyperloom-frameworks-"
_CACHE_DIR = "__pycache__"
_CACHE_SUFFIX = ".pyc"
#: How often a guard checks while a command runs: a walk of vllm's directories costs a fraction of a second, and a test
#: takes longer than this to lay its patches over the packages.
PATROL_SECONDS = 5.0
#: Where TraceLens keeps the patches the toolbox applies to the installed frameworks before it profiles.
PATCH_TREE = ("examples", "custom_workflows", "inference_analysis")
_TARGET_PREFIX = "+++ b/"


class FrameworksUnguarded(RuntimeError):
    """The installed frameworks can neither be mounted read-only nor restored, so no sandbox command may run."""


@dataclass(frozen=True)
class GuardedPackage:
    """An installed package at ``path``, named ``name``; ``generated`` are its directories the runtime writes builds
    into, which a check leaves as they are."""

    name: str
    path: Path
    generated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Restoration:
    """What one check put back, each path with how it differed, and the commands that ran since the check before
    it."""

    paths: tuple[str, ...]
    suspects: tuple[str, ...]

    def describe(self) -> str:
        by = ", ".join(self.suspects) or "no sandbox command this process knows of"
        return (
            f"the image's installed frameworks were written, and are restored as installed: {'; '.join(self.paths)}. "
            f"Written while these ran: {by}. Write into a copy (the branch worktree), never the installed package."
        )


@dataclass(frozen=True)
class _Stamp:
    size: int
    mtime_ns: int
    mode: int


def _stamp(path: Path) -> _Stamp:
    found = path.lstat()
    return _Stamp(found.st_size, found.st_mtime_ns, found.st_mode)


@dataclass(frozen=True)
class _Tree:
    """The files (a symbolic link to a directory among them) and directories below a package, relative to it."""

    files: list[str]
    dirs: list[str]

    @classmethod
    def walk(cls, root: Path, generated: tuple[str, ...]) -> _Tree:
        """The tree below ``root``, bytecode caches and ``generated`` left out."""

        tree = cls([], [])
        for top, subdirs, names in os.walk(root, onerror=_raise_unless_gone):
            here = Path(top).relative_to(root)
            kept = []
            for name in subdirs:
                rel = (here / name).as_posix()
                if name == _CACHE_DIR or rel in generated:
                    continue
                if os.path.islink(os.path.join(top, name)):
                    tree.files.append(rel)
                else:
                    kept.append(name)
                    tree.dirs.append(rel)
            subdirs[:] = kept
            tree.files.extend((here / name).as_posix() for name in names if not name.endswith(_CACHE_SUFFIX))
        return tree


class ProfilePatches:
    """The profiling patches under a TraceLens checkout at ``root``, read once, by the file each one changes or adds;
    none when there is no checkout."""

    def __init__(self, root: Path | None) -> None:
        self._root = root
        self._targets: dict[str, list[Path]] | None = None

    def applied(self, package: str, rel: str, installed: Path, pristine: Path | None) -> bool:
        """Whether ``installed``, the file at ``rel`` of the package named ``package``, is ``pristine`` (``None`` for a
        file the package did not have) with a patch that changes it applied."""

        for target, patches in self._read().items():
            parts = target.split("/")
            if package not in parts or "/".join(parts[parts.index(package) + 1 :]) != rel:
                continue
            if any(_patched(patch, target, pristine, installed) for patch in patches):
                return True
        return False

    def _read(self) -> dict[str, list[Path]]:
        if self._targets is None:
            self._targets = {}
            tree = None if self._root is None else self._root.joinpath(*PATCH_TREE)
            for patch in sorted(tree.rglob("*.patch")) if tree is not None and tree.is_dir() else ():
                for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith(_TARGET_PREFIX):
                        self._targets.setdefault(line[len(_TARGET_PREFIX) :].strip(), []).append(patch)
        return self._targets


def _patched(patch: Path, target: str, pristine: Path | None, installed: Path) -> bool:
    """Whether ``patch``'s change to ``target`` turns ``pristine`` into ``installed``."""

    with tempfile.TemporaryDirectory(prefix=PRISTINE_PREFIX) as scratch:
        result = Path(scratch) / target
        if pristine is not None:
            result.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pristine, result)
        applied = subprocess.run(
            ["git", "apply", f"--include={target}", str(patch)], cwd=scratch, capture_output=True, check=False
        )
        return applied.returncode == 0 and result.is_file() and filecmp.cmp(result, installed, shallow=False)


@dataclass
class _Baseline:
    """``package`` as it was copied to ``pristine``: the stamp of each file and the directories, by relative path;
    ``patches`` the profiling patches a file may carry and still be as installed."""

    package: GuardedPackage
    pristine: Path
    patches: ProfilePatches
    stamps: dict[str, _Stamp] = field(default_factory=dict)
    dirs: set[str] = field(default_factory=set)

    @classmethod
    def take(cls, package: GuardedPackage, pristine: Path, patches: ProfilePatches) -> _Baseline:
        if not package.path.is_dir():
            raise FileNotFoundError(f"no package at {package.path}")
        baseline = cls(package, pristine, patches)
        pristine.mkdir(parents=True)
        tree = _Tree.walk(package.path, package.generated)
        for rel in tree.dirs:
            (pristine / rel).mkdir(parents=True, exist_ok=True)
        for rel in tree.files:
            _copy(package.path / rel, pristine / rel)
            baseline.stamps[rel] = _stamp(package.path / rel)
        baseline.dirs = set(tree.dirs)
        return baseline

    def restore(self) -> list[str]:
        """Put the package back as copied; each path that differed, with how."""

        root = self.package.path
        tree = _Tree.walk(root, self.package.generated)
        found: list[str] = []
        for rel in sorted(set(tree.files) - self.stamps.keys()):
            if not self._profiled(rel, None) and _gone(_remove, root / rel):
                found.append(f"added {root / rel}")
        for rel in sorted(set(tree.dirs) - self.dirs, reverse=True):
            if _gone(shutil.rmtree, root / rel):
                found.append(f"added {root / rel}/")
        present = set(tree.files)
        for rel in sorted(self.dirs):
            (root / rel).mkdir(parents=True, exist_ok=True)
        for rel, stamp in list(self.stamps.items()):
            how = self._differs(rel, stamp, rel in present)
            if how is None or how == "changed" and self._profiled(rel, self.pristine / rel):
                continue
            if not _gone(_copy, self.pristine / rel, root / rel):
                if not os.path.lexists(self.pristine / rel):
                    raise FileNotFoundError(f"the pristine copy {self.pristine / rel} was removed")
                continue
            if how:
                found.append(f"{how} {root / rel}")
        return found

    def _profiled(self, rel: str, pristine: Path | None) -> bool:
        """Keep the file at ``rel`` as its copy when it is ``pristine`` with a profiling patch applied."""

        installed = self.package.path / rel
        try:
            if not self.patches.applied(self.package.name, rel, installed, pristine):
                return False
            _copy(installed, self.pristine / rel)
            self.stamps[rel] = _stamp(installed)
        except FileNotFoundError:
            return False
        parent = Path(rel).parent
        while parent != Path("."):
            self.dirs.add(parent.as_posix())
            parent = parent.parent
        logger.info("%s carries the toolbox's profiling patch, which is kept", installed)
        return True

    def _differs(self, rel: str, stamp: _Stamp, present: bool) -> str | None:
        """How the file at ``rel`` differs from its copy: ``None`` when its stamp is the copy's, empty when only its
        stamp differs."""

        installed = self.package.path / rel
        try:
            if not present:
                return "removed"
            if _stamp(installed) == stamp:
                return None
            return "" if _same(installed, self.pristine / rel) else "changed"
        except FileNotFoundError:
            return "removed"


@dataclass(frozen=True)
class _Command:
    """A command, named by its holder, whose own ``log`` says what a check restored while it ran."""

    name: str
    log: Path | None


class FrameworkGuard:
    """Restores ``packages`` as they were when the guard first checked them, every ``patrol_seconds`` while a command
    runs."""

    def __init__(
        self,
        packages: tuple[GuardedPackage, ...],
        patrol_seconds: float = PATROL_SECONDS,
        profile_patches: Path | None = None,
    ) -> None:
        self.packages = packages
        self.patrol_seconds = patrol_seconds
        self._patches = ProfilePatches(profile_patches)
        self._baselines: list[_Baseline] | None = None
        self._lock = threading.Lock()
        self._tickets = itertools.count()
        self._running: dict[int, _Command] = {}
        self._ended: list[_Command] = []
        self._patrol: threading.Thread | None = None

    def started(self, holder: str, command: str, log: Path | None = None) -> int:
        """Count ``command`` of ``holder``, whose output goes to ``log``, as running; the ticket :meth:`ended`
        takes."""

        with self._lock:
            ticket = next(self._tickets)
            self._running[ticket] = _Command(f"{holder or 'a sandbox'}: `{command}`", log)
            if self._patrol is None:
                self._patrol = threading.Thread(target=self._patrolled, name="framework-guard", daemon=True)
                self._patrol.start()
            return ticket

    def ended(self, ticket: int) -> None:
        with self._lock:
            held = self._running.pop(ticket, None)
            if held is not None:
                self._ended.append(held)

    def check(self) -> Restoration | None:
        """Restore every package as first checked, which is when its copy is taken; what was restored, ``None`` when
        nothing was. Raises :class:`FrameworksUnguarded` when a copy cannot be taken or a package cannot be restored."""

        with self._lock:
            baselines = self._baselines if self._baselines is not None else self._take()
            try:
                paths = [path for baseline in baselines for path in baseline.restore()]
            except OSError as error:
                raise FrameworksUnguarded(f"the installed frameworks cannot be restored: {error}") from error
            suspects = list(dict.fromkeys((*self._running.values(), *self._ended)))
            self._ended = []
        if not paths:
            return None
        restored = Restoration(tuple(paths), tuple(dict.fromkeys(command.name for command in suspects)))
        logger.error("%s", restored.describe())
        for log in dict.fromkeys(command.log for command in suspects if command.log is not None):
            try:
                with log.open("a", encoding="utf-8") as out:
                    out.write(f"\n[hyperloom] {restored.describe()}\n")
            except OSError:
                logger.exception("what was restored cannot be said in %s", log)
        return restored

    def _patrolled(self) -> None:
        while True:
            time.sleep(self.patrol_seconds)
            with self._lock:
                if not self._running:
                    self._patrol = None
                    return
            try:
                self.check()
            except FrameworksUnguarded:
                logger.exception("the installed frameworks are not restored while sandbox commands run")

    def _last_check(self) -> None:
        with self._lock:
            running = [command.name for command in self._running.values()]
        if running:
            logger.error(
                "this process exits while sandbox commands still run, and nothing restores what they write into the "
                "installed frameworks after: %s",
                ", ".join(running),
            )
        try:
            self.check()
        except FrameworksUnguarded:
            logger.exception("the installed frameworks are not restored as this process exits")

    def _take(self) -> list[_Baseline]:
        try:
            pristine = Path(tempfile.mkdtemp(prefix=PRISTINE_PREFIX))
        except OSError as error:
            raise FrameworksUnguarded(f"no local directory for the frameworks' pristine copy: {error}") from error
        atexit.register(shutil.rmtree, pristine, True)
        # atexit runs last what it was given first: the last check needs the copy.
        atexit.register(self._last_check)
        try:
            baselines = [_Baseline.take(package, pristine / package.name, self._patches) for package in self.packages]
        except OSError as error:
            shutil.rmtree(pristine, ignore_errors=True)
            raise FrameworksUnguarded(f"the installed frameworks cannot be copied to {pristine}: {error}") from error
        self._baselines = baselines
        return baselines


def _raise_unless_gone(error: OSError) -> None:
    if not isinstance(error, FileNotFoundError):
        raise error


def _gone(act: Callable[..., object], *paths: Path) -> bool:
    """Whether ``act`` on ``paths`` was done: not when a path it needed disappeared meanwhile."""

    try:
        act(*paths)
    except FileNotFoundError:
        return False
    return True


def _same(one: Path, other: Path) -> bool:
    if one.is_symlink() or other.is_symlink():
        return one.is_symlink() and other.is_symlink() and os.readlink(one) == os.readlink(other)
    if one.is_dir() or stat.S_IMODE(one.stat().st_mode) != stat.S_IMODE(other.stat().st_mode):
        return False
    return filecmp.cmp(one, other, shallow=False)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy(source: Path, target: Path) -> None:
    """Replace ``target`` with ``source`` in one rename, so a server importing it never reads half a file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    staged = target.with_name(f".{target.name}.{PRISTINE_PREFIX}{os.getpid()}")
    if source.is_symlink():
        os.symlink(os.readlink(source), staged)
    else:
        shutil.copy2(source, staged)
    os.replace(staged, target)


__all__ = ["FrameworkGuard", "FrameworksUnguarded", "GuardedPackage", "ProfilePatches", "Restoration"]
