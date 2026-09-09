# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide when a repository must be edited where it is, and serialize who does.

An editable install answers ``import`` from a ``sys.meta_path`` finder pinned to
one directory, so a lane that copies the repository and edits the copy measures
the original: its change is never loaded and the campaign reads as no
improvement. Such a repository has to be edited in place, which makes it shared
mutable state between every lane that reaches it -- hence one lock, taken by
whoever is about to write, keyed on the repository itself so it holds across
processes.

Both halves live here rather than beside one lane because a second opinion on
either is a defect. Two answers to "is this editable" would have one lane copy
while another edits in place; two lock implementations, or two lock paths, would
serialize nothing at all.
"""

from __future__ import annotations

import fcntl
import functools
import os
import re
import site
import sys


def editable_roots() -> list[str]:
    """Collect filesystem roots of PEP 660 editable-finder installs.

    Cached, and returned as a fresh list so a caller cannot disturb the cache.
    The scan opens every ``__editable__*`` file under every site-packages on the
    path -- on a serving image that is a few hundred -- and the controller asks
    this question once a second for the length of a campaign, through the
    checkpoint probe that resolves each task's workspace. An install layout does
    not change inside one run.
    """
    return list(_editable_roots_cached())


@functools.lru_cache(maxsize=1)
def _editable_roots_cached() -> tuple[str, ...]:
    """Memoize one scan for the life of the process."""
    return _scan_editable_roots()


def _scan_editable_roots() -> tuple[str, ...]:
    """Scan the interpreter's search path for editable-install roots.

    Scans site-packages for ``__editable__*.pth`` and ``__editable___*_finder.py``
    and extracts the absolute paths they map into. Such packages are imported via
    a sys.meta_path finder that points at the *live* repo and CANNOT be overridden
    by PYTHONPATH, so a git worktree copy is never imported.

    Handles two finder layouts:
      1. Path-string .pth files that contain absolute paths in quotes.
      2. Setuptools-style .pth files that ``import __editable___<pkg>_finder``;
         the finder .py has a ``MAPPING`` dict mapping package names to paths.
    """
    roots: set[str] = set()
    seen_dirs: set[str] = set()
    scan_dirs = list(sys.path)
    try:
        scan_dirs.extend(site.getsitepackages())
    except Exception:
        # Absent under a virtualenv built without the site module's framework
        # support, and raising rather than returning empty on some builds. The
        # prefix probing below reaches the same directories, so this is one of
        # several ways to find them rather than the only one.
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            scan_dirs.append(site.getusersitepackages())
        except Exception:
            # Same: a user site directory is optional, and its absence says
            # nothing about the roots the rest of the scan will find.
            pass
    # Venv / conda site-packages may not appear in sys.path; probe conventional
    # locations for sys.prefix, VIRTUAL_ENV, CONDA_PREFIX, and the interpreter.
    _pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    _prefixes = {sys.prefix, sys.exec_prefix, sys.base_prefix}
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        v = os.environ.get(var)
        if v:
            _prefixes.add(v)
    # Derive the venv from the interpreter path.
    _interp = os.path.realpath(sys.executable)
    if os.sep + "bin" + os.sep in _interp:
        _prefixes.add(_interp.rsplit(os.sep + "bin" + os.sep, 1)[0])
    for prefix in _prefixes:
        for sub in (f"lib/{_pyver}/site-packages", f"lib/{_pyver}/dist-packages"):
            cand = os.path.join(prefix, sub)
            if os.path.isdir(cand):
                scan_dirs.append(cand)
    for scan_dir in scan_dirs:
        if not scan_dir or scan_dir in seen_dirs or not os.path.isdir(scan_dir):
            continue
        seen_dirs.add(scan_dir)
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("__editable__"):
                continue
            if not (entry.endswith(".pth") or entry.endswith("_finder.py")):
                continue
            try:
                with open(os.path.join(scan_dir, entry), errors="replace") as handle:
                    text = handle.read()
            except OSError:
                # This one file, not the scan: a root named by a later finder is
                # still a root, and a swallowed scan would hand a campaign a
                # private checkout of a repository that must be edited in place.
                continue
            # Layout 0: bare absolute path on a line (no quotes, no import).
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("/") and not line.startswith("#") and "import" not in line and os.path.isdir(line):
                    roots.add(os.path.realpath(line))
            # Layout 1: quoted absolute paths directly in the file.
            for quoted in re.findall(r"['\"](/[^'\"]+)['\"]", text):
                if os.path.isdir(quoted):
                    roots.add(os.path.realpath(quoted))
            # Layout 2: .pth imports a _finder.py; read its MAPPING dict for
            # paths. The finder file lives next to the .pth in site-packages.
            if entry.endswith(".pth"):
                imported = re.search(r"import\s+(__editable___\w+_finder)", text)
                if imported:
                    finder_file = os.path.join(scan_dir, imported.group(1) + ".py")
                    try:
                        with open(finder_file, errors="replace") as finder_handle:
                            finder_text = finder_handle.read()
                    except OSError:
                        continue
                    for quoted in re.findall(r"['\"](/[^'\"]+)['\"]", finder_text):
                        if os.path.isdir(quoted):
                            roots.add(os.path.realpath(quoted))
    return tuple(sorted(roots))


def needs_inplace(kernel_repo: str) -> bool:
    """True when kernel_repo is, contains, or sits under an editable-finder root.

    In that case forge must edit the live repo in place (the finder imports the
    live path; a worktree copy would be invisible -> the loop would no-op).

    The containment half is worth stating plainly, because it decides the fate
    of a whole tree: one editable subpackage anywhere inside a monorepo makes
    the monorepo itself the borrowed workspace, cut onto a campaign branch and
    restored path by path afterwards. ``_require_tree_at`` keeps that from
    folding somebody else's uncommitted work into the patch by refusing to
    borrow a tree that is not the base commit it claims, but the scope of what
    gets borrowed is decided here.
    """
    if not kernel_repo:
        return False
    repo = os.path.realpath(kernel_repo)
    for root in editable_roots():
        # The separators matter: without them /a/repo-2 would match /a/repo and
        # a sibling checkout would be borrowed.
        if root == repo or root.startswith(repo + os.sep) or repo.startswith(root + os.sep):
            return True
    return False


class RepoLock:
    """Owned in-place repo lock; released explicitly after restore.

    Releasing twice is a no-op rather than an error. Three lanes now take this
    lock and each releases it from a ``finally``, so a second release is the
    ordinary shape of a nested cleanup -- and answering it by raising from
    ``fileno()`` on a closed file would turn tidying up into a failure.
    """

    def __init__(self, fh) -> None:
        self._fh = fh
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def fd(self) -> int:
        return self._fh.fileno()

    def close(self) -> None:
        self._released = True
        self._fh.close()


def acquire_repo_lock(repo: str) -> RepoLock | None:
    """Take a non-blocking exclusive lock on the live repo for in-place editing.

    In-place mode mutates the shared live repo, so two concurrent forge sessions
    on the same repo would race. The lock serializes them; a caller that cannot
    get it must skip in-place. Returns the held lock (release with
    release_repo_lock) or None when already held.
    """
    lock_path = os.path.join(repo, ".git", "forge_inplace.lock")
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        os.chmod(lock_path, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return RepoLock(fh)


def release_repo_lock(lock: RepoLock | None) -> None:
    """Release + close the in-place repo lock (best-effort, idempotent)."""
    if lock is None or lock.released:
        return
    try:
        fcntl.flock(lock.fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        lock.close()
    except OSError:
        pass


__all__ = [
    "RepoLock",
    "acquire_repo_lock",
    "editable_roots",
    "needs_inplace",
    "release_repo_lock",
]
