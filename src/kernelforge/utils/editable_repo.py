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

from collections.abc import Iterator
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
    """
    roots: set[str] = set()
    for scan_dir in _site_dirs_to_scan():
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("__editable__"):
                continue
            if not (entry.endswith((".pth", "_finder.py"))):
                continue
            roots |= _roots_declared_by(scan_dir, entry)
    return tuple(sorted(roots))


def _site_dirs_to_scan() -> Iterator[str]:
    """Yield each distinct existing directory an editable marker file could sit in."""
    seen: set[str] = set()
    for scan_dir in (*sys.path, *_site_module_dirs(), *_prefix_site_dirs()):
        if not scan_dir or scan_dir in seen or not os.path.isdir(scan_dir):
            continue
        seen.add(scan_dir)
        yield scan_dir


def _site_module_dirs() -> list[str]:
    """The site directories the ``site`` module itself reports, where it can answer."""
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:
        # Absent under a virtualenv built without the site module's framework
        # support, and raising rather than returning empty on some builds. The
        # prefix probing below reaches the same directories, so this is one of
        # several ways to find them rather than the only one.
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            dirs.append(site.getusersitepackages())
        except AttributeError:
            # Same: a user site directory is optional, and its absence says
            # nothing about the roots the rest of the scan will find.
            pass
    return dirs


def _prefix_site_dirs() -> list[str]:
    """Conventional site-packages under every prefix in play.

    Venv / conda site-packages may not appear in ``sys.path``, so the prefixes
    ``sys`` reports, the ones the environment names, and the one the interpreter
    path implies are each probed.
    """
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    prefixes = {sys.prefix, sys.exec_prefix, sys.base_prefix}
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        value = os.environ.get(var)
        if value:
            prefixes.add(value)
    interpreter = os.path.realpath(sys.executable)
    if os.sep + "bin" + os.sep in interpreter:
        prefixes.add(interpreter.rsplit(os.sep + "bin" + os.sep, 1)[0])
    dirs: list[str] = []
    for prefix in prefixes:
        for sub in (f"lib/{version}/site-packages", f"lib/{version}/dist-packages"):
            candidate = os.path.join(prefix, sub)
            if os.path.isdir(candidate):
                dirs.append(candidate)
    return dirs


def _roots_declared_by(scan_dir: str, entry: str) -> set[str]:
    """Absolute repository roots one editable marker file maps into.

    Three layouts reach the same answer: a bare absolute path on its own line,
    quoted absolute paths in the file itself, and a ``.pth`` that imports a
    ``_finder.py`` whose ``MAPPING`` dict holds them.
    """
    text = _read_marker(os.path.join(scan_dir, entry))
    if text is None:
        # This one file, not the scan: a root named by a later finder is still a
        # root, and a swallowed scan would hand a campaign a private checkout of
        # a repository that must be edited in place.
        return set()
    roots = _bare_path_lines(text) | _quoted_dirs(text)
    if entry.endswith(".pth"):
        roots |= _roots_behind_finder_import(scan_dir, text)
    return roots


def _roots_behind_finder_import(scan_dir: str, pth_text: str) -> set[str]:
    """Roots named by the ``_finder.py`` a ``.pth`` imports, which lives beside it."""
    imported = re.search(r"import\s+(__editable___\w+_finder)", pth_text)
    if imported is None:
        return set()
    finder_text = _read_marker(os.path.join(scan_dir, imported.group(1) + ".py"))
    return set() if finder_text is None else _quoted_dirs(finder_text)


def _read_marker(path: str) -> str | None:
    """Read a marker file, or None when it cannot be read."""
    try:
        with open(path, errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _bare_path_lines(text: str) -> set[str]:
    """Existing directories named by a bare absolute path on a line of its own."""
    roots: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("/") and "import" not in line and os.path.isdir(line):
            roots.add(os.path.realpath(line))
    return roots


def _quoted_dirs(text: str) -> set[str]:
    """Existing directories named by a quoted absolute path anywhere in the text."""
    return {os.path.realpath(quoted) for quoted in re.findall(r"['\"](/[^'\"]+)['\"]", text) if os.path.isdir(quoted)}


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
