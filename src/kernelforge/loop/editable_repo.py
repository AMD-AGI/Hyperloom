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
import os
import re
import site
import sys


def editable_roots() -> list[str]:
    """Collect filesystem roots of PEP 660 editable-finder installs.

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
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            scan_dirs.append(site.getusersitepackages())
        except Exception:
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
    for d in scan_dirs:
        if not d or d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if not n.startswith("__editable__"):
                continue
            if not (n.endswith(".pth") or n.endswith("_finder.py")):
                continue
            fpath = os.path.join(d, n)
            try:
                with open(fpath, errors="replace") as _fh:
                    txt = _fh.read()
            except OSError:
                continue
            # Layout 0: bare absolute path on a line (no quotes, no import).
            for line in txt.splitlines():
                line = line.strip()
                if line.startswith("/") and not line.startswith("#") and "import" not in line and os.path.isdir(line):
                    roots.add(os.path.realpath(line))
            # Layout 1: quoted absolute paths directly in the file.
            for m in re.findall(r"['\"](/[^'\"]+)['\"]", txt):
                if os.path.isdir(m):
                    roots.add(os.path.realpath(m))
            # Layout 2: .pth imports a _finder.py; read its MAPPING dict for
            # paths. The finder file lives next to the .pth in site-packages.
            if n.endswith(".pth"):
                fm = re.search(r"import\s+(__editable___\w+_finder)", txt)
                if fm:
                    finder_file = os.path.join(d, fm.group(1) + ".py")
                    try:
                        with open(finder_file, errors="replace") as _fh2:
                            ftxt = _fh2.read()
                    except OSError:
                        continue
                    for m in re.findall(r"['\"](/[^'\"]+)['\"]", ftxt):
                        if os.path.isdir(m):
                            roots.add(os.path.realpath(m))
    return sorted(roots)


def needs_inplace(kernel_repo: str) -> bool:
    """True when kernel_repo is (or contains/sits under) an editable-finder root.

    In that case forge must edit the live repo in place (the finder imports the
    live path; a worktree copy would be invisible -> the loop would no-op).
    """
    if not kernel_repo:
        return False
    repo = os.path.realpath(kernel_repo)
    for r in editable_roots():
        if r == repo or r.startswith(repo + os.sep) or repo.startswith(r + os.sep):
            return True
    return False


class RepoLock:
    """Owned in-place repo lock; released explicitly after restore."""

    def __init__(self, fh) -> None:
        self._fh = fh

    @property
    def fd(self) -> int:
        return self._fh.fileno()

    def close(self) -> None:
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
    """Release + close the in-place repo lock (best-effort)."""
    if lock is None:
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
