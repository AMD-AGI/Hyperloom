# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make a private checkout of an installed package the one a campaign's processes import.

A repository the host snapshots from an installed package holds that package's files at its root: there is no
directory named after the package above them, and the package's compiled extension modules are not in the snapshot.
A driver that imports the package by name therefore loads the installed copy, and the campaign measures the
unmodified original whatever it edits.

:func:`shadow_served_package` puts the checkout under the package's import name in a directory of its own, links the
installed extension modules into the checkout so the package still imports, and checks that the name now resolves to
the checkout. The directory it returns goes first on the ``PYTHONPATH`` of everything the campaign runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kernelforge.utils.git import git

#: The compiled extension modules a source snapshot leaves out.
_EXTENSION_GLOBS = ("*.so", "*.so.*")
#: What importing the checkout puts in it and no patch may carry: the links below, and the interpreter's bytecode.
_IGNORED = ("__pycache__/", *_EXTENSION_GLOBS)
_RESOLVE_ORIGIN = "import importlib.util, sys; spec = importlib.util.find_spec(sys.argv[1]); print(spec.origin or '')"
_RESOLVE_TIMEOUT_SEC = 60


class ServedPackageError(RuntimeError):
    """A checkout could not be made the package its campaign imports."""


def _link_extension_modules(workspace: Path, served_tree: Path) -> None:
    """Link each extension module of ``served_tree`` into ``workspace`` at the same relative path."""
    for pattern in _EXTENSION_GLOBS:
        for module in served_tree.rglob(pattern):
            link = workspace / module.relative_to(served_tree)
            if link.exists() or link.is_symlink():
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(module)


def _hide_from_git(workspace: Path) -> None:
    """Keep the links and the bytecode out of the checkout's status, so neither reaches a patch or the guard."""
    exclude = Path(git("rev-parse", "--git-path", "info/exclude", cwd=workspace).stdout.strip())
    exclude = exclude if exclude.is_absolute() else workspace / exclude
    listed = exclude.read_text(encoding="utf-8").splitlines() if exclude.is_file() else []
    missing = [pattern for pattern in _IGNORED if pattern not in listed]
    if missing:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("".join(f"{line}\n" for line in (*listed, *missing)), encoding="utf-8")


def _require_resolves_to(package: str, import_root: Path, workspace: Path) -> None:
    """Refuse a checkout the interpreter would not import ``package`` from, rather than measure another tree."""
    inherited = os.environ.get("PYTHONPATH", "")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(import_root), inherited)))}
    resolved = subprocess.run(
        [sys.executable, "-c", _RESOLVE_ORIGIN, package],
        env=env,
        capture_output=True,
        text=True,
        timeout=_RESOLVE_TIMEOUT_SEC,
        check=False,
    )
    origin = resolved.stdout.strip()
    if resolved.returncode != 0 or not origin or not Path(origin).resolve().is_relative_to(workspace.resolve()):
        found = origin or resolved.stderr.strip()[-400:] or "nothing"
        raise ServedPackageError(
            f"`import {package}` does not resolve to the campaign checkout {workspace} (found {found}); "
            "the campaign would measure a tree it does not edit"
        )


def shadow_served_package(workspace: Path, served_tree: Path, import_root: Path) -> Path | None:
    """Make ``workspace`` what ``import <package>`` loads, where ``<package>`` is the one served from ``served_tree``.

    Returns ``import_root``, the directory to put first on ``PYTHONPATH``, or ``None`` when ``served_tree`` is not a
    package: a tree nothing imports by name has nothing to shadow.

    Raises:
        ServedPackageError: The package's name still resolves outside ``workspace``.
    """
    if not (served_tree / "__init__.py").is_file():
        return None
    try:
        _link_extension_modules(workspace, served_tree)
        _hide_from_git(workspace)
        import_root.mkdir(parents=True, exist_ok=True)
        (import_root / served_tree.name).symlink_to(workspace, target_is_directory=True)
    except OSError as error:
        raise ServedPackageError(f"could not shadow {served_tree} with {workspace}: {error}") from error
    _require_resolves_to(served_tree.name, import_root, workspace)
    return import_root


__all__ = ["ServedPackageError", "shadow_served_package"]
