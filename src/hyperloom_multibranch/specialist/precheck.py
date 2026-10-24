# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A change checked on the CPU before a test is paid for: the files it writes compile, and its package imports with
the change laid over the installed one.

The import runs in a fresh interpreter that finds a mirror of the installed package first: every file links to the
installed one, the files the change writes are copies from the worktree, and the files it deletes are absent, so the
import sees what a test's environment holds for this package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from hyperloom.measure.compile import ComponentLayout, PackageBase
from hyperloom_multibranch.specialist.worktree import BranchWorktree, Changes

#: Importing a framework loads torch and its kernels; more than this is a hang, not a slow import.
IMPORT_TIMEOUT_S = 300.0
#: The end of a failed import's output, which names the error.
OUTPUT_CHARS = 2000


class CheckFailed(Exception):
    """The change does not compile, or its package does not import with it."""


def check_change(worktree: BranchWorktree, component: str, layout: ComponentLayout, package: PackageBase) -> int:
    """Check the worktree's change to ``component``; returns how many Python files compiled."""
    changes = worktree.changed(component)
    written = [path for path in changes.written if path.endswith(".py")]
    for path in written:
        try:
            compile((worktree.path / path).read_bytes(), path, "exec")
        except (SyntaxError, ValueError) as error:
            raise CheckFailed(f"{path} does not compile: {error}") from error
    with tempfile.TemporaryDirectory(prefix="mb-check-") as scratch:
        _mirror(package.path, Path(scratch) / layout.import_name, worktree.path, layout.upstream_prefix, changes)
        try:
            done = subprocess.run(
                [sys.executable, "-c", f"import {layout.import_name}"],
                cwd=scratch,
                env=_import_env(layout, package),
                capture_output=True,
                timeout=IMPORT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as error:
            raise CheckFailed(f"import {layout.import_name} did not finish in {IMPORT_TIMEOUT_S:g}s") from error
    if done.returncode != 0:
        output = (done.stdout + done.stderr).decode(errors="replace")[-OUTPUT_CHARS:]
        raise CheckFailed(f"import {layout.import_name} failed with the change:\n{output}")
    return len(written)


def _import_env(layout: ComponentLayout, package: PackageBase) -> dict[str, str] | None:
    if layout.import_name != "aiter" or os.environ.get("AITER_META_DIR"):
        return None
    # aiter imports csrc from its meta dir, which it derives from where the package sits; the mirror would make it the
    # scratch dir, which has no csrc, so it is pinned to the image's.
    return {**os.environ, "AITER_META_DIR": str(_aiter_meta_dir(package.path))}


def _aiter_meta_dir(package: Path) -> Path:
    """The image's aiter meta dir, resolved as aiter/jit/core.py does: the package's parent for a develop install,
    its sibling ``aiter_meta`` for a wheel."""
    try:
        develop = (package / "install_mode").read_text().strip() == "develop"
    except FileNotFoundError:
        develop = True
    return package.parent if develop else package.parent / "aiter_meta"


def _mirror(installed: Path, target: Path, worktree: Path, prefix: str, changes: Changes) -> None:
    """Lay the installed package out at ``target`` with the change over it; the change's files beside the package (a
    meta tree such as aiter's csrc) are not part of its import."""
    deleted = {_inside(path, prefix) for path in changes.deleted if _within(path, prefix)}
    written = {_inside(path, prefix): worktree / path for path in changes.written if _within(path, prefix)}
    for source in installed.rglob("*"):
        rel = source.relative_to(installed).as_posix()
        if source.is_dir() or "__pycache__" in source.parts or rel in deleted or rel in written:
            continue
        _link(source, target / rel)
    for rel, source in written.items():
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / rel)


def _inside(path: str, prefix: str) -> str:
    return str(PurePosixPath(path).relative_to(prefix))


def _within(path: str, prefix: str) -> bool:
    return PurePosixPath(prefix) in PurePosixPath(path).parents


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination)
    except OSError:
        # Linking may need a privilege the host does not grant; a copy holds the same bytes.
        shutil.copy2(source, destination)


__all__ = ["IMPORT_TIMEOUT_S", "OUTPUT_CHARS", "CheckFailed", "check_change"]
