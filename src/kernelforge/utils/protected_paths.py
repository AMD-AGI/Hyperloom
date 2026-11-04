# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which paths of a workspace a session must leave as they are, decided by rules its caller supplies."""

from __future__ import annotations

import fnmatch
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _resolved(item: str | Path, root: Path | None) -> Path:
    """``item`` as an absolute path, a relative one taken against ``root`` when there is one."""
    candidate = Path(item).expanduser()
    if candidate.is_absolute() or root is None:
        return candidate.resolve()
    return (root / candidate).resolve()


@dataclass(frozen=True)
class ProtectedPathRules:
    """The paths a caller's sessions must not change, beyond the exact ones it names per session.

    ``globs`` match a file's name or its workspace-relative path; ``dirs`` protect everything below a directory of
    that name; ``unscanned_dirs`` are skipped by :meth:`inventory`, for bookkeeping that changes on its own.
    """

    globs: tuple[str, ...] = ()
    dirs: frozenset[str] = frozenset()
    unscanned_dirs: frozenset[str] = frozenset()

    def is_protected(
        self,
        path: str | Path,
        *,
        workspace: str | Path | None = None,
        exact_paths: Iterable[str | Path] = (),
        extra_globs: Iterable[str] = (),
    ) -> bool:
        """Whether ``path`` is one a session must leave as it is."""
        raw = Path(path).expanduser()
        root = Path(workspace).expanduser().resolve() if workspace else None
        absolute = _resolved(raw, root)
        if absolute in {_resolved(item, root) for item in exact_paths if str(item or "").strip()}:
            return True

        relative = raw
        if root is not None:
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                relative = raw
        if any(part.lower() in self.dirs for part in relative.parts[:-1]):
            return True
        relative_posix = relative.as_posix()
        return any(
            fnmatch.fnmatch(relative.name, pattern) or fnmatch.fnmatch(relative_posix, pattern)
            for pattern in (*self.globs, *extra_globs)
        )

    def inventory(
        self,
        workspace: str | Path,
        *,
        exact_paths: Iterable[str | Path] = (),
        extra_globs: Iterable[str] = (),
    ) -> tuple[Path, ...]:
        """Every protected file or link under ``workspace``, recursively, plus the exact paths named."""
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise OSError(f"protected inventory workspace is not a directory: {root}")
        globs = tuple(extra_globs)
        exact = {_resolved(item, root) for item in exact_paths if str(item or "").strip()}
        inventory = set(exact)

        def raise_walk_error(error: OSError) -> None:
            raise error

        for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=raise_walk_error):
            dirnames[:] = [name for name in dirnames if name not in self.unscanned_dirs]
            parent = Path(directory)
            entries = [*filenames]
            for dirname in dirnames:
                if stat.S_ISLNK((parent / dirname).lstat().st_mode):
                    entries.append(dirname)
            for name in entries:
                candidate = parent / name
                mode = candidate.lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                    continue
                if self.is_protected(candidate, workspace=root, exact_paths=exact, extra_globs=globs):
                    inventory.add(Path(os.path.abspath(candidate)))
        return tuple(sorted(inventory, key=str))


__all__ = ["ProtectedPathRules"]
