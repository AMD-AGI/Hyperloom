# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which protected file a shell command's path names, when a basename alone cannot tell.

A task that keeps a pristine copy of the kernel as its reference (a monokernel's ``ffn.py``
beside the candidate's) names a protected file and an editable one by one basename. The
in-session gate then resolves a relative path against the directories the command runs in
instead of refusing every path with that basename.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from kernelforge.utils.git import GitError, git_bytes

# Verbs that write only their destination, the last path argument (or ``-t DIR``): their
# sources are read.
_DESTINATION_VERBS = frozenset({"cp", "install", "ln"})


def _bare(token: str) -> str:
    return token.strip().strip("'\"")


class ProtectedNames:
    """The protected files of one workspace, by the names a shell command gives them.

    ``protected_paths``: every protected file, absolute or workspace-relative;
    ``is_protected``: whether a path, absolute or workspace-relative, is a protected file.
    """

    def __init__(
        self,
        workspace_root: Path | None,
        protected_paths: Iterable[str],
        is_protected: Callable[[str], bool],
    ) -> None:
        self.workspace_root = workspace_root
        self.protected_paths = tuple(protected_paths)
        self.protected_basenames = {Path(path).name for path in self.protected_paths}
        self.is_protected = is_protected

    def mention(self, text: str) -> str:
        """Basename of the first protected file this text mentions anywhere, or \"\". A
        basename an editable file shares names neither file on its own, so only that file's
        path counts as a mention of it."""
        lowered = text.lower()
        for path in self.protected_paths:
            name = Path(path).name
            if path.lower() in lowered or (name not in self.shared and name.lower() in lowered):
                return name
        return ""

    @functools.cached_property
    def shared(self) -> frozenset[str]:
        """Basenames a protected file shares with a tracked file the session may edit,
        read once: the tracked tree is the session's starting point."""
        if self.workspace_root is None or not self.protected_basenames:
            return frozenset()
        try:
            names = git_bytes("ls-files", "-z", cwd=self.workspace_root).stdout
        except (GitError, OSError):
            return frozenset()
        shared = set()
        for name in names.split(b"\0"):
            relative = name.decode(errors="surrogateescape")
            basename = Path(relative).name
            if relative and basename in self.protected_basenames:
                if not self.is_protected(str(self.workspace_root / relative)):
                    shared.add(basename)
        return frozenset(shared)

    def by_name(self, path: str, cwds: Sequence[Path]) -> bool:
        """Whether a path no exact rule protects still names a protected file.

        A basename only protected files carry is protected wherever the command runs. One
        an editable file shares too is resolved against ``cwds``, the directories the
        command can run in; with none known it stays protected.
        """
        name = os.path.basename(path)
        if name not in self.protected_basenames:
            return False
        if name not in self.shared:
            return True
        if Path(path).expanduser().is_absolute():
            return False
        if not cwds:
            return True
        return any(self.is_protected(os.path.normpath(os.path.join(cwd, path))) for cwd in cwds)

    def places_into_protected(self, path: str, cwds: Sequence[Path]) -> bool:
        """Whether a path a copy writes inside its destination directory is a protected
        file of this workspace: outside it, no name makes a file a measurement file."""
        candidate = Path(path).expanduser()
        root = self.workspace_root
        if candidate.is_absolute():
            bases = [candidate]
        else:
            bases = [*(cwd / candidate for cwd in cwds), *([root / candidate] if root is not None else [])]
        for base in bases:
            resolved = Path(os.path.normpath(base))
            if root is not None and not resolved.is_relative_to(root):
                continue
            if self.is_protected(str(resolved)):
                return True
        return False

    def is_directory(self, path: str, cwds: Sequence[Path]) -> bool:
        """Whether ``path`` names an existing directory, taken against the command's
        directories and the workspace root when relative."""
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.is_dir()
        bases = [*cwds, *([self.workspace_root] if self.workspace_root is not None else [])]
        return any((base / candidate).is_dir() for base in bases)


def command_cwds(commands: Iterable[tuple[str, list[str]]], cwd: str | None, workspace_root: Path | None) -> list[Path]:
    """The directories a command's relative paths can be taken against: the session's
    working directory and every ``cd`` target among the command's simple commands."""
    base = Path(cwd).expanduser() if cwd else None
    found: list[Path] = [base] if base is not None else []
    anchor = base if base is not None else workspace_root
    for verb, args in commands:
        if verb not in ("cd", "pushd"):
            continue
        target = next((_bare(arg) for arg in args if not arg.startswith("-")), "")
        if not target or "$" in target:
            continue
        path = Path(target).expanduser()
        if path.is_absolute():
            found.append(path)
        elif anchor is not None:
            found.append(anchor / path)
    return list(dict.fromkeys(found))


def _target_directory(args: list[str], paths: list[str]) -> tuple[str | None, list[str]]:
    """A copy's ``-t DIR`` / ``--target-directory=DIR``, and its sources without it."""
    for index, arg in enumerate(args):
        if arg == "-t" and index + 1 < len(args):
            directory = args[index + 1]
            return directory, [path for path in paths if path != directory]
        if arg.startswith("--target-directory="):
            return arg.split("=", 1)[1], paths
    return None, paths


def written_arguments(verb: str, args: list[str], is_directory: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """The paths a shell write verb writes, as (named, placed): ``named`` are its own
    arguments; ``placed`` the paths a copy writes inside a destination directory (one
    ``is_directory`` confirms, or a ``-t`` one), which exist only as that join."""
    paths = [arg for arg in args if not arg.startswith("-")]
    if verb == "dd":
        return [arg[3:] for arg in args if arg.startswith("of=")], []
    if verb not in _DESTINATION_VERBS:
        return paths, []
    directory, sources = _target_directory(args, paths)
    if directory is None:
        if verb == "ln" and len(paths) == 1:
            return [os.path.basename(_bare(paths[0]))], []  # the link lands in the working directory
        if not paths:
            return [], []
        directory, sources = paths[-1], paths[:-1]
        target = _bare(directory)
        if not (target.endswith("/") or target in (".", "..") or is_directory(target)):
            return [target], []
    target = _bare(directory)
    return [target], [os.path.join(target, os.path.basename(_bare(source))) for source in sources]
