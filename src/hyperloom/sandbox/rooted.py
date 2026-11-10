# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A path below a sandbox's root, opened as one walk that no link it meets can lead out of.

The file API serves paths the sandbox's own processes write: a check of where a path resolves followed by an open of
the same name leaves them the moment between the two to put a link to the app's credentials where the checked file
was. Here nothing is opened by a name that is resolved twice: each component is opened relative to the descriptor of
the directory it is in, with ``O_NOFOLLOW``; a link met on the way is read and its target walked the same way, from
that directory (or from the root, for an absolute target naming the root's own path), and refused where it would
leave the root. What the walk opens last is what the caller reads or writes.
"""

from __future__ import annotations

import errno
import os
from collections import deque
from pathlib import Path

from hyperloom.sandbox.socket import SandboxDenied

__all__ = ["MAX_LINKS", "open_rooted"]

#: How many links one walk follows, as the kernel allows one path resolution (``ELOOP`` past it).
MAX_LINKS = 40
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_STEP = os.O_NOFOLLOW | os.O_CLOEXEC
#: What opening a link's own name with ``O_NOFOLLOW`` fails with: ``ELOOP``, or ``ENOTDIR`` with ``O_DIRECTORY``.
_A_LINK_MAYBE = (errno.ELOOP, errno.ENOTDIR)
_PARENT = ".."


def _parts(path: str) -> list[str]:
    return [part for part in path.split("/") if part not in ("", ".")]


def _escape(virtual: str) -> SandboxDenied:
    return SandboxDenied(f"path {virtual!r} resolves outside the sandbox")


class _Walk:
    """One walk from ``root``: the descriptors of the directories it is in, the root's first, and the links followed."""

    def __init__(self, root: Path, virtual: str) -> None:
        self.root = root
        self.virtual = virtual
        self.roots = [_parts(os.path.abspath(root)), _parts(os.path.realpath(root))]
        self.stack: list[int] = []
        self.links = 0

    def __enter__(self) -> _Walk:
        """The walk from the root, whose descriptor is opened here and closed on exit with every other."""
        self.stack.append(os.open(self.root, _DIRECTORY))
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        while self.stack:
            os.close(self.stack.pop())

    def up(self) -> None:
        """``..``: the directory above, never above the root."""
        if len(self.stack) == 1:
            raise _escape(self.virtual)
        os.close(self.stack.pop())

    def target(self, name: str) -> list[str] | None:
        """The components the link ``name`` (in the current directory) names, from where the walk goes on; ``None``
        when ``name`` is not a link. An absolute target is taken from the root, and only where it names the root's own
        path, as given or resolved."""
        try:
            target = os.readlink(name, dir_fd=self.stack[-1])
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                return None
            raise
        self.links += 1
        if self.links > MAX_LINKS:
            raise SandboxDenied(f"path {self.virtual!r} follows more than {MAX_LINKS} links")
        if not target.startswith("/"):
            return _parts(target)
        parts = _parts(target)
        inside = [parts[len(root) :] for root in self.roots if parts[: len(root)] == root]
        if not inside:
            raise _escape(self.virtual)
        while len(self.stack) > 1:
            os.close(self.stack.pop())
        return inside[0]


def open_rooted(root: Path, virtual: str, flags: int, *, make_parents: bool = False, mode: int = 0o600) -> int:
    """A descriptor of ``virtual`` (a path in the root, ``/`` its top) below the directory ``root``, opened with
    ``flags`` (plus ``O_NOFOLLOW``; a file it creates gets ``mode`` less the umask, owner-only by default), by a walk
    no link leads out of the root (see the module doc). ``make_parents`` makes each directory that is missing on the
    way. The caller closes the descriptor.

    Raises:
        SandboxDenied: A link on the way leads out of the root, or the walk follows more than :data:`MAX_LINKS`.
        OSError: The open itself failed (``FileNotFoundError`` for a missing component, ``NotADirectoryError`` for a
            file on the way, ``IsADirectoryError`` for a directory opened to write).
    """
    pending = deque(_parts(virtual))
    made: tuple[int, str] | None = None
    with _Walk(root, virtual) as walk:
        while pending:
            name = pending.popleft()
            if name == _PARENT:
                walk.up()
                continue
            last = not pending
            try:
                opened = os.open(name, (flags if last else _DIRECTORY) | _STEP, mode, dir_fd=walk.stack[-1])
            except FileNotFoundError:
                if last or not make_parents or made == (len(walk.stack), name):
                    raise
                try:
                    os.mkdir(name, dir_fd=walk.stack[-1])
                except FileExistsError:
                    pass  # made meanwhile, or a link left dangling: the next step opens or follows it
                made = (len(walk.stack), name)
                pending.appendleft(name)
                continue
            except OSError as exc:
                followed = walk.target(name) if exc.errno in _A_LINK_MAYBE else None
                if followed is None:
                    raise
                pending.extendleft(reversed(followed))
                continue
            if last:
                return opened
            walk.stack.append(opened)
            made = None
        return os.open(".", flags | os.O_CLOEXEC, dir_fd=walk.stack[-1])
