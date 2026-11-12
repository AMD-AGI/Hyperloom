# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reference material a run is given: directories outside the session that its specialists read and no phase
writes to, such as GEAK's tuning and kernel knowledge.

Each source is named on the command line and copied once, before the first phase, into ``MB-reference/<name>``. The
copy is what every branch reads, so nothing is fetched over the network once a session is running and a directory
that moves outside cannot change what this session read halfway through. A source carries the commit it was taken
at, which the manifest records with the rest of the run's settings, so a session says exactly which knowledge it
read. A session never refreshes a copy: knowledge that lands after it starts is what the next session takes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

GIT_TIMEOUT_SECONDS = 60.0
#: A name is the directory the copy lives in and the path the specialists are given, so it is one lowercase word or
#: several joined by hyphens.
REFERENCE_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class ReferenceSource:
    """A directory the session copies in under ``name``, and the commit of the checkout it was taken from.

    ``path`` is absolute, and a string because the manifest records the run's settings as they are given."""

    name: str
    path: str
    revision: str


def reference_source(spec: str) -> ReferenceSource:
    """The source ``NAME=DIR`` spells, with the commit ``DIR`` is checked out at. Blocks on git; raises ValueError
    where the name is no directory name, or ``DIR`` is no readable directory in a git checkout."""

    name, _, directory = spec.partition("=")
    if not REFERENCE_NAME.fullmatch(name):
        raise ValueError(f"--reference {spec} is not NAME=DIR, with NAME lowercase words joined by hyphens")
    path = Path(directory).expanduser()
    if not directory or not path.is_dir():
        raise ValueError(f"--reference {name} names {directory or 'nothing'}, which is no readable directory")
    return ReferenceSource(name, str(path.resolve()), _revision(path))


def copy_references(sources: Iterable[ReferenceSource], into: Path) -> None:
    """Copy each source into ``into/<name>``, following no symlink out of it. Blocks; raises ``OSError`` where a
    source can no longer be read.

    The copy is shown to every sandboxed specialist, so what a link in the source points at is never copied: a link
    that resolves to a file or directory inside the source becomes a relative link to that place in the copy, and a
    link that resolves outside the source, or to nothing, is left out with a WARNING. Anything that is neither a
    regular file, a directory nor a link, such as a FIFO, has no contents to copy and is left out too."""

    for source in sources:
        root = Path(os.path.realpath(source.path))
        left_out: list[str] = []
        _copy_dir(root, into / source.name, root, left_out)
        if left_out:
            logger.warning(
                "reference %s: %d entries left out of the copy, because each is a link out of %s or to nothing, or "
                "no regular file: %s",
                source.name,
                len(left_out),
                source.path,
                ", ".join(left_out[:5]) + (", ..." if len(left_out) > 5 else ""),
            )


def _copy_dir(root: Path, top: Path, directory: Path, left_out: list[str]) -> None:
    """Copy ``directory``, which is ``root`` or under it, to its place in the copy of ``root`` at ``top``, adding
    what it leaves out to ``left_out``."""

    copy = top / directory.relative_to(root)
    with os.scandir(directory) as entries:
        copy.mkdir()
        for entry in sorted(entries, key=lambda e: e.name):
            path = directory / entry.name
            if entry.is_symlink():
                if not _relink(root, top, path):
                    left_out.append(str(path.relative_to(root)))
            elif entry.is_dir(follow_symlinks=False):
                _copy_dir(root, top, path, left_out)
            elif entry.is_file(follow_symlinks=False):
                shutil.copy2(path, copy / entry.name, follow_symlinks=False)
            else:
                left_out.append(str(path.relative_to(root)))
    shutil.copystat(directory, copy, follow_symlinks=False)


def _relink(root: Path, top: Path, link: Path) -> bool:
    """Make ``link``'s place in the copy of ``root`` at ``top`` a relative link to where ``link`` resolves in that
    copy; False, making nothing, where it resolves outside ``root`` or to nothing."""

    target = Path(os.path.realpath(link))
    if not os.path.lexists(target) or not target.is_relative_to(root):
        return False
    copy = top / link.relative_to(root)
    os.symlink(os.path.relpath(top / target.relative_to(root), copy.parent), copy)
    return True


def _revision(path: Path) -> str:
    """The commit the checkout holding ``path`` is on; raises ValueError where git cannot say."""

    try:
        completed = subprocess.run(  # nosec B603 B607
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"--reference {path} is in no git checkout, so what it holds cannot be pinned") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"--reference {path}: the commit it is on cannot be read: {exc}") from exc
    return completed.stdout.strip()


__all__ = ["REFERENCE_NAME", "ReferenceSource", "copy_references", "reference_source"]
