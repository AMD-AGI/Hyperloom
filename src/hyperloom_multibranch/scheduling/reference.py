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

import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
    source can no longer be read."""

    for source in sources:
        shutil.copytree(Path(source.path), into / source.name, symlinks=False)


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
